use crate::package::{
    BackendAffinity, CostEstimate, CostUnit, KapslPackageSpec, MetricsToggles, ModelFormat,
    OnnxRuntimeTuningSpec, PackageThermalThresholds, Priority, QueueOverflowPolicy, SwapPolicy,
    TaskClass, TriggerMode,
};
use fs2::available_space;
use kapsl_core::loader::{LoaderError as CoreLoaderError, Manifest};
use kapsl_core::PackageLoader;
use serde_json::{Map, Value};
use std::collections::HashSet;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Errors that can occur while loading a `.aimod` package.
#[derive(Debug)]
pub enum LoadError {
    Io(io::Error),
    Package(CoreLoaderError),
    ModelFileMissing {
        package: PathBuf,
        model_file: String,
    },
    InsufficientDiskSpace {
        cache_root: PathBuf,
        required_copy_bytes: u64,
        available_bytes: u64,
        reserved_free_bytes: u64,
        cache_usage_bytes: u64,
        max_cache_bytes: Option<u64>,
    },
    InvalidMetadata {
        field: String,
        reason: String,
    },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error while loading package: {e}"),
            Self::Package(e) => write!(f, "invalid .aimod package: {e}"),
            Self::ModelFileMissing {
                package,
                model_file,
            } => write!(
                f,
                "package {} references missing model file '{}'",
                package.display(),
                model_file
            ),
            Self::InsufficientDiskSpace {
                cache_root,
                required_copy_bytes,
                available_bytes,
                reserved_free_bytes,
                cache_usage_bytes,
                max_cache_bytes,
            } => write!(
                f,
                "insufficient disk space for model cache at {} (required_copy={}B available={}B reserved_free={}B cache_usage={}B cache_limit={})",
                cache_root.display(),
                required_copy_bytes,
                available_bytes,
                reserved_free_bytes,
                cache_usage_bytes,
                max_cache_bytes
                    .map(|value| format!("{value}B"))
                    .unwrap_or_else(|| "none".to_string())
            ),
            Self::InvalidMetadata { field, reason } => {
                write!(f, "invalid metadata field '{}': {}", field, reason)
            }
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Package(e) => Some(e),
            Self::ModelFileMissing { .. }
            | Self::InsufficientDiskSpace { .. }
            | Self::InvalidMetadata { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CachePolicy {
    max_cache_bytes: Option<u64>,
    reserved_free_bytes: u64,
}

impl CachePolicy {
    fn from_env() -> Self {
        Self {
            max_cache_bytes: read_bytes_env(
                &[
                    "KAPSL_LITE_MODEL_CACHE_MAX_BYTES",
                    "KAPSL_MODEL_CACHE_MAX_BYTES",
                ],
                &[
                    "KAPSL_LITE_MODEL_CACHE_MAX_MIB",
                    "KAPSL_MODEL_CACHE_MAX_MIB",
                ],
            ),
            reserved_free_bytes: read_bytes_env(
                &[
                    "KAPSL_LITE_MODEL_CACHE_RESERVED_FREE_BYTES",
                    "KAPSL_MODEL_CACHE_RESERVED_FREE_BYTES",
                ],
                &[
                    "KAPSL_LITE_MODEL_CACHE_RESERVED_FREE_MIB",
                    "KAPSL_MODEL_CACHE_RESERVED_FREE_MIB",
                ],
            )
            .unwrap_or(0),
        }
    }
}

#[derive(Debug, Clone)]
struct CacheEntry {
    path: PathBuf,
    bytes: u64,
    modified_ns: u128,
}

#[derive(Debug, Clone)]
struct ModelAsset {
    source_path: PathBuf,
    target_name: String,
}

impl From<io::Error> for LoadError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<CoreLoaderError> for LoadError {
    fn from(e: CoreLoaderError) -> Self {
        Self::Package(e)
    }
}

/// Loads and validates a `.aimod` archive package and derives a lite runtime
/// scheduling profile from its `metadata.json`.
pub fn load_package(package_path: &Path) -> Result<KapslPackageSpec, LoadError> {
    let loader = PackageLoader::load(package_path)?;
    let model_path = loader.get_model_path();
    if !model_path.exists() || !model_path.is_file() {
        return Err(LoadError::ModelFileMissing {
            package: package_path.to_path_buf(),
            model_file: loader.manifest.model_file.clone(),
        });
    }
    let persisted_model_path =
        persist_model_file(package_path, &loader.manifest.model_file, &model_path)?;

    manifest_to_spec(&loader.manifest, package_path, &persisted_model_path)
}

fn persist_model_file(
    package_path: &Path,
    model_file: &str,
    source_model_path: &Path,
) -> Result<PathBuf, LoadError> {
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();
    package_path.hash(&mut hasher);
    model_file.hash(&mut hasher);
    if let Ok(meta) = std::fs::metadata(package_path) {
        meta.len().hash(&mut hasher);
        if let Ok(modified) = meta.modified()
            && let Ok(duration) = modified.duration_since(UNIX_EPOCH)
        {
            duration.as_nanos().hash(&mut hasher);
        }
    }
    let package_hash = hasher.finish();

    let file_name = Path::new(model_file)
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("model.bin");
    let cache_root = resolve_model_cache_root(package_path);
    let cache_dir = cache_root.join(format!("{:016x}", package_hash));
    let cache_policy = CachePolicy::from_env();
    match persist_model_assets(
        source_model_path,
        &cache_root,
        &cache_dir,
        file_name,
        cache_policy,
    ) {
        Ok(path) => Ok(path),
        Err(LoadError::Io(error)) if is_windows_mapped_section_error(&error) => {
            let fallback_cache_dir = cache_root.join(format!(
                "{:016x}-{:x}-{:x}",
                package_hash,
                std::process::id(),
                unique_cache_nonce()
            ));
            persist_model_assets(
                source_model_path,
                &cache_root,
                &fallback_cache_dir,
                file_name,
                cache_policy,
            )
        }
        Err(error) => Err(error),
    }
}

fn resolve_model_cache_root(package_path: &Path) -> PathBuf {
    for key in ["KAPSL_LITE_MODEL_CACHE_DIR", "KAPSL_MODEL_CACHE_DIR"] {
        if let Some(value) = std::env::var_os(key)
            && !value.is_empty()
        {
            return PathBuf::from(value);
        }
    }

    if let Some(parent) = package_path.parent() {
        return parent.join(".kapsl-lite-model-cache");
    }

    std::env::temp_dir().join("kapsl-lite-model-cache")
}

fn persist_model_assets(
    source_model_path: &Path,
    cache_root: &Path,
    cache_dir: &Path,
    file_name: &str,
    cache_policy: CachePolicy,
) -> Result<PathBuf, LoadError> {
    fs::create_dir_all(cache_root).map_err(LoadError::Io)?;
    fs::create_dir_all(cache_dir).map_err(LoadError::Io)?;

    let assets = collect_model_assets(source_model_path, file_name).map_err(LoadError::Io)?;
    let required_copy_bytes =
        estimate_required_copy_bytes(&assets, cache_dir).map_err(LoadError::Io)?;

    enforce_cache_policy(
        cache_root,
        cache_dir,
        required_copy_bytes,
        cache_policy,
        |path| available_space(path),
    )?;

    for asset in assets {
        copy_if_needed(&asset.source_path, &cache_dir.join(asset.target_name))
            .map_err(LoadError::Io)?;
    }

    Ok(cache_dir.join(file_name))
}

fn collect_model_assets(
    source_model_path: &Path,
    model_target_name: &str,
) -> io::Result<Vec<ModelAsset>> {
    let mut assets = Vec::new();
    let mut seen_targets = HashSet::new();

    assets.push(ModelAsset {
        source_path: source_model_path.to_path_buf(),
        target_name: model_target_name.to_string(),
    });
    seen_targets.insert(model_target_name.to_string());

    let Some(source_dir) = source_model_path.parent() else {
        return Ok(assets);
    };
    let main_model_name = source_model_path.file_name();

    for entry in fs::read_dir(source_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path.file_name() == main_model_name {
            continue;
        }
        let target_name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        if target_name.is_empty() || !seen_targets.insert(target_name.clone()) {
            continue;
        }
        assets.push(ModelAsset {
            source_path: path,
            target_name,
        });
    }

    Ok(assets)
}

fn estimate_required_copy_bytes(assets: &[ModelAsset], cache_dir: &Path) -> io::Result<u64> {
    let mut required_bytes = 0u64;
    for asset in assets {
        let source_meta = fs::metadata(&asset.source_path)?;
        let target_path = cache_dir.join(&asset.target_name);
        if should_copy_file(source_meta.len(), &target_path)? {
            required_bytes = required_bytes.saturating_add(source_meta.len());
        }
    }
    Ok(required_bytes)
}

fn enforce_cache_policy<F>(
    cache_root: &Path,
    protected_cache_dir: &Path,
    required_copy_bytes: u64,
    cache_policy: CachePolicy,
    mut query_available_space: F,
) -> Result<(), LoadError>
where
    F: FnMut(&Path) -> io::Result<u64>,
{
    if cache_policy.max_cache_bytes.is_none() && cache_policy.reserved_free_bytes == 0 {
        return Ok(());
    }

    fs::create_dir_all(cache_root).map_err(LoadError::Io)?;
    let mut entries =
        load_cache_entries(cache_root, Some(protected_cache_dir)).map_err(LoadError::Io)?;
    let mut cache_usage_bytes = directory_size_bytes(cache_root).map_err(LoadError::Io)?;

    loop {
        let available_bytes = query_available_space(cache_root).map_err(LoadError::Io)?;
        let required_available_bytes = cache_policy
            .reserved_free_bytes
            .saturating_add(required_copy_bytes);
        let free_ok = available_bytes >= required_available_bytes;
        let cap_ok = match cache_policy.max_cache_bytes {
            Some(max_bytes) => cache_usage_bytes.saturating_add(required_copy_bytes) <= max_bytes,
            None => true,
        };

        if free_ok && cap_ok {
            return Ok(());
        }

        let Some(entry) = entries.first().cloned() else {
            return Err(LoadError::InsufficientDiskSpace {
                cache_root: cache_root.to_path_buf(),
                required_copy_bytes,
                available_bytes,
                reserved_free_bytes: cache_policy.reserved_free_bytes,
                cache_usage_bytes,
                max_cache_bytes: cache_policy.max_cache_bytes,
            });
        };
        entries.remove(0);
        remove_cache_entry(&entry.path).map_err(LoadError::Io)?;
        cache_usage_bytes = cache_usage_bytes.saturating_sub(entry.bytes);
    }
}

fn load_cache_entries(
    cache_root: &Path,
    protected_path: Option<&Path>,
) -> io::Result<Vec<CacheEntry>> {
    if !cache_root.exists() {
        return Ok(Vec::new());
    }

    let mut entries = Vec::new();
    for entry in fs::read_dir(cache_root)? {
        let entry = entry?;
        let path = entry.path();
        if protected_path.is_some_and(|protected| protected == path) {
            continue;
        }
        let metadata = entry.metadata()?;
        let bytes = if metadata.is_dir() {
            directory_size_bytes(&path)?
        } else if metadata.is_file() {
            metadata.len()
        } else {
            0
        };
        entries.push(CacheEntry {
            path,
            bytes,
            modified_ns: modified_ns(&metadata),
        });
    }

    entries.sort_by(|a, b| {
        a.modified_ns
            .cmp(&b.modified_ns)
            .then_with(|| a.path.cmp(&b.path))
    });
    Ok(entries)
}

fn remove_cache_entry(path: &Path) -> io::Result<()> {
    let metadata = match fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };

    let removal = if metadata.is_dir() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    };
    match removal {
        Ok(_) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn directory_size_bytes(path: &Path) -> io::Result<u64> {
    let metadata = match fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(error),
    };
    if metadata.is_file() {
        return Ok(metadata.len());
    }
    if !metadata.is_dir() {
        return Ok(0);
    }

    let mut total = 0u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let child_path = entry.path();
        let child_meta = entry.metadata()?;
        if child_meta.is_dir() {
            total = total.saturating_add(directory_size_bytes(&child_path)?);
        } else if child_meta.is_file() {
            total = total.saturating_add(child_meta.len());
        }
    }
    Ok(total)
}

fn modified_ns(metadata: &fs::Metadata) -> u128 {
    metadata
        .modified()
        .ok()
        .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
        .map(|value| value.as_nanos())
        .unwrap_or(0)
}

fn copy_if_needed(source: &Path, destination: &Path) -> io::Result<()> {
    let source_meta = fs::metadata(source)?;
    if !should_copy_file(source_meta.len(), destination)? {
        return Ok(());
    }
    fs::copy(source, destination)?;
    Ok(())
}

fn should_copy_file(source_len: u64, destination: &Path) -> io::Result<bool> {
    match fs::metadata(destination) {
        Ok(destination_meta) => Ok(!(destination_meta.len() == source_len && source_len > 0)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(true),
        Err(error) => Err(error),
    }
}

fn read_bytes_env(byte_keys: &[&str], mib_keys: &[&str]) -> Option<u64> {
    read_u64_env(byte_keys)
        .or_else(|| read_u64_env(mib_keys).map(|value| value.saturating_mul(1024 * 1024)))
}

fn read_u64_env(keys: &[&str]) -> Option<u64> {
    for key in keys {
        let Ok(raw) = std::env::var(key) else {
            continue;
        };
        let Ok(value) = raw.trim().parse::<u64>() else {
            continue;
        };
        return Some(value);
    }
    None
}

fn unique_cache_nonce() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0)
}

fn is_windows_mapped_section_error(error: &io::Error) -> bool {
    error.raw_os_error() == Some(1224)
}

/// Scans a directory for `*.aimod` packages and loads each one.
///
/// Invalid packages are skipped with a warning so one broken artifact does not
/// block the whole lite runtime startup.
pub fn load_packages_from_dir(packages_dir: &Path) -> Result<Vec<KapslPackageSpec>, LoadError> {
    let entries = std::fs::read_dir(packages_dir)?;
    let mut specs = Vec::new();

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        if path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("aimod"))
            != Some(true)
        {
            continue;
        }

        match load_package(&path) {
            Ok(spec) => specs.push(spec),
            Err(e) => eprintln!(
                "Warning: Failed to load package from {}: {}",
                path.display(),
                e
            ),
        }
    }

    // Sort by priority descending so high-priority packages load first.
    specs.sort_by(|a, b| b.priority.cmp(&a.priority));

    Ok(specs)
}

fn manifest_to_spec(
    manifest: &Manifest,
    package_path: &Path,
    model_path: &Path,
) -> Result<KapslPackageSpec, LoadError> {
    let metadata = match manifest.metadata.as_ref() {
        Some(value) => serde_json::to_value(value).map_err(|e| LoadError::InvalidMetadata {
            field: "metadata".to_string(),
            reason: format!("failed to convert metadata to JSON object: {}", e),
        })?,
        None => Value::Object(Map::new()),
    };

    if !metadata.is_object() {
        return Err(LoadError::InvalidMetadata {
            field: "metadata".to_string(),
            reason: "expected an object".to_string(),
        });
    }

    let model_file = manifest.model_file.trim();
    if model_file.is_empty() {
        return Err(LoadError::InvalidMetadata {
            field: "model_file".to_string(),
            reason: "manifest model_file is empty".to_string(),
        });
    }

    let format = infer_model_format(&manifest.framework, model_file);
    let memory_mb = u64_field(&metadata, "memory_mb")?
        .or(manifest.hardware_requirements.min_memory_mb)
        .unwrap_or(default_memory_mb(format));
    let priority = match string_field(&metadata, "priority")? {
        Some(value) => parse_priority(value)?,
        None => Priority::Normal,
    };
    let trigger_mode = match string_field(&metadata, "trigger_mode")? {
        Some(value) => parse_trigger_mode(value)?,
        None => TriggerMode::AlwaysRunning,
    };
    let task_class = match string_field(&metadata, "task_class")? {
        Some(value) => parse_task_class(value)?,
        None => default_task_class(trigger_mode),
    };
    let preemptible = bool_field(&metadata, "preemptible")?.unwrap_or(true);
    let max_concurrent = u64_field(&metadata, "max_concurrent")?.unwrap_or(1).max(1) as usize;
    let on_demand_queue_capacity = u64_field(&metadata, "on_demand_queue_capacity")?
        .unwrap_or(64)
        .max(1) as usize;
    let on_demand_queue_policy = match string_field(&metadata, "on_demand_queue_policy")? {
        Some(value) => parse_queue_overflow_policy(value)?,
        None => QueueOverflowPolicy::DropNewest,
    };
    let cpu_threads = u64_field(&metadata, "cpu_threads")?.unwrap_or(1).max(1) as usize;
    let hardware_target = string_field(&metadata, "hardware_target")?
        .map(str::to_string)
        .or_else(|| manifest.hardware_requirements.preferred_provider.clone())
        .unwrap_or_default();
    let backend_affinity = match string_field(&metadata, "backend_affinity")? {
        Some(value) => parse_backend_affinity(value)?,
        None => infer_backend_affinity_from_hint(&hardware_target),
    };
    let cost_estimate = match metadata_field(&metadata, "cost_estimate") {
        Some(value) => Some(parse_cost_estimate(value)?),
        None => None,
    };
    let swap = match metadata_field(&metadata, "swap") {
        Some(value) => parse_swap(value)?,
        None => SwapPolicy::Disallowed,
    };
    let thermal = parse_thermal(&metadata)?;
    let metrics = parse_metrics(&metadata)?;
    let onnx_tuning = parse_onnx_tuning(&metadata, max_concurrent)?;
    let fps_target = f64_field(&metadata, "fps_target")?;
    let deadline_ms = u64_field(&metadata, "deadline_ms")?.or_else(|| {
        if task_class == TaskClass::Realtime {
            let fps = fps_target.unwrap_or(1.0).max(0.1);
            Some((1000.0 / fps).round().max(1.0) as u64)
        } else {
            None
        }
    });
    let critical_perception = bool_field(&metadata, "critical_perception")?.unwrap_or(false);
    let name = manifest
        .project_name
        .trim()
        .to_string()
        .chars()
        .next()
        .map(|_| manifest.project_name.trim().to_string())
        .unwrap_or_else(|| {
            package_path
                .file_stem()
                .and_then(|v| v.to_str())
                .unwrap_or("model")
                .to_string()
        });

    Ok(KapslPackageSpec {
        name,
        format,
        is_llm: manifest.framework.trim().eq_ignore_ascii_case("llm"),
        weights: model_file.to_string(),
        weights_path: model_path.to_string_lossy().to_string(),
        memory_mb,
        priority,
        task_class,
        trigger_mode,
        preemptible,
        max_concurrent,
        on_demand_queue_capacity,
        on_demand_queue_policy,
        cpu_threads,
        cost_estimate,
        deadline_ms,
        backend_affinity,
        hardware_target,
        critical_perception,
        swap,
        thermal,
        metrics,
        fps_target,
        onnx_tuning,
    })
}

fn parse_queue_overflow_policy(value: &str) -> Result<QueueOverflowPolicy, LoadError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "drop_oldest" | "drop-oldest" => Ok(QueueOverflowPolicy::DropOldest),
        "drop_newest" | "drop-newest" | "latest_only" | "latest-only" => {
            Ok(QueueOverflowPolicy::DropNewest)
        }
        "block" | "blocking" => Ok(QueueOverflowPolicy::Block),
        other => Err(LoadError::InvalidMetadata {
            field: "on_demand_queue_policy".to_string(),
            reason: format!(
                "expected one of drop_oldest|drop_newest|block, got '{}'",
                other
            ),
        }),
    }
}

fn infer_model_format(framework: &str, model_file: &str) -> ModelFormat {
    let framework = framework.trim().to_ascii_lowercase();
    let model_ext = Path::new(model_file)
        .extension()
        .and_then(|v| v.to_str())
        .map(|ext| ext.to_ascii_lowercase());
    if framework == "onnx" {
        return ModelFormat::Onnx;
    }
    if framework == "llm" && model_ext.as_deref() == Some("onnx") {
        return ModelFormat::Onnx;
    }
    if framework == "llm" || framework == "gguf" {
        return ModelFormat::Gguf;
    }

    if model_ext.as_deref() == Some("gguf") {
        ModelFormat::Gguf
    } else {
        ModelFormat::Onnx
    }
}

fn default_memory_mb(format: ModelFormat) -> u64 {
    match format {
        ModelFormat::Gguf => 512,
        ModelFormat::Onnx => 256,
    }
}

fn parse_priority(value: &str) -> Result<Priority, LoadError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "low" => Ok(Priority::Low),
        "normal" => Ok(Priority::Normal),
        "high" => Ok(Priority::High),
        other => Err(LoadError::InvalidMetadata {
            field: "priority".to_string(),
            reason: format!("expected one of low|normal|high, got '{}'", other),
        }),
    }
}

fn parse_trigger_mode(value: &str) -> Result<TriggerMode, LoadError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "always_running" | "always-running" | "alwaysrunning" => Ok(TriggerMode::AlwaysRunning),
        "on_demand" | "on-demand" | "ondemand" => Ok(TriggerMode::OnDemand),
        other => Err(LoadError::InvalidMetadata {
            field: "trigger_mode".to_string(),
            reason: format!("expected one of always_running|on_demand, got '{}'", other),
        }),
    }
}

fn default_task_class(trigger_mode: TriggerMode) -> TaskClass {
    match trigger_mode {
        TriggerMode::AlwaysRunning => TaskClass::Realtime,
        TriggerMode::OnDemand => TaskClass::BestEffort,
    }
}

fn parse_task_class(value: &str) -> Result<TaskClass, LoadError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "realtime" | "real_time" | "real-time" => Ok(TaskClass::Realtime),
        "interactive" => Ok(TaskClass::Interactive),
        "best_effort" | "best-effort" | "besteffort" => Ok(TaskClass::BestEffort),
        other => Err(LoadError::InvalidMetadata {
            field: "task_class".to_string(),
            reason: format!(
                "expected one of realtime|interactive|best_effort, got '{}'",
                other
            ),
        }),
    }
}

fn parse_backend_affinity(value: &str) -> Result<BackendAffinity, LoadError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "any" => Ok(BackendAffinity::Any),
        "cpu" => Ok(BackendAffinity::Cpu),
        "gpu" | "cuda" | "rocm" | "metal" => Ok(BackendAffinity::Gpu),
        "npu" => Ok(BackendAffinity::Npu),
        other => Err(LoadError::InvalidMetadata {
            field: "backend_affinity".to_string(),
            reason: format!("expected one of any|cpu|gpu|npu, got '{}'", other),
        }),
    }
}

fn infer_backend_affinity_from_hint(hardware_target: &str) -> BackendAffinity {
    let normalized = hardware_target.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return BackendAffinity::Any;
    }
    if normalized.contains("npu") {
        return BackendAffinity::Npu;
    }
    if normalized.contains("gpu")
        || normalized.contains("cuda")
        || normalized.contains("rocm")
        || normalized.contains("metal")
    {
        return BackendAffinity::Gpu;
    }
    if normalized.contains("cpu") {
        return BackendAffinity::Cpu;
    }
    BackendAffinity::Any
}

fn parse_cost_estimate(value: &Value) -> Result<CostEstimate, LoadError> {
    match value {
        Value::Number(number) => {
            let Some(ms) = number.as_u64() else {
                return Err(LoadError::InvalidMetadata {
                    field: "cost_estimate".to_string(),
                    reason: "expected non-negative integer".to_string(),
                });
            };
            Ok(CostEstimate {
                value: ms,
                unit: CostUnit::Millis,
            })
        }
        Value::String(raw) => parse_cost_estimate_string(raw),
        Value::Object(object) => {
            let amount = object.get("value").and_then(Value::as_u64).ok_or_else(|| {
                LoadError::InvalidMetadata {
                    field: "cost_estimate.value".to_string(),
                    reason: "expected non-negative integer".to_string(),
                }
            })?;
            let unit = object.get("unit").and_then(Value::as_str).ok_or_else(|| {
                LoadError::InvalidMetadata {
                    field: "cost_estimate.unit".to_string(),
                    reason: "expected string unit (ms|tps)".to_string(),
                }
            })?;
            let unit = parse_cost_unit(unit)?;
            Ok(CostEstimate {
                value: amount,
                unit,
            })
        }
        _ => Err(LoadError::InvalidMetadata {
            field: "cost_estimate".to_string(),
            reason: "expected number, string, or object".to_string(),
        }),
    }
}

fn parse_cost_estimate_string(raw: &str) -> Result<CostEstimate, LoadError> {
    let normalized = raw.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return Err(LoadError::InvalidMetadata {
            field: "cost_estimate".to_string(),
            reason: "value cannot be empty".to_string(),
        });
    }

    if let Some(number) = normalized.strip_suffix("ms") {
        let amount = number
            .trim()
            .parse::<u64>()
            .map_err(|_| LoadError::InvalidMetadata {
                field: "cost_estimate".to_string(),
                reason: format!("invalid ms value '{}'", raw),
            })?;
        return Ok(CostEstimate {
            value: amount,
            unit: CostUnit::Millis,
        });
    }

    if let Some(number) = normalized.strip_suffix("tps") {
        let amount = number
            .trim()
            .parse::<u64>()
            .map_err(|_| LoadError::InvalidMetadata {
                field: "cost_estimate".to_string(),
                reason: format!("invalid tps value '{}'", raw),
            })?;
        return Ok(CostEstimate {
            value: amount,
            unit: CostUnit::TokensPerSecond,
        });
    }

    Err(LoadError::InvalidMetadata {
        field: "cost_estimate".to_string(),
        reason: "expected suffix ms or tps".to_string(),
    })
}

fn parse_cost_unit(raw: &str) -> Result<CostUnit, LoadError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "ms" | "millis" | "milliseconds" => Ok(CostUnit::Millis),
        "tps" | "tokens_per_second" | "tokens/sec" => Ok(CostUnit::TokensPerSecond),
        other => Err(LoadError::InvalidMetadata {
            field: "cost_estimate.unit".to_string(),
            reason: format!("expected one of ms|tps, got '{}'", other),
        }),
    }
}

fn parse_swap(value: &Value) -> Result<SwapPolicy, LoadError> {
    match value {
        Value::Bool(true) => Ok(SwapPolicy::Allowed),
        Value::Bool(false) => Ok(SwapPolicy::Disallowed),
        Value::String(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "allowed" => Ok(SwapPolicy::Allowed),
            "disallowed" => Ok(SwapPolicy::Disallowed),
            other => Err(LoadError::InvalidMetadata {
                field: "swap".to_string(),
                reason: format!("expected allowed|disallowed, got '{}'", other),
            }),
        },
        _ => Err(LoadError::InvalidMetadata {
            field: "swap".to_string(),
            reason: "expected string or boolean".to_string(),
        }),
    }
}

fn parse_thermal(metadata: &Value) -> Result<PackageThermalThresholds, LoadError> {
    let Some(obj) = object_field(metadata, "thermal")? else {
        return Ok(PackageThermalThresholds::default());
    };

    Ok(PackageThermalThresholds {
        soft_c: object_f64_field(obj, "soft_c", "thermal.soft_c")?,
        hard_c: object_f64_field(obj, "hard_c", "thermal.hard_c")?,
        recovery_c: object_f64_field(obj, "recovery_c", "thermal.recovery_c")?,
    })
}

fn parse_metrics(metadata: &Value) -> Result<MetricsToggles, LoadError> {
    let Some(obj) = object_field(metadata, "metrics")? else {
        return Ok(MetricsToggles::default());
    };

    let mut toggles = MetricsToggles::default();
    if let Some(value) = object_bool_field(obj, "latency", "metrics.latency")? {
        toggles.latency = value;
    }
    if let Some(value) = object_bool_field(obj, "memory", "metrics.memory")? {
        toggles.memory = value;
    }
    if let Some(value) = object_bool_field(obj, "thermal", "metrics.thermal")? {
        toggles.thermal = value;
    }
    if let Some(value) = object_bool_field(obj, "scheduler", "metrics.scheduler")? {
        toggles.scheduler = value;
    }
    Ok(toggles)
}

fn parse_onnx_tuning(
    metadata: &Value,
    max_concurrent: usize,
) -> Result<OnnxRuntimeTuningSpec, LoadError> {
    let mut tuning = OnnxRuntimeTuningSpec::default();
    if let Some(obj) = object_field(metadata, "onnx")? {
        tuning.memory_pattern = object_bool_field(obj, "memory_pattern", "onnx.memory_pattern")?;
        tuning.disable_cpu_mem_arena =
            object_bool_field(obj, "disable_cpu_mem_arena", "onnx.disable_cpu_mem_arena")?;
        tuning.session_buckets = object_u64_field(obj, "session_buckets", "onnx.session_buckets")?
            .map(|value| value.max(1) as usize);
        tuning.bucket_dim_granularity =
            object_u64_field(obj, "bucket_dim_granularity", "onnx.bucket_dim_granularity")?
                .map(|value| value.max(1) as usize);
        tuning.bucket_max_dims = object_u64_field(obj, "bucket_max_dims", "onnx.bucket_max_dims")?
            .map(|value| value.max(1) as usize);
        tuning.peak_concurrency_hint =
            object_u64_field(obj, "peak_concurrency_hint", "onnx.peak_concurrency_hint")?
                .or(object_u64_field(
                    obj,
                    "peak_concurrency",
                    "onnx.peak_concurrency",
                )?)
                .map(|value| value.clamp(1, u32::MAX as u64) as u32);
    }

    if tuning.peak_concurrency_hint.is_none() {
        tuning.peak_concurrency_hint = Some(max_concurrent.max(1).min(u32::MAX as usize) as u32);
    }

    Ok(tuning)
}

fn metadata_field<'a>(metadata: &'a Value, key: &str) -> Option<&'a Value> {
    let obj = metadata.as_object()?;
    if let Some(value) = obj
        .get("kapsl_lite")
        .and_then(Value::as_object)
        .and_then(|lite| lite.get(key))
    {
        return Some(value);
    }
    obj.get(key)
}

fn string_field<'a>(metadata: &'a Value, key: &str) -> Result<Option<&'a str>, LoadError> {
    let Some(value) = metadata_field(metadata, key) else {
        return Ok(None);
    };
    match value {
        Value::String(v) => Ok(Some(v)),
        _ => Err(LoadError::InvalidMetadata {
            field: key.to_string(),
            reason: "expected string".to_string(),
        }),
    }
}

fn bool_field(metadata: &Value, key: &str) -> Result<Option<bool>, LoadError> {
    let Some(value) = metadata_field(metadata, key) else {
        return Ok(None);
    };
    match value {
        Value::Bool(v) => Ok(Some(*v)),
        _ => Err(LoadError::InvalidMetadata {
            field: key.to_string(),
            reason: "expected boolean".to_string(),
        }),
    }
}

fn u64_field(metadata: &Value, key: &str) -> Result<Option<u64>, LoadError> {
    let Some(value) = metadata_field(metadata, key) else {
        return Ok(None);
    };
    match value {
        Value::Number(number) => {
            if let Some(v) = number.as_u64() {
                return Ok(Some(v));
            }
            if let Some(v) = number.as_i64() {
                if v >= 0 {
                    return Ok(Some(v as u64));
                }
            }
            Err(LoadError::InvalidMetadata {
                field: key.to_string(),
                reason: "expected non-negative integer".to_string(),
            })
        }
        _ => Err(LoadError::InvalidMetadata {
            field: key.to_string(),
            reason: "expected integer".to_string(),
        }),
    }
}

fn f64_field(metadata: &Value, key: &str) -> Result<Option<f64>, LoadError> {
    let Some(value) = metadata_field(metadata, key) else {
        return Ok(None);
    };
    match value {
        Value::Number(number) => {
            number
                .as_f64()
                .map(Some)
                .ok_or_else(|| LoadError::InvalidMetadata {
                    field: key.to_string(),
                    reason: "expected number".to_string(),
                })
        }
        _ => Err(LoadError::InvalidMetadata {
            field: key.to_string(),
            reason: "expected number".to_string(),
        }),
    }
}

fn object_field<'a>(
    metadata: &'a Value,
    key: &str,
) -> Result<Option<&'a Map<String, Value>>, LoadError> {
    let Some(value) = metadata_field(metadata, key) else {
        return Ok(None);
    };
    match value {
        Value::Object(obj) => Ok(Some(obj)),
        _ => Err(LoadError::InvalidMetadata {
            field: key.to_string(),
            reason: "expected object".to_string(),
        }),
    }
}

fn object_bool_field(
    object: &Map<String, Value>,
    key: &str,
    field_path: &str,
) -> Result<Option<bool>, LoadError> {
    let Some(value) = object.get(key) else {
        return Ok(None);
    };
    match value {
        Value::Bool(v) => Ok(Some(*v)),
        _ => Err(LoadError::InvalidMetadata {
            field: field_path.to_string(),
            reason: "expected boolean".to_string(),
        }),
    }
}

fn object_f64_field(
    object: &Map<String, Value>,
    key: &str,
    field_path: &str,
) -> Result<Option<f64>, LoadError> {
    let Some(value) = object.get(key) else {
        return Ok(None);
    };
    match value {
        Value::Number(v) => v
            .as_f64()
            .map(Some)
            .ok_or_else(|| LoadError::InvalidMetadata {
                field: field_path.to_string(),
                reason: "expected number".to_string(),
            }),
        _ => Err(LoadError::InvalidMetadata {
            field: field_path.to_string(),
            reason: "expected number".to_string(),
        }),
    }
}

fn object_u64_field(
    object: &Map<String, Value>,
    key: &str,
    field_path: &str,
) -> Result<Option<u64>, LoadError> {
    let Some(value) = object.get(key) else {
        return Ok(None);
    };
    match value {
        Value::Number(number) => {
            if let Some(v) = number.as_u64() {
                return Ok(Some(v));
            }
            if let Some(v) = number.as_i64()
                && v >= 0
            {
                return Ok(Some(v as u64));
            }
            Err(LoadError::InvalidMetadata {
                field: field_path.to_string(),
                reason: "expected non-negative integer".to_string(),
            })
        }
        _ => Err(LoadError::InvalidMetadata {
            field: field_path.to_string(),
            reason: "expected integer".to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use serde_json::json;
    use std::fs;
    use std::fs::File;
    use tar::Builder;
    use tempfile::TempDir;

    fn append_tar_bytes_entry(builder: &mut Builder<GzEncoder<File>>, path: &str, bytes: &[u8]) {
        let mut header = tar::Header::new_gnu();
        header.set_size(bytes.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder.append_data(&mut header, path, bytes).unwrap();
    }

    fn write_aimod_package(
        dir: &Path,
        filename: &str,
        metadata_json: &Value,
        files: &[(&str, &[u8])],
    ) -> PathBuf {
        let metadata_bytes = serde_json::to_vec(metadata_json).unwrap();
        write_aimod_package_with_raw_metadata(dir, filename, &metadata_bytes, files)
    }

    fn write_aimod_package_with_raw_metadata(
        dir: &Path,
        filename: &str,
        metadata_bytes: &[u8],
        files: &[(&str, &[u8])],
    ) -> PathBuf {
        let package_path = dir.join(filename);
        let file = File::create(&package_path).unwrap();
        let encoder = GzEncoder::new(file, Compression::default());
        let mut archive = Builder::new(encoder);

        append_tar_bytes_entry(&mut archive, "metadata.json", metadata_bytes);
        for (path, bytes) in files {
            append_tar_bytes_entry(&mut archive, path, bytes);
        }

        archive.finish().unwrap();
        let encoder = archive.into_inner().unwrap();
        let _ = encoder.finish().unwrap();
        package_path
    }

    fn write_sized_file(path: &Path, bytes: usize) {
        let data = vec![b'x'; bytes];
        fs::write(path, data).unwrap();
    }

    #[test]
    fn load_minimal_valid_package_defaults() {
        let dir = TempDir::new().unwrap();
        let package_path = write_aimod_package(
            dir.path(),
            "test.aimod",
            &json!({
                "project_name": "test.pkg",
                "framework": "onnx",
                "version": "1.0.0",
                "created_at": "2026-02-25T00:00:00Z",
                "model_file": "model.onnx",
                "hardware_requirements": {
                    "min_memory_mb": 192
                }
            }),
            &[("model.onnx", b"dummy-model")],
        );

        let spec = load_package(&package_path).expect("should parse valid package");
        assert_eq!(spec.name, "test.pkg");
        assert_eq!(spec.format, ModelFormat::Onnx);
        assert_eq!(spec.weights, "model.onnx");
        assert!(spec.weights_path.ends_with("model.onnx"));
        assert_eq!(spec.memory_mb, 192);
        assert_eq!(spec.priority, Priority::Normal);
        assert_eq!(spec.task_class, TaskClass::Realtime);
        assert_eq!(spec.trigger_mode, TriggerMode::AlwaysRunning);
        assert_eq!(spec.backend_affinity, BackendAffinity::Any);
        assert_eq!(spec.deadline_ms, Some(1000));
        assert!(spec.cost_estimate.is_none());
        assert!(spec.preemptible);
        assert_eq!(spec.onnx_tuning.peak_concurrency_hint, Some(1));
        assert_eq!(spec.onnx_tuning.session_buckets, None);
    }

    #[test]
    fn load_applies_kapsl_lite_metadata_overrides() {
        let dir = TempDir::new().unwrap();
        let package_path = write_aimod_package(
            dir.path(),
            "override.aimod",
            &json!({
                "project_name": "edge-llm",
                "framework": "llm",
                "version": "1.0.0",
                "created_at": "2026-02-25T00:00:00Z",
                "model_file": "model.gguf",
                "metadata": {
                    "kapsl_lite": {
                        "memory_mb": 384,
                        "priority": "high",
                        "task_class": "interactive",
                        "trigger_mode": "on_demand",
                        "preemptible": false,
                        "max_concurrent": 2,
                        "on_demand_queue_capacity": 128,
                        "on_demand_queue_policy": "drop_oldest",
                        "cpu_threads": 4,
                        "hardware_target": "rpi5",
                        "backend_affinity": "gpu",
                        "critical_perception": true,
                        "deadline_ms": 80,
                        "cost_estimate": "35tps",
                        "swap": "allowed",
                        "fps_target": 2.5,
                        "thermal": {
                            "soft_c": 61.0,
                            "hard_c": 78.0,
                            "recovery_c": 56.0
                        },
                        "metrics": {
                            "latency": false,
                            "memory": true,
                            "thermal": false,
                            "scheduler": true
                        },
                        "onnx": {
                            "memory_pattern": false,
                            "disable_cpu_mem_arena": true,
                            "session_buckets": 3,
                            "bucket_dim_granularity": 128,
                            "bucket_max_dims": 5,
                            "peak_concurrency_hint": 6
                        }
                    }
                }
            }),
            &[("model.gguf", b"dummy-model")],
        );

        let spec = load_package(&package_path).expect("should parse package metadata overrides");
        assert_eq!(spec.format, ModelFormat::Gguf);
        assert_eq!(spec.memory_mb, 384);
        assert_eq!(spec.priority, Priority::High);
        assert_eq!(spec.task_class, TaskClass::Interactive);
        assert_eq!(spec.trigger_mode, TriggerMode::OnDemand);
        assert!(!spec.preemptible);
        assert_eq!(spec.max_concurrent, 2);
        assert_eq!(spec.on_demand_queue_capacity, 128);
        assert_eq!(spec.on_demand_queue_policy, QueueOverflowPolicy::DropOldest);
        assert_eq!(spec.cpu_threads, 4);
        assert_eq!(spec.hardware_target, "rpi5");
        assert_eq!(spec.backend_affinity, BackendAffinity::Gpu);
        assert!(spec.critical_perception);
        assert_eq!(spec.deadline_ms, Some(80));
        assert_eq!(
            spec.cost_estimate,
            Some(CostEstimate {
                value: 35,
                unit: CostUnit::TokensPerSecond
            })
        );
        assert_eq!(spec.swap, SwapPolicy::Allowed);
        assert_eq!(spec.fps_target, Some(2.5));
        assert_eq!(spec.thermal.soft_c, Some(61.0));
        assert_eq!(spec.thermal.hard_c, Some(78.0));
        assert_eq!(spec.thermal.recovery_c, Some(56.0));
        assert!(!spec.metrics.latency);
        assert!(spec.metrics.memory);
        assert!(!spec.metrics.thermal);
        assert!(spec.metrics.scheduler);
        assert_eq!(spec.onnx_tuning.memory_pattern, Some(false));
        assert_eq!(spec.onnx_tuning.disable_cpu_mem_arena, Some(true));
        assert_eq!(spec.onnx_tuning.session_buckets, Some(3));
        assert_eq!(spec.onnx_tuning.bucket_dim_granularity, Some(128));
        assert_eq!(spec.onnx_tuning.bucket_max_dims, Some(5));
        assert_eq!(spec.onnx_tuning.peak_concurrency_hint, Some(6));
    }

    #[test]
    fn load_fails_when_model_file_missing() {
        let dir = TempDir::new().unwrap();
        let package_path = write_aimod_package(
            dir.path(),
            "missing.aimod",
            &json!({
                "project_name": "missing-model",
                "framework": "onnx",
                "version": "1.0.0",
                "created_at": "2026-02-25T00:00:00Z",
                "model_file": "nonexistent.onnx"
            }),
            &[],
        );

        let result = load_package(&package_path);
        assert!(matches!(result, Err(LoadError::ModelFileMissing { .. })));
    }

    #[test]
    fn load_treats_llm_with_onnx_model_file_as_onnx_format() {
        let dir = TempDir::new().unwrap();
        let package_path = write_aimod_package(
            dir.path(),
            "llm-onnx.aimod",
            &json!({
                "project_name": "llm-onnx",
                "framework": "llm",
                "version": "1.0.0",
                "created_at": "2026-02-25T00:00:00Z",
                "model_file": "decoder.onnx"
            }),
            &[("decoder.onnx", b"dummy-model")],
        );

        let spec = load_package(&package_path).expect("should parse llm onnx package");
        assert_eq!(spec.format, ModelFormat::Onnx);
    }

    #[test]
    fn load_persists_onnx_external_data_files() {
        let dir = TempDir::new().unwrap();
        let package_path = write_aimod_package(
            dir.path(),
            "with-data.aimod",
            &json!({
                "project_name": "with-data",
                "framework": "onnx",
                "version": "1.0.0",
                "created_at": "2026-02-25T00:00:00Z",
                "model_file": "model.onnx"
            }),
            &[
                ("model.onnx", b"dummy-model"),
                ("model.onnx_data", b"data-part-0"),
                ("model.onnx_data_1", b"data-part-1"),
            ],
        );

        let spec = load_package(&package_path).expect("should parse package with external data");
        let persisted_model = PathBuf::from(&spec.weights_path);
        assert!(persisted_model.exists());
        let persisted_data = persisted_model.with_file_name("model.onnx_data");
        let persisted_data_1 = persisted_model.with_file_name("model.onnx_data_1");
        assert!(persisted_data.exists());
        assert!(persisted_data_1.exists());
    }

    #[test]
    fn load_fails_on_malformed_manifest() {
        let dir = TempDir::new().unwrap();
        let package_path = write_aimod_package_with_raw_metadata(
            dir.path(),
            "bad.aimod",
            b"{not-json",
            &[("model.onnx", b"dummy-model")],
        );

        let result = load_package(&package_path);
        assert!(
            matches!(result, Err(LoadError::Package(CoreLoaderError::Json(_)))),
            "expected JSON parse error from canonical package loader"
        );
    }

    #[test]
    fn scan_dir_returns_sorted_by_priority() {
        let dir = TempDir::new().unwrap();
        write_aimod_package(
            dir.path(),
            "low.aimod",
            &json!({
                "project_name": "low.pkg",
                "framework": "onnx",
                "version": "1.0.0",
                "created_at": "2026-02-25T00:00:00Z",
                "model_file": "low.onnx",
                "metadata": {
                    "kapsl_lite": {
                        "priority": "low"
                    }
                }
            }),
            &[("low.onnx", b"low")],
        );

        write_aimod_package(
            dir.path(),
            "high.aimod",
            &json!({
                "project_name": "high.pkg",
                "framework": "onnx",
                "version": "1.0.0",
                "created_at": "2026-02-25T00:00:00Z",
                "model_file": "high.onnx",
                "metadata": {
                    "kapsl_lite": {
                        "priority": "high"
                    }
                }
            }),
            &[("high.onnx", b"high")],
        );

        let specs = load_packages_from_dir(dir.path()).expect("should load all valid packages");
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].name, "high.pkg");
        assert_eq!(specs[1].name, "low.pkg");
    }

    #[test]
    fn cache_policy_evicts_oldest_entries_when_over_limit() {
        let dir = TempDir::new().unwrap();
        let cache_root = dir.path().join("cache");
        let old_entry = cache_root.join("a-old");
        let new_entry = cache_root.join("z-new");
        fs::create_dir_all(&old_entry).unwrap();
        fs::create_dir_all(&new_entry).unwrap();
        write_sized_file(&old_entry.join("model.bin"), 64);
        write_sized_file(&new_entry.join("model.bin"), 64);

        enforce_cache_policy(
            &cache_root,
            &cache_root.join("active"),
            0,
            CachePolicy {
                max_cache_bytes: Some(100),
                reserved_free_bytes: 0,
            },
            |_| Ok(u64::MAX / 4),
        )
        .expect("eviction should satisfy cache cap");

        assert!(!old_entry.exists(), "oldest cache entry should be evicted");
        assert!(new_entry.exists(), "newer cache entry should remain");
    }

    #[test]
    fn cache_policy_keeps_entries_when_requirements_already_met() {
        let dir = TempDir::new().unwrap();
        let cache_root = dir.path().join("cache");
        let only_entry = cache_root.join("single");
        fs::create_dir_all(&only_entry).unwrap();
        write_sized_file(&only_entry.join("model.bin"), 32);

        enforce_cache_policy(
            &cache_root,
            &cache_root.join("active"),
            0,
            CachePolicy {
                max_cache_bytes: Some(1024),
                reserved_free_bytes: 0,
            },
            |_| Ok(u64::MAX / 4),
        )
        .expect("no eviction required");

        assert!(only_entry.exists(), "cache entry should not be evicted");
    }

    #[test]
    fn cache_policy_returns_insufficient_space_when_unrecoverable() {
        let dir = TempDir::new().unwrap();
        let cache_root = dir.path().join("cache");
        let entry = cache_root.join("only-entry");
        fs::create_dir_all(&entry).unwrap();
        write_sized_file(&entry.join("model.bin"), 16);

        let result = enforce_cache_policy(
            &cache_root,
            &cache_root.join("active"),
            128,
            CachePolicy {
                max_cache_bytes: None,
                reserved_free_bytes: 1,
            },
            |_| Ok(0),
        );

        assert!(matches!(
            result,
            Err(LoadError::InsufficientDiskSpace {
                required_copy_bytes: 128,
                ..
            })
        ));
    }
}
