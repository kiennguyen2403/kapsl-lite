mod input_adapter;
mod loader;
mod metrics;
mod package;
mod runtime;
mod system;
mod trigger;
mod tui;

use crate::input_adapter::spawn_default_input_adapters;
use crate::loader::load_packages_from_dir;
use crate::metrics::{EventLevel, RuntimeMetrics};
use crate::package::OnnxRuntimeTuningSpec;
use crate::runtime::{RuntimeConfig, RuntimeHandle};
use clap::{Parser, Subcommand};
use signal_hook::consts::signal::{SIGINT, SIGTERM};
use signal_hook::flag;
use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

const CLI_AFTER_HELP: &str = "\
Example:
  kapsl run --models ./models/
  kapsl run --models ./models/ --memory-mb 1024 --poll-ms 1000";

#[derive(Parser, Debug)]
#[command(
    name = "kapsl",
    author,
    version,
    about = "Kapsl runtime lite — edge device inference runtime",
    after_help = CLI_AFTER_HELP
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    #[command(flatten)]
    run: RunArgs,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Load packages from a directory and start the runtime with the TUI dashboard.
    Run(RunArgs),
}

#[derive(clap::Args, Debug)]
struct RunArgs {
    /// Directory containing .aimod model packages to load.
    #[arg(
        short = 'm',
        long = "models",
        alias = "packages",
        value_name = "DIR",
        help = "Directory containing .aimod model packages"
    )]
    models: Option<PathBuf>,

    /// Total memory budget for all loaded packages (MiB).
    /// Overrides KAPSL_LITE_MEMORY_LIMIT_MIB environment variable.
    #[arg(long, value_name = "MiB")]
    memory_mb: Option<u64>,

    /// Thermal polling interval in milliseconds.
    /// Overrides KAPSL_LITE_THERMAL_POLL_INTERVAL_MS environment variable.
    #[arg(long, value_name = "MS")]
    poll_ms: Option<u64>,

    /// Global ONNX Runtime memory-pattern setting for all ONNX packages (true/false).
    #[arg(long, value_name = "BOOL")]
    onnx_memory_pattern: Option<bool>,

    /// Global ONNX Runtime CPU arena toggle for all ONNX packages (true/false).
    #[arg(long, value_name = "BOOL")]
    onnx_disable_cpu_mem_arena: Option<bool>,

    /// Global ONNX Runtime session bucket count for shape-bucketed session reuse.
    #[arg(long, value_name = "N")]
    onnx_session_buckets: Option<usize>,

    /// Global ONNX Runtime non-batch dimension bucket granularity.
    #[arg(long, value_name = "N")]
    onnx_bucket_dim_granularity: Option<usize>,

    /// Global ONNX Runtime number of leading dims used for bucket keys.
    #[arg(long, value_name = "N")]
    onnx_bucket_max_dims: Option<usize>,

    /// Global peak-concurrency hint exported in ONNX package metadata.
    #[arg(long, value_name = "N")]
    onnx_peak_concurrency_hint: Option<u32>,

    /// Per-package ONNX tuning override.
    /// Format: `<package_name|*>:k=v[,k=v...]`
    /// Keys: memory_pattern, disable_cpu_mem_arena, session_buckets,
    /// bucket_dim_granularity, bucket_max_dims, peak_concurrency.
    #[arg(long, value_name = "SPEC")]
    onnx_package_tuning: Vec<String>,
}

#[derive(Debug, Clone, Default)]
struct OnnxTuningProfile {
    global: OnnxRuntimeTuningSpec,
    per_package: HashMap<String, OnnxRuntimeTuningSpec>,
}

fn merge_onnx_tuning(
    base: &OnnxRuntimeTuningSpec,
    overrides: &OnnxRuntimeTuningSpec,
) -> OnnxRuntimeTuningSpec {
    OnnxRuntimeTuningSpec {
        memory_pattern: overrides.memory_pattern.or(base.memory_pattern),
        disable_cpu_mem_arena: overrides
            .disable_cpu_mem_arena
            .or(base.disable_cpu_mem_arena),
        session_buckets: overrides.session_buckets.or(base.session_buckets),
        bucket_dim_granularity: overrides
            .bucket_dim_granularity
            .or(base.bucket_dim_granularity),
        bucket_max_dims: overrides.bucket_max_dims.or(base.bucket_max_dims),
        peak_concurrency_hint: overrides
            .peak_concurrency_hint
            .or(base.peak_concurrency_hint),
    }
}

impl OnnxTuningProfile {
    fn resolve(&self, package_name: &str) -> OnnxRuntimeTuningSpec {
        if let Some(package_overrides) = self.per_package.get(package_name) {
            merge_onnx_tuning(&self.global, package_overrides)
        } else {
            self.global.clone()
        }
    }
}

fn parse_bool_literal(value: &str) -> Result<bool, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!("invalid boolean '{}'", value)),
    }
}

fn apply_onnx_tuning_pair(
    target: &mut OnnxRuntimeTuningSpec,
    key: &str,
    value: &str,
) -> Result<(), String> {
    match key.trim().to_ascii_lowercase().as_str() {
        "memory_pattern" | "mem_pattern" => {
            target.memory_pattern = Some(parse_bool_literal(value)?);
        }
        "disable_cpu_mem_arena" | "cpu_mem_arena_disabled" => {
            target.disable_cpu_mem_arena = Some(parse_bool_literal(value)?);
        }
        "session_buckets" => {
            let parsed = value
                .trim()
                .parse::<usize>()
                .map_err(|e| format!("invalid session_buckets '{}': {}", value, e))?;
            target.session_buckets = Some(parsed.max(1));
        }
        "bucket_dim_granularity" => {
            let parsed = value
                .trim()
                .parse::<usize>()
                .map_err(|e| format!("invalid bucket_dim_granularity '{}': {}", value, e))?;
            target.bucket_dim_granularity = Some(parsed.max(1));
        }
        "bucket_max_dims" => {
            let parsed = value
                .trim()
                .parse::<usize>()
                .map_err(|e| format!("invalid bucket_max_dims '{}': {}", value, e))?;
            target.bucket_max_dims = Some(parsed.max(1));
        }
        "peak_concurrency" | "peak_concurrency_hint" => {
            let parsed = value
                .trim()
                .parse::<u32>()
                .map_err(|e| format!("invalid peak_concurrency '{}': {}", value, e))?;
            target.peak_concurrency_hint = Some(parsed.max(1));
        }
        other => {
            return Err(format!(
                "unknown ONNX tuning key '{}'; expected one of memory_pattern, disable_cpu_mem_arena, session_buckets, bucket_dim_granularity, bucket_max_dims, peak_concurrency",
                other
            ));
        }
    }
    Ok(())
}

fn parse_onnx_package_tuning_spec(
    spec: &str,
) -> Result<(Option<String>, OnnxRuntimeTuningSpec), String> {
    let (selector_raw, config_raw) = spec.split_once(':').ok_or_else(|| {
        format!(
            "invalid --onnx-package-tuning '{}': expected '<package_name|*>:k=v[,k=v...]'",
            spec
        )
    })?;
    let selector = selector_raw.trim();
    let package_name = if selector == "*" {
        None
    } else if selector.is_empty() {
        return Err("invalid package selector: cannot be empty".to_string());
    } else {
        Some(selector.to_string())
    };

    let mut tuning = OnnxRuntimeTuningSpec::default();
    for pair in config_raw.split(',') {
        let trimmed = pair.trim();
        if trimmed.is_empty() {
            continue;
        }
        let (key, value) = trimmed
            .split_once('=')
            .ok_or_else(|| format!("invalid tuning pair '{}': expected k=v", trimmed))?;
        apply_onnx_tuning_pair(&mut tuning, key, value)?;
    }
    Ok((package_name, tuning))
}

fn build_onnx_tuning_profile(args: &RunArgs) -> Result<OnnxTuningProfile, String> {
    let mut profile = OnnxTuningProfile {
        global: OnnxRuntimeTuningSpec {
            memory_pattern: args.onnx_memory_pattern,
            disable_cpu_mem_arena: args.onnx_disable_cpu_mem_arena,
            session_buckets: args.onnx_session_buckets,
            bucket_dim_granularity: args.onnx_bucket_dim_granularity,
            bucket_max_dims: args.onnx_bucket_max_dims,
            peak_concurrency_hint: args.onnx_peak_concurrency_hint,
        },
        per_package: HashMap::new(),
    };

    for spec in &args.onnx_package_tuning {
        let (package_name, tuning) = parse_onnx_package_tuning_spec(spec)?;
        if let Some(package_name) = package_name {
            let merged = profile
                .per_package
                .get(&package_name)
                .map(|existing| merge_onnx_tuning(existing, &tuning))
                .unwrap_or(tuning);
            profile.per_package.insert(package_name, merged);
        } else {
            profile.global = merge_onnx_tuning(&profile.global, &tuning);
        }
    }

    Ok(profile)
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    let run_args = match cli.command {
        Some(Command::Run(args)) => args,
        None => cli.run,
    };

    run_command(run_args)
}

fn require_packages_arg(
    args: RunArgs,
) -> Result<(PathBuf, RuntimeConfig, OnnxTuningProfile), Box<dyn Error>> {
    let onnx_tuning_profile = build_onnx_tuning_profile(&args)
        .map_err(|e| format!("invalid ONNX tuning configuration: {}", e))?;

    let packages = args
        .models
        .ok_or_else(|| "missing required argument: --models <DIR>".to_string())?;

    // Build config from environment, then apply CLI overrides.
    let mut config = RuntimeConfig::from_env();
    if let Some(memory_mb) = args.memory_mb {
        config.memory_limit_mib = memory_mb.max(64);
    }
    if let Some(poll_ms) = args.poll_ms {
        config.thermal_poll_interval_ms = poll_ms.max(100);
    }

    Ok((packages, config, onnx_tuning_profile))
}

fn run_command(args: RunArgs) -> Result<(), Box<dyn Error>> {
    let (packages_dir, config, onnx_tuning_profile) = require_packages_arg(args)?;

    // Validate packages directory.
    if !packages_dir.exists() {
        return Err(format!("packages directory not found: {}", packages_dir.display()).into());
    }
    if !packages_dir.is_dir() {
        return Err(format!(
            "packages path is not a directory: {}",
            packages_dir.display()
        )
        .into());
    }

    // Load all .aimod specs from the directory.
    let mut packages = load_packages_from_dir(&packages_dir).map_err(|e| {
        format!(
            "failed to load packages from {}: {}",
            packages_dir.display(),
            e
        )
    })?;

    for package in &mut packages {
        let cli_overrides = onnx_tuning_profile.resolve(&package.name);
        package.onnx_tuning = merge_onnx_tuning(&package.onnx_tuning, &cli_overrides);
        if package.onnx_tuning.peak_concurrency_hint.is_none() {
            package.onnx_tuning.peak_concurrency_hint =
                Some(package.max_concurrent.max(1).min(u32::MAX as usize) as u32);
        }
    }

    if packages.is_empty() {
        return Err(format!(
            "no .aimod package files found in {}",
            packages_dir.display()
        )
        .into());
    }

    // Validate total requested memory doesn't exceed budget.
    let total_requested_mib: u64 = packages.iter().map(|p| p.memory_mb).sum();
    if total_requested_mib > config.memory_limit_mib {
        return Err(format!(
            "total package memory {}MiB exceeds budget {}MiB — reduce packages or increase --memory-mb",
            total_requested_mib, config.memory_limit_mib
        )
        .into());
    }

    let package_count = packages.len() as u64;

    // Set up shutdown signal.
    let shutdown = Arc::new(AtomicBool::new(false));
    install_signal_handlers(shutdown.clone())?;

    // Initialise metrics.
    let metrics = Arc::new(RuntimeMetrics::new(package_count, config.memory_limit_mib));
    metrics.push_scheduler_log(
        EventLevel::Normal,
        format!(
            "startup: packages={} dir={} memory_budget={}MiB",
            package_count,
            packages_dir.display(),
            config.memory_limit_mib,
        ),
    );

    // Start the runtime.
    let (runtime, trigger_bus, _registry) =
        RuntimeHandle::start(config, packages, metrics.clone(), shutdown.clone());
    let _adapter_threads =
        spawn_default_input_adapters(trigger_bus.clone(), metrics.clone(), shutdown.clone());

    // Run TUI — blocks until 'q' or shutdown signal.
    // If terminal initialization fails (for example non-interactive shells),
    // keep runtime alive in headless mode instead of exiting immediately.
    if let Err(error) = tui::run_dashboard(metrics.clone(), trigger_bus.clone(), shutdown.clone()) {
        let reason = error.to_string();
        eprintln!(
            "dashboard unavailable ({}); running in headless mode (Ctrl+C to stop)",
            reason
        );
        metrics.push_scheduler_log(
            EventLevel::Warning,
            format!(
                "dashboard unavailable: {} ; fallback=headless (ctrl+c to stop)",
                reason
            ),
        );
        run_headless_until_shutdown(shutdown.clone());
    }

    shutdown.store(true, Ordering::SeqCst);
    runtime.join();
    // Adapter threads may block in platform ingress waits (for example Windows named-pipe connect).
    // Dropping join handles here lets process shutdown complete immediately when operator quits TUI.

    Ok(())
}

fn run_headless_until_shutdown(shutdown: Arc<AtomicBool>) {
    while !shutdown.load(Ordering::Relaxed) {
        thread::sleep(Duration::from_millis(250));
    }
}

fn install_signal_handlers(shutdown: Arc<AtomicBool>) -> Result<(), Box<dyn Error>> {
    flag::register(SIGINT, shutdown.clone())?;
    flag::register(SIGTERM, shutdown)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn cli_supports_run_subcommand() {
        let cli = Cli::parse_from(["kapsl", "run", "--models", "./models"]);
        match cli.command {
            Some(Command::Run(run)) => {
                assert_eq!(
                    run.models.as_deref(),
                    Some(std::path::Path::new("./models"))
                );
            }
            None => panic!("expected run subcommand"),
        }
    }

    #[test]
    fn cli_supports_direct_run_flags() {
        let cli = Cli::parse_from(["kapsl", "--models", "./models"]);
        assert!(cli.command.is_none());
        assert_eq!(
            cli.run.models.as_deref(),
            Some(std::path::Path::new("./models"))
        );
    }

    #[test]
    fn cli_accepts_legacy_packages_alias() {
        let cli = Cli::parse_from(["kapsl", "run", "--packages", "./models"]);
        match cli.command {
            Some(Command::Run(run)) => assert_eq!(
                run.models.as_deref(),
                Some(std::path::Path::new("./models"))
            ),
            None => panic!("expected run subcommand"),
        }
    }

    #[test]
    fn onnx_tuning_profile_resolves_global_and_package_overrides() {
        let cli = Cli::parse_from([
            "kapsl",
            "run",
            "--models",
            "./models",
            "--onnx-memory-pattern",
            "false",
            "--onnx-package-tuning",
            "*:session_buckets=2",
            "--onnx-package-tuning",
            "vision.detector:disable_cpu_mem_arena=true,peak_concurrency=8",
        ]);
        let run = match cli.command {
            Some(Command::Run(run)) => run,
            None => panic!("expected run subcommand"),
        };
        let profile = build_onnx_tuning_profile(&run).expect("valid ONNX tuning profile");

        let detector = profile.resolve("vision.detector");
        assert_eq!(detector.memory_pattern, Some(false));
        assert_eq!(detector.session_buckets, Some(2));
        assert_eq!(detector.disable_cpu_mem_arena, Some(true));
        assert_eq!(detector.peak_concurrency_hint, Some(8));

        let other = profile.resolve("reasoning.edge");
        assert_eq!(other.memory_pattern, Some(false));
        assert_eq!(other.session_buckets, Some(2));
        assert_eq!(other.disable_cpu_mem_arena, None);
    }

    #[test]
    fn onnx_tuning_profile_rejects_unknown_key() {
        let cli = Cli::parse_from([
            "kapsl",
            "run",
            "--models",
            "./models",
            "--onnx-package-tuning",
            "vision.detector:not_a_real_key=1",
        ]);
        let run = match cli.command {
            Some(Command::Run(run)) => run,
            None => panic!("expected run subcommand"),
        };
        let err = build_onnx_tuning_profile(&run).expect_err("unknown key should fail");
        assert!(err.contains("unknown ONNX tuning key"));
    }
}
