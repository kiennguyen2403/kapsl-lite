use crate::metrics::{EventLevel, RuntimeMetrics};
use crate::trigger::{
    BackpressurePolicy, NormalizedInputEvent, NormalizedPayload, TriggerBus, TriggerEvent,
};
use serde_json::{Map, Value};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::thread::{self, JoinHandle};

#[cfg(unix)]
const DEFAULT_JSON_SOCKET_PATH: &str = "/tmp/kapsl-lite.sock";
#[cfg(unix)]
const DEFAULT_STREAM_SOCKET_PATH: &str = "/tmp/kapsl-lite.stream.sock";
#[cfg(windows)]
const DEFAULT_JSON_PIPE_NAME: &str = r"\\.\pipe\kapsl-lite-ingress-json";
#[cfg(windows)]
const DEFAULT_STREAM_PIPE_NAME: &str = r"\\.\pipe\kapsl-lite-ingress-stream";
const DEFAULT_ACCEPT_POLL_MS: u64 = 100;
const MAX_STREAM_HEADER_BYTES: usize = 64 * 1024;
const MAX_STREAM_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;

pub trait InputAdapter: Send + 'static {
    fn adapter_name(&self) -> &'static str;
    fn run(self, trigger_bus: TriggerBus, metrics: Arc<RuntimeMetrics>, shutdown: Arc<AtomicBool>);
}

pub fn spawn_input_adapter<A: InputAdapter>(
    adapter: A,
    trigger_bus: TriggerBus,
    metrics: Arc<RuntimeMetrics>,
    shutdown: Arc<AtomicBool>,
) -> JoinHandle<()> {
    thread::spawn(move || adapter.run(trigger_bus, metrics, shutdown))
}

#[derive(Debug, Clone)]
struct RouteRule {
    source_pattern: String,
    target_package: String,
}

impl RouteRule {
    fn matches(&self, source_id: &str) -> bool {
        let pattern = self.source_pattern.as_str();
        if let Some(prefix) = pattern.strip_suffix('*') {
            return source_id.starts_with(prefix);
        }
        source_id == pattern
    }
}

#[derive(Debug, Clone)]
struct RouteTable {
    rules: Vec<RouteRule>,
    default_target: Option<String>,
}

impl RouteTable {
    fn resolve_targets(&self, source_id: &str) -> Vec<String> {
        let mut targets: Vec<String> = self
            .rules
            .iter()
            .filter(|rule| rule.matches(source_id))
            .map(|rule| rule.target_package.clone())
            .collect();

        if targets.is_empty()
            && let Some(default_target) = &self.default_target
        {
            targets.push(default_target.clone());
        }
        targets
    }
}

#[derive(Debug)]
enum IngressRecord {
    Trigger(TriggerEvent),
    External {
        explicit_target: Option<String>,
        event: NormalizedInputEvent,
    },
}

#[cfg(unix)]
pub fn spawn_default_input_adapters(
    trigger_bus: TriggerBus,
    metrics: Arc<RuntimeMetrics>,
    shutdown: Arc<AtomicBool>,
) -> Vec<JoinHandle<()>> {
    let mut handles = Vec::new();
    if !ingress_enabled() {
        metrics.push_scheduler_log(
            EventLevel::Warning,
            "input adapter disabled: ingress skipped".to_string(),
        );
        return handles;
    }

    let route_table = parse_route_table();
    let backpressure = parse_backpressure_policy();
    let accept_poll_ms = parse_accept_poll_ms();

    let json_socket_path = std::env::var("KAPSL_LITE_INGRESS_UNIX_SOCKET")
        .map(|value| value.trim().to_string())
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_JSON_SOCKET_PATH.to_string());

    handles.push(spawn_input_adapter(
        UnixJsonIngressAdapter {
            socket_path: std::path::PathBuf::from(json_socket_path),
            accept_poll_ms,
            route_table: route_table.clone(),
            backpressure,
        },
        trigger_bus.clone(),
        metrics.clone(),
        shutdown.clone(),
    ));

    if stream_ingress_enabled() {
        let stream_socket_path = std::env::var("KAPSL_LITE_INGRESS_STREAM_SOCKET")
            .map(|value| value.trim().to_string())
            .ok()
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| DEFAULT_STREAM_SOCKET_PATH.to_string());

        handles.push(spawn_input_adapter(
            UnixStreamIngressAdapter {
                socket_path: std::path::PathBuf::from(stream_socket_path),
                accept_poll_ms,
                route_table,
                backpressure,
            },
            trigger_bus,
            metrics,
            shutdown,
        ));
    }

    handles
}

#[cfg(windows)]
pub fn spawn_default_input_adapters(
    trigger_bus: TriggerBus,
    metrics: Arc<RuntimeMetrics>,
    shutdown: Arc<AtomicBool>,
) -> Vec<JoinHandle<()>> {
    let mut handles = Vec::new();
    if !ingress_enabled() {
        metrics.push_scheduler_log(
            EventLevel::Warning,
            "input adapter disabled: ingress skipped".to_string(),
        );
        return handles;
    }

    let route_table = parse_route_table();
    let backpressure = parse_backpressure_policy();
    let accept_poll_ms = parse_accept_poll_ms();

    let json_pipe_name = std::env::var("KAPSL_LITE_INGRESS_PIPE")
        .map(|value| value.trim().to_string())
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_JSON_PIPE_NAME.to_string());

    handles.push(spawn_input_adapter(
        WindowsNamedPipeJsonIngressAdapter {
            pipe_name: json_pipe_name,
            accept_poll_ms,
            route_table: route_table.clone(),
            backpressure,
        },
        trigger_bus.clone(),
        metrics.clone(),
        shutdown.clone(),
    ));

    if stream_ingress_enabled() {
        let stream_pipe_name = std::env::var("KAPSL_LITE_INGRESS_STREAM_PIPE")
            .map(|value| value.trim().to_string())
            .ok()
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| DEFAULT_STREAM_PIPE_NAME.to_string());

        handles.push(spawn_input_adapter(
            WindowsNamedPipeStreamIngressAdapter {
                pipe_name: stream_pipe_name,
                accept_poll_ms,
                route_table,
                backpressure,
            },
            trigger_bus,
            metrics,
            shutdown,
        ));
    }

    handles
}

#[cfg(all(not(unix), not(windows)))]
pub fn spawn_default_input_adapters(
    _trigger_bus: TriggerBus,
    metrics: Arc<RuntimeMetrics>,
    _shutdown: Arc<AtomicBool>,
) -> Vec<JoinHandle<()>> {
    metrics.push_scheduler_log(
        EventLevel::Warning,
        "input adapter unavailable: no ingress transport for this target".to_string(),
    );
    Vec::new()
}

fn parse_accept_poll_ms() -> u64 {
    std::env::var("KAPSL_LITE_INGRESS_ACCEPT_POLL_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(DEFAULT_ACCEPT_POLL_MS)
        .max(10)
}

fn parse_backpressure_policy() -> BackpressurePolicy {
    match std::env::var("KAPSL_LITE_INGRESS_BACKPRESSURE")
        .unwrap_or_else(|_| "drop_oldest".to_string())
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "latest_only" | "latest-only" | "latest" => BackpressurePolicy::LatestOnly,
        "block" | "blocking" => BackpressurePolicy::Block,
        _ => BackpressurePolicy::DropOldest,
    }
}

fn parse_route_table() -> RouteTable {
    let routes_raw = std::env::var("KAPSL_LITE_INGRESS_ROUTES").unwrap_or_default();
    let rules = routes_raw
        .split([';', ','])
        .filter_map(|entry| {
            let trimmed = entry.trim();
            if trimmed.is_empty() {
                return None;
            }
            let (source_pattern, target_package) = trimmed.split_once('=')?;
            let source_pattern = source_pattern.trim();
            let target_package = target_package.trim();
            if source_pattern.is_empty() || target_package.is_empty() {
                return None;
            }
            Some(RouteRule {
                source_pattern: source_pattern.to_string(),
                target_package: target_package.to_string(),
            })
        })
        .collect::<Vec<_>>();

    let default_target = std::env::var("KAPSL_LITE_TRIGGER_TARGET_PACKAGE")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    RouteTable {
        rules,
        default_target,
    }
}

fn ingress_enabled() -> bool {
    std::env::var("KAPSL_LITE_INGRESS_ENABLED")
        .ok()
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !(normalized == "0" || normalized == "false" || normalized == "no")
        })
        .unwrap_or(true)
}

fn stream_ingress_enabled() -> bool {
    std::env::var("KAPSL_LITE_INGRESS_STREAM_ENABLED")
        .ok()
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            normalized == "1" || normalized == "true" || normalized == "yes"
        })
        .unwrap_or(false)
}

fn parse_stream_audio_ring_chunks() -> usize {
    std::env::var("KAPSL_LITE_STREAM_AUDIO_RING_CHUNKS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(8)
        .max(1)
}

fn parse_stream_image_cache_enabled() -> bool {
    std::env::var("KAPSL_LITE_STREAM_IMAGE_CACHE_ENABLED")
        .ok()
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !(normalized == "0" || normalized == "false" || normalized == "no")
        })
        .unwrap_or(true)
}

fn summarize_event(event: &TriggerEvent) -> String {
    match event {
        TriggerEvent::ManualTrigger {
            target_package,
            custom_prompt,
        } => format!(
            "manual target={} prompt=\"{}\"",
            target_package,
            truncate_text(custom_prompt, 40)
        ),
        TriggerEvent::DetectionFired {
            label,
            confidence,
            source_package,
            target_package,
        } => format!(
            "detection target={} source={} label={} confidence={:.3}",
            target_package, source_package, label, confidence
        ),
        TriggerEvent::ExternalInput {
            target_package,
            event,
        } => format!(
            "external target={} source={} payload={}B metadata_keys={}",
            target_package,
            event.source_id,
            event.payload.size_hint_bytes(),
            event.metadata.len()
        ),
    }
}

fn truncate_text(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut output: String = value.chars().take(max_chars).collect();
    output.push_str("...");
    output
}

fn parse_required_string(
    obj: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<String, String> {
    parse_optional_string(obj, field)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("missing required string field '{}'", field))
}

fn parse_optional_string(obj: &serde_json::Map<String, Value>, field: &str) -> Option<String> {
    obj.get(field)
        .and_then(Value::as_str)
        .map(|value| value.trim().to_string())
}

fn parse_optional_u64(obj: &serde_json::Map<String, Value>, field: &str) -> Option<u64> {
    obj.get(field).and_then(Value::as_u64).or_else(|| {
        obj.get(field)
            .and_then(Value::as_i64)
            .filter(|v| *v >= 0)
            .map(|v| v as u64)
    })
}

fn parse_required_f64(obj: &serde_json::Map<String, Value>, field: &str) -> Result<f64, String> {
    let value = obj
        .get(field)
        .ok_or_else(|| format!("missing required number field '{}'", field))?;
    value
        .as_f64()
        .ok_or_else(|| format!("field '{}' must be a number", field))
}

fn parse_bytes_from_value(value: &Value) -> Option<Vec<u8>> {
    let arr = value.as_array()?;
    let mut bytes = Vec::with_capacity(arr.len());
    for item in arr {
        let value = item.as_u64()?;
        if value > 255 {
            return None;
        }
        bytes.push(value as u8);
    }
    Some(bytes)
}

fn parse_metadata_map(obj: &serde_json::Map<String, Value>) -> Result<Map<String, Value>, String> {
    match obj.get("metadata") {
        None => Ok(Map::new()),
        Some(Value::Null) => Ok(Map::new()),
        Some(Value::Object(map)) => Ok(map.clone()),
        Some(_) => Err("metadata must be an object".to_string()),
    }
}

fn parse_media_envelope(
    obj: &serde_json::Map<String, Value>,
) -> Result<Option<Map<String, Value>>, String> {
    match obj.get("media") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(media)) => {
            validate_media_envelope(media)?;
            Ok(Some(media.clone()))
        }
        Some(_) => Err("media must be an object".to_string()),
    }
}

fn validate_media_envelope(media: &serde_json::Map<String, Value>) -> Result<(), String> {
    let content_type = media
        .get("content_type")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "media.content_type is required".to_string())?
        .to_ascii_lowercase();

    match content_type.split('/').next().unwrap_or_default() {
        "audio" => {
            if let Some(rate) = media.get("sample_rate_hz").and_then(Value::as_u64)
                && rate == 0
            {
                return Err("media.sample_rate_hz must be > 0".to_string());
            }
            if let Some(channels) = media.get("channels").and_then(Value::as_u64)
                && channels == 0
            {
                return Err("media.channels must be > 0".to_string());
            }
        }
        "image" => {
            if let Some(width) = media.get("width").and_then(Value::as_u64)
                && width == 0
            {
                return Err("media.width must be > 0".to_string());
            }
            if let Some(height) = media.get("height").and_then(Value::as_u64)
                && height == 0
            {
                return Err("media.height must be > 0".to_string());
            }
        }
        "text" => {}
        _ => {
            return Err(format!("unsupported media.content_type '{}'", content_type));
        }
    }

    Ok(())
}

fn validate_media_payload_compatibility(
    media: &serde_json::Map<String, Value>,
    payload: &NormalizedPayload,
) -> Result<(), String> {
    let content_type = media
        .get("content_type")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default()
        .to_ascii_lowercase();

    let family = content_type.split('/').next().unwrap_or_default();
    match family {
        "audio" | "image" => {
            if !matches!(payload, NormalizedPayload::Bytes(_)) {
                return Err(format!(
                    "media payload type mismatch: content_type '{}' requires bytes payload",
                    content_type
                ));
            }
        }
        "text" => {}
        _ => {}
    }
    Ok(())
}

fn parse_normalized_payload(
    obj: &serde_json::Map<String, Value>,
) -> Result<NormalizedPayload, String> {
    if let Some(payload) = obj.get("payload") {
        if payload.is_null() {
            return Ok(NormalizedPayload::Empty);
        }
        if let Some(bytes) = parse_bytes_from_value(payload) {
            return Ok(NormalizedPayload::Bytes(bytes));
        }
        if let Some(bytes) = payload
            .as_object()
            .and_then(|map| map.get("bytes"))
            .and_then(parse_bytes_from_value)
        {
            return Ok(NormalizedPayload::Bytes(bytes));
        }
        return Ok(NormalizedPayload::Json(payload.clone()));
    }
    if let Some(bytes) = obj.get("bytes").and_then(parse_bytes_from_value) {
        return Ok(NormalizedPayload::Bytes(bytes));
    }
    Ok(NormalizedPayload::Empty)
}

fn parse_json_line_record(line: &str) -> Result<IngressRecord, String> {
    let payload: Value =
        serde_json::from_str(line).map_err(|error| format!("invalid JSON payload: {}", error))?;
    let obj = payload
        .as_object()
        .ok_or_else(|| "payload must be a JSON object".to_string())?;

    let event_type = obj
        .get("type")
        .and_then(Value::as_str)
        .map(|value| value.trim().to_ascii_lowercase())
        .or_else(|| {
            if obj.get("label").is_some() && obj.get("confidence").is_some() {
                Some("detection".to_string())
            } else if obj.get("custom_prompt").is_some() || obj.get("prompt").is_some() {
                Some("manual".to_string())
            } else if obj.get("source_id").is_some() {
                Some("input".to_string())
            } else {
                None
            }
        })
        .ok_or_else(|| "missing type (expected manual|detection|input)".to_string())?;

    match event_type.as_str() {
        "manual" | "manual_trigger" | "manual-trigger" => {
            let target_package = parse_required_string(obj, "target_package")?;
            let custom_prompt = parse_optional_string(obj, "custom_prompt")
                .or_else(|| parse_optional_string(obj, "prompt"))
                .unwrap_or_else(|| "external trigger".to_string());
            Ok(IngressRecord::Trigger(TriggerEvent::ManualTrigger {
                target_package,
                custom_prompt,
            }))
        }
        "detection" | "detection_fired" | "detection-fired" => {
            let target_package = parse_required_string(obj, "target_package")?;
            let label = parse_required_string(obj, "label")?;
            let confidence = parse_required_f64(obj, "confidence")?;
            if !(0.0..=1.0).contains(&confidence) {
                return Err("confidence must be in [0.0, 1.0]".to_string());
            }
            let source_package = parse_optional_string(obj, "source_package")
                .unwrap_or_else(|| "external".to_string());
            Ok(IngressRecord::Trigger(TriggerEvent::DetectionFired {
                label,
                confidence,
                source_package,
                target_package,
            }))
        }
        "input" | "ingress" | "event" => {
            let source_id = parse_required_string(obj, "source_id")?;
            let timestamp_ms =
                parse_optional_u64(obj, "timestamp_ms").unwrap_or_else(unix_time_millis);
            let mut metadata = parse_metadata_map(obj)?;
            let payload = parse_normalized_payload(obj)?;
            if let Some(media) = parse_media_envelope(obj)? {
                validate_media_payload_compatibility(&media, &payload)?;
                metadata.insert("_media".to_string(), Value::Object(media));
            }
            let explicit_target = parse_optional_string(obj, "target_package");
            Ok(IngressRecord::External {
                explicit_target,
                event: NormalizedInputEvent {
                    timestamp_ms,
                    source_id,
                    payload,
                    metadata,
                },
            })
        }
        other => Err(format!(
            "unsupported type '{}' (expected manual|detection|input)",
            other
        )),
    }
}

fn publish_event(
    trigger_bus: &TriggerBus,
    metrics: &RuntimeMetrics,
    event: TriggerEvent,
    backpressure: BackpressurePolicy,
    raw_payload_for_log: Option<&str>,
) {
    let event_summary = summarize_event(&event);
    match trigger_bus.publish_with_policy(event, backpressure) {
        Ok(()) => metrics.push_scheduler_log(
            EventLevel::Normal,
            format!("ingress accepted: {}", event_summary),
        ),
        Err(error) => metrics.push_scheduler_log(
            EventLevel::Warning,
            format!(
                "ingress dropped: reason={} payload=\"{}\"",
                error.as_reason(),
                raw_payload_for_log
                    .map(|payload| truncate_text(payload, 120))
                    .unwrap_or_else(|| "<binary-frame>".to_string())
            ),
        ),
    }
}

fn dispatch_external_input(
    trigger_bus: &TriggerBus,
    metrics: &RuntimeMetrics,
    route_table: &RouteTable,
    backpressure: BackpressurePolicy,
    explicit_target: Option<String>,
    event: NormalizedInputEvent,
    raw_payload_for_log: Option<&str>,
) {
    let targets = explicit_target
        .map(|target| vec![target])
        .unwrap_or_else(|| route_table.resolve_targets(&event.source_id));

    if targets.is_empty() {
        metrics.push_scheduler_log(
            EventLevel::Warning,
            format!(
                "ingress rejected: source={} reason=no-route",
                event.source_id
            ),
        );
        return;
    }

    let target_count = targets.len();
    let mut shared_event = Some(event);

    for (index, target) in targets.into_iter().enumerate() {
        let routed_event = if index + 1 == target_count {
            shared_event
                .take()
                .expect("event should exist for final route")
        } else {
            shared_event
                .as_ref()
                .expect("event should exist for fan-out cloning")
                .clone()
        };

        publish_event(
            trigger_bus,
            metrics,
            TriggerEvent::ExternalInput {
                target_package: target,
                event: routed_event,
            },
            backpressure,
            raw_payload_for_log,
        );
    }
}

#[derive(Debug)]
struct StreamMediaState {
    media_by_source: HashMap<String, Map<String, Value>>,
    audio_windows: HashMap<String, VecDeque<usize>>,
    latest_image_sizes: HashMap<String, usize>,
    audio_ring_chunks: usize,
    image_cache_enabled: bool,
}

impl StreamMediaState {
    fn from_env() -> Self {
        Self {
            media_by_source: HashMap::new(),
            audio_windows: HashMap::new(),
            latest_image_sizes: HashMap::new(),
            audio_ring_chunks: parse_stream_audio_ring_chunks(),
            image_cache_enabled: parse_stream_image_cache_enabled(),
        }
    }

    fn resolve_media(
        &mut self,
        source_id: &str,
        header_media: Option<Map<String, Value>>,
    ) -> Option<Map<String, Value>> {
        if let Some(media) = header_media {
            self.media_by_source
                .insert(source_id.to_string(), media.clone());
            Some(media)
        } else {
            self.media_by_source.get(source_id).cloned()
        }
    }

    fn annotate_metadata(
        &mut self,
        source_id: &str,
        media_family: Option<&str>,
        payload_size: usize,
        metadata: &mut Map<String, Value>,
    ) {
        match media_family {
            Some("audio") => {
                let window = self.audio_windows.entry(source_id.to_string()).or_default();
                window.push_back(payload_size);
                while window.len() > self.audio_ring_chunks {
                    let _ = window.pop_front();
                }
                let recent_bytes = window.iter().copied().sum::<usize>() as u64;
                metadata.insert(
                    "_audio_recent_chunks".to_string(),
                    Value::from(window.len() as u64),
                );
                metadata.insert("_audio_recent_bytes".to_string(), Value::from(recent_bytes));
            }
            Some("image") if self.image_cache_enabled => {
                self.latest_image_sizes
                    .insert(source_id.to_string(), payload_size);
                metadata.insert(
                    "_image_last_frame_bytes".to_string(),
                    Value::from(payload_size as u64),
                );
            }
            _ => {}
        }
    }
}

#[cfg(unix)]
fn setup_unix_listener(
    socket_path: &std::path::Path,
    metrics: &RuntimeMetrics,
    adapter_name: &str,
) -> Option<std::os::unix::net::UnixListener> {
    use std::os::unix::net::UnixListener;

    if let Some(parent) = socket_path.parent()
        && let Err(error) = std::fs::create_dir_all(parent)
    {
        metrics.push_scheduler_log(
            EventLevel::Critical,
            format!(
                "input adapter failed: name={} path={} error=failed-to-create-parent ({})",
                adapter_name,
                socket_path.display(),
                error
            ),
        );
        return None;
    }

    if socket_path.exists() {
        let _ = std::fs::remove_file(socket_path);
    }

    let listener = match UnixListener::bind(socket_path) {
        Ok(listener) => listener,
        Err(error) => {
            metrics.push_scheduler_log(
                EventLevel::Critical,
                format!(
                    "input adapter failed: name={} path={} error=bind-failed ({})",
                    adapter_name,
                    socket_path.display(),
                    error
                ),
            );
            return None;
        }
    };

    if let Err(error) = listener.set_nonblocking(true) {
        metrics.push_scheduler_log(
            EventLevel::Critical,
            format!(
                "input adapter failed: name={} path={} error=nonblocking-failed ({})",
                adapter_name,
                socket_path.display(),
                error
            ),
        );
        let _ = std::fs::remove_file(socket_path);
        return None;
    }

    Some(listener)
}

#[cfg(windows)]
fn create_named_pipe_instance(
    pipe_name: &str,
) -> std::io::Result<windows_sys::Win32::Foundation::HANDLE> {
    use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
    use windows_sys::Win32::Storage::FileSystem::PIPE_ACCESS_DUPLEX;
    use windows_sys::Win32::System::Pipes::{
        CreateNamedPipeW, PIPE_READMODE_BYTE, PIPE_TYPE_BYTE, PIPE_UNLIMITED_INSTANCES, PIPE_WAIT,
    };

    let wide_name = to_wide_null(pipe_name);
    // SAFETY: The pipe name is null-terminated and all pointers/values follow API contract.
    let handle = unsafe {
        CreateNamedPipeW(
            wide_name.as_ptr(),
            PIPE_ACCESS_DUPLEX,
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
            PIPE_UNLIMITED_INSTANCES,
            MAX_STREAM_PAYLOAD_BYTES as u32,
            MAX_STREAM_PAYLOAD_BYTES as u32,
            0,
            std::ptr::null_mut(),
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        return Err(std::io::Error::last_os_error());
    }
    Ok(handle)
}

#[cfg(windows)]
fn connect_named_pipe_instance(
    handle: windows_sys::Win32::Foundation::HANDLE,
) -> std::io::Result<()> {
    use windows_sys::Win32::Foundation::{ERROR_PIPE_CONNECTED, GetLastError};
    use windows_sys::Win32::System::Pipes::ConnectNamedPipe;

    // SAFETY: Handle was created by CreateNamedPipeW; overlapped pointer is null for blocking connect.
    let result = unsafe { ConnectNamedPipe(handle, std::ptr::null_mut()) };
    if result != 0 {
        return Ok(());
    }

    // SAFETY: Valid immediately after failed Win32 call.
    let error = unsafe { GetLastError() };
    if error == ERROR_PIPE_CONNECTED {
        Ok(())
    } else {
        Err(std::io::Error::from_raw_os_error(error as i32))
    }
}

#[cfg(windows)]
fn disconnect_and_close_pipe(handle: windows_sys::Win32::Foundation::HANDLE) {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Pipes::DisconnectNamedPipe;

    // SAFETY: Best-effort cleanup for a valid pipe handle.
    unsafe {
        let _ = DisconnectNamedPipe(handle);
        let _ = CloseHandle(handle);
    }
}

#[cfg(windows)]
fn to_wide_null(value: &str) -> Vec<u16> {
    value.encode_utf16().chain(std::iter::once(0)).collect()
}

#[cfg(windows)]
struct WindowsNamedPipeJsonIngressAdapter {
    pipe_name: String,
    accept_poll_ms: u64,
    route_table: RouteTable,
    backpressure: BackpressurePolicy,
}

#[cfg(windows)]
impl InputAdapter for WindowsNamedPipeJsonIngressAdapter {
    fn adapter_name(&self) -> &'static str {
        "windows-named-pipe-json-ingress"
    }

    fn run(self, trigger_bus: TriggerBus, metrics: Arc<RuntimeMetrics>, shutdown: Arc<AtomicBool>) {
        use std::io::{BufRead, BufReader};
        use std::os::windows::io::FromRawHandle;
        use std::sync::atomic::Ordering;
        use std::thread::JoinHandle;
        use std::time::Duration;

        metrics.push_scheduler_log(
            EventLevel::Normal,
            format!(
                "input adapter active: name={} pipe={} protocol=json-lines backpressure={:?}",
                self.adapter_name(),
                self.pipe_name,
                self.backpressure
            ),
        );

        let mut client_threads: Vec<JoinHandle<()>> = Vec::new();
        while !shutdown.load(Ordering::Relaxed) {
            client_threads.retain(|handle| !handle.is_finished());
            let handle = match create_named_pipe_instance(&self.pipe_name) {
                Ok(handle) => handle,
                Err(error) => {
                    metrics.push_scheduler_log(
                        EventLevel::Critical,
                        format!(
                            "input adapter failed: name={} pipe={} error=create-pipe-failed ({})",
                            self.adapter_name(),
                            self.pipe_name,
                            error
                        ),
                    );
                    std::thread::sleep(Duration::from_millis(self.accept_poll_ms));
                    continue;
                }
            };

            if let Err(error) = connect_named_pipe_instance(handle) {
                metrics.push_scheduler_log(
                    EventLevel::Warning,
                    format!(
                        "input adapter warning: name={} pipe={} error=connect-failed ({})",
                        self.adapter_name(),
                        self.pipe_name,
                        error
                    ),
                );
                disconnect_and_close_pipe(handle);
                std::thread::sleep(Duration::from_millis(self.accept_poll_ms));
                continue;
            }

            let handle_value = handle as isize;
            let trigger_bus_c = trigger_bus.clone();
            let metrics_c = metrics.clone();
            let route_table_c = self.route_table.clone();
            let shutdown_c = shutdown.clone();
            let backpressure = self.backpressure;

            client_threads.push(std::thread::spawn(move || {
                let handle = handle_value as windows_sys::Win32::Foundation::HANDLE;
                // SAFETY: `handle` is a connected pipe handle owned by this thread.
                let file =
                    unsafe { std::fs::File::from_raw_handle(handle as *mut std::ffi::c_void) };
                let mut reader = BufReader::new(file);
                let mut line = String::new();
                loop {
                    line.clear();
                    match reader.read_line(&mut line) {
                        Ok(0) => break,
                        Ok(_) => {
                            let payload = line.trim();
                            if payload.is_empty() {
                                continue;
                            }
                            match parse_json_line_record(payload) {
                                Ok(IngressRecord::Trigger(event)) => {
                                    publish_event(
                                        &trigger_bus_c,
                                        &metrics_c,
                                        event,
                                        backpressure,
                                        Some(payload),
                                    );
                                }
                                Ok(IngressRecord::External {
                                    explicit_target,
                                    event,
                                }) => {
                                    dispatch_external_input(
                                        &trigger_bus_c,
                                        &metrics_c,
                                        &route_table_c,
                                        backpressure,
                                        explicit_target,
                                        event,
                                        Some(payload),
                                    );
                                }
                                Err(error) => metrics_c.push_scheduler_log(
                                    EventLevel::Warning,
                                    format!(
                                        "ingress rejected: {} payload=\"{}\"",
                                        error,
                                        truncate_text(payload, 120)
                                    ),
                                ),
                            }
                        }
                        Err(error) if error.kind() == std::io::ErrorKind::BrokenPipe => break,
                        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
                        Err(_) => break,
                    }
                    if shutdown_c.load(Ordering::Relaxed) {
                        break;
                    }
                }
                // `reader`/file drops here and closes the handle.
            }));
        }

        for handle in client_threads {
            let _ = handle.join();
        }
    }
}

#[cfg(windows)]
struct WindowsNamedPipeStreamIngressAdapter {
    pipe_name: String,
    accept_poll_ms: u64,
    route_table: RouteTable,
    backpressure: BackpressurePolicy,
}

#[cfg(windows)]
impl InputAdapter for WindowsNamedPipeStreamIngressAdapter {
    fn adapter_name(&self) -> &'static str {
        "windows-named-pipe-stream-ingress"
    }

    fn run(self, trigger_bus: TriggerBus, metrics: Arc<RuntimeMetrics>, shutdown: Arc<AtomicBool>) {
        use std::io::Read;
        use std::os::windows::io::FromRawHandle;
        use std::sync::atomic::Ordering;
        use std::thread::JoinHandle;
        use std::time::Duration;

        metrics.push_scheduler_log(
            EventLevel::Normal,
            format!(
                "input adapter active: name={} pipe={} protocol=frame-v1 backpressure={:?} audio_ring_chunks={} image_cache_enabled={}",
                self.adapter_name(),
                self.pipe_name,
                self.backpressure,
                parse_stream_audio_ring_chunks(),
                parse_stream_image_cache_enabled()
            ),
        );

        let mut client_threads: Vec<JoinHandle<()>> = Vec::new();
        while !shutdown.load(Ordering::Relaxed) {
            client_threads.retain(|handle| !handle.is_finished());
            let handle = match create_named_pipe_instance(&self.pipe_name) {
                Ok(handle) => handle,
                Err(error) => {
                    metrics.push_scheduler_log(
                        EventLevel::Critical,
                        format!(
                            "input adapter failed: name={} pipe={} error=create-pipe-failed ({})",
                            self.adapter_name(),
                            self.pipe_name,
                            error
                        ),
                    );
                    std::thread::sleep(Duration::from_millis(self.accept_poll_ms));
                    continue;
                }
            };

            if let Err(error) = connect_named_pipe_instance(handle) {
                metrics.push_scheduler_log(
                    EventLevel::Warning,
                    format!(
                        "input adapter warning: name={} pipe={} error=connect-failed ({})",
                        self.adapter_name(),
                        self.pipe_name,
                        error
                    ),
                );
                disconnect_and_close_pipe(handle);
                std::thread::sleep(Duration::from_millis(self.accept_poll_ms));
                continue;
            }

            let handle_value = handle as isize;
            let trigger_bus_c = trigger_bus.clone();
            let metrics_c = metrics.clone();
            let route_table_c = self.route_table.clone();
            let shutdown_c = shutdown.clone();
            let backpressure = self.backpressure;

            client_threads.push(std::thread::spawn(move || {
                let handle = handle_value as windows_sys::Win32::Foundation::HANDLE;
                // SAFETY: `handle` is a connected pipe handle owned by this thread.
                let mut stream =
                    unsafe { std::fs::File::from_raw_handle(handle as *mut std::ffi::c_void) };
                let mut media_state = StreamMediaState::from_env();
                let mut header_bytes = Vec::new();
                loop {
                    let mut lengths = [0u8; 8];
                    match stream.read_exact(&mut lengths) {
                        Ok(()) => {}
                        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
                        Err(error) if error.kind() == std::io::ErrorKind::BrokenPipe => break,
                        Err(_) => break,
                    }

                    let header_len =
                        u32::from_be_bytes([lengths[0], lengths[1], lengths[2], lengths[3]])
                            as usize;
                    let payload_len =
                        u32::from_be_bytes([lengths[4], lengths[5], lengths[6], lengths[7]])
                            as usize;

                    if header_len == 0 || header_len > MAX_STREAM_HEADER_BYTES {
                        metrics_c.push_scheduler_log(
                            EventLevel::Warning,
                            format!(
                                "stream ingress rejected: invalid header length {} (max={})",
                                header_len, MAX_STREAM_HEADER_BYTES
                            ),
                        );
                        break;
                    }
                    if payload_len > MAX_STREAM_PAYLOAD_BYTES {
                        metrics_c.push_scheduler_log(
                            EventLevel::Warning,
                            format!(
                                "stream ingress rejected: invalid payload length {} (max={})",
                                payload_len, MAX_STREAM_PAYLOAD_BYTES
                            ),
                        );
                        break;
                    }

                    header_bytes.resize(header_len, 0);
                    if stream.read_exact(&mut header_bytes).is_err() {
                        break;
                    }
                    let mut payload_bytes = vec![0u8; payload_len];
                    if stream.read_exact(&mut payload_bytes).is_err() {
                        break;
                    }

                    match parse_stream_frame(&header_bytes, payload_bytes, &mut media_state) {
                        Ok((explicit_target, event)) => {
                            dispatch_external_input(
                                &trigger_bus_c,
                                &metrics_c,
                                &route_table_c,
                                backpressure,
                                explicit_target,
                                event,
                                None,
                            );
                        }
                        Err(error) => metrics_c.push_scheduler_log(
                            EventLevel::Warning,
                            format!("stream ingress rejected: {}", error),
                        ),
                    }
                    if shutdown_c.load(Ordering::Relaxed) {
                        break;
                    }
                }
                // `stream` drops here and closes the handle.
            }));
        }

        for handle in client_threads {
            let _ = handle.join();
        }
    }
}

#[cfg(unix)]
struct UnixJsonIngressAdapter {
    socket_path: std::path::PathBuf,
    accept_poll_ms: u64,
    route_table: RouteTable,
    backpressure: BackpressurePolicy,
}

#[cfg(unix)]
impl InputAdapter for UnixJsonIngressAdapter {
    fn adapter_name(&self) -> &'static str {
        "unix-json-ingress"
    }

    fn run(self, trigger_bus: TriggerBus, metrics: Arc<RuntimeMetrics>, shutdown: Arc<AtomicBool>) {
        use std::io::{BufRead, BufReader, ErrorKind};
        use std::os::unix::net::UnixStream;
        use std::sync::atomic::Ordering;
        use std::time::Duration;

        let Some(listener) = setup_unix_listener(&self.socket_path, &metrics, self.adapter_name())
        else {
            return;
        };

        metrics.push_scheduler_log(
            EventLevel::Normal,
            format!(
                "input adapter active: name={} path={} protocol=json-lines backpressure={:?}",
                self.adapter_name(),
                self.socket_path.display(),
                self.backpressure
            ),
        );

        while !shutdown.load(Ordering::Relaxed) {
            match listener.accept() {
                Ok((stream, _)) => {
                    handle_json_client(
                        stream,
                        &trigger_bus,
                        &metrics,
                        &shutdown,
                        &self.route_table,
                        self.backpressure,
                    );
                }
                Err(error) if error.kind() == ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(self.accept_poll_ms));
                }
                Err(error) => {
                    metrics.push_scheduler_log(
                        EventLevel::Warning,
                        format!(
                            "input adapter warning: name={} path={} error=accept-failed ({})",
                            self.adapter_name(),
                            self.socket_path.display(),
                            error
                        ),
                    );
                    std::thread::sleep(Duration::from_millis(self.accept_poll_ms));
                }
            }
        }

        let _ = std::fs::remove_file(&self.socket_path);

        fn handle_json_client(
            stream: UnixStream,
            trigger_bus: &TriggerBus,
            metrics: &RuntimeMetrics,
            shutdown: &AtomicBool,
            route_table: &RouteTable,
            backpressure: BackpressurePolicy,
        ) {
            use std::io::ErrorKind;
            use std::sync::atomic::Ordering;
            use std::time::Duration;

            let _ = stream.set_read_timeout(Some(Duration::from_millis(250)));
            let mut reader = BufReader::new(stream);
            let mut line = String::new();
            loop {
                if shutdown.load(Ordering::Relaxed) {
                    break;
                }
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        let payload = line.trim();
                        if payload.is_empty() {
                            continue;
                        }
                        match parse_json_line_record(payload) {
                            Ok(IngressRecord::Trigger(event)) => {
                                publish_event(
                                    trigger_bus,
                                    metrics,
                                    event,
                                    backpressure,
                                    Some(payload),
                                );
                            }
                            Ok(IngressRecord::External {
                                explicit_target,
                                event,
                            }) => {
                                dispatch_external_input(
                                    trigger_bus,
                                    metrics,
                                    route_table,
                                    backpressure,
                                    explicit_target,
                                    event,
                                    Some(payload),
                                );
                            }
                            Err(error) => metrics.push_scheduler_log(
                                EventLevel::Warning,
                                format!(
                                    "ingress rejected: {} payload=\"{}\"",
                                    error,
                                    truncate_text(payload, 120)
                                ),
                            ),
                        }
                    }
                    Err(error)
                        if error.kind() == ErrorKind::WouldBlock
                            || error.kind() == ErrorKind::TimedOut =>
                    {
                        continue;
                    }
                    Err(_) => break,
                }
            }
        }
    }
}

#[cfg(unix)]
struct UnixStreamIngressAdapter {
    socket_path: std::path::PathBuf,
    accept_poll_ms: u64,
    route_table: RouteTable,
    backpressure: BackpressurePolicy,
}

#[cfg(unix)]
impl InputAdapter for UnixStreamIngressAdapter {
    fn adapter_name(&self) -> &'static str {
        "unix-stream-ingress"
    }

    fn run(self, trigger_bus: TriggerBus, metrics: Arc<RuntimeMetrics>, shutdown: Arc<AtomicBool>) {
        use std::io::{ErrorKind, Read};
        use std::os::unix::net::UnixStream;
        use std::sync::atomic::Ordering;
        use std::time::Duration;

        let Some(listener) = setup_unix_listener(&self.socket_path, &metrics, self.adapter_name())
        else {
            return;
        };

        metrics.push_scheduler_log(
            EventLevel::Normal,
            format!(
                "input adapter active: name={} path={} protocol=frame-v1 backpressure={:?} audio_ring_chunks={} image_cache_enabled={}",
                self.adapter_name(),
                self.socket_path.display(),
                self.backpressure,
                parse_stream_audio_ring_chunks(),
                parse_stream_image_cache_enabled()
            ),
        );

        while !shutdown.load(Ordering::Relaxed) {
            match listener.accept() {
                Ok((stream, _)) => {
                    handle_stream_client(
                        stream,
                        &trigger_bus,
                        &metrics,
                        &shutdown,
                        &self.route_table,
                        self.backpressure,
                    );
                }
                Err(error) if error.kind() == ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(self.accept_poll_ms));
                }
                Err(error) => {
                    metrics.push_scheduler_log(
                        EventLevel::Warning,
                        format!(
                            "input adapter warning: name={} path={} error=accept-failed ({})",
                            self.adapter_name(),
                            self.socket_path.display(),
                            error
                        ),
                    );
                    std::thread::sleep(Duration::from_millis(self.accept_poll_ms));
                }
            }
        }

        let _ = std::fs::remove_file(&self.socket_path);

        fn handle_stream_client(
            mut stream: UnixStream,
            trigger_bus: &TriggerBus,
            metrics: &RuntimeMetrics,
            shutdown: &AtomicBool,
            route_table: &RouteTable,
            backpressure: BackpressurePolicy,
        ) {
            use std::sync::atomic::Ordering;
            use std::time::Duration;

            let _ = stream.set_read_timeout(Some(Duration::from_millis(250)));
            let mut media_state = StreamMediaState::from_env();
            let mut header_bytes = Vec::new();
            loop {
                if shutdown.load(Ordering::Relaxed) {
                    break;
                }

                let mut lengths = [0u8; 8];
                match stream.read_exact(&mut lengths) {
                    Ok(()) => {}
                    Err(error)
                        if error.kind() == std::io::ErrorKind::WouldBlock
                            || error.kind() == std::io::ErrorKind::TimedOut =>
                    {
                        continue;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(_) => break,
                }

                let header_len =
                    u32::from_be_bytes([lengths[0], lengths[1], lengths[2], lengths[3]]) as usize;
                let payload_len =
                    u32::from_be_bytes([lengths[4], lengths[5], lengths[6], lengths[7]]) as usize;

                if header_len == 0 || header_len > MAX_STREAM_HEADER_BYTES {
                    metrics.push_scheduler_log(
                        EventLevel::Warning,
                        format!(
                            "stream ingress rejected: invalid header length {} (max={})",
                            header_len, MAX_STREAM_HEADER_BYTES
                        ),
                    );
                    break;
                }
                if payload_len > MAX_STREAM_PAYLOAD_BYTES {
                    metrics.push_scheduler_log(
                        EventLevel::Warning,
                        format!(
                            "stream ingress rejected: invalid payload length {} (max={})",
                            payload_len, MAX_STREAM_PAYLOAD_BYTES
                        ),
                    );
                    break;
                }

                header_bytes.resize(header_len, 0);
                if stream.read_exact(&mut header_bytes).is_err() {
                    break;
                }
                let mut payload_bytes = vec![0u8; payload_len];
                if stream.read_exact(&mut payload_bytes).is_err() {
                    break;
                }

                match parse_stream_frame(&header_bytes, payload_bytes, &mut media_state) {
                    Ok((explicit_target, event)) => {
                        dispatch_external_input(
                            trigger_bus,
                            metrics,
                            route_table,
                            backpressure,
                            explicit_target,
                            event,
                            None,
                        );
                    }
                    Err(error) => metrics.push_scheduler_log(
                        EventLevel::Warning,
                        format!("stream ingress rejected: {}", error),
                    ),
                }
            }
        }
    }
}

fn parse_stream_frame(
    header_bytes: &[u8],
    payload_bytes: Vec<u8>,
    media_state: &mut StreamMediaState,
) -> Result<(Option<String>, NormalizedInputEvent), String> {
    let header: Value = serde_json::from_slice(header_bytes)
        .map_err(|error| format!("header JSON decode failed: {}", error))?;
    let header_obj = header
        .as_object()
        .ok_or_else(|| "header must be a JSON object".to_string())?;

    let source_id = parse_required_string(header_obj, "source_id")?;
    let timestamp_ms =
        parse_optional_u64(header_obj, "timestamp_ms").unwrap_or_else(unix_time_millis);
    let mut metadata = match header_obj.get("metadata") {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(map)) => map.clone(),
        Some(_) => {
            return Err("header.metadata must be an object".to_string());
        }
    };
    let resolved_media = media_state.resolve_media(&source_id, parse_media_envelope(header_obj)?);
    if let Some(media) = resolved_media.clone() {
        metadata.insert("_media".to_string(), Value::Object(media));
    }
    let explicit_target = parse_optional_string(header_obj, "target_package");
    let media_family = resolved_media
        .as_ref()
        .and_then(|media| media.get("content_type"))
        .and_then(Value::as_str)
        .map(str::trim)
        .map(|content_type| {
            content_type
                .split('/')
                .next()
                .unwrap_or_default()
                .to_ascii_lowercase()
        });

    let payload = if payload_bytes.is_empty() {
        NormalizedPayload::Empty
    } else if media_family.as_deref() == Some("text") {
        let text_payload = std::str::from_utf8(&payload_bytes)
            .map_err(|_| {
                "media payload type mismatch: text media requires valid utf-8 bytes".to_string()
            })?
            .to_string();
        NormalizedPayload::Json(Value::String(text_payload))
    } else {
        NormalizedPayload::Bytes(payload_bytes)
    };
    if let Some(Value::Object(media)) = metadata.get("_media") {
        validate_media_payload_compatibility(media, &payload)?;
    }
    media_state.annotate_metadata(
        &source_id,
        media_family.as_deref(),
        payload.size_hint_bytes(),
        &mut metadata,
    );

    Ok((
        explicit_target,
        NormalizedInputEvent {
            timestamp_ms,
            source_id,
            payload,
            metadata,
        },
    ))
}

fn unix_time_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_manual_event() {
        let event = parse_json_line_record(
            r#"{"type":"manual","target_package":"reasoning.edge-llm","custom_prompt":"hello"}"#,
        )
        .expect("manual event should parse");

        match event {
            IngressRecord::Trigger(TriggerEvent::ManualTrigger {
                target_package,
                custom_prompt,
            }) => {
                assert_eq!(target_package, "reasoning.edge-llm");
                assert_eq!(custom_prompt, "hello");
            }
            _ => panic!("expected manual trigger"),
        }
    }

    #[test]
    fn parse_detection_event() {
        let event = parse_json_line_record(
            r#"{"type":"detection","target_package":"detector","source_package":"camera.front","label":"person","confidence":0.97}"#,
        )
        .expect("detection event should parse");

        match event {
            IngressRecord::Trigger(TriggerEvent::DetectionFired {
                label,
                confidence,
                source_package,
                target_package,
            }) => {
                assert_eq!(label, "person");
                assert_eq!(confidence, 0.97);
                assert_eq!(source_package, "camera.front");
                assert_eq!(target_package, "detector");
            }
            _ => panic!("expected detection event"),
        }
    }

    #[test]
    fn parse_external_input_event() {
        let event = parse_json_line_record(
            r#"{"type":"input","source_id":"camera/front","timestamp_ms":1,"payload":{"x":1},"metadata":{"site":"edge-a"}}"#,
        )
        .expect("external input should parse");

        match event {
            IngressRecord::External {
                explicit_target,
                event,
            } => {
                assert!(explicit_target.is_none());
                assert_eq!(event.source_id, "camera/front");
                assert_eq!(event.timestamp_ms, 1);
                assert_eq!(
                    event.metadata.get("site").and_then(Value::as_str),
                    Some("edge-a")
                );
            }
            _ => panic!("expected external input event"),
        }
    }

    #[test]
    fn parse_rejects_invalid_confidence() {
        let result = parse_json_line_record(
            r#"{"type":"detection","target_package":"detector","label":"person","confidence":2.5}"#,
        );
        assert!(result.is_err());
    }

    #[test]
    fn route_table_resolves_exact_and_wildcard() {
        let table = RouteTable {
            rules: vec![
                RouteRule {
                    source_pattern: "camera/front".to_string(),
                    target_package: "detector-a".to_string(),
                },
                RouteRule {
                    source_pattern: "sensor/*".to_string(),
                    target_package: "anomaly-model".to_string(),
                },
            ],
            default_target: Some("default-model".to_string()),
        };

        assert_eq!(table.resolve_targets("camera/front"), vec!["detector-a"]);
        assert_eq!(table.resolve_targets("sensor/temp"), vec!["anomaly-model"]);
        assert_eq!(table.resolve_targets("unknown"), vec!["default-model"]);
    }

    #[test]
    fn parse_stream_frame_basic() {
        let header = br#"{"source_id":"camera/front","target_package":"detector"}"#;
        let payload = vec![1u8, 2u8, 3u8];
        let mut media_state = StreamMediaState::from_env();
        let (target, event) =
            parse_stream_frame(header, payload, &mut media_state).expect("frame should parse");
        assert_eq!(target.as_deref(), Some("detector"));
        assert_eq!(event.source_id, "camera/front");
        match event.payload {
            NormalizedPayload::Bytes(bytes) => assert_eq!(bytes, vec![1u8, 2u8, 3u8]),
            _ => panic!("expected bytes payload"),
        }
    }

    #[test]
    fn parse_stream_frame_text_media_promotes_to_json_string() {
        let header = br#"{"source_id":"mic/front","media":{"content_type":"text/plain"}}"#;
        let payload = b"hello world".to_vec();
        let mut media_state = StreamMediaState::from_env();
        let (_, event) =
            parse_stream_frame(header, payload, &mut media_state).expect("frame should parse");
        match event.payload {
            NormalizedPayload::Json(Value::String(text)) => assert_eq!(text, "hello world"),
            _ => panic!("expected text payload"),
        }
    }

    #[test]
    fn parse_stream_frame_rejects_non_utf8_text_media() {
        let header = br#"{"source_id":"mic/front","media":{"content_type":"text/plain"}}"#;
        let payload = vec![0xff, 0xfe];
        let mut media_state = StreamMediaState::from_env();
        let result = parse_stream_frame(header, payload, &mut media_state);
        assert!(result.is_err());
    }

    #[test]
    fn parse_stream_frame_uses_cached_media_profile_for_source() {
        let mut media_state = StreamMediaState::from_env();
        let header_with_media =
            br#"{"source_id":"camera/front","media":{"content_type":"image/jpeg","width":640,"height":480}}"#;
        let header_without_media = br#"{"source_id":"camera/front"}"#;

        let first = parse_stream_frame(header_with_media, vec![1u8, 2u8], &mut media_state)
            .expect("first frame should parse");
        assert!(first.1.metadata.contains_key("_media"));

        let second = parse_stream_frame(header_without_media, vec![3u8, 4u8], &mut media_state)
            .expect("second frame should parse");
        assert!(second.1.metadata.contains_key("_media"));
    }
}
