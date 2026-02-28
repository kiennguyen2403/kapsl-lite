use crate::metrics::{EventLevel, InferenceResult, ModelRunState, RuntimeMetrics};
use crate::package::{
    BackendAffinity, KapslPackageSpec, ModelFormat, Priority, QueueOverflowPolicy, TaskClass,
    TriggerMode,
};
use crate::system::SystemSampler;
use crate::trigger::{NormalizedInputEvent, NormalizedPayload, TriggerBus, TriggerEvent};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TryRecvError, TrySendError, bounded};
use futures::executor::block_on;
use kapsl_backends::OnnxBackend;
use kapsl_backends::onnx::OnnxBackendBuilder;
use kapsl_engine_api::{
    BinaryTensorPacket, Engine, InferenceRequest, RequestMetadata, TensorDtype,
};
use kapsl_llm::llm_backend::LLMBackend;
use serde_json::{Map, Value};
use std::cmp::Ordering as CmpOrdering;
use std::collections::{BinaryHeap, HashMap};
use std::env;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

// ── RuntimeConfig ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub memory_limit_mib: u64,
    pub trigger_confidence_threshold: f64,
    pub trigger_required_free_mib: u64,
    pub trigger_bus_capacity: usize,
    pub thermal_poll_interval_ms: u64,
    pub thermal_soft_threshold_c: f64,
    pub thermal_degraded_threshold_c: f64,
    pub thermal_hard_threshold_c: f64,
    pub thermal_recovery_threshold_c: f64,
    pub thermal_throttle_fps_factor: f64,
    pub battery_poll_interval_ms: u64,
    pub battery_low_threshold_percent: f64,
    pub battery_critical_threshold_percent: f64,
    pub battery_conserve_fps_factor: f64,
    pub battery_critical_fps_factor: f64,
    pub battery_critical_max_tokens: u64,
    pub memory_guard_poll_interval_ms: u64,
    pub memory_emergency_free_mib: u64,
    pub memory_recovery_free_mib: u64,
    pub memory_emergency_max_tokens: u64,
}

impl RuntimeConfig {
    pub fn from_env() -> Self {
        let thermal_soft_threshold_c = read_env_f64("KAPSL_LITE_THERMAL_SOFT_C", 70.0).max(0.0);
        let thermal_hard_threshold_c =
            read_env_f64("KAPSL_LITE_THERMAL_HARD_C", 82.0).max(thermal_soft_threshold_c);
        let thermal_degraded_threshold_c = read_env_f64("KAPSL_LITE_THERMAL_DEGRADED_C", 76.0)
            .clamp(thermal_soft_threshold_c, thermal_hard_threshold_c);
        let thermal_recovery_threshold_c = read_env_f64("KAPSL_LITE_THERMAL_RECOVERY_C", 65.0)
            .clamp(0.0, thermal_soft_threshold_c);
        let battery_low_threshold_percent =
            read_env_f64("KAPSL_LITE_BATTERY_LOW_PERCENT", 30.0).clamp(1.0, 100.0);
        let battery_critical_threshold_percent =
            read_env_f64("KAPSL_LITE_BATTERY_CRITICAL_PERCENT", 15.0)
                .clamp(0.0, battery_low_threshold_percent);
        let memory_emergency_free_mib = read_env_u64("KAPSL_LITE_MEMORY_EMERGENCY_FREE_MIB", 64);
        let memory_recovery_free_mib = read_env_u64("KAPSL_LITE_MEMORY_RECOVERY_FREE_MIB", 96)
            .max(memory_emergency_free_mib.saturating_add(8));

        Self {
            memory_limit_mib: read_env_u64("KAPSL_LITE_MEMORY_LIMIT_MIB", 512).max(64),
            trigger_confidence_threshold: read_env_f64(
                "KAPSL_LITE_TRIGGER_CONFIDENCE_THRESHOLD",
                0.80,
            )
            .clamp(0.0, 1.0),
            trigger_required_free_mib: read_env_u64("KAPSL_LITE_TRIGGER_REQUIRED_FREE_MIB", 128),
            trigger_bus_capacity: read_env_usize("KAPSL_LITE_TRIGGER_BUS_CAPACITY", 256).max(1),
            thermal_poll_interval_ms: read_env_u64("KAPSL_LITE_THERMAL_POLL_INTERVAL_MS", 2000)
                .max(100),
            thermal_soft_threshold_c,
            thermal_degraded_threshold_c,
            thermal_hard_threshold_c,
            thermal_recovery_threshold_c,
            thermal_throttle_fps_factor: read_env_f64(
                "KAPSL_LITE_THERMAL_THROTTLE_FPS_FACTOR",
                0.50,
            )
            .clamp(0.05, 1.0),
            battery_poll_interval_ms: read_env_u64("KAPSL_LITE_BATTERY_POLL_INTERVAL_MS", 10000)
                .max(1000),
            battery_low_threshold_percent,
            battery_critical_threshold_percent,
            battery_conserve_fps_factor: read_env_f64(
                "KAPSL_LITE_BATTERY_CONSERVE_FPS_FACTOR",
                0.75,
            )
            .clamp(0.1, 1.0),
            battery_critical_fps_factor: read_env_f64(
                "KAPSL_LITE_BATTERY_CRITICAL_FPS_FACTOR",
                0.50,
            )
            .clamp(0.05, 1.0),
            battery_critical_max_tokens: read_env_u64(
                "KAPSL_LITE_BATTERY_CRITICAL_MAX_TOKENS",
                128,
            )
            .max(8),
            memory_guard_poll_interval_ms: read_env_u64(
                "KAPSL_LITE_MEMORY_GUARD_POLL_INTERVAL_MS",
                1000,
            )
            .max(100),
            memory_emergency_free_mib,
            memory_recovery_free_mib,
            memory_emergency_max_tokens: read_env_u64("KAPSL_LITE_MEMORY_EMERGENCY_MAX_TOKENS", 96)
                .max(8),
        }
    }
}

// ── InferenceBackend ──────────────────────────────────────────────────────────

/// Outcome of a single inference call.
pub struct InferenceOutcome {
    pub latency_ms: u64,
    pub output_summary: String,
    pub success: bool,
}

#[derive(Debug, Clone, Default)]
pub struct InferenceRunOptions {
    pub max_new_tokens: Option<u64>,
    pub max_wall_ms: Option<u64>,
    pub cancellation: Option<Arc<AtomicBool>>,
}

/// Abstraction over a model inference backend.  All implementations must be
/// `Send` so they can be moved into their dedicated runner thread.
pub trait InferenceBackend: Send {
    fn run(
        &mut self,
        package_name: &str,
        format: &str,
        input: Option<&NormalizedInputEvent>,
        options: InferenceRunOptions,
    ) -> InferenceOutcome;
}

/// Stub backend used when no real inference engine is linked.  Logs a notice
/// on first use and returns a zero-latency no-op outcome so the scheduler,
/// trigger system, and TUI all function end-to-end.
pub struct StubBackend {
    noticed: bool,
}

impl StubBackend {
    fn new() -> Self {
        Self { noticed: false }
    }
}

impl InferenceBackend for StubBackend {
    fn run(
        &mut self,
        package_name: &str,
        format: &str,
        input: Option<&NormalizedInputEvent>,
        options: InferenceRunOptions,
    ) -> InferenceOutcome {
        let start = Instant::now();
        if options
            .cancellation
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
        {
            return InferenceOutcome {
                latency_ms: start.elapsed().as_millis() as u64,
                output_summary: format!("stub:{} cancelled=pre-run", package_name),
                success: false,
            };
        }

        if !self.noticed {
            eprintln!(
                "[kapsl-lite] StubBackend active for package '{}' (format={}). \
                 Link a real inference backend to get actual outputs.",
                package_name, format
            );
            self.noticed = true;
        }

        let output_summary = match input {
            Some(input) => {
                let payload_bytes = input.payload.size_hint_bytes();
                if let Some(max_new_tokens) = options.max_new_tokens {
                    format!(
                        "stub:{} source={} payload={}B max_new_tokens={}",
                        package_name, input.source_id, payload_bytes, max_new_tokens
                    )
                } else {
                    format!(
                        "stub:{} source={} payload={}B",
                        package_name, input.source_id, payload_bytes
                    )
                }
            }
            None => {
                if let Some(max_new_tokens) = options.max_new_tokens {
                    format!("stub:{} max_new_tokens={}", package_name, max_new_tokens)
                } else {
                    format!("stub:{}", package_name)
                }
            }
        };

        InferenceOutcome {
            latency_ms: start.elapsed().as_millis() as u64,
            output_summary,
            success: options
                .max_wall_ms
                .map(|max_ms| start.elapsed().as_millis() as u64 <= max_ms)
                .unwrap_or(true),
        }
    }
}

struct LlmRuntimeBackend {
    engine: Box<dyn Engine>,
}

impl LlmRuntimeBackend {
    fn try_new(spec: &KapslPackageSpec) -> Result<Self, String> {
        let mut backend = LLMBackend::new();
        let model_path = Path::new(&spec.weights_path);
        if !model_path.exists() {
            return Err(format!(
                "resolved model path does not exist: {}",
                model_path.display()
            ));
        }

        block_on(backend.load(model_path))
            .map_err(|error| format!("failed to load LLM backend: {}", error))?;

        Ok(Self {
            engine: Box::new(backend),
        })
    }
}

impl InferenceBackend for LlmRuntimeBackend {
    fn run(
        &mut self,
        package_name: &str,
        _format: &str,
        input: Option<&NormalizedInputEvent>,
        options: InferenceRunOptions,
    ) -> InferenceOutcome {
        let started = Instant::now();
        let source = input
            .map(|event| event.source_id.as_str())
            .unwrap_or("runtime");

        if options
            .cancellation
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
        {
            return InferenceOutcome {
                latency_ms: started.elapsed().as_millis() as u64,
                output_summary: format!("llm:{} source={} cancelled=pre-run", package_name, source),
                success: false,
            };
        }

        let Some(prompt) = extract_prompt_from_input(input) else {
            return InferenceOutcome {
                latency_ms: started.elapsed().as_millis() as u64,
                output_summary: format!(
                    "llm:{} source={} error=missing-text-prompt",
                    package_name, source
                ),
                success: false,
            };
        };

        if prompt.trim().is_empty() {
            return InferenceOutcome {
                latency_ms: started.elapsed().as_millis() as u64,
                output_summary: format!(
                    "llm:{} source={} error=empty-text-prompt",
                    package_name, source
                ),
                success: false,
            };
        }

        let bytes = prompt.into_bytes();
        let input_packet = match BinaryTensorPacket::new(
            vec![1, bytes.len().max(1) as i64],
            TensorDtype::Utf8,
            bytes,
        ) {
            Ok(packet) => packet,
            Err(error) => {
                return InferenceOutcome {
                    latency_ms: started.elapsed().as_millis() as u64,
                    output_summary: format!(
                        "llm:{} source={} error=input-packet-build-failed ({})",
                        package_name, source, error
                    ),
                    success: false,
                };
            }
        };

        let default_max_new_tokens =
            read_env_u64("KAPSL_LITE_LLM_DEFAULT_MAX_NEW_TOKENS", 64).max(8);
        let effective_max_new_tokens = options.max_new_tokens.unwrap_or(default_max_new_tokens);

        let mut request = InferenceRequest::new(input_packet);
        let mut metadata = RequestMetadata::default();
        metadata.max_new_tokens = Some(effective_max_new_tokens.min(u32::MAX as u64) as u32);
        request.metadata = Some(metadata);

        match self.engine.infer(&request) {
            Ok(output) => {
                let text = String::from_utf8_lossy(&output.data).to_string();
                let latency_ms = started.elapsed().as_millis() as u64;
                if let Some(max_wall_ms) = options.max_wall_ms
                    && latency_ms > max_wall_ms
                {
                    return InferenceOutcome {
                        latency_ms,
                        output_summary: format!(
                            "llm:{} source={} cancelled=deadline_exceeded wall={}ms max={}ms",
                            package_name, source, latency_ms, max_wall_ms
                        ),
                        success: false,
                    };
                }
                InferenceOutcome {
                    latency_ms,
                    output_summary: format!(
                        "llm:{} source={} text=\"{}\"",
                        package_name,
                        source,
                        truncate_llm_text_for_summary(&text),
                    ),
                    success: true,
                }
            }
            Err(error) => InferenceOutcome {
                latency_ms: started.elapsed().as_millis() as u64,
                output_summary: format!("llm:{} source={} error={}", package_name, source, error),
                success: false,
            },
        }
    }
}

struct AuxiliaryOnnxStage {
    name: String,
    engine: Box<dyn Engine>,
    input_shape: Option<Vec<i64>>,
    input_dtype: TensorDtype,
}

impl AuxiliaryOnnxStage {
    fn infer(&self, input: BinaryTensorPacket) -> Result<BinaryTensorPacket, String> {
        self.engine
            .infer(&InferenceRequest::new(input))
            .map_err(|error| format!("{} inference failed: {}", self.name, error))
    }
}

struct OnnxRuntimeBackend {
    engine: Box<dyn Engine>,
    input_name: Option<String>,
    input_shape: Option<Vec<i64>>,
    input_dtype: TensorDtype,
    text_embed_stage: Option<AuxiliaryOnnxStage>,
    vision_stage: Option<AuxiliaryOnnxStage>,
}

impl OnnxRuntimeBackend {
    fn tuned_builder(spec: &KapslPackageSpec) -> OnnxBackendBuilder {
        let mut builder = OnnxBackend::builder();
        if let Some(value) = spec.onnx_tuning.memory_pattern {
            builder = builder.with_memory_pattern(value);
        }
        if let Some(value) = spec.onnx_tuning.disable_cpu_mem_arena {
            builder = builder.with_disable_cpu_mem_arena(value);
        }
        if let Some(value) = spec.onnx_tuning.session_buckets {
            builder = builder.with_max_bucket_sessions(value);
        }
        if let Some(value) = spec.onnx_tuning.bucket_dim_granularity {
            builder = builder.with_bucket_dim_granularity(value);
        }
        if let Some(value) = spec.onnx_tuning.bucket_max_dims {
            builder = builder.with_bucket_max_dims(value);
        }
        if let Some(value) = spec.onnx_tuning.peak_concurrency_hint {
            builder = builder.with_peak_concurrency_hint(value);
        }
        builder
    }

    fn try_new(spec: &KapslPackageSpec) -> Result<Self, String> {
        let mut engine: Box<dyn Engine> = Box::new(Self::tuned_builder(spec).build());
        let model_path = Path::new(&spec.weights_path);
        if !model_path.exists() {
            return Err(format!(
                "resolved model path does not exist: {}",
                model_path.display()
            ));
        }

        block_on(engine.load(model_path))
            .map_err(|error| format!("failed to load ONNX model: {}", error))?;

        let model_info = engine.model_info();
        let input_name = model_info
            .as_ref()
            .and_then(|info| info.input_names.first().cloned());
        let input_shape = model_info
            .as_ref()
            .and_then(|info| info.input_shapes.first().cloned());
        let input_dtype = model_info
            .as_ref()
            .and_then(|info| info.input_dtypes.first())
            .and_then(|dtype| TensorDtype::from_str(dtype).ok())
            .unwrap_or(TensorDtype::Float32);

        let mut text_embed_stage = None;
        let mut vision_stage = None;
        if spec.is_llm && input_name.as_deref() == Some("inputs_embeds") {
            text_embed_stage = Self::try_load_aux_stage(
                spec,
                model_path,
                "embed_tokens",
                &[
                    "embed_tokens_q4f16.onnx",
                    "embed_tokens_fp16.onnx",
                    "embed_tokens.onnx",
                ],
            )?;
            vision_stage = Self::try_load_aux_stage(
                spec,
                model_path,
                "vision_encoder",
                &[
                    "vision_encoder_q4f16.onnx",
                    "vision_encoder_fp16.onnx",
                    "vision_encoder.onnx",
                ],
            )?;
        }

        Ok(Self {
            engine,
            input_name,
            input_shape,
            input_dtype,
            text_embed_stage,
            vision_stage,
        })
    }

    fn resolve_stage_path(model_path: &Path, candidates: &[&str]) -> Option<PathBuf> {
        let parent = model_path.parent()?;
        for candidate in candidates {
            let path = parent.join(candidate);
            if path.exists() {
                return Some(path);
            }
        }

        let mut prefixes = candidates
            .iter()
            .filter_map(|name| name.strip_suffix(".onnx"))
            .collect::<Vec<_>>();
        prefixes.sort_unstable();
        prefixes.dedup();
        for entry in std::fs::read_dir(parent).ok()?.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|v| v.to_str()) else {
                continue;
            };
            if !name.ends_with(".onnx") {
                continue;
            }
            if prefixes.iter().any(|prefix| name.starts_with(prefix)) {
                return Some(path);
            }
        }
        None
    }

    fn try_load_aux_stage(
        spec: &KapslPackageSpec,
        model_path: &Path,
        stage_name: &str,
        candidates: &[&str],
    ) -> Result<Option<AuxiliaryOnnxStage>, String> {
        let Some(stage_path) = Self::resolve_stage_path(model_path, candidates) else {
            return Ok(None);
        };

        let mut engine: Box<dyn Engine> = Box::new(Self::tuned_builder(spec).build());
        block_on(engine.load(&stage_path))
            .map_err(|error| format!("failed to load {} stage: {}", stage_name, error))?;
        let model_info = engine.model_info();
        let input_shape = model_info
            .as_ref()
            .and_then(|info| info.input_shapes.first().cloned());
        let input_dtype = model_info
            .as_ref()
            .and_then(|info| info.input_dtypes.first())
            .and_then(|dtype| TensorDtype::from_str(dtype).ok())
            .unwrap_or(TensorDtype::Float32);

        Ok(Some(AuxiliaryOnnxStage {
            name: stage_name.to_string(),
            engine,
            input_shape,
            input_dtype,
        }))
    }

    fn build_input_tensor(
        &self,
        input: Option<&NormalizedInputEvent>,
    ) -> Result<BinaryTensorPacket, String> {
        let fallback_elements = fixed_shape_elements(self.input_shape.as_deref()).unwrap_or(1);

        match self.input_dtype {
            TensorDtype::Uint8 => {
                let mut values = extract_bytes_payload(input);
                if values.is_empty() {
                    values = vec![0u8; fallback_elements];
                }

                let desired_elements = values.len().max(1);
                let shape = resolve_shape(self.input_shape.as_deref(), desired_elements);
                let required_elements = shape_elements(&shape).max(1);
                values.resize(required_elements, 0);
                if values.len() > required_elements {
                    values.truncate(required_elements);
                }

                BinaryTensorPacket::new(shape, TensorDtype::Uint8, values)
                    .map_err(|error| format!("failed to build uint8 input tensor: {}", error))
            }
            TensorDtype::Int64 => {
                let mut values = extract_i64_payload(input);
                if values.is_empty() {
                    values = vec![0i64; fallback_elements];
                }

                let desired_elements = values.len().max(1);
                let shape = resolve_shape(self.input_shape.as_deref(), desired_elements);
                let required_elements = shape_elements(&shape).max(1);
                values.resize(required_elements, 0);
                if values.len() > required_elements {
                    values.truncate(required_elements);
                }

                let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<i64>());
                for value in values {
                    bytes.extend_from_slice(&value.to_ne_bytes());
                }

                BinaryTensorPacket::new(shape, TensorDtype::Int64, bytes)
                    .map_err(|error| format!("failed to build int64 input tensor: {}", error))
            }
            TensorDtype::Int32 => {
                let mut values = extract_i32_payload(input);
                if values.is_empty() {
                    values = vec![0i32; fallback_elements];
                }

                let desired_elements = values.len().max(1);
                let shape = resolve_shape(self.input_shape.as_deref(), desired_elements);
                let required_elements = shape_elements(&shape).max(1);
                values.resize(required_elements, 0);
                if values.len() > required_elements {
                    values.truncate(required_elements);
                }

                let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<i32>());
                for value in values {
                    bytes.extend_from_slice(&value.to_ne_bytes());
                }

                BinaryTensorPacket::new(shape, TensorDtype::Int32, bytes)
                    .map_err(|error| format!("failed to build int32 input tensor: {}", error))
            }
            TensorDtype::Float64 => {
                let mut values = extract_numeric_payload(input);
                if values.is_empty() {
                    values = vec![0.0f64; fallback_elements];
                }

                let desired_elements = values.len().max(1);
                let shape = resolve_shape(self.input_shape.as_deref(), desired_elements);
                let required_elements = shape_elements(&shape).max(1);
                values.resize(required_elements, 0.0);
                if values.len() > required_elements {
                    values.truncate(required_elements);
                }

                let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f64>());
                for value in values {
                    bytes.extend_from_slice(&value.to_ne_bytes());
                }

                BinaryTensorPacket::new(shape, TensorDtype::Float64, bytes)
                    .map_err(|error| format!("failed to build float64 input tensor: {}", error))
            }
            _ => {
                let mut values = extract_numeric_payload(input)
                    .into_iter()
                    .map(|value| value as f32)
                    .collect::<Vec<f32>>();
                if values.is_empty() {
                    values = vec![0.0f32; fallback_elements];
                }

                let desired_elements = values.len().max(1);
                let shape = resolve_shape(self.input_shape.as_deref(), desired_elements);
                let required_elements = shape_elements(&shape).max(1);
                values.resize(required_elements, 0.0);
                if values.len() > required_elements {
                    values.truncate(required_elements);
                }

                let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
                for value in values {
                    bytes.extend_from_slice(&value.to_ne_bytes());
                }

                BinaryTensorPacket::new(shape, TensorDtype::Float32, bytes)
                    .map_err(|error| format!("failed to build float32 input tensor: {}", error))
            }
        }
    }

    fn build_request(
        &self,
        input: Option<&NormalizedInputEvent>,
    ) -> Result<InferenceRequest, String> {
        let is_llm_embed_input = self.input_name.as_deref() == Some("inputs_embeds");
        if is_llm_embed_input && self.text_embed_stage.is_some() {
            if let Ok(request) = self.build_multimodal_embed_request(input) {
                return Ok(request);
            }
        }

        let tensor = self.build_input_tensor(input)?;
        Ok(InferenceRequest::new(tensor))
    }

    fn build_multimodal_embed_request(
        &self,
        input: Option<&NormalizedInputEvent>,
    ) -> Result<InferenceRequest, String> {
        let text_stage = self
            .text_embed_stage
            .as_ref()
            .ok_or_else(|| "missing text embedding stage".to_string())?;

        let prompt =
            extract_prompt_from_input(input).unwrap_or_else(|| "describe the image".to_string());
        let mut token_ids = prompt
            .bytes()
            .map(|value| (value as i64).max(1))
            .collect::<Vec<_>>();
        if token_ids.is_empty() {
            token_ids.push(1);
        }
        let token_tensor =
            build_token_id_tensor(&token_ids, &text_stage.input_shape, text_stage.input_dtype)?;
        let text_embed = text_stage.infer(token_tensor)?;
        let (mut text_data, text_seq_len, text_hidden) = parse_embeddings_tensor(&text_embed)?;

        if let Some(vision_stage) = &self.vision_stage
            && let Some(pixel_tensor) = build_image_tensor_from_input(input, vision_stage)?
            && let Ok(image_embed) = vision_stage.infer(pixel_tensor)
            && let Ok((image_data, image_seq_len, image_hidden)) =
                parse_embeddings_tensor(&image_embed)
            && image_hidden == text_hidden
        {
            let mut merged = Vec::with_capacity((image_seq_len + text_seq_len) * text_hidden);
            merged.extend_from_slice(&image_data);
            merged.extend_from_slice(&text_data);
            text_data = merged;
        }

        let mut bytes = Vec::with_capacity(text_data.len() * std::mem::size_of::<f32>());
        for value in text_data {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        let shape = vec![
            1,
            (bytes.len() / (text_hidden * std::mem::size_of::<f32>())) as i64,
            text_hidden as i64,
        ];
        let main_input = BinaryTensorPacket::new(shape, TensorDtype::Float32, bytes)
            .map_err(|error| format!("failed to build inputs_embeds tensor: {}", error))?;
        Ok(InferenceRequest::new(main_input))
    }
}

impl InferenceBackend for OnnxRuntimeBackend {
    fn run(
        &mut self,
        package_name: &str,
        _format: &str,
        input: Option<&NormalizedInputEvent>,
        options: InferenceRunOptions,
    ) -> InferenceOutcome {
        let started = Instant::now();
        let source = input
            .map(|event| event.source_id.as_str())
            .unwrap_or("runtime");

        if options
            .cancellation
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
        {
            return InferenceOutcome {
                latency_ms: started.elapsed().as_millis() as u64,
                output_summary: format!(
                    "onnx:{} source={} cancelled=pre-run",
                    package_name, source
                ),
                success: false,
            };
        }

        let request = match self.build_request(input) {
            Ok(request) => request,
            Err(error) => {
                return InferenceOutcome {
                    latency_ms: started.elapsed().as_millis() as u64,
                    output_summary: format!(
                        "onnx:{} source={} error={}",
                        package_name, source, error
                    ),
                    success: false,
                };
            }
        };

        if options
            .cancellation
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
        {
            return InferenceOutcome {
                latency_ms: started.elapsed().as_millis() as u64,
                output_summary: format!(
                    "onnx:{} source={} cancelled=pre-infer",
                    package_name, source
                ),
                success: false,
            };
        }

        match self.engine.infer(&request) {
            Ok(output) => {
                let latency_ms = started.elapsed().as_millis() as u64;
                if let Some(max_wall_ms) = options.max_wall_ms
                    && latency_ms > max_wall_ms
                {
                    return InferenceOutcome {
                        latency_ms,
                        output_summary: format!(
                            "onnx:{} source={} cancelled=deadline_exceeded wall={}ms max={}ms",
                            package_name, source, latency_ms, max_wall_ms
                        ),
                        success: false,
                    };
                }
                InferenceOutcome {
                    latency_ms,
                    output_summary: format!(
                        "onnx:{} source={} {}",
                        package_name,
                        source,
                        summarize_output(&output)
                    ),
                    success: true,
                }
            }
            Err(error) => InferenceOutcome {
                latency_ms: started.elapsed().as_millis() as u64,
                output_summary: format!("onnx:{} source={} error={}", package_name, source, error),
                success: false,
            },
        }
    }
}

fn create_backend_for_spec(
    spec: &KapslPackageSpec,
    metrics: &RuntimeMetrics,
) -> Box<dyn InferenceBackend> {
    let llm_backend_mode = std::env::var("KAPSL_LITE_LLM_BACKEND_MODE")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .unwrap_or_else(|| "onnx".to_string());
    let prefer_llm_backend = matches!(llm_backend_mode.as_str(), "llm" | "prefer" | "force");

    match spec.format {
        ModelFormat::Onnx if spec.is_llm && prefer_llm_backend => {
            match LlmRuntimeBackend::try_new(spec) {
                Ok(backend) => {
                    metrics.push_scheduler_log(
                        EventLevel::Normal,
                        format!(
                            "backend active: package={} backend=llm path={} mode=text-generation",
                            spec.name, spec.weights_path
                        ),
                    );
                    Box::new(backend)
                }
                Err(error) => {
                    metrics.push_scheduler_log(
                    EventLevel::Warning,
                    format!(
                        "backend fallback: package={} backend=onnx-runtime reason=llm-backend-unavailable ({})",
                        spec.name, error
                    ),
                );
                    match OnnxRuntimeBackend::try_new(spec) {
                        Ok(backend) => Box::new(backend),
                        Err(onnx_error) => {
                            metrics.push_scheduler_log(
                                EventLevel::Warning,
                                format!(
                                    "backend fallback: package={} backend=stub reason={}",
                                    spec.name, onnx_error
                                ),
                            );
                            Box::new(StubBackend::new())
                        }
                    }
                }
            }
        }
        ModelFormat::Onnx if spec.is_llm => {
            metrics.push_scheduler_log(
                EventLevel::Normal,
                format!(
                    "backend preference: package={} backend=onnx-runtime reason=llm-backend-mode({})",
                    spec.name, llm_backend_mode
                ),
            );
            match OnnxRuntimeBackend::try_new(spec) {
                Ok(backend) => {
                    metrics.push_scheduler_log(
                        EventLevel::Normal,
                        format!(
                            "backend active: package={} backend=onnx-runtime path={} memory_pattern={:?} cpu_arena_disabled={:?} session_buckets={:?} bucket_dim_granularity={:?} bucket_max_dims={:?} peak_concurrency_hint={:?}",
                            spec.name,
                            spec.weights_path,
                            spec.onnx_tuning.memory_pattern,
                            spec.onnx_tuning.disable_cpu_mem_arena,
                            spec.onnx_tuning.session_buckets,
                            spec.onnx_tuning.bucket_dim_granularity,
                            spec.onnx_tuning.bucket_max_dims,
                            spec.onnx_tuning.peak_concurrency_hint
                        ),
                    );
                    Box::new(backend)
                }
                Err(error) => {
                    metrics.push_scheduler_log(
                        EventLevel::Warning,
                        format!(
                            "backend fallback: package={} backend=stub reason={}",
                            spec.name, error
                        ),
                    );
                    Box::new(StubBackend::new())
                }
            }
        }
        ModelFormat::Onnx => match OnnxRuntimeBackend::try_new(spec) {
            Ok(backend) => {
                metrics.push_scheduler_log(
                    EventLevel::Normal,
                    format!(
                        "backend active: package={} backend=onnx-runtime path={} memory_pattern={:?} cpu_arena_disabled={:?} session_buckets={:?} bucket_dim_granularity={:?} bucket_max_dims={:?} peak_concurrency_hint={:?}",
                        spec.name,
                        spec.weights_path,
                        spec.onnx_tuning.memory_pattern,
                        spec.onnx_tuning.disable_cpu_mem_arena,
                        spec.onnx_tuning.session_buckets,
                        spec.onnx_tuning.bucket_dim_granularity,
                        spec.onnx_tuning.bucket_max_dims,
                        spec.onnx_tuning.peak_concurrency_hint
                    ),
                );
                Box::new(backend)
            }
            Err(error) => {
                metrics.push_scheduler_log(
                    EventLevel::Warning,
                    format!(
                        "backend fallback: package={} backend=stub reason={}",
                        spec.name, error
                    ),
                );
                Box::new(StubBackend::new())
            }
        },
        ModelFormat::Gguf => {
            metrics.push_scheduler_log(
                EventLevel::Warning,
                format!(
                    "backend fallback: package={} backend=stub reason=gguf backend unavailable in lite runtime",
                    spec.name
                ),
            );
            Box::new(StubBackend::new())
        }
    }
}

fn truncate_llm_text_for_summary(value: &str) -> String {
    const LIMIT: usize = 180;
    if value.chars().count() <= LIMIT {
        value.to_string()
    } else {
        let mut output: String = value.chars().take(LIMIT).collect();
        output.push_str("...");
        output
    }
}

fn extract_prompt_from_input(input: Option<&NormalizedInputEvent>) -> Option<String> {
    let event = input?;
    if let Some(prompt) = event
        .metadata
        .get("prompt")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return Some(prompt.to_string());
    }
    if let Some(text) = event
        .metadata
        .get("text")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return Some(text.to_string());
    }
    match &event.payload {
        NormalizedPayload::Json(Value::String(text)) => Some(text.clone()),
        NormalizedPayload::Json(Value::Object(map)) => map
            .get("prompt")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| map.get("text").and_then(Value::as_str).map(str::to_string))
            .or_else(|| Some(Value::Object(map.clone()).to_string())),
        NormalizedPayload::Json(value) => Some(value.to_string()),
        NormalizedPayload::Bytes(bytes) => std::str::from_utf8(bytes).ok().map(str::to_string),
        NormalizedPayload::Empty => None,
    }
}

fn extract_bytes_payload(input: Option<&NormalizedInputEvent>) -> Vec<u8> {
    let Some(event) = input else {
        return Vec::new();
    };
    match &event.payload {
        NormalizedPayload::Bytes(bytes) => bytes.clone(),
        NormalizedPayload::Json(Value::String(text)) => text.as_bytes().to_vec(),
        NormalizedPayload::Json(value) => {
            let mut values = Vec::new();
            flatten_json_numbers(value, &mut values);
            values
                .into_iter()
                .map(|value| value.round().clamp(0.0, 255.0) as u8)
                .collect()
        }
        NormalizedPayload::Empty => Vec::new(),
    }
}

fn extract_numeric_payload(input: Option<&NormalizedInputEvent>) -> Vec<f64> {
    let Some(event) = input else {
        return Vec::new();
    };
    match &event.payload {
        NormalizedPayload::Bytes(bytes) => {
            bytes.iter().map(|value| *value as f64 / 255.0).collect()
        }
        NormalizedPayload::Json(value) => {
            let mut values = Vec::new();
            flatten_json_numbers(value, &mut values);
            values
        }
        NormalizedPayload::Empty => Vec::new(),
    }
}

fn extract_i64_payload(input: Option<&NormalizedInputEvent>) -> Vec<i64> {
    extract_numeric_payload(input)
        .into_iter()
        .map(|value| value.round() as i64)
        .collect()
}

fn extract_i32_payload(input: Option<&NormalizedInputEvent>) -> Vec<i32> {
    extract_numeric_payload(input)
        .into_iter()
        .map(|value| value.round().clamp(i32::MIN as f64, i32::MAX as f64) as i32)
        .collect()
}

fn build_token_id_tensor(
    token_ids: &[i64],
    shape_hint: &Option<Vec<i64>>,
    dtype: TensorDtype,
) -> Result<BinaryTensorPacket, String> {
    let sequence_len = token_ids.len().max(1);
    let shape = resolve_shape(shape_hint.as_deref(), sequence_len);
    let required_elements = shape_elements(&shape).max(1);

    match dtype {
        TensorDtype::Int32 => {
            let mut values = token_ids
                .iter()
                .copied()
                .map(|value| value.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
                .collect::<Vec<_>>();
            values.resize(required_elements, 0);
            if values.len() > required_elements {
                values.truncate(required_elements);
            }
            let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<i32>());
            for value in values {
                bytes.extend_from_slice(&value.to_ne_bytes());
            }
            BinaryTensorPacket::new(shape, TensorDtype::Int32, bytes)
                .map_err(|error| format!("failed to build int32 token tensor: {}", error))
        }
        _ => {
            let mut values = token_ids.to_vec();
            values.resize(required_elements, 0);
            if values.len() > required_elements {
                values.truncate(required_elements);
            }
            let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<i64>());
            for value in values {
                bytes.extend_from_slice(&value.to_ne_bytes());
            }
            BinaryTensorPacket::new(shape, TensorDtype::Int64, bytes)
                .map_err(|error| format!("failed to build int64 token tensor: {}", error))
        }
    }
}

fn parse_embeddings_tensor(
    tensor: &BinaryTensorPacket,
) -> Result<(Vec<f32>, usize, usize), String> {
    if tensor.dtype != TensorDtype::Float32 {
        return Err(format!(
            "embedding tensor must be float32, got {}",
            tensor.dtype.as_str()
        ));
    }
    if tensor.shape.len() != 3 {
        return Err(format!(
            "embedding tensor must be rank-3 [batch,seq,hidden], got {:?}",
            tensor.shape
        ));
    }
    let batch =
        usize::try_from(tensor.shape[0]).map_err(|_| "invalid embedding batch".to_string())?;
    let seq_len =
        usize::try_from(tensor.shape[1]).map_err(|_| "invalid embedding seq".to_string())?;
    let hidden =
        usize::try_from(tensor.shape[2]).map_err(|_| "invalid embedding hidden".to_string())?;
    if batch == 0 || seq_len == 0 || hidden == 0 {
        return Err("embedding tensor has zero dimension".to_string());
    }

    let mut values = Vec::with_capacity(tensor.data.len() / std::mem::size_of::<f32>());
    for chunk in tensor.data.chunks_exact(std::mem::size_of::<f32>()) {
        values.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let expected = batch * seq_len * hidden;
    if values.len() < expected {
        return Err(format!(
            "embedding tensor data too short: got {} expected {}",
            values.len(),
            expected
        ));
    }
    let first_batch_len = seq_len * hidden;
    Ok((values[..first_batch_len].to_vec(), seq_len, hidden))
}

fn metadata_media_map(event: &NormalizedInputEvent) -> Option<&Map<String, Value>> {
    event
        .metadata
        .get("_media")
        .and_then(Value::as_object)
        .or_else(|| event.metadata.get("media").and_then(Value::as_object))
}

fn metadata_u64(event: &NormalizedInputEvent, key: &str) -> Option<u64> {
    event.metadata.get(key).and_then(Value::as_u64)
}

fn build_image_tensor_from_input(
    input: Option<&NormalizedInputEvent>,
    stage: &AuxiliaryOnnxStage,
) -> Result<Option<BinaryTensorPacket>, String> {
    let Some(event) = input else {
        return Ok(None);
    };
    let image_bytes = match &event.payload {
        NormalizedPayload::Bytes(bytes) => bytes.clone(),
        NormalizedPayload::Json(Value::Object(map)) => map
            .get("bytes")
            .and_then(|value| value.as_array())
            .map(|array| {
                array
                    .iter()
                    .filter_map(|item| item.as_u64())
                    .map(|value| value.min(255) as u8)
                    .collect::<Vec<u8>>()
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    };
    if image_bytes.is_empty() {
        return Ok(None);
    }

    let media = metadata_media_map(event);
    let content_type = media
        .and_then(|value| value.get("content_type"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !content_type.is_empty() && !content_type.starts_with("image/") {
        return Ok(None);
    }

    let source_width = media
        .and_then(|value| value.get("width"))
        .and_then(Value::as_u64)
        .or_else(|| metadata_u64(event, "width"))
        .map(|value| value as usize)
        .unwrap_or(224);
    let source_height = media
        .and_then(|value| value.get("height"))
        .and_then(Value::as_u64)
        .or_else(|| metadata_u64(event, "height"))
        .map(|value| value as usize)
        .unwrap_or(224);
    let channels = media
        .and_then(|value| value.get("channels"))
        .and_then(Value::as_u64)
        .or_else(|| metadata_u64(event, "channels"))
        .map(|value| value as usize)
        .unwrap_or(3)
        .max(1);

    let target_h = stage
        .input_shape
        .as_ref()
        .and_then(|shape| shape.get(2))
        .copied()
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)
        .unwrap_or(source_height.max(1));
    let target_w = stage
        .input_shape
        .as_ref()
        .and_then(|shape| shape.get(3))
        .copied()
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)
        .unwrap_or(source_width.max(1));

    let expected_source = source_width
        .saturating_mul(source_height)
        .saturating_mul(channels);
    if expected_source == 0 {
        return Ok(None);
    }

    let mut source = image_bytes;
    if source.len() < expected_source {
        source.resize(expected_source, 0);
    } else if source.len() > expected_source {
        source.truncate(expected_source);
    }

    let mut chw = vec![0f32; 3 * target_h * target_w];
    for y in 0..target_h {
        let src_y = y * source_height / target_h.max(1);
        for x in 0..target_w {
            let src_x = x * source_width / target_w.max(1);
            let src_base = (src_y * source_width + src_x) * channels;
            let (r, g, b) = if channels == 1 {
                let v = source[src_base];
                (v, v, v)
            } else {
                let r = source[src_base];
                let g = source.get(src_base + 1).copied().unwrap_or(r);
                let b = source.get(src_base + 2).copied().unwrap_or(g);
                (r, g, b)
            };
            let idx = y * target_w + x;
            chw[idx] = r as f32 / 255.0;
            chw[target_h * target_w + idx] = g as f32 / 255.0;
            chw[2 * target_h * target_w + idx] = b as f32 / 255.0;
        }
    }

    let mut bytes = Vec::with_capacity(chw.len() * std::mem::size_of::<f32>());
    for value in chw {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    let shape = vec![1, 3, target_h as i64, target_w as i64];
    let tensor = BinaryTensorPacket::new(shape, TensorDtype::Float32, bytes)
        .map_err(|error| format!("failed to build vision tensor: {}", error))?;
    Ok(Some(tensor))
}

fn flatten_json_numbers(value: &Value, out: &mut Vec<f64>) {
    match value {
        Value::Number(number) => {
            if let Some(value) = number.as_f64() {
                out.push(value);
            }
        }
        Value::Array(items) => {
            for item in items {
                flatten_json_numbers(item, out);
            }
        }
        Value::Object(map) => {
            if let Some(data) = map.get("data") {
                flatten_json_numbers(data, out);
            } else {
                for item in map.values() {
                    flatten_json_numbers(item, out);
                }
            }
        }
        _ => {}
    }
}

fn fixed_shape_elements(shape_hint: Option<&[i64]>) -> Option<usize> {
    let shape = shape_hint?;
    if shape.is_empty() {
        return None;
    }
    let mut elements = 1usize;
    for dim in shape {
        if *dim <= 0 {
            return None;
        }
        elements = elements.checked_mul(*dim as usize)?;
    }
    Some(elements.max(1))
}

fn shape_elements(shape: &[i64]) -> usize {
    if shape.is_empty() {
        return 1;
    }
    let mut total = 1usize;
    for dim in shape {
        total = total.saturating_mul((*dim).max(1) as usize);
    }
    total.max(1)
}

fn resolve_shape(shape_hint: Option<&[i64]>, desired_elements: usize) -> Vec<i64> {
    let desired_elements = desired_elements.max(1);
    match shape_hint {
        None => vec![1, desired_elements as i64],
        Some(shape) if shape.is_empty() => vec![desired_elements as i64],
        Some(shape) => {
            let mut resolved = shape.to_vec();
            let dynamic_idx = resolved.iter().position(|dim| *dim <= 0);
            for dim in &mut resolved {
                if *dim <= 0 {
                    *dim = 1;
                }
            }

            if let Some(idx) = dynamic_idx {
                let mut other = 1usize;
                for (current_idx, dim) in resolved.iter().enumerate() {
                    if current_idx == idx {
                        continue;
                    }
                    other = other.saturating_mul((*dim).max(1) as usize);
                }
                let needed = desired_elements.div_ceil(other.max(1));
                resolved[idx] = needed.max(1) as i64;
            }

            resolved
        }
    }
}

fn summarize_output(output: &BinaryTensorPacket) -> String {
    let shape_readable = describe_output_shape(&output.shape);
    let base = format!(
        "output_dtype={} output_shape={} output_bytes={}",
        output.dtype.as_str(),
        shape_readable,
        output.data.len()
    );

    if output.dtype == TensorDtype::Float32 && output.data.len() >= std::mem::size_of::<f32>() {
        let mut best_idx = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        for (index, chunk) in output
            .data
            .chunks_exact(std::mem::size_of::<f32>())
            .enumerate()
        {
            let value = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if value > best_value {
                best_value = value;
                best_idx = index;
            }
        }
        let mut summary = format!("{} top_index_flat={}", base, best_idx);
        if let Some(last_dim) = output
            .shape
            .last()
            .and_then(|dim| usize::try_from(*dim).ok())
            && last_dim > 0
        {
            summary.push_str(&format!(" top_class_id={}", best_idx % last_dim));
        }
        if let Some(coords) = unravel_flat_index(best_idx, &output.shape) {
            summary.push_str(&format!(" top_coords={:?}", coords));
        }
        summary.push_str(&format!(" top_score={:.6}", best_value));
        return summary;
    }

    base
}

fn describe_output_shape(shape: &[i64]) -> String {
    if shape.is_empty() {
        return "scalar".to_string();
    }
    let dims = shape
        .iter()
        .map(|dim| dim.to_string())
        .collect::<Vec<_>>()
        .join("x");
    let last_dim = shape.last().copied().unwrap_or_default();

    if last_dim >= 4096 {
        return match shape {
            [batch, step, vocab] => {
                format!(
                    "logits(batch={},step={},vocab={}) [{}]",
                    batch, step, vocab, dims
                )
            }
            [batch, vocab] => format!("logits(batch={},vocab={}) [{}]", batch, vocab, dims),
            [vocab] => format!("logits(vocab={}) [{}]", vocab, dims),
            _ => format!("tensor({})", dims),
        };
    }

    if shape.len() == 4 {
        return format!(
            "tensor4d(n={},c={},h={},w={}) [{}]",
            shape[0], shape[1], shape[2], shape[3], dims
        );
    }

    format!("tensor({})", dims)
}

fn unravel_flat_index(flat_index: usize, shape: &[i64]) -> Option<Vec<usize>> {
    if shape.is_empty() {
        return Some(Vec::new());
    }

    let mut remaining = flat_index;
    let mut coords = vec![0usize; shape.len()];
    for axis in (0..shape.len()).rev() {
        let dim = usize::try_from(shape[axis]).ok()?;
        if dim == 0 {
            return None;
        }
        coords[axis] = remaining % dim;
        remaining /= dim;
    }

    Some(coords)
}

// ── PackageStatus ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackageStatus {
    Loaded,
    Running,
    Suspended,
}

// ── PackageEntry ──────────────────────────────────────────────────────────────

#[derive(Debug)]
struct PackageEntry {
    spec: KapslPackageSpec,
    status: PackageStatus,
    /// Trigger sender for on-demand packages;  `None` for always-running.
    trigger_tx: Option<Sender<TriggerEvent>>,
    /// Drop receiver used by `drop_oldest` queue policy.
    trigger_drop_rx: Option<Receiver<TriggerEvent>>,
    queue_policy: QueueOverflowPolicy,
    cancel_flag: Arc<AtomicBool>,
}

// ── PackageRegistry ───────────────────────────────────────────────────────────

/// Shared, lock-protected registry of all loaded packages and their current
/// status.  Also tracks total memory usage against the configured budget.
#[derive(Debug)]
pub struct PackageRegistry {
    packages: Mutex<Vec<PackageEntry>>,
    memory_used_mib: AtomicU64,
    memory_limit_mib: u64,
}

impl PackageRegistry {
    pub fn new(memory_limit_mib: u64) -> Self {
        Self {
            packages: Mutex::new(Vec::new()),
            memory_used_mib: AtomicU64::new(0),
            memory_limit_mib,
        }
    }

    /// Register a package in the registry.  Returns `Err` if adding the package
    /// would exceed the memory budget.
    pub fn register(
        &self,
        spec: KapslPackageSpec,
        trigger_tx: Option<Sender<TriggerEvent>>,
        trigger_drop_rx: Option<Receiver<TriggerEvent>>,
        cancel_flag: Arc<AtomicBool>,
    ) -> Result<(), String> {
        let needed_mib = spec.memory_mb;
        let current = self.memory_used_mib.load(Ordering::SeqCst);
        if current + needed_mib > self.memory_limit_mib {
            return Err(format!(
                "package '{}' requires {}MiB but only {}MiB available (limit={}MiB)",
                spec.name,
                needed_mib,
                self.memory_limit_mib.saturating_sub(current),
                self.memory_limit_mib,
            ));
        }

        self.memory_used_mib.fetch_add(needed_mib, Ordering::SeqCst);
        let queue_policy = spec.on_demand_queue_policy;
        self.packages.lock().unwrap().push(PackageEntry {
            spec,
            status: PackageStatus::Loaded,
            trigger_tx,
            trigger_drop_rx,
            queue_policy,
            cancel_flag,
        });
        Ok(())
    }

    pub fn set_status(&self, package_name: &str, status: PackageStatus) {
        let mut pkgs = self.packages.lock().unwrap();
        if let Some(entry) = pkgs.iter_mut().find(|e| e.spec.name == package_name) {
            entry.status = status;
        }
    }

    /// Suspend all preemptible packages — used for thermal hard guardrails.
    pub fn suspend_preemptible(&self, metrics: &RuntimeMetrics) {
        let mut pkgs = self.packages.lock().unwrap();
        for entry in pkgs.iter_mut() {
            if entry.spec.preemptible && entry.status != PackageStatus::Suspended {
                entry.status = PackageStatus::Suspended;
                entry.cancel_flag.store(true, Ordering::Relaxed);
                metrics.set_model_paused(&entry.spec.name, true);
                metrics.push_scheduler_log(
                    EventLevel::Critical,
                    format!(
                        "package suspended: name={} priority={} reason=thermal-hard",
                        entry.spec.name,
                        entry.spec.priority.as_str()
                    ),
                );
            }
        }
    }

    /// Thermal safe mode keeps only critical perception realtime loops running.
    pub fn enter_safe_mode(&self, metrics: &RuntimeMetrics) {
        let mut pkgs = self.packages.lock().unwrap();
        for entry in pkgs.iter_mut() {
            let keep_running =
                entry.spec.task_class == TaskClass::Realtime && entry.spec.critical_perception;
            if keep_running {
                continue;
            }
            if entry.status != PackageStatus::Suspended {
                entry.status = PackageStatus::Suspended;
                entry.cancel_flag.store(true, Ordering::Relaxed);
                metrics.set_model_paused(&entry.spec.name, true);
                metrics.push_scheduler_log(
                    EventLevel::Critical,
                    format!(
                        "package suspended: name={} class={} reason=thermal-safe-mode",
                        entry.spec.name,
                        entry.spec.task_class.as_str()
                    ),
                );
            }
        }
    }

    /// Restore all previously suspended packages — called on thermal recovery.
    pub fn restore_suspended(&self, metrics: &RuntimeMetrics) {
        let mut pkgs = self.packages.lock().unwrap();
        for entry in pkgs.iter_mut() {
            if entry.status == PackageStatus::Suspended {
                entry.status = PackageStatus::Loaded;
                entry.cancel_flag.store(false, Ordering::Relaxed);
                metrics.set_model_paused(&entry.spec.name, false);
                metrics.push_scheduler_log(
                    EventLevel::Normal,
                    format!(
                        "package restored: name={} priority={}",
                        entry.spec.name,
                        entry.spec.priority.as_str()
                    ),
                );
            }
        }
    }

    pub fn is_suspended(&self, package_name: &str) -> bool {
        let pkgs = self.packages.lock().unwrap();
        pkgs.iter()
            .find(|e| e.spec.name == package_name)
            .map(|e| e.status == PackageStatus::Suspended)
            .unwrap_or(false)
    }

    pub fn package_count(&self) -> usize {
        self.packages.lock().unwrap().len()
    }

    pub fn memory_used_mib(&self) -> u64 {
        self.memory_used_mib.load(Ordering::Relaxed)
    }

    pub fn dispatch_trigger(
        &self,
        package_name: &str,
        event: TriggerEvent,
    ) -> Result<(), &'static str> {
        let (tx, drop_rx, queue_policy) = {
            let pkgs = self.packages.lock().unwrap();
            let Some(entry) = pkgs.iter().find(|e| e.spec.name == package_name) else {
                return Err("package-not-found-or-not-on-demand");
            };
            let Some(tx) = entry.trigger_tx.clone() else {
                return Err("package-not-found-or-not-on-demand");
            };
            (tx, entry.trigger_drop_rx.clone(), entry.queue_policy)
        };

        match queue_policy {
            QueueOverflowPolicy::DropNewest => match tx.try_send(event) {
                Ok(()) => Ok(()),
                Err(TrySendError::Full(_)) => Err("package-channel-full"),
                Err(TrySendError::Disconnected(_)) => Err("package-channel-disconnected"),
            },
            QueueOverflowPolicy::Block => {
                tx.send(event).map_err(|_| "package-channel-disconnected")
            }
            QueueOverflowPolicy::DropOldest => match tx.try_send(event) {
                Ok(()) => Ok(()),
                Err(TrySendError::Disconnected(_)) => Err("package-channel-disconnected"),
                Err(TrySendError::Full(pending)) => {
                    if let Some(rx) = drop_rx {
                        match rx.try_recv() {
                            Ok(_) | Err(TryRecvError::Empty) => {}
                            Err(TryRecvError::Disconnected) => {
                                return Err("package-channel-disconnected");
                            }
                        }
                    }
                    match tx.try_send(pending) {
                        Ok(()) => Ok(()),
                        Err(TrySendError::Full(_)) => Err("package-channel-full"),
                        Err(TrySendError::Disconnected(_)) => Err("package-channel-disconnected"),
                    }
                }
            },
        }
    }

    /// Returns full scheduling metadata for a package by name.
    pub fn package_spec(&self, package_name: &str) -> Option<KapslPackageSpec> {
        let pkgs = self.packages.lock().unwrap();
        pkgs.iter()
            .find(|e| e.spec.name == package_name)
            .map(|e| e.spec.clone())
    }

    pub fn cancel_flag(&self, package_name: &str) -> Option<Arc<AtomicBool>> {
        let pkgs = self.packages.lock().unwrap();
        pkgs.iter()
            .find(|entry| entry.spec.name == package_name)
            .map(|entry| entry.cancel_flag.clone())
    }

    pub fn request_cancel_preemptible(&self, metrics: &RuntimeMetrics, reason: &str) {
        let mut pkgs = self.packages.lock().unwrap();
        for entry in pkgs.iter_mut() {
            if entry.spec.preemptible {
                entry.cancel_flag.store(true, Ordering::Relaxed);
                metrics.push_scheduler_log(
                    EventLevel::Warning,
                    format!(
                        "cooperative cancel requested: package={} reason={}",
                        entry.spec.name, reason
                    ),
                );
            }
        }
    }

    pub fn clear_cancel_requests(&self) {
        let mut pkgs = self.packages.lock().unwrap();
        for entry in pkgs.iter_mut() {
            if entry.status != PackageStatus::Suspended {
                entry.cancel_flag.store(false, Ordering::Relaxed);
            }
        }
    }
}

// ── ThermalAction ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum ThermalAction {
    Normal = 0,
    T1ConstrainBestEffort = 1,
    T2DegradeRealtime = 2,
    T3SafeMode = 3,
}

impl ThermalAction {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::T1ConstrainBestEffort,
            2 => Self::T2DegradeRealtime,
            3 => Self::T3SafeMode,
            _ => Self::Normal,
        }
    }
}

// ── BatteryAction ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum BatteryAction {
    Normal = 0,
    Conserve = 1,
    Critical = 2,
}

impl BatteryAction {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Conserve,
            2 => Self::Critical,
            _ => Self::Normal,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum MemoryAction {
    Normal = 0,
    Emergency = 1,
}

impl MemoryAction {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Emergency,
            _ => Self::Normal,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct BatterySnapshot {
    capacity_percent: f64,
    is_discharging: bool,
}

// ── SchedulerState ────────────────────────────────────────────────────────────

#[derive(Debug)]
struct SchedulerState {
    thermal_action: AtomicU8,
    battery_action: AtomicU8,
    memory_action: AtomicU8,
    thermal_fps_scale_milli: AtomicU64,
    battery_fps_scale_milli: AtomicU64,
    best_effort_gate_divisor: AtomicU64,
    best_effort_gate_counter: AtomicU64,
    battery_response_token_cap: AtomicU64,
    memory_response_token_cap: AtomicU64,
    memory_infer_guard: Mutex<()>,
    memory_load_guard: Mutex<()>,
}

impl SchedulerState {
    fn new() -> Self {
        Self {
            thermal_action: AtomicU8::new(ThermalAction::Normal as u8),
            battery_action: AtomicU8::new(BatteryAction::Normal as u8),
            memory_action: AtomicU8::new(MemoryAction::Normal as u8),
            thermal_fps_scale_milli: AtomicU64::new(1000),
            battery_fps_scale_milli: AtomicU64::new(1000),
            best_effort_gate_divisor: AtomicU64::new(1),
            best_effort_gate_counter: AtomicU64::new(0),
            battery_response_token_cap: AtomicU64::new(0),
            memory_response_token_cap: AtomicU64::new(0),
            memory_infer_guard: Mutex::new(()),
            memory_load_guard: Mutex::new(()),
        }
    }

    fn thermal_action(&self) -> ThermalAction {
        ThermalAction::from_u8(self.thermal_action.load(Ordering::Relaxed))
    }

    fn set_thermal_action(&self, action: ThermalAction) {
        self.thermal_action.store(action as u8, Ordering::Relaxed);
    }

    fn battery_action(&self) -> BatteryAction {
        BatteryAction::from_u8(self.battery_action.load(Ordering::Relaxed))
    }

    fn set_battery_action(&self, action: BatteryAction) {
        self.battery_action.store(action as u8, Ordering::Relaxed);
    }

    fn memory_action(&self) -> MemoryAction {
        MemoryAction::from_u8(self.memory_action.load(Ordering::Relaxed))
    }

    fn set_memory_action(&self, action: MemoryAction) {
        self.memory_action.store(action as u8, Ordering::Relaxed);
    }

    fn set_thermal_fps_factor(&self, factor: f64) {
        self.thermal_fps_scale_milli
            .store(factor_to_milli(factor), Ordering::Relaxed);
    }

    fn set_battery_fps_factor(&self, factor: f64) {
        self.battery_fps_scale_milli
            .store(factor_to_milli(factor), Ordering::Relaxed);
    }

    fn effective_fps_scale_milli(&self) -> u64 {
        self.thermal_fps_scale_milli
            .load(Ordering::Relaxed)
            .min(self.battery_fps_scale_milli.load(Ordering::Relaxed))
            .max(50)
    }

    fn scaled_cycle_ms(&self, base_cycle_ms: u64) -> u64 {
        base_cycle_ms
            .saturating_mul(1000)
            .checked_div(self.effective_fps_scale_milli())
            .unwrap_or(base_cycle_ms)
            .max(1)
    }

    fn set_best_effort_gate_divisor(&self, divisor: u64) {
        self.best_effort_gate_divisor
            .store(divisor.max(1), Ordering::Relaxed);
    }

    fn allow_best_effort_dispatch(&self) -> bool {
        let divisor = self.best_effort_gate_divisor.load(Ordering::Relaxed).max(1);
        if divisor <= 1 {
            return true;
        }
        self.best_effort_gate_counter
            .fetch_add(1, Ordering::Relaxed)
            % divisor
            == 0
    }

    fn set_response_token_cap(&self, max_tokens: Option<u64>) {
        self.battery_response_token_cap
            .store(max_tokens.unwrap_or(0), Ordering::Relaxed);
    }

    fn set_memory_token_cap(&self, max_tokens: Option<u64>) {
        self.memory_response_token_cap
            .store(max_tokens.unwrap_or(0), Ordering::Relaxed);
    }

    fn response_token_cap(&self) -> Option<u64> {
        let battery = self.battery_response_token_cap.load(Ordering::Relaxed);
        let memory = self.memory_response_token_cap.load(Ordering::Relaxed);
        match (battery, memory) {
            (0, 0) => None,
            (0, value) | (value, 0) => Some(value),
            (battery, memory) => Some(battery.min(memory)),
        }
    }
}

// ── RuntimeHandle ─────────────────────────────────────────────────────────────

pub struct RuntimeHandle {
    threads: Vec<JoinHandle<()>>,
}

impl RuntimeHandle {
    pub fn start(
        config: RuntimeConfig,
        packages: Vec<KapslPackageSpec>,
        metrics: Arc<RuntimeMetrics>,
        shutdown: Arc<AtomicBool>,
    ) -> (Self, TriggerBus, Arc<PackageRegistry>) {
        let (trigger_bus, trigger_rx) = TriggerBus::new(config.trigger_bus_capacity);
        let scheduler_state = Arc::new(SchedulerState::new());
        let registry = Arc::new(PackageRegistry::new(config.memory_limit_mib));

        metrics.set_status("loading", EventLevel::Normal);

        let mut threads = Vec::new();

        // Launch a runner thread for each package.
        for spec in packages {
            let package_name = spec.name.clone();
            let format_str = spec.format.as_str().to_string();
            let trigger_mode = spec.trigger_mode;
            let memory_mb = spec.memory_mb;
            let priority = spec.priority;
            let task_class = spec.task_class;

            if !backend_affinity_available(spec.backend_affinity) {
                metrics.push_scheduler_log(
                    EventLevel::Critical,
                    format!(
                        "package rejected: name={} reason=backend-affinity-unavailable affinity={}",
                        package_name,
                        spec.backend_affinity.as_str()
                    ),
                );
                continue;
            }

            match trigger_mode {
                TriggerMode::AlwaysRunning => {
                    // Register with no trigger channel.

                    match registry.register(
                        spec.clone(),
                        None,
                        None,
                        Arc::new(AtomicBool::new(false)),
                    ) {
                        Ok(()) => {
                            metrics.register_model(&package_name, ModelRunState::Idle);
                            metrics.set_memory_budget_used_mib(registry.memory_used_mib());
                            metrics.push_scheduler_log(
                                EventLevel::Normal,
                                format!(
                                    "package loaded: name={} format={} weights={} memory={}MiB priority={} class={} mode=always_running affinity={} deadline_ms={:?} cost={}",
                                    package_name,
                                    format_str,
                                    spec.weights,
                                    memory_mb,
                                    priority.as_str(),
                                    task_class.as_str(),
                                    spec.backend_affinity.as_str(),
                                    spec.deadline_ms,
                                    spec.cost_estimate
                                        .map(|cost| cost.as_label())
                                        .unwrap_or_else(|| "n/a".to_string())
                                ),
                            );
                        }
                        Err(reason) => {
                            metrics.push_scheduler_log(
                                EventLevel::Critical,
                                format!(
                                    "package rejected: name={} reason={}",
                                    package_name, reason
                                ),
                            );
                            continue;
                        }
                    }

                    let metrics_c = metrics.clone();
                    let registry_c = registry.clone();
                    let shutdown_c = shutdown.clone();
                    let cycle_ms = spec.cycle_interval_ms();
                    let scheduler_c = scheduler_state.clone();
                    let spec_c = spec.clone();

                    threads.push(thread::spawn(move || {
                        always_running_loop(
                            spec_c,
                            cycle_ms,
                            scheduler_c,
                            registry_c,
                            metrics_c,
                            shutdown_c,
                        );
                    }));
                }

                TriggerMode::OnDemand => {
                    let queue_capacity = spec.on_demand_queue_capacity.max(1);
                    let queue_policy = spec.on_demand_queue_policy;
                    let (pkg_tx, pkg_rx) = bounded::<TriggerEvent>(queue_capacity);
                    let max_workers = spec.max_concurrent.max(1);

                    match registry.register(
                        spec.clone(),
                        Some(pkg_tx),
                        Some(pkg_rx.clone()),
                        Arc::new(AtomicBool::new(false)),
                    ) {
                        Ok(()) => {
                            metrics.register_model(&package_name, ModelRunState::Idle);
                            metrics.set_memory_budget_used_mib(registry.memory_used_mib());
                            metrics.push_scheduler_log(
                                EventLevel::Normal,
                                format!(
                                    "package loaded: name={} format={} weights={} memory={}MiB priority={} class={} mode=on_demand max_concurrency={} queue_capacity={} queue_policy={} affinity={} deadline_ms={:?} cost={}",
                                    package_name,
                                    format_str,
                                    spec.weights,
                                    memory_mb,
                                    priority.as_str(),
                                    task_class.as_str(),
                                    max_workers,
                                    queue_capacity,
                                    queue_policy.as_str(),
                                    spec.backend_affinity.as_str(),
                                    spec.deadline_ms,
                                    spec.cost_estimate
                                        .map(|cost| cost.as_label())
                                        .unwrap_or_else(|| "n/a".to_string())
                                ),
                            );
                        }
                        Err(reason) => {
                            metrics.push_scheduler_log(
                                EventLevel::Critical,
                                format!(
                                    "package rejected: name={} reason={}",
                                    package_name, reason
                                ),
                            );
                            continue;
                        }
                    }

                    for worker_idx in 0..max_workers {
                        let metrics_c = metrics.clone();
                        let registry_c = registry.clone();
                        let scheduler_c = scheduler_state.clone();
                        let shutdown_c = shutdown.clone();
                        let spec_c = spec.clone();
                        let rx_c = pkg_rx.clone();

                        threads.push(thread::spawn(move || {
                            on_demand_loop(
                                spec_c,
                                worker_idx,
                                rx_c,
                                scheduler_c,
                                registry_c,
                                metrics_c,
                                shutdown_c,
                            );
                        }));
                    }
                }
            }
        }

        metrics.set_status("loaded", EventLevel::Normal);
        metrics.set_memory_budget_used_mib(registry.memory_used_mib());
        metrics.push_scheduler_log(
            EventLevel::Normal,
            format!(
                "runtime ready: packages={} memory_used={}MiB memory_limit={}MiB trigger_threshold={:.2}",
                registry.package_count(),
                registry.memory_used_mib(),
                config.memory_limit_mib,
                config.trigger_confidence_threshold,
            ),
        );

        // System sampler thread — reads real /proc stats on Linux.
        {
            let metrics_c = metrics.clone();
            let shutdown_c = shutdown.clone();
            threads.push(thread::spawn(move || {
                system_sampler_loop(metrics_c, shutdown_c);
            }));
        }

        // Thermal controller thread — polls /sys thermal zone.
        {
            let config_c = config.clone();
            let metrics_c = metrics.clone();
            let registry_c = registry.clone();
            let scheduler_c = scheduler_state.clone();
            let shutdown_c = shutdown.clone();
            threads.push(thread::spawn(move || {
                thermal_controller_loop(config_c, metrics_c, registry_c, scheduler_c, shutdown_c);
            }));
        }

        // Battery controller thread — polls /sys power_supply battery state.
        {
            let config_c = config.clone();
            let metrics_c = metrics.clone();
            let scheduler_c = scheduler_state.clone();
            let shutdown_c = shutdown.clone();
            threads.push(thread::spawn(move || {
                battery_controller_loop(config_c, metrics_c, scheduler_c, shutdown_c);
            }));
        }

        // Trigger dispatcher — routes events from trigger bus to per-package channels.
        {
            let config_c = config.clone();
            let metrics_c = metrics.clone();
            let registry_c = registry.clone();
            let scheduler_c = scheduler_state.clone();
            let shutdown_c = shutdown.clone();
            threads.push(thread::spawn(move || {
                memory_guard_loop(config_c, metrics_c, registry_c, scheduler_c, shutdown_c);
            }));
        }

        {
            let config_c = config;
            let metrics_c = metrics.clone();
            let registry_c = registry.clone();
            let scheduler_c = scheduler_state;
            let shutdown_c = shutdown.clone();
            threads.push(thread::spawn(move || {
                trigger_dispatcher_loop(
                    config_c,
                    trigger_rx,
                    metrics_c,
                    registry_c,
                    scheduler_c,
                    shutdown_c,
                );
            }));
        }

        (Self { threads }, trigger_bus, registry)
    }

    pub fn join(self) {
        for handle in self.threads {
            let _ = handle.join();
        }
    }
}

// ── always_running_loop ───────────────────────────────────────────────────────

/// Drives continuous inference for an `always_running` package.  Respects the
/// package's FPS target cycle interval and skips cycles when suspended.
fn always_running_loop(
    spec: KapslPackageSpec,
    cycle_ms: u64,
    scheduler_state: Arc<SchedulerState>,
    registry: Arc<PackageRegistry>,
    metrics: Arc<RuntimeMetrics>,
    shutdown: Arc<AtomicBool>,
) {
    let package_name = spec.name.clone();
    let format_str = spec.format.as_str().to_string();
    let cancel_flag = registry.cancel_flag(&package_name);
    let mut backend = create_backend_for_spec(&spec, metrics.as_ref());
    let mut next_tick = Instant::now();
    let mut memory_paused = false;

    registry.set_status(&package_name, PackageStatus::Running);

    while !shutdown.load(Ordering::Relaxed) {
        let effective_cycle_ms = scheduler_state.scaled_cycle_ms(cycle_ms);

        // Skip inference when thermally suspended.
        if registry.is_suspended(&package_name) {
            metrics.set_model_paused(&package_name, true);
            sleep_with_shutdown(Duration::from_millis(effective_cycle_ms), &shutdown);
            next_tick = Instant::now();
            continue;
        }

        // In memory emergency, cooperative-canceling every preemptible always-running cycle
        // produces fail spam and CPU churn. Pause the loop until headroom recovers.
        if spec.preemptible && scheduler_state.memory_action() == MemoryAction::Emergency {
            if !memory_paused {
                memory_paused = true;
                metrics.push_scheduler_log(
                    EventLevel::Warning,
                    format!(
                        "always-running paused: package={} reason=memory-emergency preemptible=true",
                        package_name
                    ),
                );
            }
            metrics.set_model_paused(&package_name, true);
            sleep_with_shutdown(Duration::from_millis(effective_cycle_ms), &shutdown);
            next_tick = Instant::now();
            continue;
        }

        if memory_paused {
            memory_paused = false;
            metrics.push_scheduler_log(
                EventLevel::Normal,
                format!(
                    "always-running resumed: package={} reason=memory-recovered",
                    package_name
                ),
            );
        }
        metrics.set_model_paused(&package_name, false);

        metrics.mark_model_running(&package_name);
        metrics.mark_job_started();
        let started = Instant::now();

        let outcome = if scheduler_state.memory_action() == MemoryAction::Emergency {
            let _infer_guard = scheduler_state
                .memory_infer_guard
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            backend.run(
                &package_name,
                &format_str,
                None,
                InferenceRunOptions {
                    max_new_tokens: None,
                    max_wall_ms: spec.deadline_ms,
                    cancellation: cancel_flag.clone(),
                },
            )
        } else {
            backend.run(
                &package_name,
                &format_str,
                None,
                InferenceRunOptions {
                    max_new_tokens: None,
                    max_wall_ms: spec.deadline_ms,
                    cancellation: cancel_flag.clone(),
                },
            )
        };
        let wall_ms = started.elapsed().as_millis() as u64;

        metrics.mark_job_finished(wall_ms, outcome.success);
        metrics.mark_model_finished(&package_name);
        metrics.push_inference_result(InferenceResult {
            timestamp_ms: unix_time_millis(),
            package_name: package_name.clone(),
            latency_ms: outcome.latency_ms.max(wall_ms),
            success: outcome.success,
            output_summary: outcome.output_summary,
        });

        // Pace to the configured FPS target.
        next_tick += Duration::from_millis(effective_cycle_ms);
        let now = Instant::now();
        if next_tick > now {
            sleep_with_shutdown(next_tick - now, &shutdown);
        } else {
            // We fell behind; reset rather than spinning.
            next_tick = Instant::now();
        }
    }

    metrics.set_model_idle(&package_name);
}

// ── on_demand_loop ────────────────────────────────────────────────────────────

/// Waits for a `TriggerEvent` on `trigger_rx`, then dispatches one inference
/// call for the package.  Fire-and-forget: does not block the trigger sender.
fn on_demand_loop(
    spec: KapslPackageSpec,
    worker_idx: usize,
    trigger_rx: Receiver<TriggerEvent>,
    scheduler_state: Arc<SchedulerState>,
    registry: Arc<PackageRegistry>,
    metrics: Arc<RuntimeMetrics>,
    shutdown: Arc<AtomicBool>,
) {
    let package_name = spec.name.clone();
    let format_str = spec.format.as_str().to_string();
    let cancel_flag = registry.cancel_flag(&package_name);
    let mut backend: Option<Box<dyn InferenceBackend>> = None;
    let mut last_token_cap = None;
    let mut emergency_worker_parked = false;

    loop {
        if shutdown.load(Ordering::Relaxed) && trigger_rx.is_empty() {
            break;
        }

        let in_memory_emergency = scheduler_state.memory_action() == MemoryAction::Emergency;
        if in_memory_emergency && backend.is_some() && trigger_rx.is_empty() {
            backend = None;
            metrics.mark_emergency_backend_unloaded();
            metrics.mark_model_emergency_backend_unloaded(&package_name);
            metrics.push_scheduler_log(
                EventLevel::Warning,
                format!(
                    "on-demand backend unloaded: package={} worker={} reason=memory-emergency-idle",
                    package_name, worker_idx
                ),
            );
        }

        if in_memory_emergency && worker_idx > 0 {
            if !emergency_worker_parked {
                emergency_worker_parked = true;
                if backend.is_some() {
                    backend = None;
                    metrics.mark_emergency_backend_unloaded();
                    metrics.mark_model_emergency_backend_unloaded(&package_name);
                    metrics.push_scheduler_log(
                        EventLevel::Warning,
                        format!(
                            "on-demand backend unloaded: package={} worker={} reason=memory-emergency-worker-park",
                            package_name, worker_idx
                        ),
                    );
                }
                metrics.push_scheduler_log(
                    EventLevel::Warning,
                    format!(
                        "on-demand worker parked: package={} worker={} reason=memory-emergency effective_max_concurrency=1",
                        package_name, worker_idx
                    ),
                );
                metrics.mark_emergency_worker_parked();
                metrics.mark_model_emergency_worker_parked(&package_name);
            }
            sleep_with_shutdown(Duration::from_millis(100), &shutdown);
            continue;
        }

        if emergency_worker_parked {
            emergency_worker_parked = false;
            metrics.push_scheduler_log(
                EventLevel::Normal,
                format!(
                    "on-demand worker resumed: package={} worker={} reason=memory-recovered",
                    package_name, worker_idx
                ),
            );
        }

        let event = match trigger_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(e) => e,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };
        metrics.set_model_queue_depth(&package_name, trigger_rx.len());

        // Honour thermal suspension even for on-demand packages.
        if registry.is_suspended(&package_name) {
            metrics.set_model_paused(&package_name, true);
            metrics.push_scheduler_log(
                EventLevel::Critical,
                format!(
                    "on-demand skipped: package={} worker={} reason=thermal-suspended trigger={:?}",
                    package_name,
                    worker_idx,
                    event.target_package()
                ),
            );
            continue;
        }
        metrics.set_model_paused(&package_name, false);

        if scheduler_state.memory_action() == MemoryAction::Emergency
            && spec.task_class == TaskClass::BestEffort
        {
            metrics.push_scheduler_log(
                EventLevel::Critical,
                format!(
                    "on-demand skipped: package={} worker={} reason=memory-emergency class={}",
                    package_name,
                    worker_idx,
                    spec.task_class.as_str(),
                ),
            );
            continue;
        }

        let token_cap = scheduler_state.response_token_cap();
        if token_cap != last_token_cap {
            if let Some(max_tokens) = token_cap {
                metrics.push_scheduler_log(
                    EventLevel::Warning,
                    format!(
                        "response budget active: package={} worker={} max_new_tokens={} audio_quality=low",
                        package_name, worker_idx, max_tokens
                    ),
                );
            }
            last_token_cap = token_cap;
        }

        if backend.is_none() {
            if scheduler_state.memory_action() == MemoryAction::Emergency {
                let _load_guard = scheduler_state
                    .memory_load_guard
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if backend.is_none() {
                    metrics.mark_emergency_serialized_load();
                    metrics.mark_model_emergency_serialized_load(&package_name);
                    metrics.push_scheduler_log(
                        EventLevel::Warning,
                        format!(
                            "on-demand backend loading: package={} worker={} mode=serialized reason=memory-emergency",
                            package_name, worker_idx
                        ),
                    );
                    backend = Some(create_backend_for_spec(&spec, metrics.as_ref()));
                }
            } else {
                backend = Some(create_backend_for_spec(&spec, metrics.as_ref()));
            }
        }

        metrics.mark_model_running(&package_name);
        metrics.mark_job_started();
        let started = Instant::now();

        let manual_input = match &event {
            TriggerEvent::ManualTrigger { custom_prompt, .. } => Some(NormalizedInputEvent {
                timestamp_ms: unix_time_millis(),
                source_id: "manual/operator".to_string(),
                payload: NormalizedPayload::Json(Value::String(custom_prompt.clone())),
                metadata: Map::new(),
            }),
            _ => None,
        };
        let input_event = match &event {
            TriggerEvent::ExternalInput { event, .. } => Some(event),
            _ => manual_input.as_ref(),
        };
        let Some(backend_ref) = backend.as_mut() else {
            metrics.push_scheduler_log(
                EventLevel::Warning,
                format!(
                    "on-demand skipped: package={} worker={} reason=backend-not-loaded",
                    package_name, worker_idx
                ),
            );
            continue;
        };
        let outcome = if scheduler_state.memory_action() == MemoryAction::Emergency {
            let _infer_guard = scheduler_state
                .memory_infer_guard
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            backend_ref.run(
                &package_name,
                &format_str,
                input_event,
                InferenceRunOptions {
                    max_new_tokens: scheduler_state.response_token_cap(),
                    max_wall_ms: spec.deadline_ms,
                    cancellation: cancel_flag.clone(),
                },
            )
        } else {
            backend_ref.run(
                &package_name,
                &format_str,
                input_event,
                InferenceRunOptions {
                    max_new_tokens: scheduler_state.response_token_cap(),
                    max_wall_ms: spec.deadline_ms,
                    cancellation: cancel_flag.clone(),
                },
            )
        };
        let wall_ms = started.elapsed().as_millis() as u64;

        metrics.mark_job_finished(wall_ms, outcome.success);
        metrics.mark_model_finished(&package_name);
        metrics.set_model_queue_depth(&package_name, trigger_rx.len());
        metrics.push_inference_result(InferenceResult {
            timestamp_ms: unix_time_millis(),
            package_name: package_name.clone(),
            latency_ms: outcome.latency_ms.max(wall_ms),
            success: outcome.success,
            output_summary: outcome.output_summary,
        });

        metrics.push_scheduler_log(
            EventLevel::Normal,
            format!(
                "trigger completed: package={} worker={} latency={}ms",
                package_name, worker_idx, wall_ms
            ),
        );

        if scheduler_state.memory_action() == MemoryAction::Emergency && backend.is_some() {
            backend = None;
            metrics.mark_emergency_backend_unloaded();
            metrics.mark_model_emergency_backend_unloaded(&package_name);
            metrics.push_scheduler_log(
                EventLevel::Warning,
                format!(
                    "on-demand backend unloaded: package={} worker={} reason=memory-emergency-post-run",
                    package_name, worker_idx
                ),
            );
        }
    }

    metrics.set_model_idle(&package_name);
}

// ── system_sampler_loop ───────────────────────────────────────────────────────

fn system_sampler_loop(metrics: Arc<RuntimeMetrics>, shutdown: Arc<AtomicBool>) {
    let mut sampler = SystemSampler::new();

    while !shutdown.load(Ordering::Relaxed) {
        let sample = sampler.sample();
        metrics.set_system_stats(sample.cpu_percent, sample.memory_rss_mib);
        sleep_with_shutdown(Duration::from_secs(1), &shutdown);
    }
}

// ── thermal_controller_loop ───────────────────────────────────────────────────

fn thermal_controller_loop(
    config: RuntimeConfig,
    metrics: Arc<RuntimeMetrics>,
    registry: Arc<PackageRegistry>,
    scheduler_state: Arc<SchedulerState>,
    shutdown: Arc<AtomicBool>,
) {
    let mut current_action = ThermalAction::Normal;
    let mut cpufreq_throttled = false;

    while !shutdown.load(Ordering::Relaxed) {
        let temperature_c = read_thermal_celsius();
        metrics.set_temperature_c(temperature_c);
        if let Some(freq_ratio) = read_cpufreq_throttle_ratio() {
            let now_throttled = freq_ratio < 0.90;
            if now_throttled != cpufreq_throttled {
                cpufreq_throttled = now_throttled;
                metrics.push_scheduler_log(
                    if now_throttled {
                        EventLevel::Warning
                    } else {
                        EventLevel::Normal
                    },
                    format!(
                        "cpufreq state changed: throttled={} ratio={:.2}",
                        now_throttled, freq_ratio
                    ),
                );
            }
        }

        let next_action = evaluate_thermal_action(current_action, temperature_c, &config);
        if next_action != current_action {
            apply_thermal_action_change(
                &config,
                next_action,
                temperature_c,
                &metrics,
                &registry,
                &scheduler_state,
            );
            current_action = next_action;
        }

        sleep_with_shutdown(
            Duration::from_millis(config.thermal_poll_interval_ms),
            &shutdown,
        );
    }
}

fn evaluate_thermal_action(
    current: ThermalAction,
    temperature_c: f64,
    config: &RuntimeConfig,
) -> ThermalAction {
    match current {
        ThermalAction::Normal => {
            if temperature_c >= config.thermal_hard_threshold_c {
                ThermalAction::T3SafeMode
            } else if temperature_c >= config.thermal_degraded_threshold_c {
                ThermalAction::T2DegradeRealtime
            } else if temperature_c >= config.thermal_soft_threshold_c {
                ThermalAction::T1ConstrainBestEffort
            } else {
                ThermalAction::Normal
            }
        }
        ThermalAction::T1ConstrainBestEffort => {
            if temperature_c >= config.thermal_hard_threshold_c {
                ThermalAction::T3SafeMode
            } else if temperature_c >= config.thermal_degraded_threshold_c {
                ThermalAction::T2DegradeRealtime
            } else if temperature_c <= config.thermal_recovery_threshold_c {
                ThermalAction::Normal
            } else {
                ThermalAction::T1ConstrainBestEffort
            }
        }
        ThermalAction::T2DegradeRealtime => {
            if temperature_c >= config.thermal_hard_threshold_c {
                ThermalAction::T3SafeMode
            } else if temperature_c <= config.thermal_recovery_threshold_c {
                ThermalAction::Normal
            } else if temperature_c < config.thermal_degraded_threshold_c {
                ThermalAction::T1ConstrainBestEffort
            } else {
                ThermalAction::T2DegradeRealtime
            }
        }
        ThermalAction::T3SafeMode => {
            if temperature_c <= config.thermal_recovery_threshold_c {
                ThermalAction::Normal
            } else if temperature_c < config.thermal_hard_threshold_c {
                ThermalAction::T2DegradeRealtime
            } else {
                ThermalAction::T3SafeMode
            }
        }
    }
}

fn apply_thermal_action_change(
    config: &RuntimeConfig,
    next_action: ThermalAction,
    temperature_c: f64,
    metrics: &RuntimeMetrics,
    registry: &PackageRegistry,
    scheduler_state: &SchedulerState,
) {
    match next_action {
        ThermalAction::Normal => {
            scheduler_state.set_thermal_action(ThermalAction::Normal);
            scheduler_state.set_thermal_fps_factor(1.0);
            scheduler_state.set_best_effort_gate_divisor(1);
            registry.restore_suspended(metrics);
            metrics.set_status("restored", EventLevel::Normal);
            metrics.push_scheduler_log(
                EventLevel::Normal,
                format!(
                    "thermal action changed: restored at {:.1}C — preemptible packages resumed",
                    temperature_c
                ),
            );
        }
        ThermalAction::T1ConstrainBestEffort => {
            scheduler_state.set_thermal_action(ThermalAction::T1ConstrainBestEffort);
            scheduler_state.set_thermal_fps_factor(1.0);
            scheduler_state.set_best_effort_gate_divisor(2);
            metrics.set_status("thermal_t1", EventLevel::Warning);
            metrics.push_scheduler_log(
                EventLevel::Warning,
                format!(
                    "thermal action changed: t1-constrain-best-effort at {:.1}C — best_effort_gate=2",
                    temperature_c
                ),
            );
        }
        ThermalAction::T2DegradeRealtime => {
            scheduler_state.set_thermal_action(ThermalAction::T2DegradeRealtime);
            scheduler_state.set_thermal_fps_factor(config.thermal_throttle_fps_factor);
            scheduler_state.set_best_effort_gate_divisor(4);
            metrics.set_status("throttled", EventLevel::Warning);
            metrics.push_scheduler_log(
                EventLevel::Warning,
                format!(
                    "thermal action changed: t2-degrade-realtime at {:.1}C — fps_factor={:.2} best_effort_gate=4",
                    temperature_c, config.thermal_throttle_fps_factor
                ),
            );
        }
        ThermalAction::T3SafeMode => {
            scheduler_state.set_thermal_action(ThermalAction::T3SafeMode);
            scheduler_state.set_thermal_fps_factor(config.thermal_throttle_fps_factor * 0.8);
            scheduler_state.set_best_effort_gate_divisor(16);
            registry.suspend_preemptible(metrics);
            registry.enter_safe_mode(metrics);
            metrics.set_status("suspended", EventLevel::Critical);
            metrics.push_scheduler_log(
                EventLevel::Critical,
                format!(
                    "thermal action changed: t3-safe-mode at {:.1}C — only critical perception remains active",
                    temperature_c
                ),
            );
        }
    }
}

// ── battery_controller_loop ───────────────────────────────────────────────────

fn battery_controller_loop(
    config: RuntimeConfig,
    metrics: Arc<RuntimeMetrics>,
    scheduler_state: Arc<SchedulerState>,
    shutdown: Arc<AtomicBool>,
) {
    let mut current_action = BatteryAction::Normal;

    while !shutdown.load(Ordering::Relaxed) {
        let snapshot = read_battery_snapshot();
        metrics.set_battery_snapshot(
            snapshot.map(|value| value.capacity_percent),
            snapshot.map(|value| value.is_discharging),
        );
        let next_action = evaluate_battery_action(current_action, snapshot, &config);

        if next_action != current_action {
            apply_battery_action_change(&config, next_action, snapshot, &metrics, &scheduler_state);
            current_action = next_action;
        }

        sleep_with_shutdown(
            Duration::from_millis(config.battery_poll_interval_ms),
            &shutdown,
        );
    }
}

fn evaluate_battery_action(
    current: BatteryAction,
    snapshot: Option<BatterySnapshot>,
    config: &RuntimeConfig,
) -> BatteryAction {
    const RECOVERY_HYSTERESIS_PERCENT: f64 = 2.0;

    let Some(snapshot) = snapshot else {
        return BatteryAction::Normal;
    };

    if !snapshot.is_discharging {
        return BatteryAction::Normal;
    }

    match current {
        BatteryAction::Normal => {
            if snapshot.capacity_percent <= config.battery_critical_threshold_percent {
                BatteryAction::Critical
            } else if snapshot.capacity_percent <= config.battery_low_threshold_percent {
                BatteryAction::Conserve
            } else {
                BatteryAction::Normal
            }
        }
        BatteryAction::Conserve => {
            if snapshot.capacity_percent <= config.battery_critical_threshold_percent {
                BatteryAction::Critical
            } else if snapshot.capacity_percent
                > config.battery_low_threshold_percent + RECOVERY_HYSTERESIS_PERCENT
            {
                BatteryAction::Normal
            } else {
                BatteryAction::Conserve
            }
        }
        BatteryAction::Critical => {
            if snapshot.capacity_percent
                > config.battery_critical_threshold_percent + RECOVERY_HYSTERESIS_PERCENT
            {
                if snapshot.capacity_percent <= config.battery_low_threshold_percent {
                    BatteryAction::Conserve
                } else {
                    BatteryAction::Normal
                }
            } else {
                BatteryAction::Critical
            }
        }
    }
}

fn apply_battery_action_change(
    config: &RuntimeConfig,
    next_action: BatteryAction,
    snapshot: Option<BatterySnapshot>,
    metrics: &RuntimeMetrics,
    scheduler_state: &SchedulerState,
) {
    scheduler_state.set_battery_action(next_action);
    metrics.set_battery_action_code(next_action as u8);

    let context = match snapshot {
        Some(snapshot) => format!(
            "battery={:.0}% discharging={}",
            snapshot.capacity_percent, snapshot.is_discharging
        ),
        None => "battery=unavailable".to_string(),
    };

    match next_action {
        BatteryAction::Normal => {
            scheduler_state.set_battery_fps_factor(1.0);
            scheduler_state.set_response_token_cap(None);
            metrics.push_scheduler_log(
                EventLevel::Normal,
                format!("battery action changed: normal ({})", context),
            );
        }
        BatteryAction::Conserve => {
            scheduler_state.set_battery_fps_factor(config.battery_conserve_fps_factor);
            scheduler_state.set_response_token_cap(None);
            metrics.push_scheduler_log(
                EventLevel::Warning,
                format!(
                    "battery action changed: conserve ({}) — best_effort disabled fps_factor={:.2}",
                    context, config.battery_conserve_fps_factor
                ),
            );
        }
        BatteryAction::Critical => {
            scheduler_state.set_battery_fps_factor(config.battery_critical_fps_factor);
            scheduler_state.set_response_token_cap(Some(config.battery_critical_max_tokens));
            metrics.push_scheduler_log(
                EventLevel::Critical,
                format!(
                    "battery action changed: critical ({}) — best_effort+interactive constrained fps_factor={:.2} max_new_tokens={}",
                    context,
                    config.battery_critical_fps_factor,
                    config.battery_critical_max_tokens
                ),
            );
        }
    }
}

// ── trigger_dispatcher_loop ───────────────────────────────────────────────────

fn memory_guard_loop(
    config: RuntimeConfig,
    metrics: Arc<RuntimeMetrics>,
    registry: Arc<PackageRegistry>,
    scheduler_state: Arc<SchedulerState>,
    shutdown: Arc<AtomicBool>,
) {
    let mut current_action = MemoryAction::Normal;

    while !shutdown.load(Ordering::Relaxed) {
        let available_mib = metrics.available_memory_mib();
        let next_action = match current_action {
            MemoryAction::Normal if available_mib <= config.memory_emergency_free_mib => {
                MemoryAction::Emergency
            }
            MemoryAction::Emergency if available_mib >= config.memory_recovery_free_mib => {
                MemoryAction::Normal
            }
            _ => current_action,
        };

        if next_action != current_action {
            match next_action {
                MemoryAction::Normal => {
                    scheduler_state.set_memory_action(MemoryAction::Normal);
                    scheduler_state.set_memory_token_cap(None);
                    registry.clear_cancel_requests();
                    metrics.set_memory_emergency_active(false);
                    metrics.push_scheduler_log(
                        EventLevel::Normal,
                        format!(
                            "memory action changed: normal available={}MiB recovery={}MiB",
                            available_mib, config.memory_recovery_free_mib
                        ),
                    );
                }
                MemoryAction::Emergency => {
                    scheduler_state.set_memory_action(MemoryAction::Emergency);
                    scheduler_state
                        .set_memory_token_cap(Some(config.memory_emergency_max_tokens.max(8)));
                    registry.request_cancel_preemptible(&metrics, "memory-emergency");
                    metrics.set_memory_emergency_active(true);
                    metrics.set_status("memory_emergency", EventLevel::Critical);
                    metrics.push_scheduler_log(
                        EventLevel::Critical,
                        format!(
                            "memory action changed: emergency available={}MiB emergency={}MiB max_new_tokens={}",
                            available_mib,
                            config.memory_emergency_free_mib,
                            config.memory_emergency_max_tokens
                        ),
                    );
                }
            }
            current_action = next_action;
        }

        sleep_with_shutdown(
            Duration::from_millis(config.memory_guard_poll_interval_ms),
            &shutdown,
        );
    }
}

#[derive(Debug)]
struct PendingTrigger {
    event: TriggerEvent,
    target_package: String,
    package_priority: u8,
    event_priority: u8,
    enqueue_seq: u64,
}

impl PartialEq for PendingTrigger {
    fn eq(&self, other: &Self) -> bool {
        self.package_priority == other.package_priority
            && self.event_priority == other.event_priority
            && self.enqueue_seq == other.enqueue_seq
    }
}

impl Eq for PendingTrigger {}

impl PartialOrd for PendingTrigger {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for PendingTrigger {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        self.package_priority
            .cmp(&other.package_priority)
            .then_with(|| self.event_priority.cmp(&other.event_priority))
            .then_with(|| other.enqueue_seq.cmp(&self.enqueue_seq))
    }
}

fn package_priority_rank(priority: Priority) -> u8 {
    match priority {
        Priority::Low => 1,
        Priority::Normal => 2,
        Priority::High => 3,
    }
}

fn event_priority_rank(event: &TriggerEvent) -> u8 {
    match event {
        TriggerEvent::DetectionFired { .. } => 1,
        TriggerEvent::ManualTrigger { .. } => 2,
        TriggerEvent::ExternalInput { .. } => 3,
    }
}

fn trigger_dispatcher_loop(
    config: RuntimeConfig,
    trigger_rx: Receiver<TriggerEvent>,
    metrics: Arc<RuntimeMetrics>,
    registry: Arc<PackageRegistry>,
    scheduler_state: Arc<SchedulerState>,
    shutdown: Arc<AtomicBool>,
) {
    let mut pending = BinaryHeap::<PendingTrigger>::new();
    let mut pending_by_target = HashMap::<String, usize>::new();
    let mut enqueue_seq = 0u64;

    while !shutdown.load(Ordering::Relaxed) || !trigger_rx.is_empty() || !pending.is_empty() {
        let recv_wait = if pending.is_empty() {
            Duration::from_millis(100)
        } else {
            Duration::from_millis(5)
        };

        match trigger_rx.recv_timeout(recv_wait) {
            Ok(event) => {
                let target = event.target_package().to_string();
                let package_priority = registry
                    .package_spec(&target)
                    .map(|spec| package_priority_rank(spec.priority))
                    .unwrap_or(0);
                let event_priority = event_priority_rank(&event);

                pending.push(PendingTrigger {
                    event,
                    target_package: target.clone(),
                    package_priority,
                    event_priority,
                    enqueue_seq,
                });
                enqueue_seq = enqueue_seq.saturating_add(1);

                let queued = pending_by_target.entry(target.clone()).or_insert(0);
                *queued = queued.saturating_add(1);
                metrics.set_model_queue_depth(&target, *queued);
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                if shutdown.load(Ordering::Relaxed) && pending.is_empty() {
                    break;
                }
            }
        }

        metrics.set_queue_depth(pending.len());
        while let Some(next) = pending.pop() {
            if let Some(queued) = pending_by_target.get_mut(&next.target_package) {
                *queued = queued.saturating_sub(1);
                metrics.set_model_queue_depth(&next.target_package, *queued);
            }
            handle_trigger_event(&config, next.event, &metrics, &registry, &scheduler_state);
        }
    }
}

fn handle_trigger_event(
    config: &RuntimeConfig,
    event: TriggerEvent,
    metrics: &RuntimeMetrics,
    registry: &PackageRegistry,
    scheduler_state: &SchedulerState,
) {
    let target = event.target_package().to_string();
    let event_priority = event.priority_label();
    let event_preemptible = event.is_preemptible();
    let Some(package_spec) = registry.package_spec(&target) else {
        metrics.push_scheduler_log(
            EventLevel::Warning,
            format!(
                "trigger rejected: target={} reason=package-not-found-or-not-loaded priority={} preemptible={}",
                target, event_priority, event_preemptible
            ),
        );
        return;
    };

    if let Some(deadline_ms) = package_spec.deadline_ms
        && let Some(age_ms) = trigger_age_ms(&event)
        && age_ms > deadline_ms
    {
        metrics.push_scheduler_log(
            EventLevel::Warning,
            format!(
                "trigger dropped: target={} reason=deadline-missed age={}ms deadline={}ms",
                target, age_ms, deadline_ms
            ),
        );
        return;
    }

    let battery_action = scheduler_state.battery_action();
    if battery_action == BatteryAction::Conserve && package_spec.task_class == TaskClass::BestEffort
    {
        metrics.push_scheduler_log(
            EventLevel::Warning,
            format!(
                "trigger deferred: target={} reason=battery-conserve class={} priority={} preemptible={}",
                target,
                package_spec.task_class.as_str(),
                event_priority,
                event_preemptible,
            ),
        );
        return;
    }

    if battery_action == BatteryAction::Critical && package_spec.task_class == TaskClass::BestEffort
    {
        metrics.push_scheduler_log(
            EventLevel::Critical,
            format!(
                "trigger rejected: target={} reason=battery-critical class={} priority={} preemptible={}",
                target,
                package_spec.task_class.as_str(),
                event_priority,
                event_preemptible,
            ),
        );
        return;
    }

    if scheduler_state.memory_action() == MemoryAction::Emergency
        && package_spec.task_class != TaskClass::Realtime
    {
        metrics.push_scheduler_log(
            EventLevel::Critical,
            format!(
                "trigger rejected: target={} reason=memory-emergency class={} priority={} preemptible={}",
                target,
                package_spec.task_class.as_str(),
                event_priority,
                event_preemptible,
            ),
        );
        return;
    }

    let thermal_action = scheduler_state.thermal_action();
    if matches!(
        thermal_action,
        ThermalAction::T1ConstrainBestEffort | ThermalAction::T2DegradeRealtime
    ) && package_spec.task_class == TaskClass::BestEffort
        && !scheduler_state.allow_best_effort_dispatch()
    {
        metrics.push_scheduler_log(
            EventLevel::Warning,
            format!(
                "trigger deferred: target={} reason=thermal-load-shed action={:?} class={} priority={} preemptible={}",
                target,
                thermal_action,
                package_spec.task_class.as_str()
                ,
                event_priority,
                event_preemptible
            ),
        );
        return;
    }

    if thermal_action == ThermalAction::T3SafeMode
        && !(package_spec.task_class == TaskClass::Realtime && package_spec.critical_perception)
    {
        metrics.push_scheduler_log(
            EventLevel::Critical,
            format!(
                "trigger rejected: target={} reason=thermal-safe-mode class={} critical_perception={} priority={} preemptible={}",
                target,
                package_spec.task_class.as_str(),
                package_spec.critical_perception,
                event_priority,
                event_preemptible,
            ),
        );
        return;
    }

    // Reject if not enough free memory to safely dispatch.
    let available_mib = metrics.available_memory_mib();

    let required_free_mib = if scheduler_state.memory_action() == MemoryAction::Emergency {
        config
            .trigger_required_free_mib
            .max(config.memory_recovery_free_mib)
    } else {
        config.trigger_required_free_mib
    };

    if available_mib < required_free_mib {
        metrics.push_scheduler_log(
            EventLevel::Warning,
            format!(
                "trigger rejected: target={} reason=insufficient-memory available={}MiB required={}MiB",
                target, available_mib, required_free_mib,
            ),
        );
        return;
    }

    match &event {
        TriggerEvent::DetectionFired {
            label,
            confidence,
            source_package,
            ..
        } => {
            if *confidence < config.trigger_confidence_threshold {
                metrics.push_scheduler_log(
                    EventLevel::Normal,
                    format!(
                        "trigger filtered: target={} label={} confidence={:.2} threshold={:.2}",
                        target, label, confidence, config.trigger_confidence_threshold
                    ),
                );
                return;
            }
            metrics.push_scheduler_log(
                EventLevel::Normal,
                format!(
                    "trigger accepted: label={} confidence={:.2} source={} target={} class={}",
                    label,
                    confidence,
                    source_package,
                    target,
                    package_spec.task_class.as_str()
                ),
            );
        }
        TriggerEvent::ManualTrigger { custom_prompt, .. } => {
            metrics.push_scheduler_log(
                EventLevel::Normal,
                format!(
                    "trigger accepted: manual target={} class={} prompt=\"{}\"",
                    target,
                    package_spec.task_class.as_str(),
                    truncate_prompt(custom_prompt)
                ),
            );
        }
        TriggerEvent::ExternalInput { event, .. } => {
            metrics.push_scheduler_log(
                        EventLevel::Normal,
                        format!(
                            "trigger accepted: external target={} class={} source={} payload={}B metadata_keys={}",
                            target,
                            package_spec.task_class.as_str(),
                            event.source_id,
                            event.payload.size_hint_bytes(),
                            event.metadata.len()
                        ),
                    );
        }
    }

    if let Err(reason) = registry.dispatch_trigger(&target, event) {
        metrics.push_scheduler_log(
            EventLevel::Warning,
            format!("trigger dropped: target={} reason={}", target, reason),
        );
    }
}

// ── thermal reading ───────────────────────────────────────────────────────────

fn read_thermal_celsius() -> f64 {
    #[cfg(target_os = "linux")]
    {
        let root = std::path::Path::new("/sys/class/thermal");
        if let Ok(entries) = std::fs::read_dir(root) {
            let mut max_celsius = None;
            for entry in entries.flatten() {
                let file_name = entry.file_name();
                let name = file_name.to_string_lossy();
                if !name.starts_with("thermal_zone") {
                    continue;
                }

                if let Ok(raw) = std::fs::read_to_string(entry.path().join("temp"))
                    && let Ok(raw_temp) = raw.trim().parse::<f64>()
                {
                    let celsius = if raw_temp > 200.0 {
                        raw_temp / 1000.0
                    } else {
                        raw_temp
                    };
                    max_celsius = Some(
                        max_celsius
                            .map(|current| current.max(celsius))
                            .unwrap_or(celsius),
                    );
                }
            }
            if let Some(value) = max_celsius {
                return value;
            }
        }

        45.0
    }

    #[cfg(not(target_os = "linux"))]
    {
        #[cfg(target_os = "windows")]
        {
            return read_windows_thermal_celsius().unwrap_or(45.0);
        }

        #[cfg(not(target_os = "windows"))]
        {
            45.0
        }
    }
}

#[cfg(target_os = "windows")]
fn read_windows_thermal_celsius() -> Option<f64> {
    use std::sync::OnceLock;
    use std::time::{Duration, Instant};

    const MIN_REFRESH: Duration = Duration::from_secs(10);
    static CACHE: OnceLock<Mutex<Option<(Instant, f64)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));

    if let Ok(guard) = cache.lock()
        && let Some((last_read, value)) = *guard
        && last_read.elapsed() < MIN_REFRESH
    {
        return Some(value);
    }

    let value = query_windows_thermal_celsius()?;
    if let Ok(mut guard) = cache.lock() {
        *guard = Some((Instant::now(), value));
    }
    Some(value)
}

#[cfg(target_os = "windows")]
fn query_windows_thermal_celsius() -> Option<f64> {
    query_windows_acpi_thermal_celsius().or_else(query_windows_perf_counter_thermal_celsius)
}

#[cfg(target_os = "windows")]
fn query_windows_acpi_thermal_celsius() -> Option<f64> {
    let script = r#"$temps = Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature -ErrorAction SilentlyContinue | Select-Object -ExpandProperty CurrentTemperature; if ($temps) { $temps }"#;
    let output = std::process::Command::new("powershell.exe")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let mut max_celsius = None;
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let Some(raw) = line.trim().parse::<f64>().ok() else {
            continue;
        };
        // MSAcpi_ThermalZoneTemperature reports tenths of Kelvin.
        let celsius = (raw / 10.0) - 273.15;
        if !celsius.is_finite() || !(-50.0..=150.0).contains(&celsius) {
            continue;
        }
        max_celsius = Some(
            max_celsius
                .map(|current: f64| current.max(celsius))
                .unwrap_or(celsius),
        );
    }

    max_celsius
}

#[cfg(target_os = "windows")]
fn query_windows_perf_counter_thermal_celsius() -> Option<f64> {
    let script = r#"$samples = (Get-Counter '\Thermal Zone Information(*)\Temperature' -ErrorAction SilentlyContinue).CounterSamples | Select-Object -ExpandProperty CookedValue; if ($samples) { $samples }"#;
    let output = std::process::Command::new("powershell.exe")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let mut max_celsius = None;
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let Some(raw) = line.trim().parse::<f64>().ok() else {
            continue;
        };
        let Some(celsius) = normalize_windows_counter_temp(raw) else {
            continue;
        };
        max_celsius = Some(
            max_celsius
                .map(|current: f64| current.max(celsius))
                .unwrap_or(celsius),
        );
    }

    max_celsius
}

#[cfg(target_os = "windows")]
fn normalize_windows_counter_temp(raw: f64) -> Option<f64> {
    if !raw.is_finite() {
        return None;
    }

    // Windows thermal counter units vary by platform/firmware:
    // - some devices report tenths of Celsius (e.g. 361 => 36.1C)
    // - some can report Celsius directly.
    let celsius = if raw >= 2000.0 {
        // Very large values are usually tenths of Kelvin.
        (raw / 10.0) - 273.15
    } else if raw >= 150.0 {
        // Common case for counter samples on laptops/desktops.
        raw / 10.0
    } else {
        raw
    };

    if (-50.0..=150.0).contains(&celsius) {
        Some(celsius)
    } else {
        None
    }
}

fn read_battery_snapshot() -> Option<BatterySnapshot> {
    #[cfg(target_os = "linux")]
    {
        let supply_dir = std::path::Path::new("/sys/class/power_supply");
        let entries = std::fs::read_dir(supply_dir).ok()?;

        for entry in entries.flatten() {
            let path = entry.path();
            let device_name = entry.file_name();
            let device_name = device_name.to_string_lossy();

            let is_battery = match std::fs::read_to_string(path.join("type")) {
                Ok(raw_type) => raw_type.trim().eq_ignore_ascii_case("battery"),
                Err(_) => device_name.starts_with("BAT"),
            };
            if !is_battery {
                continue;
            }

            let capacity_percent = std::fs::read_to_string(path.join("capacity"))
                .ok()
                .and_then(|raw| raw.trim().parse::<f64>().ok())
                .map(|percent| percent.clamp(0.0, 100.0))?;

            let is_discharging = std::fs::read_to_string(path.join("status"))
                .ok()
                .map(|status| status.trim().eq_ignore_ascii_case("discharging"))
                .unwrap_or(false);

            return Some(BatterySnapshot {
                capacity_percent,
                is_discharging,
            });
        }

        None
    }

    #[cfg(not(target_os = "linux"))]
    {
        #[cfg(target_os = "windows")]
        {
            return read_windows_battery_snapshot();
        }

        #[cfg(not(target_os = "windows"))]
        {
            None
        }
    }
}

#[cfg(target_os = "windows")]
fn read_windows_battery_snapshot() -> Option<BatterySnapshot> {
    let script = r#"$b = Get-CimInstance -ClassName Win32_Battery -ErrorAction SilentlyContinue | Select-Object -First 1 EstimatedChargeRemaining,BatteryStatus,PowerOnline; if ($b) { $b | ConvertTo-Json -Compress }"#;
    let output = std::process::Command::new("powershell.exe")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let raw = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if raw.is_empty() {
        return None;
    }

    let value: Value = serde_json::from_str(&raw).ok()?;
    let capacity_percent = value
        .get("EstimatedChargeRemaining")
        .and_then(|field| field.as_f64().or_else(|| field.as_u64().map(|v| v as f64)))
        .map(|percent| percent.clamp(0.0, 100.0))?;

    let battery_status = value
        .get("BatteryStatus")
        .and_then(|field| field.as_u64().or_else(|| field.as_i64().map(|v| v as u64)));
    let power_online = value.get("PowerOnline").and_then(Value::as_bool);

    let is_discharging = if let Some(online) = power_online {
        !online
    } else {
        matches!(battery_status, Some(1 | 4 | 5))
    };

    Some(BatterySnapshot {
        capacity_percent,
        is_discharging,
    })
}

fn read_cpufreq_throttle_ratio() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        let root = std::path::Path::new("/sys/devices/system/cpu");
        let entries = std::fs::read_dir(root).ok()?;
        let mut ratios = Vec::new();

        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();
            if !name.starts_with("cpu") || name == "cpufreq" {
                continue;
            }
            let cpufreq_dir = entry.path().join("cpufreq");
            let cur = std::fs::read_to_string(cpufreq_dir.join("scaling_cur_freq"))
                .ok()
                .and_then(|raw| raw.trim().parse::<f64>().ok());
            let max = std::fs::read_to_string(cpufreq_dir.join("cpuinfo_max_freq"))
                .ok()
                .and_then(|raw| raw.trim().parse::<f64>().ok());
            if let (Some(cur), Some(max)) = (cur, max)
                && max > 0.0
            {
                ratios.push((cur / max).clamp(0.0, 1.0));
            }
        }

        if ratios.is_empty() {
            None
        } else {
            Some(ratios.iter().copied().sum::<f64>() / ratios.len() as f64)
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn factor_to_milli(factor: f64) -> u64 {
    (factor.clamp(0.05, 1.0) * 1000.0).round() as u64
}

fn trigger_age_ms(event: &TriggerEvent) -> Option<u64> {
    match event {
        TriggerEvent::ExternalInput { event, .. } => {
            let now = unix_time_millis();
            Some(now.saturating_sub(event.timestamp_ms))
        }
        _ => None,
    }
}

fn backend_affinity_available(affinity: BackendAffinity) -> bool {
    if affinity == BackendAffinity::Any {
        return true;
    }

    let raw = env::var("KAPSL_LITE_AVAILABLE_BACKENDS").unwrap_or_else(|_| "cpu".to_string());
    let normalized = raw.to_ascii_lowercase();
    let has = |needle: &str| {
        normalized
            .split([',', ';', ' '])
            .map(str::trim)
            .filter(|token| !token.is_empty())
            .any(|token| token == needle)
    };

    match affinity {
        BackendAffinity::Any => true,
        BackendAffinity::Cpu => has("cpu"),
        BackendAffinity::Gpu => has("gpu") || has("cuda") || has("rocm") || has("metal"),
        BackendAffinity::Npu => has("npu"),
    }
}

fn truncate_prompt(value: &str) -> String {
    const LIMIT: usize = 48;
    let mut output: String = value.chars().take(LIMIT).collect();
    if value.chars().count() > LIMIT {
        output.push_str("...");
    }
    output
}

fn read_env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

fn read_env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn read_env_f64(name: &str, default: f64) -> f64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(default)
}

fn unix_time_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

fn sleep_with_shutdown(duration: Duration, shutdown: &AtomicBool) {
    let step = Duration::from_millis(50);
    let start = Instant::now();

    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        let elapsed = start.elapsed();
        if elapsed >= duration {
            break;
        }

        let remaining = duration - elapsed;
        thread::sleep(remaining.min(step));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::package::{
        BackendAffinity, KapslPackageSpec, MetricsToggles, ModelFormat, OnnxRuntimeTuningSpec,
        PackageThermalThresholds, Priority, QueueOverflowPolicy, SwapPolicy, TaskClass,
        TriggerMode,
    };
    use crossbeam_channel::bounded;

    fn test_config() -> RuntimeConfig {
        RuntimeConfig {
            memory_limit_mib: 512,
            trigger_confidence_threshold: 0.8,
            trigger_required_free_mib: 64,
            trigger_bus_capacity: 8,
            thermal_poll_interval_ms: 2000,
            thermal_soft_threshold_c: 70.0,
            thermal_degraded_threshold_c: 76.0,
            thermal_hard_threshold_c: 82.0,
            thermal_recovery_threshold_c: 65.0,
            thermal_throttle_fps_factor: 0.5,
            battery_poll_interval_ms: 10000,
            battery_low_threshold_percent: 30.0,
            battery_critical_threshold_percent: 15.0,
            battery_conserve_fps_factor: 0.75,
            battery_critical_fps_factor: 0.5,
            battery_critical_max_tokens: 128,
            memory_guard_poll_interval_ms: 1000,
            memory_emergency_free_mib: 32,
            memory_recovery_free_mib: 16,
            memory_emergency_max_tokens: 1024,
        }
    }

    fn on_demand_spec(name: &str) -> KapslPackageSpec {
        KapslPackageSpec {
            name: name.to_string(),
            format: ModelFormat::Onnx,
            is_llm: false,
            weights: "model.onnx".to_string(),
            weights_path: "model.onnx".to_string(),
            memory_mb: 64,
            priority: Priority::Normal,
            task_class: TaskClass::BestEffort,
            trigger_mode: TriggerMode::OnDemand,
            preemptible: true,
            max_concurrent: 1,
            on_demand_queue_capacity: 4,
            on_demand_queue_policy: QueueOverflowPolicy::DropNewest,
            cpu_threads: 1,
            cost_estimate: None,
            deadline_ms: Some(500),
            backend_affinity: BackendAffinity::Any,
            hardware_target: String::new(),
            critical_perception: false,
            swap: SwapPolicy::Disallowed,
            thermal: PackageThermalThresholds::default(),
            metrics: MetricsToggles::default(),
            fps_target: None,
            onnx_tuning: OnnxRuntimeTuningSpec::default(),
        }
    }

    #[test]
    fn battery_action_transitions_follow_thresholds() {
        let config = test_config();

        let normal = evaluate_battery_action(
            BatteryAction::Normal,
            Some(BatterySnapshot {
                capacity_percent: 70.0,
                is_discharging: true,
            }),
            &config,
        );
        assert_eq!(normal, BatteryAction::Normal);

        let conserve = evaluate_battery_action(
            BatteryAction::Normal,
            Some(BatterySnapshot {
                capacity_percent: 25.0,
                is_discharging: true,
            }),
            &config,
        );
        assert_eq!(conserve, BatteryAction::Conserve);

        let critical = evaluate_battery_action(
            BatteryAction::Conserve,
            Some(BatterySnapshot {
                capacity_percent: 12.0,
                is_discharging: true,
            }),
            &config,
        );
        assert_eq!(critical, BatteryAction::Critical);

        let charging = evaluate_battery_action(
            BatteryAction::Critical,
            Some(BatterySnapshot {
                capacity_percent: 5.0,
                is_discharging: false,
            }),
            &config,
        );
        assert_eq!(charging, BatteryAction::Normal);
    }

    #[test]
    fn battery_conserve_defers_low_priority_detection_triggers() {
        let config = test_config();
        let scheduler_state = SchedulerState::new();
        scheduler_state.set_battery_action(BatteryAction::Conserve);

        let metrics = RuntimeMetrics::new(1, 512);
        let registry = PackageRegistry::new(512);
        let (tx, rx) = bounded(1);
        registry
            .register(
                on_demand_spec("reasoning"),
                Some(tx),
                Some(rx.clone()),
                Arc::new(AtomicBool::new(false)),
            )
            .unwrap();

        handle_trigger_event(
            &config,
            TriggerEvent::DetectionFired {
                label: "person".to_string(),
                confidence: 0.95,
                source_package: "vision.detector".to_string(),
                target_package: "reasoning".to_string(),
            },
            &metrics,
            &registry,
            &scheduler_state,
        );

        assert!(
            rx.try_recv().is_err(),
            "detection trigger should be deferred"
        );
        assert!(
            metrics
                .scheduler_logs()
                .iter()
                .any(|entry| entry.message.contains("reason=battery-conserve"))
        );
    }

    #[test]
    fn battery_critical_rejects_manual_triggers() {
        let config = test_config();
        let scheduler_state = SchedulerState::new();
        scheduler_state.set_battery_action(BatteryAction::Critical);

        let metrics = RuntimeMetrics::new(1, 512);
        let registry = PackageRegistry::new(512);
        let (tx, rx) = bounded(1);
        registry
            .register(
                on_demand_spec("reasoning"),
                Some(tx),
                Some(rx.clone()),
                Arc::new(AtomicBool::new(false)),
            )
            .unwrap();

        handle_trigger_event(
            &config,
            TriggerEvent::ManualTrigger {
                target_package: "reasoning".to_string(),
                custom_prompt: "run".to_string(),
            },
            &metrics,
            &registry,
            &scheduler_state,
        );

        assert!(rx.try_recv().is_err(), "manual trigger should be rejected");
        assert!(
            metrics
                .scheduler_logs()
                .iter()
                .any(|entry| entry.message.contains("reason=battery-critical"))
        );
    }
}
