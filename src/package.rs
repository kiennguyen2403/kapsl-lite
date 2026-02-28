use serde::Deserialize;

/// Runtime scheduling profile derived from a `.aimod` package manifest.
#[derive(Debug, Clone, Deserialize)]
pub struct KapslPackageSpec {
    /// Human-readable package name, used as identifier in logs and trigger routing.
    pub name: String,

    /// Model file format — determines which backend is used.
    pub format: ModelFormat,

    /// True when manifest.framework is `llm`; used to select generation backend.
    #[serde(default)]
    pub is_llm: bool,

    /// Model path from `metadata.json::model_file` in the `.aimod` package.
    pub weights: String,

    /// Absolute extracted model path resolved by the package loader.
    pub weights_path: String,

    /// Hard memory cap for this package in megabytes.  The scheduler will never
    /// exceed this limit when loading.
    pub memory_mb: u64,

    /// Scheduling priority — affects queue order and preemption decisions.
    pub priority: Priority,

    /// Scheduler class used by governors and admission control.
    #[serde(default)]
    pub task_class: TaskClass,

    /// When this package's inference loop runs.
    pub trigger_mode: TriggerMode,

    /// If true, the thermal controller may suspend this package under hard thermal
    /// pressure and restore it once temperatures recover.
    pub preemptible: bool,

    /// Maximum number of inference requests that may execute concurrently for this
    /// package.  Edge devices typically set this to 1.
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,

    /// Queue capacity for on-demand dispatch.
    #[serde(default = "default_on_demand_queue_capacity")]
    pub on_demand_queue_capacity: usize,

    /// Overflow strategy when on-demand queue is full.
    #[serde(default)]
    pub on_demand_queue_policy: QueueOverflowPolicy,

    /// Number of CPU threads allocated to this package's inference backend.
    #[serde(default = "default_cpu_threads")]
    pub cpu_threads: usize,

    /// Estimated runtime cost for admission and power/thermal policy.
    #[serde(default)]
    pub cost_estimate: Option<CostEstimate>,

    /// Per-task freshness deadline in milliseconds.
    pub deadline_ms: Option<u64>,

    /// Preferred execution backend family.
    #[serde(default)]
    pub backend_affinity: BackendAffinity,

    /// Free-form hardware target declaration (e.g. "rpi5", "jetson-nano").
    #[serde(default)]
    pub hardware_target: String,

    /// Marks perception workloads that may continue in thermal safe mode.
    #[serde(default)]
    pub critical_perception: bool,

    /// Whether the OS may swap this package's weights to disk.
    #[serde(default)]
    pub swap: SwapPolicy,

    /// Per-package thermal thresholds that override the global runtime defaults
    /// when present.
    #[serde(default)]
    pub thermal: PackageThermalThresholds,

    /// Selectively enable or disable individual metric signals for this package.
    #[serde(default)]
    pub metrics: MetricsToggles,

    /// Target FPS for vision model packages.  Ignored for LLM packages.
    pub fps_target: Option<f64>,

    /// ONNX Runtime memory/session tuning for this package.
    #[serde(default)]
    pub onnx_tuning: OnnxRuntimeTuningSpec,
}

impl KapslPackageSpec {
    /// Returns the effective FPS target, defaulting to 1.0 if not set.
    pub fn effective_fps(&self) -> f64 {
        self.fps_target.unwrap_or(1.0).max(0.1)
    }

    /// Interval between inference cycles for `always_running` packages.
    pub fn cycle_interval_ms(&self) -> u64 {
        (1000.0 / self.effective_fps()).round() as u64
    }
}

#[derive(Debug, Clone, Deserialize, Default, PartialEq, Eq)]
pub struct OnnxRuntimeTuningSpec {
    pub memory_pattern: Option<bool>,
    pub disable_cpu_mem_arena: Option<bool>,
    pub session_buckets: Option<usize>,
    pub bucket_dim_granularity: Option<usize>,
    pub bucket_max_dims: Option<usize>,
    pub peak_concurrency_hint: Option<u32>,
}

// ── TaskClass ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TaskClass {
    Realtime,
    Interactive,
    #[default]
    BestEffort,
}

impl TaskClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Realtime => "realtime",
            Self::Interactive => "interactive",
            Self::BestEffort => "best_effort",
        }
    }
}

// ── CostEstimate ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct CostEstimate {
    pub value: u64,
    pub unit: CostUnit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostUnit {
    Millis,
    TokensPerSecond,
}

impl CostEstimate {
    pub fn as_label(self) -> String {
        match self.unit {
            CostUnit::Millis => format!("{}ms", self.value),
            CostUnit::TokensPerSecond => format!("{}tps", self.value),
        }
    }
}

// ── BackendAffinity ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum BackendAffinity {
    #[default]
    Any,
    Cpu,
    Gpu,
    Npu,
}

impl BackendAffinity {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Any => "any",
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
            Self::Npu => "npu",
        }
    }
}

// ── QueueOverflowPolicy ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum QueueOverflowPolicy {
    DropOldest,
    #[default]
    DropNewest,
    Block,
}

impl QueueOverflowPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DropOldest => "drop_oldest",
            Self::DropNewest => "drop_newest",
            Self::Block => "block",
        }
    }
}

// ── ModelFormat ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelFormat {
    Gguf,
    Onnx,
}

impl ModelFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gguf => "gguf",
            Self::Onnx => "onnx",
        }
    }
}

// ── Priority ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
}

impl Priority {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Normal => "normal",
            Self::High => "high",
        }
    }
}

// ── TriggerMode ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TriggerMode {
    /// Package runs continuous inference at `fps_target` Hz.
    AlwaysRunning,
    /// Package only runs when a `TriggerEvent` targets its name.
    OnDemand,
}

// ── SwapPolicy ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SwapPolicy {
    #[default]
    Disallowed,
    Allowed,
}

// ── PackageThermalThresholds ─────────────────────────────────────────────────

/// Per-package overrides for thermal thresholds (°C).  All fields are optional;
/// when absent the global runtime thresholds apply.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct PackageThermalThresholds {
    pub soft_c: Option<f64>,
    pub hard_c: Option<f64>,
    pub recovery_c: Option<f64>,
}

// ── MetricsToggles ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct MetricsToggles {
    #[serde(default = "bool_true")]
    pub latency: bool,
    #[serde(default = "bool_true")]
    pub memory: bool,
    #[serde(default = "bool_true")]
    pub thermal: bool,
    #[serde(default = "bool_true")]
    pub scheduler: bool,
}

impl Default for MetricsToggles {
    fn default() -> Self {
        Self {
            latency: true,
            memory: true,
            thermal: true,
            scheduler: true,
        }
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn default_max_concurrent() -> usize {
    1
}

fn default_cpu_threads() -> usize {
    1
}

fn default_on_demand_queue_capacity() -> usize {
    64
}

fn bool_true() -> bool {
    true
}
