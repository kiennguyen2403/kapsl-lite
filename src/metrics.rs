use std::collections::{BTreeMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const MAX_LATENCY_SAMPLES: usize = 512;
const MAX_INFERENCE_RESULTS: usize = 64;
const MAX_LOG_ENTRIES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventLevel {
    Normal,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelRunState {
    Idle,
    Queued,
    Running,
    Paused,
}

impl ModelRunState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Paused => "paused",
        }
    }
}

/// Result of a single real inference call, reported by any backend.
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub timestamp_ms: u64,
    pub package_name: String,
    pub latency_ms: u64,
    pub success: bool,
    pub output_summary: String,
}

#[derive(Debug, Clone)]
pub struct SchedulerLogEntry {
    pub timestamp_ms: u64,
    pub level: EventLevel,
    pub message: String,
}

#[derive(Debug, Clone)]
struct RuntimeStatus {
    label: String,
    level: EventLevel,
}

#[derive(Debug, Clone)]
struct ModelRuntimeState {
    state: ModelRunState,
    queued: u64,
    active_workers: u64,
    emergency_worker_parked_total: u64,
    emergency_backend_unloaded_total: u64,
    emergency_serialized_load_total: u64,
}

impl ModelRuntimeState {
    fn new(initial_state: ModelRunState) -> Self {
        Self {
            state: initial_state,
            queued: 0,
            active_workers: 0,
            emergency_worker_parked_total: 0,
            emergency_backend_unloaded_total: 0,
            emergency_serialized_load_total: 0,
        }
    }

    fn recompute_state(&mut self) {
        if self.state == ModelRunState::Paused {
            return;
        }
        self.state = if self.active_workers > 0 {
            ModelRunState::Running
        } else if self.queued > 0 {
            ModelRunState::Queued
        } else {
            ModelRunState::Idle
        };
    }
}

#[derive(Debug, Clone)]
pub struct ModelStateSnapshot {
    pub package_name: String,
    pub state: ModelRunState,
    pub queued: u64,
    pub active_workers: u64,
    pub emergency_worker_parked_total: u64,
    pub emergency_backend_unloaded_total: u64,
    pub emergency_serialized_load_total: u64,
}

#[derive(Debug)]
pub struct RuntimeMetrics {
    started_at: Instant,
    inferences_total: AtomicU64,
    success_total: AtomicU64,
    failure_total: AtomicU64,
    active_jobs: AtomicU64,
    queue_depth: AtomicU64,
    queue_peak: AtomicU64,
    cpu_milli_percent: AtomicU64,
    memory_rss_mib: AtomicU64,
    memory_budget_used_mib: AtomicU64,
    temperature_milli_c: AtomicU64,
    memory_limit_mib: AtomicU64,
    package_count: AtomicU64,
    latencies_ms: Mutex<VecDeque<u64>>,
    inference_results: Mutex<VecDeque<InferenceResult>>,
    inference_result_sink: Option<Mutex<File>>,
    scheduler_log: Mutex<VecDeque<SchedulerLogEntry>>,
    status: Mutex<RuntimeStatus>,
    model_states: Mutex<BTreeMap<String, ModelRuntimeState>>,
    memory_emergency_active: AtomicBool,
    emergency_worker_parked_total: AtomicU64,
    emergency_backend_unloaded_total: AtomicU64,
    emergency_serialized_load_total: AtomicU64,
}

#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub uptime_secs: u64,
    pub inferences_total: u64,
    pub success_total: u64,
    pub failure_total: u64,
    pub active_jobs: u64,
    pub queue_depth: u64,
    pub queue_peak: u64,
    pub avg_latency_ms: u64,
    pub p95_latency_ms: u64,
    pub fps: f64,
    pub cpu_percent: f64,
    pub memory_rss_mib: u64,
    pub memory_budget_used_mib: u64,
    pub memory_limit_mib: u64,
    pub temperature_c: f64,
    pub package_count: u64,
    pub status_label: String,
    pub status_level: EventLevel,
    pub running_models: u64,
    pub queued_models: u64,
    pub paused_models: u64,
    pub idle_models: u64,
    pub memory_emergency_active: bool,
    pub emergency_worker_parked_total: u64,
    pub emergency_backend_unloaded_total: u64,
    pub emergency_serialized_load_total: u64,
}

impl RuntimeMetrics {
    pub fn new(package_count: u64, memory_limit_mib: u64) -> Self {
        Self {
            started_at: Instant::now(),
            inferences_total: AtomicU64::new(0),
            success_total: AtomicU64::new(0),
            failure_total: AtomicU64::new(0),
            active_jobs: AtomicU64::new(0),
            queue_depth: AtomicU64::new(0),
            queue_peak: AtomicU64::new(0),
            cpu_milli_percent: AtomicU64::new(0),
            memory_rss_mib: AtomicU64::new(0),
            memory_budget_used_mib: AtomicU64::new(0),
            temperature_milli_c: AtomicU64::new(0),
            memory_limit_mib: AtomicU64::new(memory_limit_mib.max(1)),
            package_count: AtomicU64::new(package_count.max(1)),
            latencies_ms: Mutex::new(VecDeque::with_capacity(MAX_LATENCY_SAMPLES)),
            inference_results: Mutex::new(VecDeque::with_capacity(MAX_INFERENCE_RESULTS)),
            inference_result_sink: open_inference_result_sink_from_env(),
            scheduler_log: Mutex::new(VecDeque::with_capacity(MAX_LOG_ENTRIES)),
            status: Mutex::new(RuntimeStatus {
                label: "starting".to_string(),
                level: EventLevel::Normal,
            }),
            model_states: Mutex::new(BTreeMap::new()),
            memory_emergency_active: AtomicBool::new(false),
            emergency_worker_parked_total: AtomicU64::new(0),
            emergency_backend_unloaded_total: AtomicU64::new(0),
            emergency_serialized_load_total: AtomicU64::new(0),
        }
    }

    pub fn register_model<S: Into<String>>(&self, package_name: S, initial_state: ModelRunState) {
        let mut model_states = self
            .model_states
            .lock()
            .expect("model state lock should not be poisoned");
        model_states.insert(package_name.into(), ModelRuntimeState::new(initial_state));
    }

    pub fn set_model_queue_depth(&self, package_name: &str, depth: usize) {
        let mut model_states = self
            .model_states
            .lock()
            .expect("model state lock should not be poisoned");
        let entry = model_states
            .entry(package_name.to_string())
            .or_insert_with(|| ModelRuntimeState::new(ModelRunState::Idle));
        entry.queued = depth as u64;
        entry.recompute_state();
    }

    pub fn mark_model_running(&self, package_name: &str) {
        let mut model_states = self
            .model_states
            .lock()
            .expect("model state lock should not be poisoned");
        let entry = model_states
            .entry(package_name.to_string())
            .or_insert_with(|| ModelRuntimeState::new(ModelRunState::Idle));
        entry.active_workers = entry.active_workers.saturating_add(1);
        entry.recompute_state();
    }

    pub fn mark_model_finished(&self, package_name: &str) {
        let mut model_states = self
            .model_states
            .lock()
            .expect("model state lock should not be poisoned");
        let Some(entry) = model_states.get_mut(package_name) else {
            return;
        };
        entry.active_workers = entry.active_workers.saturating_sub(1);
        entry.recompute_state();
    }

    pub fn set_model_idle(&self, package_name: &str) {
        let mut model_states = self
            .model_states
            .lock()
            .expect("model state lock should not be poisoned");
        let entry = model_states
            .entry(package_name.to_string())
            .or_insert_with(|| ModelRuntimeState::new(ModelRunState::Idle));
        entry.active_workers = 0;
        entry.queued = 0;
        if entry.state != ModelRunState::Paused {
            entry.state = ModelRunState::Idle;
        }
    }

    pub fn set_model_paused(&self, package_name: &str, paused: bool) {
        let mut model_states = self
            .model_states
            .lock()
            .expect("model state lock should not be poisoned");
        let entry = model_states
            .entry(package_name.to_string())
            .or_insert_with(|| ModelRuntimeState::new(ModelRunState::Idle));
        if paused {
            entry.state = ModelRunState::Paused;
        } else {
            if entry.state == ModelRunState::Paused {
                entry.state = ModelRunState::Idle;
            }
            entry.recompute_state();
        }
    }

    pub fn mark_model_emergency_worker_parked(&self, package_name: &str) {
        let mut model_states = self
            .model_states
            .lock()
            .expect("model state lock should not be poisoned");
        let entry = model_states
            .entry(package_name.to_string())
            .or_insert_with(|| ModelRuntimeState::new(ModelRunState::Idle));
        entry.emergency_worker_parked_total = entry.emergency_worker_parked_total.saturating_add(1);
    }

    pub fn mark_model_emergency_backend_unloaded(&self, package_name: &str) {
        let mut model_states = self
            .model_states
            .lock()
            .expect("model state lock should not be poisoned");
        let entry = model_states
            .entry(package_name.to_string())
            .or_insert_with(|| ModelRuntimeState::new(ModelRunState::Idle));
        entry.emergency_backend_unloaded_total =
            entry.emergency_backend_unloaded_total.saturating_add(1);
    }

    pub fn mark_model_emergency_serialized_load(&self, package_name: &str) {
        let mut model_states = self
            .model_states
            .lock()
            .expect("model state lock should not be poisoned");
        let entry = model_states
            .entry(package_name.to_string())
            .or_insert_with(|| ModelRuntimeState::new(ModelRunState::Idle));
        entry.emergency_serialized_load_total =
            entry.emergency_serialized_load_total.saturating_add(1);
    }

    pub fn set_queue_depth(&self, depth: usize) {
        let depth = depth as u64;
        self.queue_depth.store(depth, Ordering::Relaxed);
        self.queue_peak.fetch_max(depth, Ordering::Relaxed);
    }

    pub fn mark_job_started(&self) {
        self.active_jobs.fetch_add(1, Ordering::Relaxed);
    }

    pub fn mark_job_finished(&self, end_to_end_latency_ms: u64, success: bool) {
        self.inferences_total.fetch_add(1, Ordering::Relaxed);
        if success {
            self.success_total.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failure_total.fetch_add(1, Ordering::Relaxed);
        }

        let _ = self
            .active_jobs
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_sub(1))
            });

        let mut latencies = self
            .latencies_ms
            .lock()
            .expect("latency sample lock should not be poisoned");
        if latencies.len() == MAX_LATENCY_SAMPLES {
            latencies.pop_front();
        }
        latencies.push_back(end_to_end_latency_ms);
    }

    pub fn set_system_stats(&self, cpu_percent: f64, memory_rss_mib: u64) {
        let cpu_milli_percent = (cpu_percent.max(0.0) * 1000.0).round() as u64;
        self.cpu_milli_percent
            .store(cpu_milli_percent, Ordering::Relaxed);
        self.memory_rss_mib.store(memory_rss_mib, Ordering::Relaxed);
    }

    pub fn set_memory_budget_used_mib(&self, memory_budget_used_mib: u64) {
        self.memory_budget_used_mib
            .store(memory_budget_used_mib, Ordering::Relaxed);
    }

    /// Fast-path headroom used by trigger admission checks.
    pub fn available_memory_mib(&self) -> u64 {
        let memory_limit_mib = self.memory_limit_mib.load(Ordering::Relaxed).max(1);
        let memory_rss_mib = self.memory_rss_mib.load(Ordering::Relaxed);
        memory_limit_mib.saturating_sub(memory_rss_mib)
    }

    pub fn set_temperature_c(&self, temperature_c: f64) {
        let temperature_milli_c = (temperature_c.max(0.0) * 1000.0).round() as u64;
        self.temperature_milli_c
            .store(temperature_milli_c, Ordering::Relaxed);
    }

    pub fn set_status<S: Into<String>>(&self, label: S, level: EventLevel) {
        let mut status = self
            .status
            .lock()
            .expect("status lock should not be poisoned");
        status.label = label.into();
        status.level = level;
    }

    pub fn set_memory_emergency_active(&self, active: bool) {
        self.memory_emergency_active
            .store(active, Ordering::Relaxed);
    }

    pub fn mark_emergency_worker_parked(&self) {
        self.emergency_worker_parked_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn mark_emergency_backend_unloaded(&self) {
        self.emergency_backend_unloaded_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn mark_emergency_serialized_load(&self) {
        self.emergency_serialized_load_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn push_scheduler_log<S: Into<String>>(&self, level: EventLevel, message: S) {
        let mut logs = self
            .scheduler_log
            .lock()
            .expect("scheduler log lock should not be poisoned");
        if logs.len() == MAX_LOG_ENTRIES {
            logs.pop_front();
        }

        logs.push_back(SchedulerLogEntry {
            timestamp_ms: unix_time_millis(),
            level,
            message: message.into(),
        });
    }

    /// Records a real inference result from any backend.
    pub fn push_inference_result(&self, result: InferenceResult) {
        if let Some(sink) = &self.inference_result_sink
            && let Ok(mut file) = sink.lock()
        {
            let line = serde_json::json!({
                "timestamp_ms": result.timestamp_ms,
                "package_name": result.package_name,
                "latency_ms": result.latency_ms,
                "success": result.success,
                "output_summary": result.output_summary,
            });
            let _ = writeln!(file, "{}", line);
            let _ = file.flush();
        }

        let mut feed = self
            .inference_results
            .lock()
            .expect("inference results lock should not be poisoned");
        if feed.len() == MAX_INFERENCE_RESULTS {
            feed.pop_front();
        }
        feed.push_back(result);
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        let uptime = self.started_at.elapsed();
        let uptime_secs = uptime.as_secs().max(1);

        let inferences_total = self.inferences_total.load(Ordering::Relaxed);
        let success_total = self.success_total.load(Ordering::Relaxed);
        let failure_total = self.failure_total.load(Ordering::Relaxed);
        let active_jobs = self.active_jobs.load(Ordering::Relaxed);
        let queue_depth = self.queue_depth.load(Ordering::Relaxed);
        let queue_peak = self.queue_peak.load(Ordering::Relaxed);
        let cpu_percent = self.cpu_milli_percent.load(Ordering::Relaxed) as f64 / 1000.0;
        let memory_rss_mib = self.memory_rss_mib.load(Ordering::Relaxed);
        let memory_budget_used_mib = self.memory_budget_used_mib.load(Ordering::Relaxed);
        let memory_limit_mib = self.memory_limit_mib.load(Ordering::Relaxed).max(1);
        let temperature_c = self.temperature_milli_c.load(Ordering::Relaxed) as f64 / 1000.0;
        let package_count = self.package_count.load(Ordering::Relaxed).max(1);
        let memory_emergency_active = self.memory_emergency_active.load(Ordering::Relaxed);
        let emergency_worker_parked_total =
            self.emergency_worker_parked_total.load(Ordering::Relaxed);
        let emergency_backend_unloaded_total = self
            .emergency_backend_unloaded_total
            .load(Ordering::Relaxed);
        let emergency_serialized_load_total =
            self.emergency_serialized_load_total.load(Ordering::Relaxed);

        let (status_label, status_level) = {
            let status = self
                .status
                .lock()
                .expect("status lock should not be poisoned");
            (status.label.clone(), status.level)
        };

        let (running_models, queued_models, paused_models, idle_models) = {
            let model_states = self
                .model_states
                .lock()
                .expect("model state lock should not be poisoned");
            let mut running = 0u64;
            let mut queued = 0u64;
            let mut paused = 0u64;
            let mut idle = 0u64;
            for state in model_states.values() {
                match state.state {
                    ModelRunState::Running => running += 1,
                    ModelRunState::Queued => queued += 1,
                    ModelRunState::Paused => paused += 1,
                    ModelRunState::Idle => idle += 1,
                }
            }
            (running, queued, paused, idle)
        };

        let mut sample_copy = {
            let latencies = self
                .latencies_ms
                .lock()
                .expect("latency sample lock should not be poisoned");
            latencies.iter().copied().collect::<Vec<_>>()
        };

        sample_copy.sort_unstable();

        let avg_latency_ms = if sample_copy.is_empty() {
            0
        } else {
            let total: u64 = sample_copy.iter().sum();
            total / sample_copy.len() as u64
        };

        let p95_latency_ms = percentile(&sample_copy, 95.0);

        MetricsSnapshot {
            uptime_secs,
            inferences_total,
            success_total,
            failure_total,
            active_jobs,
            queue_depth,
            queue_peak,
            avg_latency_ms,
            p95_latency_ms,
            fps: inferences_total as f64 / uptime_secs as f64,
            cpu_percent,
            memory_rss_mib,
            memory_budget_used_mib,
            memory_limit_mib,
            temperature_c,
            package_count,
            status_label,
            status_level,
            running_models,
            queued_models,
            paused_models,
            idle_models,
            memory_emergency_active,
            emergency_worker_parked_total,
            emergency_backend_unloaded_total,
            emergency_serialized_load_total,
        }
    }

    /// Returns recent inference results for TUI display (most recent last).
    pub fn inference_results(&self) -> Vec<InferenceResult> {
        let feed = self
            .inference_results
            .lock()
            .expect("inference results lock should not be poisoned");
        feed.iter().cloned().collect()
    }

    pub fn scheduler_logs(&self) -> Vec<SchedulerLogEntry> {
        let logs = self
            .scheduler_log
            .lock()
            .expect("scheduler log lock should not be poisoned");
        logs.iter().cloned().collect()
    }

    pub fn model_states(&self) -> Vec<ModelStateSnapshot> {
        let model_states = self
            .model_states
            .lock()
            .expect("model state lock should not be poisoned");
        model_states
            .iter()
            .map(|(package_name, state)| ModelStateSnapshot {
                package_name: package_name.clone(),
                state: state.state,
                queued: state.queued,
                active_workers: state.active_workers,
                emergency_worker_parked_total: state.emergency_worker_parked_total,
                emergency_backend_unloaded_total: state.emergency_backend_unloaded_total,
                emergency_serialized_load_total: state.emergency_serialized_load_total,
            })
            .collect()
    }
}

fn percentile(sorted: &[u64], pct: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }

    let clamped_pct = pct.clamp(0.0, 100.0);
    let rank = ((clamped_pct / 100.0) * (sorted.len().saturating_sub(1)) as f64).round() as usize;
    sorted[rank]
}

fn unix_time_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

fn open_inference_result_sink_from_env() -> Option<Mutex<File>> {
    let path = std::env::var("KAPSL_LITE_INFERENCE_RESULTS_PATH")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())?;
    let path = PathBuf::from(path);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .ok()?;
    Some(Mutex::new(file))
}

#[cfg(test)]
mod tests {
    use super::{EventLevel, ModelRunState, RuntimeMetrics, percentile};

    #[test]
    fn percentile_works_for_simple_case() {
        let values = vec![10, 20, 30, 40, 50];
        assert_eq!(percentile(&values, 95.0), 50);
        assert_eq!(percentile(&values, 50.0), 30);
        assert_eq!(percentile(&values, 0.0), 10);
    }

    #[test]
    fn queue_peak_tracks_maximum_depth() {
        let metrics = RuntimeMetrics::new(1, 512);
        metrics.set_queue_depth(2);
        metrics.set_queue_depth(9);
        metrics.set_queue_depth(5);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.queue_peak, 9);
        assert_eq!(snapshot.queue_depth, 5);
    }

    #[test]
    fn status_roundtrip_is_stable() {
        let metrics = RuntimeMetrics::new(1, 512);
        metrics.set_status("restored", EventLevel::Normal);
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.status_label, "restored");
        assert_eq!(snapshot.status_level, EventLevel::Normal);
    }

    #[test]
    fn inference_result_roundtrip() {
        use super::InferenceResult;
        let metrics = RuntimeMetrics::new(1, 512);
        metrics.push_inference_result(InferenceResult {
            timestamp_ms: 1,
            package_name: "test.pkg".to_string(),
            latency_ms: 42,
            success: true,
            output_summary: "detected: cat".to_string(),
        });
        let results = metrics.inference_results();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].package_name, "test.pkg");
        assert_eq!(results[0].latency_ms, 42);
    }

    #[test]
    fn model_state_transitions_roundtrip() {
        let metrics = RuntimeMetrics::new(1, 512);
        metrics.register_model("reasoning.edge", ModelRunState::Idle);
        metrics.set_model_queue_depth("reasoning.edge", 2);
        metrics.mark_model_running("reasoning.edge");

        let states = metrics.model_states();
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].state, ModelRunState::Running);
        assert_eq!(states[0].queued, 2);
        assert_eq!(states[0].active_workers, 1);

        metrics.mark_model_finished("reasoning.edge");
        metrics.set_model_queue_depth("reasoning.edge", 0);
        metrics.set_model_paused("reasoning.edge", true);

        let states = metrics.model_states();
        assert_eq!(states[0].state, ModelRunState::Paused);
    }
}
