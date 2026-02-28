use crossbeam_channel::{Receiver, Sender, TryRecvError, TrySendError, bounded};
use serde_json::{Map, Value};

#[derive(Debug, Clone)]
pub enum NormalizedPayload {
    Bytes(Vec<u8>),
    Json(Value),
    Empty,
}

impl NormalizedPayload {
    pub fn size_hint_bytes(&self) -> usize {
        match self {
            Self::Bytes(bytes) => bytes.len(),
            Self::Json(value) => json_size_hint_bytes(value),
            Self::Empty => 0,
        }
    }
}

fn json_size_hint_bytes(value: &Value) -> usize {
    match value {
        Value::Null => 4,
        Value::Bool(true) => 4,
        Value::Bool(false) => 5,
        Value::Number(number) => number.to_string().len(),
        Value::String(text) => text.len() + 2,
        Value::Array(items) => {
            if items.is_empty() {
                2
            } else {
                items.iter().map(json_size_hint_bytes).sum::<usize>()
                    + items.len().saturating_sub(1)
                    + 2
            }
        }
        Value::Object(map) => {
            if map.is_empty() {
                2
            } else {
                map.iter()
                    .map(|(key, item)| key.len() + 2 + 1 + json_size_hint_bytes(item))
                    .sum::<usize>()
                    + map.len().saturating_sub(1)
                    + 2
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct NormalizedInputEvent {
    pub timestamp_ms: u64,
    pub source_id: String,
    pub payload: NormalizedPayload,
    pub metadata: Map<String, Value>,
}

#[derive(Debug, Clone)]
pub enum TriggerEvent {
    DetectionFired {
        label: String,
        confidence: f64,
        source_package: String,
        target_package: String,
    },
    ManualTrigger {
        target_package: String,
        custom_prompt: String,
    },
    ExternalInput {
        target_package: String,
        event: NormalizedInputEvent,
    },
}

impl TriggerEvent {
    pub fn target_package(&self) -> &str {
        match self {
            Self::DetectionFired { target_package, .. } => target_package,
            Self::ManualTrigger { target_package, .. } => target_package,
            Self::ExternalInput { target_package, .. } => target_package,
        }
    }

    pub fn priority_label(&self) -> &'static str {
        match self {
            Self::DetectionFired { .. } => "low",
            Self::ManualTrigger { .. } => "normal",
            Self::ExternalInput { .. } => "normal",
        }
    }

    pub fn is_preemptible(&self) -> bool {
        matches!(
            self,
            Self::DetectionFired { .. } | Self::ManualTrigger { .. } | Self::ExternalInput { .. }
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackpressurePolicy {
    DropOldest,
    LatestOnly,
    Block,
}

#[derive(Debug, Clone)]
pub struct TriggerBus {
    tx: Sender<TriggerEvent>,
    drop_rx: Receiver<TriggerEvent>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerPublishError {
    Full,
    Disconnected,
}

impl TriggerPublishError {
    pub fn as_reason(self) -> &'static str {
        match self {
            Self::Full => "event bus full",
            Self::Disconnected => "event bus disconnected",
        }
    }
}

impl TriggerBus {
    pub fn new(capacity: usize) -> (Self, Receiver<TriggerEvent>) {
        let (tx, rx) = bounded(capacity.max(1));
        (
            Self {
                tx,
                drop_rx: rx.clone(),
            },
            rx,
        )
    }

    pub fn publish(&self, event: TriggerEvent) -> Result<(), TriggerPublishError> {
        self.publish_with_policy(event, BackpressurePolicy::LatestOnly)
    }

    pub fn publish_with_policy(
        &self,
        event: TriggerEvent,
        policy: BackpressurePolicy,
    ) -> Result<(), TriggerPublishError> {
        match policy {
            BackpressurePolicy::LatestOnly => match self.tx.try_send(event) {
                Ok(()) => Ok(()),
                Err(TrySendError::Full(_)) => Err(TriggerPublishError::Full),
                Err(TrySendError::Disconnected(_)) => Err(TriggerPublishError::Disconnected),
            },
            BackpressurePolicy::Block => self
                .tx
                .send(event)
                .map_err(|_| TriggerPublishError::Disconnected),
            BackpressurePolicy::DropOldest => match self.tx.try_send(event) {
                Ok(()) => Ok(()),
                Err(TrySendError::Disconnected(_)) => Err(TriggerPublishError::Disconnected),
                Err(TrySendError::Full(event)) => {
                    match self.drop_rx.try_recv() {
                        Ok(_) | Err(TryRecvError::Empty) => {}
                        Err(TryRecvError::Disconnected) => {
                            return Err(TriggerPublishError::Disconnected);
                        }
                    }
                    match self.tx.try_send(event) {
                        Ok(()) => Ok(()),
                        Err(TrySendError::Full(_)) => Err(TriggerPublishError::Full),
                        Err(TrySendError::Disconnected(_)) => {
                            Err(TriggerPublishError::Disconnected)
                        }
                    }
                }
            },
        }
    }
}
