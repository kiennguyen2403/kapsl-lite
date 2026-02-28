use crate::metrics::{
    EventLevel, InferenceResult, MetricsSnapshot, ModelStateSnapshot, RuntimeMetrics,
    SchedulerLogEntry,
};
use crate::trigger::{TriggerBus, TriggerEvent};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, Borders, Gauge, Paragraph};
use std::io::{self, Stdout};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const REFRESH_FRAME: Duration = Duration::from_millis(100);
const COLOR_GREEN: Color = Color::Rgb(80, 200, 120);
const COLOR_AMBER: Color = Color::Rgb(255, 191, 0);
const COLOR_RED: Color = Color::Rgb(255, 99, 71);

pub fn run_dashboard(
    metrics: Arc<RuntimeMetrics>,
    trigger_bus: TriggerBus,
    shutdown: Arc<AtomicBool>,
) -> io::Result<()> {
    let mut terminal = init_terminal()?;
    let manual_target_package = std::env::var("KAPSL_LITE_TRIGGER_TARGET_PACKAGE")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "mistral-llm".to_string());

    let result = run_loop(
        &mut terminal,
        metrics,
        trigger_bus,
        manual_target_package,
        shutdown,
    );
    let cleanup_result = restore_terminal(&mut terminal);

    result.and(cleanup_result)
}

fn init_terminal() -> io::Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    Terminal::new(backend)
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> io::Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()
}

fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    metrics: Arc<RuntimeMetrics>,
    trigger_bus: TriggerBus,
    manual_target_package: String,
    shutdown: Arc<AtomicBool>,
) -> io::Result<()> {
    while !shutdown.load(Ordering::Relaxed) {
        let snapshot = metrics.snapshot();
        let results = metrics.inference_results();
        let logs = metrics.scheduler_logs();
        let model_states = metrics.model_states();

        terminal.draw(|frame| {
            render_dashboard(frame, &snapshot, &results, &logs, &model_states);
        })?;

        if event::poll(REFRESH_FRAME)?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
        {
            match key.code {
                KeyCode::Char('q') | KeyCode::Char('Q') => shutdown.store(true, Ordering::SeqCst),
                KeyCode::Char('m') | KeyCode::Char('M') => {
                    let prompt = format!("manual operator trigger at {}", unix_time_millis());
                    match trigger_bus.publish(TriggerEvent::ManualTrigger {
                        target_package: manual_target_package.clone(),
                        custom_prompt: prompt,
                    }) {
                        Ok(()) => metrics.push_scheduler_log(
                            EventLevel::Normal,
                            format!(
                                "manual trigger queued: target={} prompt=operator request",
                                manual_target_package
                            ),
                        ),
                        Err(error) => metrics.push_scheduler_log(
                            EventLevel::Warning,
                            format!(
                                "manual trigger dropped: target={} reason={}",
                                manual_target_package,
                                error.as_reason()
                            ),
                        ),
                    }
                }
                _ => {}
            }
        }
    }

    Ok(())
}

fn render_dashboard(
    frame: &mut ratatui::Frame,
    snapshot: &MetricsSnapshot,
    results: &[InferenceResult],
    logs: &[SchedulerLogEntry],
    model_states: &[ModelStateSnapshot],
) {
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(1)])
        .split(frame.area());

    let panels = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(34),
            Constraint::Percentage(32),
            Constraint::Percentage(34),
        ])
        .split(root[0]);

    render_package_panel(frame, panels[0], results, model_states);
    render_centre_panel(frame, panels[1], snapshot);
    render_scheduler_panel(frame, panels[2], logs);
    render_footer(frame, root[1]);
}

/// Left panel: shows per-model state and recent inference results.
fn render_package_panel(
    frame: &mut ratatui::Frame,
    area: Rect,
    results: &[InferenceResult],
    model_states: &[ModelStateSnapshot],
) {
    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(8), Constraint::Min(3)])
        .split(area);

    let state_capacity = sections[0].height.saturating_sub(2) as usize;
    let mut state_lines = Vec::with_capacity(state_capacity.max(1));

    if model_states.is_empty() {
        state_lines.push(Line::styled(
            "no models",
            Style::default().fg(Color::DarkGray),
        ));
    } else {
        for state in model_states.iter().take(state_capacity.max(1)) {
            let state_color = match state.state.as_str() {
                "running" => COLOR_GREEN,
                "queued" => COLOR_AMBER,
                "paused" => COLOR_RED,
                _ => Color::Gray,
            };
            state_lines.push(Line::styled(
                format!(
                    "{:<14} {:<7} q={} w={} e={}/{}/{}",
                    truncate_name(&state.package_name, 14),
                    state.state.as_str(),
                    state.queued,
                    state.active_workers,
                    state.emergency_worker_parked_total,
                    state.emergency_backend_unloaded_total,
                    state.emergency_serialized_load_total
                ),
                Style::default().fg(state_color),
            ));
        }
    }

    let state_widget = Paragraph::new(state_lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Models (e=park/unload/load)"),
    );
    frame.render_widget(state_widget, sections[0]);

    let capacity = sections[1].height.saturating_sub(2) as usize;
    let mut lines = Vec::with_capacity(capacity.max(1));

    if results.is_empty() {
        lines.push(Line::styled(
            "waiting for inference...",
            Style::default().fg(Color::DarkGray),
        ));
    } else {
        for result in results.iter().rev().take(capacity.max(1)) {
            let latency_color = if result.latency_ms < 100 {
                COLOR_GREEN
            } else if result.latency_ms < 500 {
                COLOR_AMBER
            } else {
                COLOR_RED
            };

            // Format: "package-name           42ms  stub:pkg"
            let summary = if result.output_summary.len() > 20 {
                format!("{}…", &result.output_summary[..20])
            } else {
                result.output_summary.clone()
            };

            lines.push(Line::styled(
                format!(
                    "{:<20} {:>5}ms  {}",
                    truncate_name(&result.package_name, 20),
                    result.latency_ms,
                    summary
                ),
                Style::default().fg(latency_color),
            ));
        }
    }

    let widget =
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title("Package Feed"));
    frame.render_widget(widget, sections[1]);
}

fn render_centre_panel(frame: &mut ratatui::Frame, area: Rect, snapshot: &MetricsSnapshot) {
    let centre = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(5)])
        .split(area);

    let memory_used_mib = snapshot.memory_budget_used_mib.max(snapshot.memory_rss_mib);
    let memory_ratio =
        (memory_used_mib as f64 / snapshot.memory_limit_mib.max(1) as f64).clamp(0.0, 1.0);

    let memory_level = if memory_ratio >= 0.90 {
        EventLevel::Critical
    } else if memory_ratio >= 0.75 {
        EventLevel::Warning
    } else {
        EventLevel::Normal
    };

    let gauge = Gauge::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Memory Budget"),
        )
        .gauge_style(Style::default().fg(color_for_level(memory_level)))
        .ratio(memory_ratio)
        .label(format!(
            "{} / {} MiB",
            memory_used_mib, snapshot.memory_limit_mib
        ));
    frame.render_widget(gauge, centre[0]);

    let temperature_level = temperature_level(snapshot.temperature_c);
    let emergency_level = if snapshot.memory_emergency_active {
        EventLevel::Critical
    } else {
        EventLevel::Normal
    };
    let lines = vec![
        Line::raw(format!(
            "uptime: {}s  infer: {} (ok={} fail={})",
            snapshot.uptime_secs,
            snapshot.inferences_total,
            snapshot.success_total,
            snapshot.failure_total
        )),
        Line::styled(
            format!(
                "cpu: {:.1}%  avg latency: {}ms",
                snapshot.cpu_percent, snapshot.avg_latency_ms
            ),
            Style::default().fg(COLOR_GREEN),
        ),
        Line::raw(format!(
            "memory: budget={}MiB rss={}MiB",
            snapshot.memory_budget_used_mib, snapshot.memory_rss_mib
        )),
        Line::styled(
            format!("temperature: {:.1} C", snapshot.temperature_c),
            Style::default().fg(color_for_level(temperature_level)),
        ),
        Line::styled(
            format!("fps: {:.2}", snapshot.fps),
            Style::default().fg(COLOR_GREEN),
        ),
        Line::raw(format!("packages: {}", snapshot.package_count)),
        Line::raw(format!(
            "active={} p95={}ms",
            snapshot.active_jobs, snapshot.p95_latency_ms
        )),
        Line::raw(format!(
            "queue: depth={} peak={}",
            snapshot.queue_depth, snapshot.queue_peak
        )),
        Line::raw(format!(
            "models: run={} queued={} paused={} idle={}",
            snapshot.running_models,
            snapshot.queued_models,
            snapshot.paused_models,
            snapshot.idle_models
        )),
        Line::styled(
            format!(
                "emergency: {} parked={} unload={} serialized_load={}",
                if snapshot.memory_emergency_active {
                    "active"
                } else {
                    "inactive"
                },
                snapshot.emergency_worker_parked_total,
                snapshot.emergency_backend_unloaded_total,
                snapshot.emergency_serialized_load_total
            ),
            Style::default().fg(color_for_level(emergency_level)),
        ),
        Line::styled(
            format!("status: {}", snapshot.status_label),
            Style::default().fg(color_for_level(snapshot.status_level)),
        ),
    ];

    let widget =
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title("Runtime Core"));
    frame.render_widget(widget, centre[1]);
}

fn render_scheduler_panel(frame: &mut ratatui::Frame, area: Rect, logs: &[SchedulerLogEntry]) {
    let capacity = area.height.saturating_sub(2) as usize;
    let start = logs.len().saturating_sub(capacity);
    let visible = &logs[start..];

    let mut lines = Vec::with_capacity(capacity.max(1));

    for _ in 0..capacity.saturating_sub(visible.len()) {
        lines.push(Line::raw(""));
    }

    for entry in visible {
        let stamp = format_timestamp(entry.timestamp_ms);
        lines.push(Line::styled(
            format!("{} {}", stamp, entry.message),
            Style::default().fg(color_for_level(entry.level)),
        ));
    }

    if lines.is_empty() {
        lines.push(Line::styled(
            "scheduler idle",
            Style::default().fg(Color::DarkGray),
        ));
    }

    let widget = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Scheduler Log"),
    );
    frame.render_widget(widget, area);
}

fn render_footer(frame: &mut ratatui::Frame, area: Rect) {
    let footer = format!(
        "v{} | target {} | q quit | m manual-trigger",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::ARCH,
    );

    let widget = Paragraph::new(footer).style(Style::default().fg(Color::DarkGray));
    frame.render_widget(widget, area);
}

fn color_for_level(level: EventLevel) -> Color {
    match level {
        EventLevel::Normal => COLOR_GREEN,
        EventLevel::Warning => COLOR_AMBER,
        EventLevel::Critical => COLOR_RED,
    }
}

fn temperature_level(temperature_c: f64) -> EventLevel {
    if temperature_c >= 82.0 {
        EventLevel::Critical
    } else if temperature_c >= 72.0 {
        EventLevel::Warning
    } else {
        EventLevel::Normal
    }
}

fn format_timestamp(unix_ms: u64) -> String {
    let seconds = (unix_ms / 1000) % 86_400;
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    let secs = seconds % 60;
    format!("{:02}:{:02}:{:02}", hours, minutes, secs)
}

fn truncate_name(name: &str, limit: usize) -> String {
    if name.len() <= limit {
        name.to_string()
    } else {
        format!("{}…", &name[..limit.saturating_sub(1)])
    }
}

fn unix_time_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}
