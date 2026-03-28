use std::collections::VecDeque;
use std::io;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use ratatui::prelude::*;
use ratatui::widgets::*;
use ratatui::widgets::canvas::{Canvas, Circle, Points};
use serde::Deserialize;
use tungstenite::{connect, Message};

const WS_URL: &str = "ws://127.0.0.1:7881";
const HISTORY: usize = 120; // ~60s at 2Hz
const PHASE_HISTORY: usize = 60; // phase plot trail length
const POLL_MS: u64 = 500;

// --- Protocol types ---

#[derive(Deserialize, Default, Clone)]
struct HandleInfo {
    name: String,
    entity: String,
    mode: String,
    decay_weight: f64,
    tick_count: u64,
    last_tick_ago: Option<f64>,
}

#[derive(Deserialize)]
struct ListResponse {
    handles: Vec<HandleInfo>,
}

#[derive(Deserialize)]
struct TrajectoryResponse {
    outputs: Vec<f64>,
    h_norms: Vec<Vec<f64>>,
}

#[derive(Deserialize, Clone)]
struct ResonanceResponse {
    name_a: String,
    name_b: String,
    correlation: Option<f64>,
    divergence: Option<f64>,
}

// --- App state ---

struct HandleState {
    info: HandleInfo,
    output_history: VecDeque<f64>,
    // Phase plot: h1_norm (fast) vs h3_norm (slow)
    phase_history: VecDeque<(f64, f64)>,
    h1_norm: f64,
    h2_norm: f64,
    h3_norm: f64,
    prev_ticks: u64,
    tick_rate: f64,
}

impl HandleState {
    fn new(info: HandleInfo) -> Self {
        Self {
            prev_ticks: info.tick_count,
            info,
            output_history: VecDeque::with_capacity(HISTORY),
            phase_history: VecDeque::with_capacity(PHASE_HISTORY),
            h1_norm: 0.0,
            h2_norm: 0.0,
            h3_norm: 0.0,
            tick_rate: 0.0,
        }
    }

    fn entity_color(&self) -> Color {
        match self.info.entity.as_str() {
            "astrid" => Color::Magenta,
            "minime" => Color::Cyan,
            "claude" => Color::Yellow,
            _ => Color::White,
        }
    }
}

// Resonance pair with correlation history
struct ResonancePair {
    name_a: String,
    name_b: String,
    corr_history: VecDeque<f64>,
    latest_div: f64,
}

impl ResonancePair {
    fn new(name_a: String, name_b: String) -> Self {
        Self {
            name_a,
            name_b,
            corr_history: VecDeque::with_capacity(HISTORY),
            latest_div: 0.0,
        }
    }

    fn key(&self) -> (String, String) {
        (self.name_a.clone(), self.name_b.clone())
    }
}

struct App {
    handles: Vec<HandleState>,
    resonance_pairs: Vec<ResonancePair>,
    connected: bool,
    error: Option<String>,
}

impl App {
    fn new() -> Self {
        Self {
            handles: Vec::new(),
            resonance_pairs: Vec::new(),
            connected: false,
            error: None,
        }
    }

    fn find_or_create_pair(&mut self, a: &str, b: &str) -> &mut ResonancePair {
        let idx = self
            .resonance_pairs
            .iter()
            .position(|p| p.name_a == a && p.name_b == b);
        match idx {
            Some(i) => &mut self.resonance_pairs[i],
            None => {
                self.resonance_pairs
                    .push(ResonancePair::new(a.to_string(), b.to_string()));
                self.resonance_pairs.last_mut().unwrap()
            }
        }
    }
}

// --- WebSocket polling ---

type WsConn = tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<std::net::TcpStream>>;

fn send_recv(ws: &mut WsConn, msg: &serde_json::Value) -> Option<serde_json::Value> {
    ws.send(Message::Text(msg.to_string())).ok()?;
    match ws.read() {
        Ok(Message::Text(t)) => serde_json::from_str(&t).ok(),
        _ => None,
    }
}

fn poll(ws: &mut WsConn, app: &mut App) {
    let list_msg = serde_json::json!({"type": "list_handles"});
    let Some(list_val) = send_recv(ws, &list_msg) else {
        return;
    };
    let Ok(list): Result<ListResponse, _> = serde_json::from_value(list_val) else {
        return;
    };

    // Ensure state exists for each handle
    for h in &list.handles {
        if !app.handles.iter().any(|s| s.info.name == h.name) {
            app.handles.push(HandleState::new(h.clone()));
        }
    }
    app.handles
        .retain(|s| list.handles.iter().any(|h| h.name == s.info.name));

    // Update each handle
    for state in &mut app.handles {
        let Some(h) = list.handles.iter().find(|h| h.name == state.info.name) else {
            continue;
        };

        let delta = h.tick_count.saturating_sub(state.prev_ticks);
        state.tick_rate = delta as f64 / (POLL_MS as f64 / 1000.0);
        state.prev_ticks = h.tick_count;
        state.info = h.clone();

        let traj_msg = serde_json::json!({"type": "trajectory", "name": h.name, "last_n": 5});
        if let Some(traj_val) = send_recv(ws, &traj_msg) {
            if let Ok(traj) = serde_json::from_value::<TrajectoryResponse>(traj_val) {
                if let Some(&latest) = traj.outputs.last() {
                    state.output_history.push_back(latest);
                    if state.output_history.len() > HISTORY {
                        state.output_history.pop_front();
                    }
                }
                if let Some(norms) = traj.h_norms.last() {
                    if norms.len() >= 3 {
                        state.h1_norm = norms[0];
                        state.h2_norm = norms[1];
                        state.h3_norm = norms[2];

                        // Record phase point (h1 fast vs h3 slow)
                        state.phase_history.push_back((norms[0], norms[2]));
                        if state.phase_history.len() > PHASE_HISTORY {
                            state.phase_history.pop_front();
                        }
                    }
                }
            }
        }
    }

    // Resonance — update pairs with history
    let names: Vec<String> = app.handles.iter().map(|s| s.info.name.clone()).collect();
    for i in 0..names.len() {
        for j in (i + 1)..names.len() {
            let msg =
                serde_json::json!({"type": "resonance", "name_a": names[i], "name_b": names[j]});
            if let Some(val) = send_recv(ws, &msg) {
                if let Ok(res) = serde_json::from_value::<ResonanceResponse>(val) {
                    let pair = app.find_or_create_pair(&names[i], &names[j]);
                    let corr = res.correlation.unwrap_or(0.0);
                    pair.corr_history.push_back(corr);
                    if pair.corr_history.len() > HISTORY {
                        pair.corr_history.pop_front();
                    }
                    pair.latest_div = res.divergence.unwrap_or(0.0);
                }
            }
        }
    }
}

// --- Drawing helpers ---

fn sparkline_data(history: &VecDeque<f64>, width: usize) -> Vec<u64> {
    if history.is_empty() {
        return vec![0; width];
    }
    let min = history.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = history.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(0.001);

    let skip = history.len().saturating_sub(width);
    history
        .iter()
        .skip(skip)
        .map(|&v| ((v - min) / range * 7.0) as u64)
        .collect()
}

/// Map correlation [-1, 1] to sparkline range [0, 7], center at 3.5
fn corr_sparkline_data(history: &VecDeque<f64>, width: usize) -> Vec<u64> {
    if history.is_empty() {
        return vec![3; width]; // center line
    }
    let skip = history.len().saturating_sub(width);
    history
        .iter()
        .skip(skip)
        .map(|&v| ((v + 1.0) / 2.0 * 7.0).clamp(0.0, 7.0) as u64)
        .collect()
}

fn corr_color(c: f64) -> Color {
    if c > 0.3 {
        Color::Green
    } else if c < -0.3 {
        Color::Red
    } else {
        Color::DarkGray
    }
}

// --- UI ---

fn draw(frame: &mut Frame, app: &App) {
    let area = frame.area();

    if !app.connected {
        let msg = app
            .error
            .as_deref()
            .unwrap_or("connecting to ws://127.0.0.1:7881...");
        let p = Paragraph::new(msg)
            .style(Style::default().fg(Color::Red))
            .block(Block::bordered().title(" reservoir-tui "));
        frame.render_widget(p, area);
        return;
    }

    // Top-level layout: handles+phase on top, resonance on bottom
    let n_handles = app.handles.len();
    let handle_height = 5u16;
    let phase_height = 16u16;
    let resonance_pair_count = app.resonance_pairs.len() as u16;
    let resonance_height = (resonance_pair_count * 4 + 2).min(16);

    let top_bottom = Layout::vertical([
        Constraint::Min(n_handles as u16 * handle_height + phase_height),
        Constraint::Length(resonance_height),
    ])
    .split(area);

    // Top section: handle rows + phase plot
    let mut top_constraints: Vec<Constraint> = app
        .handles
        .iter()
        .map(|_| Constraint::Length(handle_height))
        .collect();
    top_constraints.push(Constraint::Min(phase_height)); // phase plot
    let top_chunks = Layout::vertical(top_constraints).split(top_bottom[0]);

    // --- Handle rows ---
    for (i, state) in app.handles.iter().enumerate() {
        let chunk = top_chunks[i];
        let cols =
            Layout::horizontal([Constraint::Min(30), Constraint::Length(42)]).split(chunk);

        let spark_width = cols[0].width as usize;
        let data = sparkline_data(&state.output_history, spark_width);
        let color = state.entity_color();

        let ago = state
            .info
            .last_tick_ago
            .map(|a| format!("{a:.0}s"))
            .unwrap_or("?".into());
        let title = format!(
            " {} ({}) out={:+.3} {}Hz last={ago} ",
            state.info.name,
            state.info.entity,
            state.output_history.back().copied().unwrap_or(0.0),
            state.tick_rate as u32,
        );

        let spark = Sparkline::default()
            .block(
                Block::bordered()
                    .title(title)
                    .border_style(Style::default().fg(color)),
            )
            .data(&data)
            .style(Style::default().fg(color));
        frame.render_widget(spark, cols[0]);

        // Right side: mode + h-norm gauges
        let norm_max = 20.0_f64;
        let gauge = |v: f64, label: &str, c: Color| -> Gauge {
            let ratio = (v / norm_max).clamp(0.0, 1.0);
            Gauge::default()
                .gauge_style(Style::default().fg(c))
                .ratio(ratio)
                .label(format!("{label} {v:.1}"))
        };

        let right_chunks = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(0),
        ])
        .split(cols[1]);

        let mode_line = format!(
            " mode={:<8} decay={:.3} ticks={}",
            state.info.mode, state.info.decay_weight, state.info.tick_count,
        );
        frame.render_widget(
            Paragraph::new(mode_line).style(Style::default().fg(Color::DarkGray)),
            right_chunks[0],
        );
        frame.render_widget(gauge(state.h1_norm, "h1", Color::Red), right_chunks[1]);
        frame.render_widget(gauge(state.h2_norm, "h2", Color::Green), right_chunks[2]);
        frame.render_widget(gauge(state.h3_norm, "h3", Color::Blue), right_chunks[3]);
    }

    // --- Phase plot: h1 (fast) vs h3 (slow) for all handles ---
    let phase_chunk = top_chunks[n_handles];
    // Collect phase data for the canvas closure (needs 'static)
    struct PhaseData {
        name: String,
        color: Color,
        faded: Color,
        trail: Vec<(f64, f64)>,
        current: Option<(f64, f64)>,
    }
    let phase_data: Vec<PhaseData> = app
        .handles
        .iter()
        .map(|state| {
            let color = state.entity_color();
            let faded = match color {
                Color::Magenta => Color::Indexed(53),
                Color::Cyan => Color::Indexed(30),
                Color::Yellow => Color::Indexed(58),
                _ => Color::Indexed(240),
            };
            let len = state.phase_history.len();
            let trail: Vec<(f64, f64)> = if len > 1 {
                state.phase_history.iter().take(len - 1).copied().collect()
            } else {
                vec![]
            };
            let current = state.phase_history.back().copied();
            PhaseData {
                name: state.info.name.clone(),
                color,
                faded,
                trail,
                current,
            }
        })
        .collect();

    let canvas = Canvas::default()
        .block(
            Block::bordered()
                .title(" phase: h1 (fast) vs h3 (slow) ")
                .border_style(Style::default().fg(Color::DarkGray)),
        )
        .x_bounds([0.0, 20.0])
        .y_bounds([0.0, 20.0])
        .paint(move |ctx| {
            // Crosshairs at center
            ctx.draw(&Points {
                coords: &[
                    (10.0, 0.0), (10.0, 5.0), (10.0, 10.0), (10.0, 15.0), (10.0, 20.0),
                    (0.0, 10.0), (5.0, 10.0), (10.0, 10.0), (15.0, 10.0), (20.0, 10.0),
                ],
                color: Color::Indexed(236),
            });

            for pd in &phase_data {
                if pd.current.is_none() {
                    continue;
                }

                // Trail
                if !pd.trail.is_empty() {
                    ctx.draw(&Points {
                        coords: &pd.trail,
                        color: pd.faded,
                    });
                }

                // Current position
                let (x, y) = pd.current.unwrap();
                ctx.draw(&Circle {
                    x,
                    y,
                    radius: 0.4,
                    color: pd.color,
                });

                // Label
                ctx.print(x + 0.5, y + 0.5, Line::from(
                    Span::styled(pd.name.clone(), Style::default().fg(pd.color)),
                ));
            }
        });
    frame.render_widget(canvas, phase_chunk);

    // --- Resonance: correlation sparklines ---
    let res_chunk = top_bottom[1];

    if app.resonance_pairs.is_empty() {
        let empty = Paragraph::new(" no resonance data yet")
            .block(Block::bordered().title(" resonance "));
        frame.render_widget(empty, res_chunk);
        return;
    }

    // Each pair gets: label line + sparkline (3 rows each)
    let pair_height = 3u16;
    let mut pair_constraints: Vec<Constraint> = app
        .resonance_pairs
        .iter()
        .map(|_| Constraint::Length(pair_height))
        .collect();
    pair_constraints.push(Constraint::Min(0));

    let res_inner = Block::bordered()
        .title(" resonance (correlation over time) ")
        .border_style(Style::default().fg(Color::DarkGray));
    let res_inner_area = res_inner.inner(res_chunk);
    frame.render_widget(res_inner, res_chunk);

    let pair_chunks = Layout::vertical(pair_constraints).split(res_inner_area);

    for (i, pair) in app.resonance_pairs.iter().enumerate() {
        if i >= pair_chunks.len() - 1 {
            break;
        }
        let chunk = pair_chunks[i];

        let latest_corr = pair.corr_history.back().copied().unwrap_or(0.0);
        let color = corr_color(latest_corr);

        // Split into label + sparkline
        let cols =
            Layout::horizontal([Constraint::Length(32), Constraint::Min(20)]).split(chunk);

        // Label
        let label = format!(
            " {:>12} <-> {:<12} {:+.3}",
            pair.name_a, pair.name_b, latest_corr,
        );
        let label_widget = Paragraph::new(label)
            .style(Style::default().fg(color))
            .alignment(Alignment::Left);
        // Center vertically in the 3-row space
        let label_area = Layout::vertical([
            Constraint::Min(0),
            Constraint::Length(1),
            Constraint::Min(0),
        ]).split(cols[0]);
        frame.render_widget(label_widget, label_area[1]);

        // Correlation sparkline: [-1, 1] mapped to [0, 7]
        let spark_width = cols[1].width as usize;
        let data = corr_sparkline_data(&pair.corr_history, spark_width);
        let spark = Sparkline::default()
            .data(&data)
            .style(Style::default().fg(color));
        frame.render_widget(spark, cols[1]);
    }
}

// --- Main ---

fn main() -> io::Result<()> {
    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;

    let mut app = App::new();
    let mut ws_conn: Option<WsConn> = None;
    let mut last_poll = Instant::now() - Duration::from_secs(10);

    loop {
        if last_poll.elapsed() >= Duration::from_millis(POLL_MS) {
            last_poll = Instant::now();

            if ws_conn.is_none() {
                match connect(WS_URL) {
                    Ok((ws, _)) => {
                        ws_conn = Some(ws);
                        app.connected = true;
                        app.error = None;
                    }
                    Err(e) => {
                        app.connected = false;
                        app.error = Some(format!("connect failed: {e}"));
                    }
                }
            }

            if let Some(ref mut ws) = ws_conn {
                poll(ws, &mut app);
            }
        }

        terminal.draw(|frame| draw(frame, &app))?;

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press
                    && (key.code == KeyCode::Char('q') || key.code == KeyCode::Esc)
                {
                    break;
                }
            }
        }
    }

    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}
