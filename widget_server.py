"""HTTP + SSE widget for v2.

Endpoints:
  GET /          single-page HTML dashboard
  GET /snapshot  JSON: { state, pastes, transcript, recent_events, now_accepted_ms }
                 — polled by the dashboard every 500ms for steady-state.
  GET /transcript JSON: server-side merged transcript tail (HQ-preferred).
  GET /events    SSE stream — live deltas for the debug timeline only.

Server-side state tracking from the bus:
  - capture mode: idle / recording / aside (from user.actions)
  - mute toggled (from user.actions)
  - paste log (rolling list of paste.actions)
  - recent_events ring buffer for the debug timeline

The transcript is fetched from the TranscriptStreamManager directly so
the merge logic lives in one place. Same pattern as v1.
"""

from __future__ import annotations

import json
import logging
import queue
import sys
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from .clock import AcceptedClock
from .event_bus import EventBus
from .runtime_config import (
    RuntimeConfig, apply_dict_to_config, save_runtime_config, to_dict,
)
from .transcript_stream_manager import TranscriptStreamManager
from .types import Event


logger = logging.getLogger(__name__)


# Fields on FasterWhisperConfig that, when changed, require a model reload.
# device is auto-resolved post-load and isn't user-settable from the UI;
# changes to it would require a restart, so we deliberately ignore it.
_FW_RELOAD_FIELDS = ("model", "compute", "beam_size", "condition_on_previous_text")


def _snapshot_fw(fw) -> dict:
    return {k: getattr(fw, k, None) for k in _FW_RELOAD_FIELDS}


def _fw_changed(prev: dict, new_fw) -> bool:
    return any(prev.get(k) != getattr(new_fw, k, None) for k in _FW_RELOAD_FIELDS)


def _print_url_banner(label: str, url: str, extras: list[tuple[str, str]] | None = None) -> None:
    """Print a high-visibility startup banner to stderr.

    The widget URL is rendered as an OSC 8 hyperlink + ANSI bold/underline/cyan
    when stderr is a TTY, so it stands out from log spam and is one-click-openable
    in modern terminals (iTerm2, Terminal.app, VS Code, Ghostty, …).

    ``extras`` is a list of (label, value) pairs printed under the URL — used to
    surface session dir, hotkeys, models, etc. on startup.
    """
    stream = sys.stderr
    is_tty = False
    try:
        is_tty = stream.isatty()
    except Exception:
        is_tty = False
    extras = extras or []

    # Determine box width from longest content line.
    plain_lines = [f"{label}: {url}"]
    for k, v in extras:
        plain_lines.append(f"{k}: {v}")
    width = max(len(s) for s in plain_lines) + 4

    if is_tty:
        link = f"\x1b]8;;{url}\x1b\\\x1b[1;4;36m{url}\x1b[0m\x1b]8;;\x1b\\"
        bar = "\x1b[1;36m" + "─" * width + "\x1b[0m"
        bold_label = f"\x1b[1m{label}\x1b[0m"
        url_line = f"  {bold_label}: {link}"
        extra_lines = [
            f"  \x1b[1m{k}\x1b[0m: {v}" for k, v in extras
        ]
    else:
        bar = "─" * width
        url_line = f"  {label}: {url}"
        extra_lines = [f"  {k}: {v}" for k, v in extras]

    body = "\n".join([bar, url_line, *extra_lines, bar])
    msg = f"\n{body}\n"
    try:
        stream.write(msg)
        stream.flush()
    except Exception:
        pass


_INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>voice_dictation</title>
<style>
:root {
  --bg: #15171c;
  --panel: #1f222a;
  --panel2: #2a2e38;
  --fg: #e6e6ea;
  --muted: #8a8d97;
  --fast: #5fa8d3;
  --hq: #b39ddb;
  --hq-text: #cfb8f3;
  --recording: #d64545;
  --aside: #c792ea;
  --muted-color: #ff9800;
  --good: #6abf4b;
  --bad: #ef5350;
  --marker: #f0a04b;
  --debug-flag: #ff5252;
  --paste: #aed581;
  --press-start: #6abf4b;
  --press-end: #d64545;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 13px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
}
header {
  display: flex; align-items: center; gap: 16px;
  padding: 8px 14px; border-bottom: 1px solid #2a2e38;
  background: #181b22;
}
header .title { font-weight: 600; font-size: 14px; }
header .meta { color: var(--muted); font-size: 11px; }
.pill {
  display: inline-block; padding: 2px 10px; border-radius: 12px;
  font-size: 11px; font-weight: 600; letter-spacing: 0.02em;
}
.pill.passive    { background: #2a2e38; color: var(--muted); }
.pill.recording  { background: var(--recording); color: #fff; }
.pill.aside      { background: var(--aside); color: #1b1d23; }
.pill.muted      { background: var(--muted-color); color: #1b1d23; }
.pill.marker-1   { background: #2a2e38; color: var(--marker); border: 1px solid var(--marker); }
.pill.marker-2   { background: #2a2e38; color: var(--marker); border: 1px solid var(--marker); }
.pill.none       { background: #2a2e38; color: var(--muted); }

main { padding: 10px 12px 60px; max-width: 1400px; margin: 0 auto; }
section {
  background: var(--panel); border-radius: 6px; margin-bottom: 10px;
  overflow: hidden;
}
section > h2 {
  margin: 0; padding: 8px 12px; font-size: 13px; font-weight: 500;
  color: var(--muted); cursor: pointer; user-select: none;
  background: #1a1d24; border-bottom: 1px solid #2a2e38;
}
section.collapsed .body { display: none; }
section .body { padding: 8px 12px; }
.chev::before { content: "▾"; display: inline-block; width: 14px; }
section.collapsed .chev::before { content: "▸"; }

/* MUTED border highlight on body when capture is muted. */
body.muted-state {
  box-shadow: inset 0 0 0 3px var(--muted-color);
}

/* Paste log */
#pastes-body .empty { color: var(--muted); padding: 4px 0; }
.paste-row {
  display: flex; gap: 8px; padding: 3px 0; border-bottom: 1px solid #242832;
  font-family: monospace; font-size: 12px;
  align-items: baseline;
  /* one line per paste; clip the tail rather than wrap */
  overflow: hidden; white-space: nowrap;
}
.paste-row:last-child { border: 0; }
.paste-row .t {
  color: var(--muted); flex: 0 0 130px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.paste-row .text {
  color: var(--paste); flex: 1 1 auto; min-width: 0;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.paste-row .copy {
  flex: 0 0 auto; padding: 1px 8px; font-size: 11px;
  background: var(--panel2); color: var(--muted);
  border: 0; border-radius: 3px; cursor: pointer;
}
.paste-row .copy:hover { background: #3a3f4d; color: var(--fg); }
.paste-row .copy.copied { background: var(--good); color: #fff; }

/* Transcript */
.section-toolbar {
  display: flex; justify-content: flex-end; padding: 0 0 6px;
}
.section-toolbar button {
  font-size: 11px; padding: 2px 10px; border: 0; border-radius: 3px;
  background: var(--panel2); color: var(--muted); cursor: pointer;
}
.section-toolbar button:hover { background: #3a3f4d; color: var(--fg); }
.section-toolbar button.copied { background: var(--good); color: #fff; }
#transcript {
  font-family: -apple-system, sans-serif;
  font-size: 14px; line-height: 1.6;
  max-height: 280px; overflow-y: auto;
  white-space: pre-wrap; word-break: break-word;
  color: var(--fg);
}
#transcript .empty { color: var(--muted); }
.tx-pill {
  display: inline-block; padding: 0 5px; margin: 0 3px;
  border-radius: 3px; font-size: 11px;
  font-family: monospace; font-weight: 600;
  vertical-align: middle;
}
.tx-pill.rec-start { background: var(--recording); color: #fff; }
.tx-pill.rec-end   { background: var(--recording); color: #fff; }
.tx-pill.aside     { background: var(--aside); color: #1b1d23; }
.tx-pill.cancel    { background: #555; color: #ccc; }
.tx-pill.marker    { background: var(--marker); color: #1b1d23; }
.tx-pill.dbg       { background: var(--debug-flag); color: #fff; }
.tx-pill.hq        { background: var(--hq-text); color: #1b1d23; }

/* Recent events */
.events-controls {
  display: flex; gap: 4px; flex-wrap: wrap;
  padding-bottom: 6px; border-bottom: 1px solid #242832; margin-bottom: 6px;
}
.chip {
  font-size: 11px; padding: 2px 8px; border-radius: 10px;
  background: #2a2e38; color: var(--muted);
  cursor: pointer; user-select: none;
}
.chip.on { background: #34384a; color: var(--fg); }
#events-table {
  font-family: monospace; font-size: 11px;
  max-height: 260px; overflow-y: auto;
}
.ev-row { display: flex; gap: 6px; padding: 1px 0; }
.ev-row .t { color: var(--muted); flex: 0 0 70px; }
.ev-row .kind {
  flex: 0 0 110px; font-weight: 600;
}
.ev-row .body { color: var(--fg); white-space: pre; overflow: hidden;
  text-overflow: ellipsis; }
/* Event colorization by kind */
.ev-row[data-kind="segment"] .kind  { color: var(--fast); }
.ev-row[data-kind="segment-hq"] .kind { color: var(--hq-text); }
.ev-row[data-kind="press"] .kind    { color: var(--press-end); }
.ev-row[data-kind="action"] .kind   { color: var(--good); }
.ev-row[data-kind="paste"] .kind    { color: var(--paste); }
.ev-row[data-kind="marker"] .kind   { color: var(--marker); }
.ev-row[data-kind="mute"] .kind     { color: var(--muted-color); }
.ev-row[data-kind="debug_flag"] .kind { color: var(--debug-flag); }
.ev-row[data-kind="silence"] .kind  { color: var(--muted); }

/* Debug timeline */
#timeline-legend {
  display: flex; gap: 14px; flex-wrap: wrap;
  font-size: 11px; color: var(--muted); margin-bottom: 6px;
  align-items: center;
}
#timeline-legend .swatch {
  display: inline-block; width: 12px; height: 10px;
  border-radius: 2px; vertical-align: middle; margin-right: 3px;
}
#timeline-legend .stick {
  display: inline-block; width: 2px; height: 12px; vertical-align: middle;
  margin-right: 3px;
}
.timeline-controls {
  display: flex; gap: 10px; flex-wrap: wrap;
  padding-bottom: 6px; margin-bottom: 6px; font-size: 11px;
  color: var(--muted); align-items: center;
}
.timeline-row {
  display: flex; gap: 0; align-items: stretch;
  background: var(--panel2); border-radius: 4px; overflow: hidden;
}
#timeline-labels {
  position: relative; width: 64px; flex: 0 0 64px;
  border-right: 1px solid #1a1d24;
  background: #1a1d24;
}
#timeline-labels .tl-label {
  position: absolute; left: 4px;
  font-size: 10px; color: var(--muted);
  line-height: 1; padding: 1px 0;
  white-space: nowrap;
}
#timeline-svg {
  flex: 1 1 auto; height: 160px;
  cursor: ew-resize; user-select: none;
  overflow: hidden;
}

/* Stats */
#stats-body {
  display: grid; gap: 8px;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
}
.stat-card {
  background: var(--panel2); border-radius: 4px; padding: 8px 10px;
}
.stat-card .stat-title {
  font-size: 11px; color: var(--fast); font-family: monospace;
  margin-bottom: 4px; text-transform: lowercase;
}
.stat-card .stat-row {
  display: flex; justify-content: space-between; gap: 8px;
  font-family: monospace; font-size: 11px; padding: 1px 0;
  color: var(--fg);
}
.stat-card .stat-row .k { color: var(--muted); }
.stat-card .stat-row .v.warn { color: var(--marker); }
.stat-card .stat-row .v.bad  { color: var(--bad); }

/* Header */
header .pill-row { display: flex; gap: 10px; flex-wrap: wrap; }
header .pill-row > div { display: flex; gap: 6px; align-items: center; }
header .pill-row label { color: var(--muted); font-size: 11px; }

/* Settings */
#settings-gear {
  position: fixed; right: 12px; bottom: 12px;
  width: 36px; height: 36px; border-radius: 50%;
  background: var(--panel2); color: var(--fg); border: 0;
  cursor: pointer; font-size: 18px; z-index: 100;
}
#settings-gear:hover { background: #3a3f4d; }
#settings-overlay {
  position: fixed; inset: 0; background: rgba(0,0,0,0.5);
  display: flex; align-items: center; justify-content: center; z-index: 101;
}
#settings-overlay[hidden] { display: none; }
#settings-modal {
  background: var(--panel); border-radius: 8px; padding: 16px;
  width: min(720px, 90vw); max-height: 80vh;
  display: flex; flex-direction: column; gap: 10px;
}
#settings-modal header {
  display: flex; align-items: center; justify-content: space-between;
  border: 0; background: none; padding: 0;
}
#settings-modal h3 { margin: 0; font-size: 15px; }
#settings-modal #settings-close {
  background: none; color: var(--muted); border: 0; font-size: 20px;
  cursor: pointer;
}
#settings-form {
  flex: 1; min-height: 360px; overflow-y: auto;
  background: #0e1015; border: 1px solid #2a2e38; border-radius: 4px;
  padding: 8px;
}
#settings-form .cfg-section {
  margin-bottom: 14px;
  background: #14171c; border-radius: 4px; padding: 8px 10px;
}
#settings-form .cfg-section > h4 {
  margin: 0 0 6px;
  font-size: 12px; color: var(--fast); text-transform: lowercase;
  letter-spacing: 0.04em; cursor: pointer; user-select: none;
}
#settings-form .cfg-section.collapsed > .cfg-body { display: none; }
#settings-form .cfg-section > h4::before {
  content: "▾ "; color: var(--muted);
}
#settings-form .cfg-section.collapsed > h4::before { content: "▸ "; }
#settings-form .cfg-row {
  display: grid;
  grid-template-columns: minmax(150px, 30%) 1fr;
  gap: 8px; align-items: center;
  padding: 3px 0;
}
#settings-form .cfg-row label {
  font-size: 11px; color: var(--fg);
  font-family: monospace;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
#settings-form .cfg-row label.required::after { content: " *"; color: var(--bad); }
#settings-form .cfg-row .cfg-help {
  grid-column: 2 / 3;
  font-size: 10px; color: var(--muted);
  margin: -2px 0 4px;
}
#settings-form input[type=text],
#settings-form input[type=number],
#settings-form select,
#settings-form textarea {
  background: #1a1d24; color: var(--fg);
  border: 1px solid #2a2e38; border-radius: 3px;
  padding: 3px 6px; font-size: 12px;
  font-family: monospace; width: 100%;
}
#settings-form input[type=checkbox] {
  justify-self: start; width: 16px; height: 16px;
}
#settings-form textarea {
  min-height: 70px; resize: vertical;
}
#settings-form .cfg-nested {
  margin-left: 12px; padding-left: 8px;
  border-left: 1px solid #2a2e38;
}
#settings-form .cfg-nested > .cfg-nested-title {
  font-size: 11px; color: var(--muted);
  font-family: monospace; margin: 6px 0 2px;
}
#settings-json-fallback {
  display: none;
  width: 100%; min-height: 200px; resize: vertical;
  font-family: monospace; font-size: 12px;
  background: #0e1015; color: var(--fg);
  border: 1px solid #2a2e38; border-radius: 4px; padding: 8px;
}
#settings-mode-toggle {
  font-size: 11px; color: var(--muted); cursor: pointer;
  background: none; border: 0; padding: 0;
}
#settings-mode-toggle:hover { color: var(--fg); }
#settings-actions { display: flex; gap: 8px; align-items: center; }
#settings-actions button {
  padding: 6px 12px; border-radius: 4px; border: 0;
  background: var(--panel2); color: var(--fg); cursor: pointer;
}
#settings-actions button.primary { background: var(--fast); color: #1b1d23; }
#settings-status { color: var(--muted); font-size: 11px; flex: 1; }
#settings-status.ok { color: var(--good); }
#settings-status.err { color: var(--bad); }
</style>
</head>
<body>
<header>
  <span class="title">🎤 voice_dictation</span>
  <div class="pill-row">
    <div><label>capture:</label> <span id="capture-pill" class="pill passive">passive</span></div>
    <div><label>marker:</label> <span id="marker-pill" class="pill none">none</span></div>
    <div><span id="mute-pill" class="pill muted" style="display:none">MUTED</span></div>
  </div>
  <span class="meta" style="margin-left:auto" id="uptime-meta">uptime: —</span>
  <span class="meta" id="status-meta">connecting…</span>
</header>

<main>
  <section id="pastes-section">
    <h2 onclick="toggleSection('pastes-section')"><span class="chev"></span>Paste log</h2>
    <div class="body" id="pastes-body"><div class="empty">no pastes yet</div></div>
  </section>

  <section id="transcript-section">
    <h2 onclick="toggleSection('transcript-section')"><span class="chev"></span>Transcript (tail)</h2>
    <div class="body">
      <div class="section-toolbar">
        <button id="transcript-copy" title="Copy full transcript tail">copy</button>
      </div>
      <div id="transcript"><div class="empty">no audio yet</div></div>
    </div>
  </section>

  <section id="events-section">
    <h2 onclick="toggleSection('events-section')"><span class="chev"></span>Recent events</h2>
    <div class="body">
      <div class="events-controls" id="events-filters">
        <label style="color:var(--muted)">filter:</label>
        <span class="chip on" data-kind="segment">segment</span>
        <span class="chip on" data-kind="press">press</span>
        <span class="chip on" data-kind="action">action</span>
        <span class="chip on" data-kind="paste">paste</span>
        <span class="chip on" data-kind="marker">marker</span>
        <span class="chip on" data-kind="mute">mute</span>
        <span class="chip on" data-kind="debug_flag">🚩 debug_flag</span>
        <span class="chip" data-kind="silence">silence</span>
      </div>
      <div id="events-table"><div class="empty">no events yet</div></div>
    </div>
  </section>

  <section id="stats-section">
    <h2 onclick="toggleSection('stats-section')"><span class="chev"></span>Stats</h2>
    <div class="body">
      <div id="stats-body"><div class="empty" style="color:var(--muted)">loading…</div></div>
    </div>
  </section>

  <section id="timeline-section">
    <h2 onclick="toggleSection('timeline-section')"><span class="chev"></span>Debug timeline</h2>
    <div class="body">
      <div id="timeline-legend">
        <span><span class="swatch" style="background:var(--fast)"></span>fast segment</span>
        <span><span class="swatch" style="background:var(--hq)"></span>hq segment</span>
        <span><span class="stick" style="background:var(--press-start)"></span>press (open)</span>
        <span><span class="stick" style="background:var(--press-end)"></span>press (end)</span>
        <span><span class="swatch" style="background:var(--marker)"></span>marker</span>
        <span><span class="swatch" style="background:var(--paste)"></span>paste</span>
        <span style="color:var(--debug-flag)">🚩 debug flag</span>
        <span class="meta" id="tl-window-label" style="margin-left:auto"></span>
      </div>
      <div class="timeline-controls">
        <label>window:</label>
        <button onclick="setTlWindow(15)">15s</button>
        <button onclick="setTlWindow(30)">30s</button>
        <button onclick="setTlWindow(60)">60s</button>
        <button onclick="setTlWindow(120)">2m</button>
      </div>
      <div class="timeline-row">
        <div id="timeline-labels">
          <span class="tl-label" style="top: 25px">fast</span>
          <span class="tl-label" style="top: 50px">hq</span>
          <span class="tl-label" style="top: 80px">press</span>
          <span class="tl-label" style="top: 105px">paste</span>
          <span class="tl-label" style="top: 130px">marker</span>
        </div>
        <svg id="timeline-svg" viewBox="0 0 1000 160" preserveAspectRatio="none"></svg>
      </div>
    </div>
  </section>
</main>

<button id="settings-gear" title="Settings (edit local_config.json)" aria-label="Settings">⚙</button>
<div id="settings-overlay" hidden>
  <div id="settings-modal" role="dialog" aria-labelledby="settings-title">
    <header>
      <h3 id="settings-title">Settings — <span style="color:var(--muted);font-weight:normal;font-size:12px">local_config.json</span></h3>
      <button id="settings-close" aria-label="Close">×</button>
    </header>
    <div style="display:flex;justify-content:space-between;align-items:center;color:var(--muted);font-size:11px;margin:0">
      <span>Most fields take effect on next app launch. Unknown keys are ignored.</span>
      <button id="settings-mode-toggle" type="button" title="Toggle structured / raw-JSON edit">⇄ raw JSON</button>
    </div>
    <div id="settings-form"></div>
    <textarea id="settings-json-fallback" spellcheck="false" autocomplete="off"></textarea>
    <div id="settings-actions">
      <span id="settings-status"></span>
      <button id="settings-reload">Reload</button>
      <button id="settings-save" class="primary">Save</button>
    </div>
  </div>
</div>

<script>
let TL_WINDOW_MS = 30_000;
let nowAcceptedMs = 0;
let recentEvents = [];      // for debug timeline + recent-events render
let eventFilters = new Set(["segment","press","action","paste","marker","mute","debug_flag"]);

function fmtMs(ms) {
  if (ms == null) return "—";
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s/60)}:${String(s%60).padStart(2,"0")}.${String(ms%1000).padStart(3,"0")}`;
}

function fmtDur(ms) {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s/60);
  if (m > 0) return `${m}m${s%60}s`;
  return `${s}s`;
}

function toggleSection(id) {
  document.getElementById(id).classList.toggle("collapsed");
}

function setTlWindow(seconds) {
  TL_WINDOW_MS = seconds * 1000;
  renderTimeline();
}

// Map a bus event topic+payload to a normalized kind for filter chips.
function eventKind(e) {
  if (e.topic === "audio.silence") return "silence";
  if (e.topic.startsWith("segments.")) return "segment";
  if (e.topic === "user.presses") return "press";
  if (e.topic === "user.actions") {
    if (e.action === "paste_window_complete") return "paste";
    if (e.action === "marker" || e.action === "marker_opened" ||
        e.action === "marker_closed") return "marker";
    if (e.action === "debug_flag") return "debug_flag";
    if (e.action === "mute_toggle") return "mute";
    return "action";
  }
  if (e.topic === "paste.actions") return "paste";
  return null; // window events etc. — not shown in this view
}

function eventLabel(e) {
  if (e.topic.startsWith("segments.")) {
    const stream = e.topic.split(".")[1];
    return stream + ":" + (e.text || "").slice(0, 80);
  }
  if (e.topic === "user.presses") {
    return `${e.key} idx=${e.press_idx}${e.is_clipboard_end ? " [end]" : ""}`;
  }
  if (e.topic === "user.actions") {
    return e.action + " " + JSON.stringify(e).slice(0, 80);
  }
  if (e.topic === "paste.actions") {
    return "paste #" + e.paste_idx + ": " + (e.text || "").slice(0, 80);
  }
  if (e.topic === "audio.silence") {
    return `boundary @ ${fmtMs(e.boundary_at_accepted_ms)} dur=${e.duration_real_ms}ms`;
  }
  return JSON.stringify(e).slice(0, 120);
}

function ingestEvent(e) {
  recentEvents.push(e);
  if (recentEvents.length > 5000) recentEvents = recentEvents.slice(-5000);
}

function renderRecentEvents() {
  const el = document.getElementById("events-table");
  const cutoff = nowAcceptedMs - 5 * 60_000;
  const visible = [];
  for (let i = recentEvents.length - 1; i >= 0 && visible.length < 200; i--) {
    const e = recentEvents[i];
    if (e.emit_accepted_ms < cutoff) break;
    const kind = eventKind(e);
    if (!kind || !eventFilters.has(kind)) continue;
    visible.push({ ev: e, kind });
  }
  if (!visible.length) {
    el.innerHTML = '<div class="empty">no events match filters</div>';
    return;
  }
  el.innerHTML = visible.map(v =>
    `<div class="ev-row" data-kind="${v.kind}">` +
    `<span class="t">${fmtMs(v.ev.emit_accepted_ms)}</span>` +
    `<span class="kind">${v.kind}</span>` +
    `<span class="body">${escapeHtml(eventLabel(v.ev))}</span>` +
    `</div>`
  ).join("");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => (
    {"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]
  ));
}

function renderTimeline() {
  const svg = document.getElementById("timeline-svg");
  const W = 1000, H = 160;
  const startMs = Math.max(0, nowAcceptedMs - TL_WINDOW_MS);
  // Don't clamp — let the SVG viewport clip naturally so bars slide off
  // the left edge instead of pinning to x=0.
  const x = ms => ((ms - startMs) / TL_WINDOW_MS) * W;
  const lbl = (ms) => `${Math.floor((startMs)/1000)}s – ${Math.floor((startMs+TL_WINDOW_MS)/1000)}s`;
  document.getElementById("tl-window-label").textContent = lbl(nowAcceptedMs);

  // Lanes (y centers).
  const LANES = {
    "segments.fast": 30,
    "segments.hq":   55,
    "user.presses":  85,
    "paste.actions": 110,
    "marker":        135,
  };
  const parts = [];
  // Faint baselines per lane. Labels live in the HTML gutter to the left
  // of the SVG so they don't overlap data shapes.
  for (const y of Object.values(LANES)) {
    parts.push(`<line x1="0" x2="${W}" y1="${y}" y2="${y}" stroke="#2c303b" stroke-width="1"/>`);
  }

  for (const e of recentEvents) {
    if (e.emit_accepted_ms < startMs - 5000) continue;
    if (e.emit_accepted_ms > nowAcceptedMs + 1000) continue;
    if (e.topic === "segments.fast" || e.topic === "segments.hq") {
      const y = LANES[e.topic];
      const x0 = x(e.content_accepted_ms_start);
      const x1 = x(e.content_accepted_ms_end);
      if (x1 < 0 || x0 > W) continue;
      const color = e.topic === "segments.hq" ? "var(--hq)" : "var(--fast)";
      parts.push(
        `<rect x="${x0}" y="${y-7}" width="${Math.max(2,x1-x0)}" height="14" ` +
        `fill="${color}" rx="2"><title>${escapeHtml(e.text||"")}</title></rect>`
      );
    } else if (e.topic === "user.presses") {
      const cx = x(e.content_accepted_ms);
      if (cx < 0 || cx > W) continue;
      const color = e.is_clipboard_end ? "var(--press-end)" : "var(--press-start)";
      parts.push(
        `<rect x="${cx-1}" y="${LANES["user.presses"]-7}" width="3" height="14" ` +
        `fill="${color}"><title>${escapeHtml(e.key + " " + (e.is_clipboard_end?"end":"start"))}</title></rect>`
      );
    } else if (e.topic === "paste.actions") {
      const x0 = x(e.start_accepted_ms);
      const x1 = x(e.end_accepted_ms);
      if (x1 < 0 || x0 > W) continue;
      parts.push(
        `<rect x="${x0}" y="${LANES["paste.actions"]-7}" width="${Math.max(2,x1-x0)}" height="14" ` +
        `fill="var(--paste)" rx="2"><title>${escapeHtml(e.text||"")}</title></rect>`
      );
    } else if (e.topic === "user.actions" && e.action === "marker") {
      const cx = x(e.accepted_ms || e.emit_accepted_ms);
      if (cx < 0 || cx > W) continue;
      parts.push(
        `<circle cx="${cx}" cy="${LANES["marker"]}" r="5" fill="var(--marker)">` +
        `<title>marker ${e.key}</title></circle>`
      );
    } else if (e.topic === "user.actions" && e.action === "debug_flag") {
      const cx = x(e.accepted_ms || e.emit_accepted_ms);
      if (cx < 0 || cx > W) continue;
      parts.push(
        `<text x="${cx-6}" y="${LANES["marker"]+5}" fill="var(--debug-flag)" font-size="14">🚩</text>`
      );
    }
  }
  svg.innerHTML = parts.join("");
}

function renderState(state) {
  const cap = document.getElementById("capture-pill");
  cap.className = "pill " + (state.capture || "passive");
  let txt = state.capture || "passive";
  if (state.capture === "recording" && state.recording_seconds != null) {
    txt = `● recording ${state.recording_seconds.toFixed(1)}s`;
  } else if (state.capture === "aside" && state.aside_seconds != null) {
    txt = `○ aside ${state.aside_seconds.toFixed(1)}s`;
  }
  cap.textContent = txt;

  const mute = document.getElementById("mute-pill");
  if (state.muted) {
    mute.style.display = "inline-block";
    document.body.classList.add("muted-state");
  } else {
    mute.style.display = "none";
    document.body.classList.remove("muted-state");
  }

  const markEl = document.getElementById("marker-pill");
  const open = state.open_markers || [];
  if (open.length > 0) {
    const parts = open.map(m =>
      `${m.key} (${(m.age_ms/1000).toFixed(1)}s)`
    );
    markEl.className = "pill marker-" + open[0].key;
    markEl.textContent = parts.join(", ");
  } else {
    markEl.className = "pill none";
    markEl.textContent = "—";
  }

  document.getElementById("uptime-meta").textContent =
    "uptime: " + fmtDur(state.now_accepted_ms || 0) + " (accepted)";
  document.getElementById("status-meta").textContent =
    `events: ${recentEvents.length}`;
}

// Keyed by paste_idx so the copy button can look up the full (unescaped)
// text without parsing the rendered DOM.
const pasteTexts = new Map();

function renderPasteLog(pastes) {
  const el = document.getElementById("pastes-body");
  pasteTexts.clear();
  if (!pastes || !pastes.length) {
    el.innerHTML = '<div class="empty">no pastes yet</div>';
    return;
  }
  for (const p of pastes) pasteTexts.set(p.paste_idx, p.text || "");
  el.innerHTML = pastes.slice().reverse().map(p =>
    `<div class="paste-row">` +
    `<span class="t">#${p.paste_idx} @ ${fmtMs(p.start_accepted_ms)}</span>` +
    `<span class="text">${escapeHtml(p.text||"")}</span>` +
    `<button class="copy" data-paste-idx="${p.paste_idx}" title="copy">copy</button>` +
    `</div>`
  ).join("");
}

async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (e) {
    // Fallback: use a hidden textarea.
    const ta = document.createElement("textarea");
    ta.value = text; ta.style.position = "fixed"; ta.style.left = "-9999px";
    document.body.appendChild(ta); ta.select();
    let ok = false;
    try { ok = document.execCommand("copy"); } catch {}
    document.body.removeChild(ta);
    return ok;
  }
}

function flashCopied(btn, label) {
  const orig = btn.textContent;
  btn.classList.add("copied");
  btn.textContent = label || "copied";
  setTimeout(() => {
    btn.classList.remove("copied");
    btn.textContent = orig;
  }, 1200);
}

// Click delegation for paste-row copy buttons.
document.getElementById("pastes-body").addEventListener("click", async (e) => {
  const btn = e.target.closest("button.copy");
  if (!btn) return;
  const idx = Number(btn.dataset.pasteIdx);
  const text = pasteTexts.get(idx) || "";
  if (await copyToClipboard(text)) flashCopied(btn);
});

document.getElementById("transcript-copy").addEventListener("click", async (e) => {
  // Plain-text version of the transcript: drop the HQ/fast styling, keep text only.
  const el = document.getElementById("transcript");
  const text = el.innerText || el.textContent || "";
  if (await copyToClipboard(text.trim())) flashCopied(e.target);
});

// Build the transcript view by interleaving:
//   - segments from the manager's HQ-preferred merged tail
//   - user-action markers (recording start/end, paste, marker open/close,
//     debug_flag) anchored at their accepted_ms
//   - an "HQ caught up to here" cursor at the HQ watermark
function renderTranscript(transcript) {
  const el = document.getElementById("transcript");
  const segs = (transcript && transcript.segments) || [];
  if (!segs.length) {
    el.innerHTML = '<div class="empty">no audio yet</div>';
    return;
  }
  const minMs = segs[0].start_accepted_ms;
  // Upper bound for marker rendering: prefer the manager's "processed up
  // to here" watermark over the last segment's end. The press timestamp
  // for a close-recording action lands at the current accepted_ms cursor,
  // which can sit a few frames past the last admitted VAD frame — so
  // using last-segment-end as the bound would briefly hide the end
  // marker until more audio extends the segment-end horizon.
  const segMaxMs = segs[segs.length - 1].end_accepted_ms;
  const maxMs = Math.max(
    segMaxMs,
    transcript.processed_up_to_accepted_ms || 0,
  );
  const hqUpTo = transcript.hq_covered_end_accepted_ms;

  // Collect user-action markers in [minMs, maxMs] from the live event stream.
  const markers = [];
  for (const e of recentEvents) {
    if (e.topic !== "user.actions") continue;
    let at = null, pill = null, label = null;
    if (e.action === "recording_started") {
      at = e.start_accepted_ms; pill = "rec-start"; label = "[● rec";
    } else if (e.action === "paste_window_complete") {
      at = e.end_accepted_ms; pill = "rec-end"; label = "rec ●]";
    } else if (e.action === "aside_started") {
      at = e.start_accepted_ms; pill = "aside"; label = "[aside";
    } else if (e.action === "aside_ended") {
      at = e.end_accepted_ms; pill = "aside"; label = "aside]";
    } else if (e.action === "cancel") {
      at = e.end_accepted_ms; pill = "cancel"; label = "✕ cancel";
    } else if (e.action === "marker_opened") {
      at = e.accepted_ms; pill = "marker"; label = `[${e.key}`;
    } else if (e.action === "marker_closed") {
      at = e.close_accepted_ms; pill = "marker"; label = `${e.key}]`;
    } else if (e.action === "debug_flag") {
      at = e.accepted_ms; pill = "dbg"; label = "🚩";
    } else {
      continue;
    }
    if (at == null) continue;
    if (at < minMs || at > maxMs) continue;
    markers.push({type: "marker", at, pill, label});
  }

  // Add HQ cursor as a virtual marker.
  if (hqUpTo > minMs && hqUpTo < maxMs) {
    markers.push({type: "hq-cursor", at: hqUpTo});
  }
  markers.sort((a, b) => a.at - b.at);

  // Walk segments in order. For each segment, splice in any markers
  // whose timestamp falls inside the segment's range — using word
  // timings (word midpoints) to choose the insertion point between
  // words. Markers earlier than the current segment's start are
  // emitted before the segment (between-segment placement).
  const parts = [];
  function renderMarker(m) {
    if (m.type === "hq-cursor") {
      parts.push(`<span class="tx-pill hq" title="HQ has been processed up to this point">HQ ↤</span> `);
    } else {
      parts.push(`<span class="tx-pill ${m.pill}">${escapeHtml(m.label)}</span> `);
    }
  }
  let mi = 0;
  for (const s of segs) {
    // Markers strictly before this segment start: emit first.
    while (mi < markers.length && markers[mi].at < s.start_accepted_ms) {
      renderMarker(markers[mi]); mi++;
    }
    const words = s.words || [];
    // Markers inside this segment: split at word boundaries.
    if (!words.length) {
      parts.push(`<span>${escapeHtml(s.text)} </span>`);
      while (mi < markers.length && markers[mi].at <= s.end_accepted_ms) {
        renderMarker(markers[mi]); mi++;
      }
      continue;
    }
    let wi = 0;
    while (wi < words.length) {
      // Emit any markers that fall before/at this word's midpoint.
      while (mi < markers.length) {
        const m = markers[mi];
        if (m.at > s.end_accepted_ms) break;
        const w = words[wi];
        const mid = (w.start_accepted_ms + w.end_accepted_ms) / 2;
        if (m.at <= mid) { renderMarker(m); mi++; }
        else break;
      }
      parts.push(`<span>${escapeHtml(words[wi].text)}</span>`);
      wi++;
    }
    // Trailing markers that landed after the last word but still within
    // the segment range.
    while (mi < markers.length && markers[mi].at <= s.end_accepted_ms) {
      renderMarker(markers[mi]); mi++;
    }
    parts.push(" ");
  }
  // Any markers past the last segment.
  while (mi < markers.length) { renderMarker(markers[mi]); mi++; }
  el.innerHTML = parts.join("");
  el.scrollTop = el.scrollHeight;
}

async function pollSnapshot() {
  try {
    const r = await fetch("/snapshot");
    const s = await r.json();
    nowAcceptedMs = s.now_accepted_ms;
    renderState(s.state);
    renderPasteLog(s.pastes);
    renderTranscript(s.transcript);
    renderRecentEvents();
    renderTimeline();
  } catch (e) {
    document.getElementById("status-meta").textContent = "disconnected";
  }
}

// SSE: only used for live debug-timeline deltas.
function startSSE() {
  const sse = new EventSource("/events");
  sse.addEventListener("event", e => {
    const ev = JSON.parse(e.data);
    ingestEvent(ev);
  });
  sse.addEventListener("tick", e => {
    nowAcceptedMs = JSON.parse(e.data).now_accepted_ms;
  });
  sse.onerror = () => { document.getElementById("status-meta").textContent = "sse error"; };
}

// Wire filter chips.
document.querySelectorAll("#events-filters .chip").forEach(ch => {
  ch.addEventListener("click", () => {
    const k = ch.dataset.kind;
    if (eventFilters.has(k)) {
      eventFilters.delete(k); ch.classList.remove("on");
    } else {
      eventFilters.add(k); ch.classList.add("on");
    }
    renderRecentEvents();
  });
});
// Initialize chip state from defaults.
document.querySelectorAll("#events-filters .chip").forEach(ch => {
  if (eventFilters.has(ch.dataset.kind)) ch.classList.add("on");
  else ch.classList.remove("on");
});

// Start collapsed/expanded as in v1.
document.getElementById("pastes-section").classList.add("collapsed");
document.getElementById("transcript-section").classList.remove("collapsed");
document.getElementById("events-section").classList.add("collapsed");
document.getElementById("stats-section").classList.add("collapsed");
document.getElementById("timeline-section").classList.remove("collapsed");

// ---- Stats panel --------------------------------------------------------

function fmtNumber(n) {
  if (typeof n !== "number" || !isFinite(n)) return String(n);
  if (Number.isInteger(n)) return n.toLocaleString();
  return n.toFixed(2);
}

function statRow(k, v, kind) {
  const cls = kind ? `v ${kind}` : "v";
  return `<div class="stat-row"><span class="k">${escapeHtml(k)}</span>` +
         `<span class="${cls}">${escapeHtml(fmtNumber(v))}</span></div>`;
}

function classifyValue(name, key, value, allValues) {
  // Mark queue near-full or any dropped count > 0 as warnings.
  if (typeof value === "number") {
    if (key === "dropped_queue_full" && value > 0) return "warn";
    if (key.endsWith("_q_size")) {
      const max = allValues[key.replace("_size", "_maxsize")];
      if (typeof max === "number" && max > 0 && value / max >= 0.75) return "warn";
    }
  }
  return null;
}

function renderStatCard(name, modStats) {
  const rows = Object.entries(modStats).map(([k, v]) => {
    if (typeof v === "object" && v !== null) {
      // Render nested obj briefly.
      return statRow(k, JSON.stringify(v));
    }
    const kind = classifyValue(name, k, v, modStats);
    return statRow(k, v, kind);
  });
  return `<div class="stat-card">` +
         `<div class="stat-title">${escapeHtml(name)}</div>` +
         rows.join("") + `</div>`;
}

function renderBusCard(busStats) {
  if (!busStats || !Object.keys(busStats).length) return "";
  const rows = Object.entries(busStats).map(([name, info]) => {
    const depth = info.queue_depth;
    const dropped = info.dropped;
    const kind = (dropped > 0) ? "warn" : null;
    return `<div class="stat-row">` +
           `<span class="k">${escapeHtml(name)}</span>` +
           `<span class="v ${kind || ''}">q=${depth} drop=${dropped}</span>` +
           `</div>`;
  });
  return `<div class="stat-card">` +
         `<div class="stat-title">event_bus</div>` + rows.join("") + `</div>`;
}

async function pollStats() {
  // Skip when the section is collapsed (cheap).
  if (document.getElementById("stats-section").classList.contains("collapsed")) {
    return;
  }
  try {
    const r = await fetch("/stats");
    const data = await r.json();
    const cards = [];
    for (const [name, modStats] of Object.entries(data.modules || {})) {
      cards.push(renderStatCard(name, modStats));
    }
    cards.push(renderBusCard(data.bus));
    document.getElementById("stats-body").innerHTML = cards.join("");
  } catch (e) {
    document.getElementById("stats-body").innerHTML =
      `<div style="color:var(--bad)">stats error: ${escapeHtml(String(e))}</div>`;
  }
}
setInterval(pollStats, 800);

pollSnapshot();
setInterval(pollSnapshot, 500);
startSSE();
setInterval(renderTimeline, 250);

// Settings modal — structured form generated from the config dict.
const settingsOverlay = document.getElementById("settings-overlay");
const settingsForm = document.getElementById("settings-form");
const settingsJson = document.getElementById("settings-json-fallback");
const settingsStatus = document.getElementById("settings-status");

let configRaw = null;     // last loaded config (for unknown-shape preservation)
let editMode = "form";    // "form" | "json"

// Model picker list — populated from the /models endpoint at load time.
// Hub ids (auto-download via faster-whisper) followed by any local
// CT2 fine-tuned model directories under voice_dictation/models/.
// Entries are either a plain string (hub id) or
// {label, value, group} for grouped/local entries.
let FW_MODELS = [
  "tiny.en", "base.en", "small.en", "medium.en",
  "large-v3", "distil-large-v3",
];

async function refreshModelList() {
  try {
    const r = await fetch("/models");
    if (!r.ok) return;
    const data = await r.json();
    const out = [];
    for (const id of data.hub || []) out.push(id);
    for (const m of data.local || []) {
      if (!m.valid) continue;
      // Label distinguishes a local fine-tune from a hub id.
      out.push({label: "📁 " + m.name, value: m.path});
    }
    if (out.length) FW_MODELS = out;
  } catch (e) { /* ignore — defaults still work */ }
}

// Per-path hints for controls that aren't trivially inferable from the
// value type. Path uses dot-notation against the config root.
const CFG_HINTS = {
  "audio_persist.format":           {select: ["wav", "mp3"]},
  "fast.fw.model":                  {select: FW_MODELS},
  "hq.fw.model":                    {select: FW_MODELS},
  "fast.fw.compute":                {select: ["int8", "int8_float16", "float16", "float32"]},
  "hq.fw.compute":                  {select: ["int8", "int8_float16", "float16", "float32"]},
  "fast.fw.device":                 {select: ["", "cpu", "cuda"], help: "empty = auto-detect"},
  "hq.fw.device":                   {select: ["", "cpu", "cuda"], help: "empty = auto-detect"},
  "audio.format":                   {select: ["int16"]},
  "preprocessor.aggressiveness":    {select: [0, 1, 2, 3], help: "webrtcvad 0-3"},
  "fast.label":                     {readonly: true},
  "hq.label":                       {readonly: true},
};

// Field help text (path → string). Kept separate so hints stay terse.
const CFG_HELP = {
  "audio.sample_rate":              "webrtcvad requires 8/16/32/48 kHz; 16 kHz is the default",
  "audio_persist.chunk_seconds":    "size of each persisted audio chunk in seconds of accepted audio",
  "audio_persist.mp3_bitrate_kbps": "only used when format=mp3",
  "preprocessor.silence_boundary_ms": "ms of continuous silence after voice to fire a boundary event",
  "preprocessor.voiced_gate_frames":   "size of sliding-window voiced gate (in 20ms sub-frames)",
  "preprocessor.voiced_gate_min_voiced": "min voiced sub-frames out of voiced_gate_frames to admit",
  "fast.max_window_seconds":        "hard ceiling for a window before force-flush",
  "fast.min_window_seconds":        "minimum window size before silence-boundary flush (clipboard-end bypasses)",
  "fast.flush_on_markers":          "if true, silence/clipboard-end markers trigger early flush",
  "fast.window_q_maxsize":          "max in-flight windows queued for the transcriber",
  "fast.min_rms":                   "skip windows whose RMS amplitude is below this (filters background noise hallucinations)",
  "fast.fw.beam_size":              "1 = greedy (fast), 5 = beam search (better)",
  "fast.fw.condition_on_previous_text": "feed prior decoded text as initial prompt",
  "hq.max_window_seconds":          "hard ceiling for a window before force-flush",
  "hq.min_window_seconds":          "minimum window size before silence-boundary flush",
  "hq.flush_on_markers":            "HQ typically waits for max_window — set true to also flush on markers",
  "hq.min_rms":                     "skip windows whose RMS amplitude is below this",
  "manager.complete_tolerance_ms":  "paste fires when watermark + tolerance reaches press time",
  "paste.delay":                    "seconds between clipboard.copy and Cmd+V/Ctrl+V tap",
  "app.hf_hub_offline":             "skip HuggingFace 'check for updates' on model load",
  "app.persist_raw_keystrokes":     "record pre-double-tap keystrokes (debug only)",
  "app.debug_flag_key":             "single character; double-tap stamps a debug flag",
};

function setSettingsStatus(text, kind) {
  settingsStatus.className = kind || "";
  settingsStatus.textContent = text || "";
}

function isPlainObject(v) {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

// Generate a form control for one leaf value at ``path``.
function buildControl(path, value) {
  const hint = CFG_HINTS[path] || {};
  if (hint.select) {
    const sel = document.createElement("select");
    sel.dataset.cfgPath = path;
    sel.dataset.cfgType = typeof value === "number" ? "number" : "string";
    const current = String(value == null ? "" : value);
    // Options may be plain strings/numbers OR {label, value} objects (for
    // local fine-tuned models where the display name differs from the path).
    const optValue = o => String(o && typeof o === "object" ? o.value : o);
    const optLabel = o => {
      if (o && typeof o === "object") return o.label || String(o.value);
      const s = String(o);
      return s || "(auto)";
    };
    const inList = hint.select.some(o => optValue(o) === current);
    if (!inList) {
      const o = document.createElement("option");
      o.value = current;
      o.textContent = current ? `${current} (custom)` : "(auto)";
      o.selected = true;
      sel.appendChild(o);
    }
    for (const opt of hint.select) {
      const o = document.createElement("option");
      o.value = optValue(opt);
      o.textContent = optLabel(opt);
      if (optValue(opt) === current) o.selected = true;
      sel.appendChild(o);
    }
    if (hint.readonly) sel.disabled = true;
    return sel;
  }
  if (hint.datalist) {
    // Combobox: free-text input with suggestion list. Future model ids
    // still work even if they're not in the suggestion set.
    const wrap = document.createDocumentFragment();
    const el = document.createElement("input");
    el.type = "text";
    el.value = value == null ? "" : String(value);
    el.dataset.cfgPath = path;
    el.dataset.cfgType = "string";
    // Unique id so multiple combos on the page don't share a list.
    const listId = "dl-" + path.replace(/\./g, "-");
    el.setAttribute("list", listId);
    if (hint.readonly) el.readOnly = true;
    const dl = document.createElement("datalist");
    dl.id = listId;
    for (const opt of hint.datalist) {
      const o = document.createElement("option");
      o.value = String(opt);
      dl.appendChild(o);
    }
    wrap.appendChild(el);
    wrap.appendChild(dl);
    return wrap;
  }
  if (typeof value === "boolean") {
    const el = document.createElement("input");
    el.type = "checkbox";
    el.checked = value;
    el.dataset.cfgPath = path;
    el.dataset.cfgType = "boolean";
    return el;
  }
  if (typeof value === "number") {
    const el = document.createElement("input");
    el.type = "number";
    el.value = value;
    el.step = Number.isInteger(value) ? "1" : "any";
    el.dataset.cfgPath = path;
    el.dataset.cfgType = Number.isInteger(value) ? "int" : "float";
    if (hint.readonly) el.readOnly = true;
    return el;
  }
  if (Array.isArray(value)) {
    const ta = document.createElement("textarea");
    ta.value = JSON.stringify(value, null, 2);
    ta.dataset.cfgPath = path;
    ta.dataset.cfgType = "json";
    if (hint.readonly) ta.readOnly = true;
    return ta;
  }
  // string fallback
  const el = document.createElement("input");
  el.type = "text";
  el.value = value == null ? "" : String(value);
  el.dataset.cfgPath = path;
  el.dataset.cfgType = "string";
  if (hint.readonly) el.readOnly = true;
  return el;
}

function buildRow(path, key, value) {
  const row = document.createElement("div");
  row.className = "cfg-row";
  const label = document.createElement("label");
  label.textContent = key;
  label.title = path;
  row.appendChild(label);
  row.appendChild(buildControl(path, value));
  const help = (CFG_HINTS[path] && CFG_HINTS[path].help) || CFG_HELP[path];
  if (help) {
    const h = document.createElement("div");
    h.className = "cfg-help";
    h.textContent = help;
    row.appendChild(h);
  }
  return row;
}

// Render a section card for a top-level config key; recurses into nested objects.
function buildSection(rootEl, name, obj, pathPrefix) {
  const sec = document.createElement("div");
  sec.className = "cfg-section";
  const h = document.createElement("h4");
  h.textContent = name;
  h.addEventListener("click", () => sec.classList.toggle("collapsed"));
  sec.appendChild(h);
  const body = document.createElement("div");
  body.className = "cfg-body";
  sec.appendChild(body);
  buildBody(body, obj, pathPrefix);
  rootEl.appendChild(sec);
}

function buildBody(parent, obj, pathPrefix) {
  for (const [k, v] of Object.entries(obj)) {
    const path = pathPrefix ? `${pathPrefix}.${k}` : k;
    if (isPlainObject(v)) {
      // Nested object — render as inline sub-section.
      const wrap = document.createElement("div");
      wrap.className = "cfg-nested";
      const t = document.createElement("div");
      t.className = "cfg-nested-title";
      t.textContent = k;
      wrap.appendChild(t);
      buildBody(wrap, v, path);
      parent.appendChild(wrap);
    } else {
      parent.appendChild(buildRow(path, k, v));
    }
  }
}

function renderConfigForm(config) {
  settingsForm.innerHTML = "";
  // Top-level keys get their own collapsible section.
  for (const [k, v] of Object.entries(config)) {
    if (isPlainObject(v)) {
      buildSection(settingsForm, k, v, k);
    } else {
      // Scalar at root — drop into a "general" section.
      let gen = settingsForm.querySelector(".cfg-section[data-gen]");
      if (!gen) {
        gen = document.createElement("div");
        gen.className = "cfg-section";
        gen.dataset.gen = "1";
        const h = document.createElement("h4");
        h.textContent = "(top level)";
        h.addEventListener("click", () => gen.classList.toggle("collapsed"));
        gen.appendChild(h);
        const body = document.createElement("div");
        body.className = "cfg-body";
        gen.appendChild(body);
        settingsForm.appendChild(gen);
      }
      gen.querySelector(".cfg-body").appendChild(buildRow(k, k, v));
    }
  }
}

// Walk the form and rebuild a config dict with type-coerced values.
function collectConfigForm() {
  // Start from a deep clone of the last-loaded config so unknown keys
  // (e.g. fields the JS doesn't render) are preserved.
  const out = JSON.parse(JSON.stringify(configRaw || {}));
  function setPath(dotted, value) {
    const parts = dotted.split(".");
    let cur = out;
    for (let i = 0; i < parts.length - 1; i++) {
      if (!isPlainObject(cur[parts[i]])) cur[parts[i]] = {};
      cur = cur[parts[i]];
    }
    cur[parts[parts.length - 1]] = value;
  }
  for (const el of settingsForm.querySelectorAll("[data-cfg-path]")) {
    const path = el.dataset.cfgPath;
    const type = el.dataset.cfgType;
    let v;
    if (type === "boolean") v = el.checked;
    else if (type === "int") v = parseInt(el.value, 10);
    else if (type === "float") v = parseFloat(el.value);
    else if (type === "number") v = (el.value === "" ? "" : Number(el.value));
    else if (type === "json") {
      try { v = JSON.parse(el.value); }
      catch (e) { throw new Error(`Field ${path}: invalid JSON (${e.message})`); }
    } else v = el.value;
    setPath(path, v);
  }
  return out;
}

function setMode(mode) {
  editMode = mode;
  if (mode === "json") {
    settingsForm.style.display = "none";
    settingsJson.style.display = "block";
    settingsJson.value = JSON.stringify(configRaw, null, 2);
  } else {
    settingsForm.style.display = "block";
    settingsJson.style.display = "none";
  }
}

async function loadSettings() {
  setSettingsStatus("loading…");
  try {
    // Refresh model list first so the form's model selects render with
    // the latest local fine-tuned models on first paint.
    await refreshModelList();
    CFG_HINTS["fast.fw.model"].select = FW_MODELS;
    CFG_HINTS["hq.fw.model"].select = FW_MODELS;
    const r = await fetch("/config");
    const data = await r.json();
    if (data.error) { setSettingsStatus(data.error, "err"); return; }
    configRaw = data.config;
    renderConfigForm(configRaw);
    if (editMode === "json") {
      settingsJson.value = JSON.stringify(configRaw, null, 2);
    }
    setSettingsStatus("loaded", "ok");
  } catch (e) {
    setSettingsStatus("load failed: " + e, "err");
  }
}

async function saveSettings() {
  let parsed;
  try {
    if (editMode === "json") {
      parsed = JSON.parse(settingsJson.value);
    } else {
      parsed = collectConfigForm();
    }
  } catch (e) {
    setSettingsStatus(e.message || ("invalid input: " + e), "err");
    return;
  }
  setSettingsStatus("saving…");
  try {
    const r = await fetch("/config", {
      method: "PUT",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({config: parsed}),
    });
    const data = await r.json();
    if (data.error) { setSettingsStatus(data.error, "err"); return; }
    configRaw = parsed;
    const msg = data.reloaded && data.reloaded.length
      ? `saved → reloading model on next window: ${data.reloaded.join(", ")}`
      : `saved → ${data.note || "restart required for some fields"}`;
    setSettingsStatus(msg, "ok");
  } catch (e) {
    setSettingsStatus("save failed: " + e, "err");
  }
}

document.getElementById("settings-gear").addEventListener("click", () => {
  settingsOverlay.hidden = false;
  loadSettings();
});
document.getElementById("settings-close").addEventListener("click", () => {
  settingsOverlay.hidden = true;
});
document.getElementById("settings-reload").addEventListener("click", loadSettings);
document.getElementById("settings-save").addEventListener("click", saveSettings);
document.getElementById("settings-mode-toggle").addEventListener("click", () => {
  setMode(editMode === "form" ? "json" : "form");
});
settingsOverlay.addEventListener("click", (e) => {
  if (e.target === settingsOverlay) settingsOverlay.hidden = true;
});
</script>
</body>
</html>
"""


class _SSEClient:
    def __init__(self, max_queued: int = 2048) -> None:
        self.queue: queue.Queue = queue.Queue(maxsize=max_queued)
        self.stop = threading.Event()


class WidgetServer:
    def __init__(
        self,
        bus: EventBus,
        clock: AcceptedClock,
        manager: TranscriptStreamManager,
        *,
        port: int = 0,
        recent_events_ring_size: int = 5000,
        paste_log_size: int = 200,
        cfg: Optional[RuntimeConfig] = None,
        stats_modules: Optional[dict] = None,
        models_dir: Optional[str] = None,
    ) -> None:
        self._bus = bus
        self._clock = clock
        self._manager = manager
        self._port = port
        self._cfg = cfg
        # Map of label → object with stats() — used by /stats endpoint.
        # Order is preserved for UI rendering.
        self._stats_modules: dict = stats_modules or {}
        # Where to scan for local CT2 fine-tuned model directories. The
        # /models endpoint lists each subdirectory containing both
        # ``model.bin`` and ``tokenizer.json`` (faster-whisper's minimum
        # requirements). Falls back to ``voice_dictation/models/``
        # relative to the package root.
        if models_dir is None:
            from pathlib import Path as _Path
            models_dir = str(_Path(__file__).parent / "models")
        self._models_dir = models_dir

        # Recent-events ring (for the debug timeline + recent-events panel).
        self._events: deque = deque(maxlen=recent_events_ring_size)
        self._events_lock = threading.Lock()

        # Server-tracked state.
        self._paste_log: deque = deque(maxlen=paste_log_size)
        self._state_lock = threading.Lock()
        self._capture: str = "passive"
        self._recording_started_at: Optional[int] = None
        self._aside_started_at: Optional[int] = None
        self._muted: bool = False
        # Currently-open marker intervals: key → opened_at_accepted_ms.
        self._open_markers: dict[str, int] = {}

        # Bus subscriptions.
        self._sub_all = bus.subscribe("*", self._on_event, name="widget.events",
                                      max_queued=8192)

        # SSE clients.
        self._clients: list[_SSEClient] = []
        self._clients_lock = threading.Lock()

        # HTTP / lifecycle.
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None
        self._tick_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------

    @property
    def actual_port(self) -> int:
        if self._httpd is None:
            return 0
        return self._httpd.server_address[1]

    def start(self) -> None:
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args, **_kw) -> None:
                pass

            def _send_json(self, body: dict, code: int = 200) -> None:
                data = json.dumps(body).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                if self.path == "/":
                    body = _INDEX_HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == "/snapshot":
                    self._send_json(outer._build_snapshot())
                    return
                if self.path == "/transcript":
                    self._send_json(outer._manager.get_canonical_tail(10_000))
                    return
                if self.path == "/events":
                    outer._serve_sse(self)
                    return
                if self.path == "/config":
                    self._send_json(outer._get_config())
                    return
                if self.path == "/stats":
                    self._send_json(outer._get_stats())
                    return
                if self.path == "/models":
                    self._send_json(outer._get_models())
                    return
                self.send_error(404)

            def do_PUT(self) -> None:
                if self.path == "/config":
                    length = int(self.headers.get("Content-Length", "0") or "0")
                    raw = self.rfile.read(length) if length else b""
                    try:
                        body = json.loads(raw.decode("utf-8")) if raw else {}
                    except Exception:
                        self._send_json({"error": "invalid json"}, code=400)
                        return
                    try:
                        result = outer._put_config(body)
                    except Exception as e:
                        self._send_json({"error": str(e)}, code=500)
                        return
                    self._send_json(result)
                    return
                self.send_error(404)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", self._port), _Handler)
        self._http_thread = threading.Thread(
            target=self._httpd.serve_forever, name="widget-http", daemon=True,
        )
        self._http_thread.start()
        self._tick_thread = threading.Thread(
            target=self._tick_loop, name="widget-tick", daemon=True,
        )
        self._tick_thread.start()
        logger.info("WidgetServer started on http://127.0.0.1:%d/", self.actual_port)

    def shutdown(self) -> None:
        self._stop.set()
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None
        with self._clients_lock:
            for c in self._clients:
                c.stop.set()
            self._clients.clear()
        self._sub_all.shutdown()

    # ------------------------------------------------------------------
    # Bus intake
    # ------------------------------------------------------------------

    def _on_event(self, ev: Event) -> None:
        record = self._serialize(ev)
        with self._events_lock:
            self._events.append(record)
        self._update_state(ev)
        # Fan out to SSE clients.
        with self._clients_lock:
            clients = list(self._clients)
        for c in clients:
            try:
                c.queue.put_nowait(("event", record))
            except queue.Full:
                try:
                    c.queue.get_nowait()
                    c.queue.put_nowait(("event", record))
                except queue.Empty:
                    pass

    def _update_state(self, ev: Event) -> None:
        with self._state_lock:
            if ev.topic == "user.actions":
                a = ev.payload.get("action")
                if a == "recording_started":
                    self._capture = "recording"
                    self._recording_started_at = ev.emit_accepted_ms
                elif a == "paste_window_complete":
                    self._capture = "passive"
                    self._recording_started_at = None
                elif a == "cancel":
                    self._capture = "passive"
                    self._recording_started_at = None
                elif a == "aside_started":
                    self._capture = "aside"
                    self._aside_started_at = ev.emit_accepted_ms
                elif a == "aside_ended":
                    # Restore prior state: if a recording is still open,
                    # go back to "recording"; otherwise passive.
                    self._capture = (
                        "recording" if self._recording_started_at is not None
                        else "passive"
                    )
                    self._aside_started_at = None
                elif a == "mute_toggle":
                    self._muted = not self._muted
                elif a == "marker_opened":
                    k = ev.payload.get("key")
                    if k is not None:
                        self._open_markers[k] = ev.payload.get(
                            "accepted_ms", ev.emit_accepted_ms,
                        )
                elif a == "marker_closed":
                    k = ev.payload.get("key")
                    if k is not None:
                        self._open_markers.pop(k, None)
            elif ev.topic == "paste.actions":
                self._paste_log.append({
                    "paste_idx": ev.payload.get("paste_idx"),
                    "start_accepted_ms": ev.payload.get("start_accepted_ms"),
                    "end_accepted_ms": ev.payload.get("end_accepted_ms"),
                    "text": ev.payload.get("text", ""),
                    "is_aside": ev.payload.get("is_aside", False),
                })

    @staticmethod
    def _serialize(ev: Event) -> dict:
        return {
            "topic": ev.topic,
            "emit_accepted_ms": ev.emit_accepted_ms,
            "seq": ev.seq,
            **ev.payload,
        }

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def _get_stats(self) -> dict:
        modules: dict[str, dict] = {}
        for name, mod in self._stats_modules.items():
            try:
                modules[name] = mod.stats() if hasattr(mod, "stats") else {}
            except Exception as e:
                modules[name] = {"error": str(e)}
        bus_stats = self._bus.stats()
        return {
            "now_accepted_ms": self._clock.now_accepted_ms(),
            "modules": modules,
            "bus": bus_stats,
        }

    def _get_models(self) -> dict:
        """List available Whisper models.

        Returns:
          ``hub``: canonical HuggingFace ids that faster-whisper auto-downloads.
          ``local``: subdirectories of ``self._models_dir`` that look like
                     a valid faster-whisper CT2 model (contain ``model.bin``
                     and a tokenizer file). Each entry is
                     ``{name, path, valid}`` where ``path`` is what should be
                     written into ``fast.fw.model`` / ``hq.fw.model``.
        """
        from pathlib import Path as _P
        hub = [
            "tiny.en", "base.en", "small.en", "medium.en",
            "large-v3", "distil-large-v3",
        ]
        local: list[dict] = []
        d = _P(self._models_dir)
        if d.is_dir():
            for sub in sorted(d.iterdir()):
                if not sub.is_dir():
                    continue
                has_bin = (sub / "model.bin").exists()
                has_tok = (
                    (sub / "tokenizer.json").exists()
                    or (sub / "vocabulary.json").exists()
                )
                local.append({
                    "name": sub.name,
                    "path": str(sub),
                    "valid": has_bin and has_tok,
                })
        return {
            "hub": hub,
            "local": local,
            "models_dir": str(d),
        }

    def _get_config(self) -> dict:
        if self._cfg is None:
            return {"error": "no config wired"}
        data = to_dict(self._cfg)
        # Strip resolved-path / auto-detect fields the user shouldn't edit.
        try:
            data["app"]["sessions_dir"] = ""
            data["fast"]["fw"]["device"] = ""
            data["hq"]["fw"]["device"] = ""
        except Exception:
            pass
        return {"config": data}

    def _put_config(self, body: dict) -> dict:
        if self._cfg is None:
            raise RuntimeError("no config wired")
        data = body.get("config") if "config" in body else body
        if not isinstance(data, dict):
            raise ValueError("expected {'config': {...}} or plain {...}")
        # Snapshot fields that affect cached model state so we can hot-reload.
        prev_fast_fw = _snapshot_fw(self._cfg.fast.fw)
        prev_hq_fw = _snapshot_fw(self._cfg.hq.fw)
        apply_dict_to_config(self._cfg, data)
        path = save_runtime_config(self._cfg)
        logger.info("config updated and saved to %s", path)
        # Hot-reload any transcribers whose fw config materially changed.
        reloaded: list[str] = []
        fast_tr = self._stats_modules.get("transcriber.fast")
        if fast_tr is not None and _fw_changed(prev_fast_fw, self._cfg.fast.fw):
            fast_tr.invalidate_model()
            reloaded.append("fast")
        hq_tr = self._stats_modules.get("transcriber.hq")
        if hq_tr is not None and _fw_changed(prev_hq_fw, self._cfg.hq.fw):
            hq_tr.invalidate_model()
            reloaded.append("hq")
        if reloaded:
            note = (f"applied · reloading model on next window: "
                    f"{', '.join(reloaded)} (other fields may still require restart)")
        else:
            note = ("saved · changes to window sizes / preprocessor / device "
                    "still require restart")
        return {"saved": path, "note": note, "reloaded": reloaded}

    def _build_snapshot(self) -> dict:
        now_ms = self._clock.now_accepted_ms()
        with self._state_lock:
            state = {
                "capture": self._capture,
                "muted": self._muted,
                # List of {key, opened_at_accepted_ms, age_ms} for the UI pill.
                "open_markers": [
                    {"key": k, "opened_at_accepted_ms": v,
                     "age_ms": max(0, now_ms - v)}
                    for k, v in self._open_markers.items()
                ],
                "now_accepted_ms": now_ms,
            }
            if self._recording_started_at is not None:
                state["recording_seconds"] = (
                    now_ms - self._recording_started_at
                ) / 1000.0
            if self._aside_started_at is not None:
                state["aside_seconds"] = (
                    now_ms - self._aside_started_at
                ) / 1000.0
            pastes = list(self._paste_log)
        transcript = self._manager.get_canonical_tail(10_000)
        return {
            "state": state,
            "pastes": pastes,
            "transcript": transcript,
            "now_accepted_ms": now_ms,
        }

    # ------------------------------------------------------------------
    # SSE
    # ------------------------------------------------------------------

    def _serve_sse(self, handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "keep-alive")
        handler.end_headers()
        client = _SSEClient()
        with self._clients_lock:
            self._clients.append(client)
        try:
            while not client.stop.is_set() and not self._stop.is_set():
                try:
                    name, data = client.queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                body = f"event: {name}\ndata: {json.dumps(data)}\n\n".encode("utf-8")
                try:
                    handler.wfile.write(body)
                    handler.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
        finally:
            with self._clients_lock:
                try:
                    self._clients.remove(client)
                except ValueError:
                    pass

    def _tick_loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(0.5)
            with self._clients_lock:
                clients = list(self._clients)
            payload = {"now_accepted_ms": self._clock.now_accepted_ms()}
            for c in clients:
                try:
                    c.queue.put_nowait(("tick", payload))
                except queue.Full:
                    pass
