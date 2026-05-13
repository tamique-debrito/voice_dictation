"""Embedded HTTP status widget for the persistent dictation app.

A tiny stdlib HTTP server bound to a free localhost port. Serves:

* ``GET /``       — static HTML dashboard (single string below).
* ``GET /status`` — current ``snapshot_provider()`` payload as JSON.

The dashboard polls ``/status`` every 500 ms and renders three panels
(status bar, paste log, transcript tail). All CSS/JS is inline so the page
needs no external assets and works fully offline.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Persistent Monitoring</title>
<style>
  :root {
    --bg: #1b1d23;
    --panel: #24262d;
    --panel-2: #2c2f37;
    --fg: #e6e6ea;
    --muted: #8a8d97;
    --accent: #7e57c2;
    --r-active: #d64545;
    --aside-active: #c792ea;
    --marker: #f0a04b;
    --good: #6abf4b;
    --bad: #d64545;
    /* HQ transcript text: a brighter, slightly cooler tone than the
       default fast-stream color so the eye can distinguish the
       HQ-decoded prefix from the fast-only tail at a glance. */
    --src-hq: #b3e5fc;
    --src-fast: #cfcfd5;
  }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--fg);
    font-size: 14px;
  }
  header {
    display: grid;
    grid-template-columns: 1fr auto auto auto auto;
    gap: 16px;
    align-items: center;
    padding: 10px 16px;
    background: var(--panel);
    border-bottom: 1px solid #0d0e11;
    position: sticky;
    top: 0;
    z-index: 10;
  }
  header .title { font-weight: 600; }
  header .meta  { color: var(--muted); font-size: 12px; }
  .pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    background: #3a3d45;
    color: var(--fg);
    font-size: 12px;
    font-weight: 500;
  }
  .pill.passive { background: #3a3d45; color: var(--muted); }
  .pill.r       { background: var(--r-active); color: #fff; }
  .pill.aside   { background: var(--aside-active); color: #1b1d23; }
  .pill.r-aside { background: linear-gradient(90deg, var(--r-active) 50%, var(--aside-active) 50%); color: #1b1d23; }
  .pill.marker  { background: var(--marker); color: #1b1d23; }
  .pill.none    { background: transparent; color: var(--muted); border: 1px dashed #4a4d55; }
  .pill.muted   { background: #e7c84a; color: #1b1d23; font-weight: 700; letter-spacing: 0.05em; }
  body.is-muted { box-shadow: inset 0 0 0 3px #e7c84a; }
  body.is-muted header { background: #3a341a; }

  #disconnected {
    display: none;
    background: var(--bad);
    color: white;
    padding: 6px 12px;
    text-align: center;
    font-weight: 600;
  }

  main { padding: 12px 16px; display: grid; gap: 12px; }
  section {
    background: var(--panel);
    border-radius: 6px;
    overflow: hidden;
  }
  section > h2 {
    margin: 0;
    padding: 8px 12px;
    background: var(--panel-2);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    user-select: none;
  }
  section > h2:hover { color: var(--fg); }
  section > h2 .chev {
    display: inline-block;
    width: 1em;
    text-align: center;
    font-size: 10px;
    transition: transform 0.12s;
  }
  section[data-state="expanded"] > h2 .chev { transform: rotate(90deg); }
  section .body {
    overflow-y: auto;
    transition: max-height 0.15s ease-out;
  }
  section[data-state="collapsed"] .body { max-height: 100px; }
  section[data-state="expanded"]  .body { max-height: 500px; }
  .empty { padding: 16px; color: var(--muted); font-style: italic; }

  table.pastes { width: 100%; border-collapse: collapse; }
  table.pastes td {
    padding: 6px 12px;
    border-top: 1px solid #1b1d23;
    vertical-align: top;
  }
  table.pastes tr:first-child td { border-top: 0; }
  table.pastes td.ts { color: var(--muted); white-space: nowrap; font-variant-numeric: tabular-nums; }
  table.pastes td.label { font-weight: 600; }
  table.pastes td.label.r { color: var(--r-active); }
  table.pastes td.label.aside { color: var(--aside-active); }
  table.pastes td.preview { color: var(--fg); }
  table.pastes tr { cursor: copy; }
  table.pastes tr:hover td { background: var(--panel-2); }
  table.pastes tr.copied td { background: #2a4a2a; transition: background 0.4s; }

  .copyable { cursor: copy; position: relative; }
  .copyable.copied {
    box-shadow: inset 0 0 0 2px var(--good);
    transition: box-shadow 0.4s;
  }
  #copy-toast {
    position: fixed;
    bottom: 16px; left: 50%;
    transform: translateX(-50%);
    background: var(--good); color: #1b1d23;
    padding: 6px 14px;
    border-radius: 4px;
    font-size: 13px;
    font-weight: 600;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s;
    z-index: 100;
  }
  #copy-toast.show { opacity: 1; }

  #transcript {
    padding: 12px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 13px;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .marker-pill {
    display: inline-block;
    padding: 0 6px;
    margin: 0 2px;
    border-radius: 3px;
    background: var(--marker);
    color: #1b1d23;
    font-size: 11px;
    font-weight: 600;
    vertical-align: baseline;
  }
  .marker-pill.recording { background: var(--r-active); color: #fff; }
  .marker-pill.aside     { background: var(--aside-active); color: #1b1d23; }

  /* Settings: floating gear + modal */
  #settings-gear {
    position: fixed;
    right: 16px;
    bottom: 16px;
    width: 44px;
    height: 44px;
    border-radius: 50%;
    border: 1px solid #3a3d45;
    background: var(--panel);
    color: var(--fg);
    font-size: 20px;
    cursor: pointer;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    z-index: 100;
    line-height: 1;
  }
  #settings-gear:hover { background: var(--panel-2); }
  #settings-overlay {
    position: fixed; inset: 0;
    background: rgba(0,0,0,0.55);
    display: flex; align-items: center; justify-content: center;
    z-index: 200;
  }
  #settings-overlay[hidden] { display: none; }
  #settings-modal {
    background: var(--panel);
    border: 1px solid #0d0e11;
    border-radius: 8px;
    width: min(640px, 92vw);
    max-height: 86vh;
    display: flex; flex-direction: column;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6);
  }
  #settings-modal header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 16px;
    border-bottom: 1px solid #0d0e11;
  }
  #settings-modal h3 { margin: 0; font-size: 15px; }
  #settings-close {
    background: transparent; border: none; color: var(--muted);
    font-size: 22px; cursor: pointer; line-height: 1; padding: 0 4px;
  }
  #settings-close:hover { color: var(--fg); }
  #settings-modal .modal-body {
    overflow-y: auto;
    padding: 12px 16px;
    flex: 1;
  }
  #settings-modal .hint { color: var(--muted); font-size: 12px; margin: 0 0 12px; }
  #settings-modal fieldset {
    border: 1px solid #3a3d45; border-radius: 6px;
    padding: 10px 12px 4px; margin-bottom: 12px;
  }
  #settings-modal legend { color: var(--muted); font-size: 12px; padding: 0 6px; }
  #settings-modal label {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 8px; font-size: 13px; gap: 12px;
  }
  #settings-modal label.checkbox { justify-content: flex-start; }
  #settings-modal label.checkbox input { margin-right: 8px; }
  #settings-modal input[type="number"],
  #settings-modal select {
    background: #15171c; color: var(--fg);
    border: 1px solid #3a3d45; border-radius: 4px;
    padding: 4px 8px; font-size: 13px; min-width: 140px;
  }
  #settings-modal .deferred-note { color: var(--muted); font-size: 11px; }
  #settings-modal details { margin-top: 8px; }
  #settings-modal details summary { color: var(--muted); font-size: 12px; cursor: pointer; }
  #settings-modal textarea {
    width: 100%; height: 200px; margin-top: 8px;
    background: #15171c; color: var(--fg);
    border: 1px solid #3a3d45; border-radius: 4px;
    padding: 8px; font-family: monospace; font-size: 12px;
    box-sizing: border-box;
  }
  #settings-modal footer {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 16px;
    border-top: 1px solid #0d0e11;
  }
  #settings-modal footer button {
    padding: 6px 14px; border: none; border-radius: 4px;
    background: #3a3d45; color: var(--fg); cursor: pointer;
  }
  #settings-modal footer button.primary { background: var(--accent); color: #fff; font-weight: 600; }
  #settings-modal footer #settings-status { flex: 1; color: var(--muted); font-size: 12px; }

  /* Merged transcript: HQ prefix in one color, fast tail in another. */
  .src-hq   { color: var(--src-hq); }
  .src-fast { color: var(--src-fast); }
  .hq-edge {
    display: inline-block;
    padding: 1px 6px;
    margin: 0 6px;
    border-radius: 8px;
    background: #324a5a;
    color: #b3e5fc;
    font-size: 11px;
    vertical-align: middle;
  }

  /* Debug panels */
  .debug-controls {
    padding: 8px 12px;
    background: var(--panel-2);
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    align-items: center;
    border-bottom: 1px solid #1b1d23;
    /* Anchor the filter chips to the top of the scrollable section body
       so they remain visible when the events list grows long. */
    position: sticky;
    top: 0;
    z-index: 1;
  }
  .debug-controls label { color: var(--muted); font-size: 12px; }
  .chip {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    background: #3a3d45;
    color: var(--muted);
    font-size: 11px;
    cursor: pointer;
    user-select: none;
    border: 1px solid transparent;
  }
  .chip.on { background: var(--accent); color: #fff; border-color: #b39ddb; }
  table.events {
    width: 100%; border-collapse: collapse;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px;
  }
  table.events td {
    padding: 3px 8px;
    border-top: 1px solid #1b1d23;
    vertical-align: top;
    white-space: nowrap;
  }
  table.events td.summary { white-space: pre-wrap; word-break: break-word; }
  table.events td.ts { color: var(--muted); font-variant-numeric: tabular-nums; width: 70px; }
  table.events td.kind { color: var(--accent); font-weight: 600; width: 110px; }
  table.events tr.kind-segment td.kind   { color: var(--good); }
  table.events tr.kind-audio_window td.kind { color: #5fa8d3; }
  table.events tr.kind-press td.kind     { color: var(--marker); }
  table.events tr.kind-paste td.kind     { color: var(--r-active); }
  table.events tr.kind-marker td.kind    { color: var(--marker); }
  table.events tr.kind-cursor_capture td.kind { color: #c792ea; }
  table.events tr.kind-mute td.kind      { color: #e7c84a; }

  #timeline-svg {
    width: 100%;
    background: var(--panel-2);
    display: block;
  }
  #timeline-legend {
    padding: 6px 12px;
    background: var(--panel);
    color: var(--muted);
    font-size: 11px;
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
  }
  #timeline-legend .swatch {
    display: inline-block; width: 10px; height: 10px;
    border-radius: 2px; margin-right: 4px; vertical-align: middle;
  }
</style>
</head>
<body>
<header>
  <div>
    <span class="title">🎤 Persistent Monitoring</span>
    <span class="meta" id="meta"></span>
  </div>
  <div>capture: <span id="capture" class="pill passive">passive</span></div>
  <div>marker:  <span id="marker" class="pill none">none</span></div>
  <div><span id="mute" class="pill muted" style="display:none">MUTED</span></div>
  <div class="meta" id="uptime"></div>
</header>
<div id="disconnected">⚠ disconnected from monitoring service</div>
<div id="copy-toast">copied</div>
<main>
  <section id="pastes-section" class="collapsible" data-state="collapsed">
    <h2><span class="chev">▸</span>Paste log</h2>
    <div class="body" id="pastes-body"><div class="empty">no pastes yet</div></div>
  </section>
  <section id="transcript-section" class="collapsible" data-state="collapsed">
    <h2><span class="chev">▸</span>Transcript (tail)</h2>
    <div class="body" id="transcript-body">
      <div id="transcript" class="empty copyable" title="click to copy full transcript tail">no audio yet</div>
    </div>
  </section>
  <section id="events-section" class="collapsible" data-state="collapsed">
    <h2><span class="chev">▸</span>Recent events</h2>
    <div class="body" id="events-body">
      <div class="debug-controls">
        <label>filter:</label>
        <span class="chip on" data-kind="audio_window">audio_window</span>
        <span class="chip on" data-kind="segment">segment</span>
        <span class="chip on" data-kind="press">press</span>
        <span class="chip on" data-kind="marker">marker</span>
        <span class="chip on" data-kind="cursor_capture">cursor_capture</span>
        <span class="chip on" data-kind="paste">paste</span>
        <span class="chip on" data-kind="mute">mute</span>
      </div>
      <div id="events-table" class="copyable" title="click to copy all visible events"><div class="empty">no events yet</div></div>
    </div>
  </section>
  <section id="timeline-section" class="collapsible" data-state="expanded">
    <h2><span class="chev">▸</span>Debug timeline</h2>
    <div class="body" id="timeline-body">
      <div id="timeline-legend">
        <span><span class="swatch" style="background:#5fa8d3"></span>audio window</span>
        <span><span class="swatch" style="background:#6abf4b"></span>segment</span>
        <span><span class="swatch" style="background:#7e57c2"></span>word</span>
        <span><span class="swatch" style="background:#f0a04b"></span>press / marker</span>
        <span><span class="swatch" style="background:#d64545"></span>paste</span>
        <span class="muted" id="timeline-window-label"></span>
      </div>
      <svg id="timeline-svg" viewBox="0 0 1000 220" preserveAspectRatio="none"></svg>
    </div>
  </section>
</main>

<button id="settings-gear" title="Settings" aria-label="Open settings">⚙</button>

<div id="settings-overlay" hidden>
  <div id="settings-modal" role="dialog" aria-labelledby="settings-title">
    <header>
      <h3 id="settings-title">Settings</h3>
      <button id="settings-close" aria-label="Close">×</button>
    </header>
    <div class="modal-body">
      <p class="hint">
        Model / beam / prev-text changes hot-swap live (brief pause while the
        new model loads). Other fields apply on next launch.
      </p>

      <fieldset>
        <legend>General</legend>
        <label class="checkbox">
          <input type="checkbox" data-path="persistent.hf_hub_offline">
          Skip HuggingFace update check at model load
          <span class="deferred-note">(uncheck to allow downloads — slows startup; requires restart)</span>
        </label>
        <label>Chunk token target
          <input type="number" min="100" max="20000" step="100" data-path="persistent.chunk_token_target">
        </label>
        <label>Paste delay (s)
          <input type="number" step="0.05" min="0" max="2" data-path="paste.delay">
        </label>
        <label>Paste drain timeout (s)
          <input type="number" step="0.5" min="1" max="60" data-path="paste.drain_timeout">
        </label>
        <label class="checkbox">
          <input type="checkbox" data-path="persistent.debug_recording">
          Record debug events + status snapshots for replay
          <span class="deferred-note">(applies on next launch)</span>
        </label>
        <label>Snapshot interval (s)
          <input type="number" step="0.5" min="0.5" max="60" data-path="persistent.debug_snapshot_interval_s">
        </label>
      </fieldset>

      <fieldset>
        <legend>Audio capture</legend>
        <label>Sample rate (Hz)
          <input type="number" min="8000" max="48000" step="1000" data-path="audio.sample_rate">
        </label>
        <label>Channels
          <input type="number" min="1" max="2" data-path="audio.channels">
        </label>
        <label>Format
          <select data-path="audio.format">
            <option>int16</option><option>int32</option><option>float32</option>
          </select>
        </label>
        <label>Chunk size (frames)
          <input type="number" min="64" max="8192" step="64" data-path="audio.chunk_size">
        </label>
      </fieldset>

      <fieldset>
        <legend>VAD &amp; gating</legend>
        <label>Silence ms
          <input type="number" min="100" max="5000" step="50" data-path="vad.silence_ms">
        </label>
        <label>Aggressiveness (0–3)
          <input type="number" min="0" max="3" data-path="vad.aggressiveness">
        </label>
        <label>Min voiced ms
          <input type="number" min="0" max="5000" step="50" data-path="vad.min_voiced_ms">
        </label>
        <label>Min voiced fraction (0–1)
          <input type="number" min="0" max="1" step="0.05" data-path="vad.min_voiced_frac">
        </label>
      </fieldset>

      <fieldset>
        <legend>Fast stream</legend>
        <label class="checkbox">
          <input type="checkbox" data-path="persistent.fast.enabled">
          Enabled
        </label>
        <label>Model
          <select data-path="persistent.fast.fw.model">
            <option>tiny.en</option><option>base.en</option>
            <option>small.en</option><option>medium.en</option>
            <option>large-v3</option><option>distil-large-v3</option>
          </select>
        </label>
        <label>Compute type
          <select data-path="persistent.fast.fw.compute">
            <option>int8</option><option>int8_float16</option>
            <option>float16</option><option>float32</option>
          </select>
        </label>
        <label>Device
          <select data-path="persistent.fast.fw.device">
            <option value="">auto-detect</option>
            <option>cpu</option><option>cuda</option>
          </select>
        </label>
        <label>Beam size
          <input type="number" min="1" max="10" data-path="persistent.fast.fw.beam_size">
        </label>
        <label class="checkbox">
          <input type="checkbox" data-path="persistent.fast.fw.condition_on_previous_text">
          Condition on previous text
        </label>
        <label>No-speech threshold
          <input type="number" min="0" max="1" step="0.05" data-path="persistent.fast.fw.no_speech_threshold">
        </label>
        <label>Log-prob threshold
          <input type="number" step="0.1" data-path="persistent.fast.fw.log_prob_threshold">
        </label>
        <label>Max window seconds
          <input type="number" step="0.1" min="5" max="120" data-path="persistent.fast.max_window_seconds">
        </label>
        <label>Window queue maxsize (drops on full)
          <input type="number" min="1" max="512" data-path="persistent.fast.window_q_maxsize">
        </label>
      </fieldset>

      <fieldset>
        <legend>HQ stream (background, higher quality)</legend>
        <label class="checkbox">
          <input type="checkbox" data-path="persistent.hq.enabled">
          Enable HQ stream
        </label>
        <label>Model
          <select data-path="persistent.hq.fw.model">
            <option>tiny.en</option><option>base.en</option>
            <option>small.en</option><option>medium.en</option>
            <option>large-v3</option><option>distil-large-v3</option>
          </select>
        </label>
        <label>Compute type
          <select data-path="persistent.hq.fw.compute">
            <option>int8</option><option>int8_float16</option>
            <option>float16</option><option>float32</option>
          </select>
        </label>
        <label>Device
          <select data-path="persistent.hq.fw.device">
            <option value="">auto-detect</option>
            <option>cpu</option><option>cuda</option>
          </select>
        </label>
        <label>Beam size
          <input type="number" min="1" max="10" data-path="persistent.hq.fw.beam_size">
        </label>
        <label class="checkbox">
          <input type="checkbox" data-path="persistent.hq.fw.condition_on_previous_text">
          Condition on previous text
        </label>
        <label>No-speech threshold
          <input type="number" min="0" max="1" step="0.05" data-path="persistent.hq.fw.no_speech_threshold">
        </label>
        <label>Log-prob threshold
          <input type="number" step="0.1" data-path="persistent.hq.fw.log_prob_threshold">
        </label>
        <label>Max window seconds
          <input type="number" step="0.1" min="5" max="240" data-path="persistent.hq.max_window_seconds">
        </label>
        <label>Window queue maxsize (drops on full)
          <input type="number" min="1" max="512" data-path="persistent.hq.window_q_maxsize">
        </label>
      </fieldset>

      <fieldset>
        <legend>Ports</legend>
        <label>Widget port (0 = ephemeral)
          <input type="number" min="0" max="65535" data-path="persistent.widget_port">
        </label>
        <label>Transcript stream port
          <input type="number" min="0" max="65535" data-path="persistent.transcript_stream_port">
        </label>
      </fieldset>

      <fieldset>
        <legend>Markers</legend>
        <p class="hint" style="margin:0 0 8px">
          Hotkey keys that toggle annotation markers in the transcript.
        </p>
        <table id="markers-table" style="width:100%;font-size:12px;border-collapse:collapse">
          <thead>
            <tr style="color:var(--muted);text-align:left">
              <th style="width:50px">Key</th>
              <th style="width:140px">Type</th>
              <th>Description</th>
              <th style="width:32px"></th>
            </tr>
          </thead>
          <tbody id="markers-tbody"></tbody>
        </table>
        <button type="button" id="markers-add"
                style="margin-top:6px;padding:4px 10px;background:#3a3d45;color:#fff;border:none;border-radius:4px;cursor:pointer">+ Add marker</button>
      </fieldset>

      <fieldset>
        <legend>Hotkeys</legend>
        <label class="checkbox">
          <input type="checkbox" data-path="hotkeys.no_modifiers">
          Bare double-tap (no modifier keys held)
        </label>
        <label>Modifiers (comma-separated)
          <input type="text" data-path="hotkeys.modifiers" data-kind="list">
        </label>
        <label>Toggle recording key
          <input type="text" maxlength="1" data-path="hotkeys.keys.toggle_recording">
        </label>
        <label>Discard recording key
          <input type="text" maxlength="1" data-path="hotkeys.keys.discard_recording">
        </label>
        <label>Toggle aside key
          <input type="text" maxlength="1" data-path="hotkeys.keys.toggle_aside">
        </label>
      </fieldset>

      <details>
        <summary>Raw JSON (advanced)</summary>
        <textarea id="settings-json" spellcheck="false"></textarea>
      </details>
    </div>
    <footer>
      <span id="settings-status"></span>
      <button id="settings-reload">Reload current</button>
      <button id="settings-save" class="primary">Save &amp; apply</button>
    </footer>
  </div>
</div>

<script>
(function() {
  const $ = (id) => document.getElementById(id);
  const escape = (s) => s.replace(/[&<>"']/g, c => (
    {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const MARKER_RE = /&lt;&lt;&lt;MARKER:([a-z_]+):(start|end)&gt;&gt;&gt;/g;

  // ---- Click-to-copy ---------------------------------------------------
  let toastTimer = null;
  function flashToast(msg) {
    const t = $('copy-toast');
    t.textContent = msg;
    t.classList.add('show');
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove('show'), 900);
  }
  async function copyText(text, el, label) {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      flashToast(`copied ${label} (${text.length}c)`);
      if (el) {
        el.classList.add('copied');
        setTimeout(() => el.classList.remove('copied'), 400);
      }
    } catch (e) {
      flashToast('copy failed (clipboard permission?)');
    }
  }

  // Plaintext-ize the transcript: strip the marker tokens so the copied
  // text is the actual content, not the raw <<<MARKER:...>>>; keep them
  // as inline [type:kind] hints so structure is still readable.
  function transcriptToPlain(text) {
    if (!text) return '';
    return text.replace(/<<<MARKER:([a-z_]+):(start|end)>>>/g,
                        (_, t, k) => `[${t}:${k}]`);
  }

  // Wire up collapse toggles. Both sections default to "collapsed" via HTML.
  document.querySelectorAll('section.collapsible').forEach(sec => {
    sec.querySelector('h2').addEventListener('click', () => {
      const next = sec.dataset.state === 'expanded' ? 'collapsed' : 'expanded';
      sec.dataset.state = next;
      // After the state flip the inner area resizes; re-pin to bottom so the
      // tail stays visible.
      const body = sec.querySelector('.body');
      requestAnimationFrame(() => { body.scrollTop = body.scrollHeight; });
    });
  });

  // For each scroll container, remember whether the user manually scrolled
  // away from the bottom. If they're at the bottom, we keep pinning; if
  // they've scrolled up to read, we leave them alone.
  const scrollState = new WeakMap();
  function attachScrollWatcher(body) {
    if (scrollState.has(body)) return;
    scrollState.set(body, { atBottom: true });
    body.addEventListener('scroll', () => {
      const slack = 8;
      const atBottom = (body.scrollHeight - body.clientHeight) <= (body.scrollTop + slack);
      scrollState.get(body).atBottom = atBottom;
    });
  }
  function pinToBottomIfStuck(body) {
    const state = scrollState.get(body);
    if (!state || state.atBottom) {
      body.scrollTop = body.scrollHeight;
    }
  }

  let lastTranscript = '';
  function renderSpanHtml(text, sourceClass) {
    const escaped = escape(text);
    const withMarkers = escaped.replace(MARKER_RE, (m, type, kind) => {
      const cls = ['marker-pill', type].join(' ');
      return `<span class="${cls}" title="${type}:${kind}">${type}:${kind}</span>`;
    });
    return `<span class="${sourceClass}">${withMarkers}</span>`;
  }
  function renderTranscript(s) {
    const el = $('transcript');
    const body = $('transcript-body');
    attachScrollWatcher(body);
    // Prefer merged spans when the HQ stream is producing them; fall back
    // to the plain fast tail otherwise.
    const spans = s.transcript_tail_merged;
    const plain = s.transcript_tail || '';
    const isEmpty = spans
      ? spans.every(sp => !sp.text)
      : !plain;
    lastTranscript = spans
      ? spans.map(sp => sp.text).join(' ')
      : plain;
    if (isEmpty) {
      el.className = 'empty copyable';
      el.textContent = 'no audio yet';
      return;
    }
    el.className = 'copyable';
    el.title = 'click to copy full transcript tail';
    if (spans && spans.length) {
      const parts = [];
      for (let i = 0; i < spans.length; i++) {
        const sp = spans[i];
        if (i > 0 && spans[i - 1].source === 'hq' && sp.source === 'fast') {
          const edgeSec = s.hq && s.hq.leading_edge_seconds != null
            ? s.hq.leading_edge_seconds.toFixed(1) + 's'
            : '';
          parts.push(`<span class="hq-edge" title="HQ leading edge">HQ→${edgeSec}</span>`);
        }
        const cls = sp.source === 'hq' ? 'src-hq' : 'src-fast';
        parts.push(renderSpanHtml(sp.text || '', cls));
      }
      el.innerHTML = parts.join(' ');
    } else {
      el.innerHTML = renderSpanHtml(plain, 'src-fast');
    }
    pinToBottomIfStuck(body);
  }
  // Single delegated click handler for the transcript element.
  document.addEventListener('DOMContentLoaded', () => {});
  // Attach immediately — the element exists at parse time.
  $('transcript').addEventListener('click', (ev) => {
    // Avoid stealing text-selection clicks.
    const sel = window.getSelection();
    if (sel && sel.toString().length > 0) return;
    copyText(transcriptToPlain(lastTranscript), $('transcript'), 'transcript');
  });

  function renderPastes(pastes) {
    const body = $('pastes-body');
    attachScrollWatcher(body);
    if (!pastes || pastes.length === 0) {
      body.innerHTML = '<div class="empty">no pastes yet</div>';
      return;
    }
    // Server sends chronological (oldest first, newest last). Render as-is
    // so the tail is at the bottom of the scrolling region.
    const rows = pastes.map((p, i) => {
      const labelCls = p.label === 'r' ? 'r' : 'aside';
      return `<tr data-idx="${i}" title="click to copy">` +
             `<td class="ts">${escape(p.ts)}</td>` +
             `<td class="label ${labelCls}">${escape(p.label)}</td>` +
             `<td class="preview">${escape(p.preview)}</td>` +
             `</tr>`;
    }).join('');
    body.innerHTML = `<table class="pastes"><tbody>${rows}</tbody></table>`;
    body.querySelectorAll('tr[data-idx]').forEach(tr => {
      tr.addEventListener('click', (ev) => {
        const sel = window.getSelection();
        if (sel && sel.toString().length > 0) return;
        const idx = parseInt(tr.dataset.idx, 10);
        const p = pastes[idx];
        copyText(p.full || p.preview || '', tr, `paste ${p.label}`);
      });
    });
    pinToBottomIfStuck(body);
  }

  function renderHeader(s) {
    let meta = `${s.session_id} · ${s.model} (${s.device}) · ${s.chunk_count} chunk(s)`;
    if (s.hq) {
      const now = s.now_seconds || 0;
      const edge = s.hq.leading_edge_seconds || 0;
      const lag = Math.max(0, now - edge);
      meta += ` · HQ:${s.hq.model.replace(/^faster-whisper:/, '')} (lag ${lag.toFixed(1)}s)`;
    }
    // Per-stream drop counters. Only render when something has actually
    // been dropped so the header stays clean in healthy sessions.
    if (s.stream_stats) {
      for (const label of Object.keys(s.stream_stats)) {
        const st = s.stream_stats[label];
        const totalDrops = (st.dropped_queue_full || 0) + (st.dropped_silent || 0);
        if (totalDrops > 0) {
          meta += ` · ⚠ ${label} dropped ${st.dropped_queue_full || 0}q+${st.dropped_silent || 0}s`;
        }
      }
    }
    $('meta').textContent = meta;
    const cap = $('capture');
    cap.textContent = s.capture.mode;
    cap.className = 'pill ' + (s.capture.mode === 'r+aside' ? 'r-aside' : s.capture.mode);
    const muted = !!(s.capture && s.capture.muted);
    $('mute').style.display = muted ? '' : 'none';
    document.body.classList.toggle('is-muted', muted);
    const m = $('marker');
    if (s.open_marker) {
      m.textContent = s.open_marker + (s.open_marker_since ? ` (since ${s.open_marker_since})` : '');
      m.className = 'pill marker';
    } else {
      m.textContent = 'none';
      m.className = 'pill none';
    }
    const secs = Math.floor(s.uptime_seconds || 0);
    const hh = String(Math.floor(secs / 3600)).padStart(2, '0');
    const mm = String(Math.floor((secs % 3600) / 60)).padStart(2, '0');
    const ss = String(secs % 60).padStart(2, '0');
    $('uptime').textContent = `${hh}:${mm}:${ss}`;
  }

  // ---- Debug events ---------------------------------------------------
  const EVENT_KINDS = ['audio_window','segment','press','marker','cursor_capture','paste','mute'];
  const enabledKinds = new Set(EVENT_KINDS);
  document.querySelectorAll('.chip[data-kind]').forEach(c => {
    c.addEventListener('click', () => {
      const k = c.dataset.kind;
      if (enabledKinds.has(k)) { enabledKinds.delete(k); c.classList.remove('on'); }
      else { enabledKinds.add(k); c.classList.add('on'); }
    });
  });

  function summarize(evt) {
    const d = evt.data || {};
    switch (evt.kind) {
      case 'audio_window':
        return `[${d.start?.toFixed(2)}–${d.end?.toFixed(2)}s] ${d.reason}` +
               ` voiced=${d.voiced_ms}ms (${(d.voiced_frac*100).toFixed(0)}%)` +
               (d.trigger ? ` trigger=${d.trigger}` : '');
      case 'segment':
        const wc = (d.words||[]).length;
        return `"${d.text}" [${d.start?.toFixed(2)}–${d.end?.toFixed(2)}s] ${wc}w`;
      case 'press':
        return `${d.key} ${d.phase} @${d.session_time?.toFixed(2)}s`;
      case 'marker':
        return `${d.type} ${d.action} (key=${d.key||'-'})`;
      case 'cursor_capture':
        return `${d.label} press@${d.press_time?.toFixed(2)}s landed in ${d.fired_latency_ms}ms cur=${d.cursor}`;
      case 'paste':
        return `${d.label} ${d.char_count}c drain=${d.drain_wait_ms}ms` +
               (d.timed_out ? ' TIMEOUT' : '') +
               ` press=[${d.start_press_time?.toFixed(2)}–${d.end_press_time?.toFixed(2)}s]\\n${d.preview}`;
      case 'mute':
        return d.state;
      default:
        return JSON.stringify(d);
    }
  }

  let lastEventsRendered = [];
  function renderEvents(events) {
    const host = $('events-table');
    attachScrollWatcher($('events-body'));
    if (!events || events.length === 0) {
      host.innerHTML = '<div class="empty">no events yet</div>';
      lastEventsRendered = [];
      return;
    }
    const filtered = events.filter(e => enabledKinds.has(e.kind)).slice(-300);
    lastEventsRendered = filtered;
    const rows = filtered.map(e => (
      `<tr class="kind-${e.kind}">` +
      `<td class="ts">${e.ts.toFixed(2)}</td>` +
      `<td class="kind">${e.kind}</td>` +
      `<td class="summary">${escape(summarize(e))}</td>` +
      `</tr>`
    )).join('');
    host.innerHTML = `<table class="events"><tbody>${rows}</tbody></table>`;
    pinToBottomIfStuck($('events-body'));
  }
  function eventsToPlain(evts) {
    return evts.map(e =>
      `${e.ts.toFixed(2).padStart(7)}  ${e.kind.padEnd(15)} ${summarize(e)}`
    ).join('\\n');
  }
  $('events-table').addEventListener('click', (ev) => {
    const sel = window.getSelection();
    if (sel && sel.toString().length > 0) return;
    if (!lastEventsRendered.length) return;
    copyText(eventsToPlain(lastEventsRendered), $('events-table'),
             `${lastEventsRendered.length} events`);
  });

  // ---- Timeline -------------------------------------------------------
  // Last WINDOW_SECONDS of session-relative time, rendered as parallel lanes.
  const WINDOW_SECONDS = 60;
  const LANES = {
    audio_window: { y: 20,  h: 20, color: '#5fa8d3' },
    segment:      { y: 50,  h: 20, color: '#6abf4b' },
    word:         { y: 75,  h: 8,  color: '#7e57c2' },
    paste:        { y: 95,  h: 20, color: '#d64545' },
    press:        { y: 0,   h: 220, color: '#f0a04b' },
    marker:       { y: 130, h: 20, color: '#f0a04b' },
    cursor_capture: { y: 160, h: 12, color: '#c792ea' },
  };

  function renderTimeline(events, nowSec) {
    const svg = $('timeline-svg');
    const W = 1000, H = 220;
    const tEnd = nowSec || (events.length ? events[events.length-1].ts : 0);
    const tStart = Math.max(0, tEnd - WINDOW_SECONDS);
    const span = Math.max(0.01, tEnd - tStart);
    const x = (t) => ((t - tStart) / span) * W;
    $('timeline-window-label').textContent =
      ` window: ${tStart.toFixed(1)}s → ${tEnd.toFixed(1)}s (${WINDOW_SECONDS}s)`;

    const parts = [];
    // Grid: 5s ticks
    for (let t = Math.ceil(tStart/5)*5; t <= tEnd; t += 5) {
      const xx = x(t);
      parts.push(`<line x1="${xx}" y1="0" x2="${xx}" y2="${H}" stroke="#1b1d23" stroke-width="1"/>`);
      parts.push(`<text x="${xx+2}" y="${H-4}" fill="#8a8d97" font-size="10">${t.toFixed(0)}s</text>`);
    }

    for (const e of events) {
      const d = e.data || {};
      if (e.kind === 'audio_window' && d.start != null && d.end != null) {
        const x0 = x(d.start), x1 = Math.max(x(d.end), x0+1);
        const isDrop = d.reason && d.reason.startsWith('dropped');
        const fill = d.reason === 'forced' ? '#ffa726'
                   : d.reason === 'silence' ? '#5fa8d3'
                   : d.reason === 'max_window' ? '#ec407a'
                   : d.reason === 'dropped_silent' ? '#3a3d45'
                   : d.reason === 'dropped_queue_full' ? '#d64545'
                   : '#5fa8d3';
        const opacity = isDrop ? 0.5 : 0.85;
        const stroke = isDrop ? ' stroke="#ff5252" stroke-width="1.5" stroke-dasharray="3,2"' : '';
        const title = escape(summarize(e));
        parts.push(`<rect x="${x0}" y="20" width="${x1-x0}" height="20" fill="${fill}" opacity="${opacity}"${stroke}><title>${title}</title></rect>`);
        if (isDrop) {
          // Diagonal X across the bar so a dropped window is unmistakable
          // even at a glance, plus a small ⚠ glyph just above it tagged
          // with the drop reason.
          parts.push(`<line x1="${x0}" y1="20" x2="${x1}" y2="40" stroke="#ff5252" stroke-width="1.5"><title>${title}</title></line>`);
          parts.push(`<line x1="${x0}" y1="40" x2="${x1}" y2="20" stroke="#ff5252" stroke-width="1.5"><title>${title}</title></line>`);
          const sym = d.reason === 'dropped_queue_full' ? '⚠Q' : '⚠S';
          parts.push(`<text x="${x0+2}" y="18" fill="#ff5252" font-size="9" font-weight="bold">${sym}</text>`);
        }
      } else if (e.kind === 'segment' && d.start != null && d.end != null) {
        const x0 = x(d.start), x1 = Math.max(x(d.end), x0+1);
        parts.push(`<rect x="${x0}" y="50" width="${x1-x0}" height="20" fill="#6abf4b" opacity="0.7"><title>${escape(summarize(e))}</title></rect>`);
        for (const w of (d.words||[])) {
          const wx = x(w.s);
          parts.push(`<line x1="${wx}" y1="75" x2="${wx}" y2="83" stroke="#7e57c2" stroke-width="1"><title>${escape(w.text)} [${w.s.toFixed(2)}–${w.e.toFixed(2)}] p=${w.p}</title></line>`);
        }
      } else if (e.kind === 'press') {
        const xx = x(d.session_time);
        const stroke = d.phase === 'start' ? '#6abf4b' : d.phase === 'end' ? '#d64545' : '#f0a04b';
        parts.push(`<line x1="${xx}" y1="0" x2="${xx}" y2="${H-15}" stroke="${stroke}" stroke-width="2" opacity="0.8"><title>${escape(summarize(e))}</title></line>`);
        parts.push(`<text x="${xx+2}" y="12" fill="${stroke}" font-size="10">${d.key}/${d.phase}</text>`);
      } else if (e.kind === 'paste') {
        const x0 = x(d.start_press_time), x1 = Math.max(x(d.end_press_time), x0+1);
        parts.push(`<rect x="${x0}" y="95" width="${x1-x0}" height="20" fill="#d64545" opacity="${d.timed_out?0.4:0.7}" stroke="#fff" stroke-width="0.5"><title>${escape(summarize(e))}</title></rect>`);
      } else if (e.kind === 'marker') {
        // Markers don't carry timing; place at event ts
        const xx = x(e.ts);
        parts.push(`<circle cx="${xx}" cy="140" r="4" fill="#f0a04b"><title>${escape(summarize(e))}</title></circle>`);
      } else if (e.kind === 'cursor_capture') {
        const xx = x(d.press_time);
        parts.push(`<circle cx="${xx}" cy="166" r="3" fill="#c792ea"><title>${escape(summarize(e))}</title></circle>`);
      }
    }
    svg.innerHTML = parts.join('');
  }

  async function poll() {
    // Abort hung polls so a wedged keep-alive connection can't permanently
    // jam the per-origin pool (manifests as a stuck "disconnected" banner
    // that hard-refresh can't recover from, but a new browser can).
    const ctrl = new AbortController();
    const tid = setTimeout(() => ctrl.abort(), 2000);
    try {
      const r = await fetch('/status', { cache: 'no-store', signal: ctrl.signal });
      if (!r.ok) throw new Error('status ' + r.status);
      const s = await r.json();
      $('disconnected').style.display = 'none';
      renderHeader(s);
      renderPastes(s.paste_log);
      renderTranscript(s);
      renderEvents(s.debug_events || []);
      renderTimeline(s.debug_events || [], s.now_seconds || 0);
    } catch (e) {
      $('disconnected').style.display = 'block';
    } finally {
      clearTimeout(tid);
    }
  }
  poll();
  setInterval(poll, 500);
  // Background tabs get setInterval throttled to ~1/min on macOS, which
  // leaves the "disconnected" banner stuck until the next throttled tick.
  // Kick a fresh poll the moment the tab is visible or focused again so
  // reconnection is instant on focus instead of waiting up to a minute.
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) poll();
  });
  window.addEventListener('focus', poll);

  // ---- Settings: gear + modal -----------------------------------------
  async function loadConfig() {
    const r = await fetch('/config', { cache: 'no-store' });
    if (!r.ok) throw new Error('GET /config ' + r.status);
    return await r.json();
  }
  function setSettingsStatus(msg, isError) {
    const el = $('settings-status');
    el.textContent = msg;
    el.style.color = isError ? 'var(--bad)' : 'var(--good)';
  }
  function getByPath(obj, path) {
    return path.split('.').reduce((o, k) => (o == null ? undefined : o[k]), obj);
  }
  function setByPath(obj, path, value) {
    const parts = path.split('.');
    let cur = obj;
    for (let i = 0; i < parts.length - 1; i++) {
      if (cur[parts[i]] == null || typeof cur[parts[i]] !== 'object') {
        cur[parts[i]] = {};
      }
      cur = cur[parts[i]];
    }
    cur[parts[parts.length - 1]] = value;
  }
  function fillFormFromConfig(cfg) {
    document.querySelectorAll('#settings-modal [data-path]').forEach(el => {
      const path = el.dataset.path;
      const v = getByPath(cfg, path);
      if (v === undefined) return;
      if (el.type === 'checkbox') el.checked = !!v;
      else if (el.dataset.kind === 'list' && Array.isArray(v)) el.value = v.join(', ');
      else el.value = v;
    });
    renderMarkersTable(Array.isArray(cfg.markers) ? cfg.markers : []);
    $('settings-json').value = JSON.stringify(cfg, null, 2);
  }
  function buildPatchFromForm() {
    // Send only the structured-form fields. The advanced JSON textarea is
    // a read-only mirror unless the user expanded it (in which case we
    // merge it as base and overlay the form values on top).
    let patch = {};
    const raw = $('settings-json').value.trim();
    if (raw) {
      try { patch = JSON.parse(raw); } catch (e) { /* fall through */ }
    }
    document.querySelectorAll('#settings-modal [data-path]').forEach(el => {
      const path = el.dataset.path;
      let v;
      if (el.type === 'checkbox') v = el.checked;
      else if (el.type === 'number') v = el.value === '' ? null : Number(el.value);
      else if (el.dataset.kind === 'list') {
        v = el.value.split(',').map(s => s.trim()).filter(Boolean);
      } else v = el.value;
      if (v === null) return;
      setByPath(patch, path, v);
    });
    patch.markers = readMarkersTable();
    return patch;
  }
  // ---- Markers table editor (dynamic rows) ----
  function renderMarkersTable(markers) {
    const tbody = $('markers-tbody');
    tbody.innerHTML = '';
    markers.forEach(m => tbody.appendChild(buildMarkerRow(m)));
  }
  function buildMarkerRow(m) {
    const tr = document.createElement('tr');
    const cellStyle = 'padding:3px 4px';
    const inputStyle =
      'width:100%;background:#15171c;color:var(--fg);' +
      'border:1px solid #3a3d45;border-radius:3px;padding:3px 6px;' +
      'font-size:12px;box-sizing:border-box';
    tr.innerHTML =
      `<td style="${cellStyle}"><input class="m-key" maxlength="1" value="${escape(m.key || '')}" style="${inputStyle}"></td>` +
      `<td style="${cellStyle}"><input class="m-type" value="${escape(m.type || '')}" style="${inputStyle}"></td>` +
      `<td style="${cellStyle}"><input class="m-desc" value="${escape(m.description || '')}" style="${inputStyle}"></td>` +
      `<td style="${cellStyle};text-align:center">` +
      `<button type="button" class="m-del" title="remove" ` +
      `style="background:transparent;border:none;color:var(--bad);cursor:pointer;font-size:16px;line-height:1">×</button></td>`;
    tr.querySelector('.m-del').addEventListener('click', () => tr.remove());
    return tr;
  }
  function readMarkersTable() {
    const out = [];
    $('markers-tbody').querySelectorAll('tr').forEach(tr => {
      const key = tr.querySelector('.m-key').value.trim();
      const type = tr.querySelector('.m-type').value.trim();
      const description = tr.querySelector('.m-desc').value.trim();
      if (key && type) out.push({ key, type, description });
    });
    return out;
  }
  $('markers-add').addEventListener('click', () => {
    $('markers-tbody').appendChild(buildMarkerRow({ key: '', type: '', description: '' }));
  });
  async function refreshSettingsForm() {
    try {
      const cfg = await loadConfig();
      fillFormFromConfig(cfg);
      setSettingsStatus('loaded', false);
    } catch (e) {
      setSettingsStatus('load failed: ' + e.message, true);
    }
  }
  function openSettings() {
    $('settings-overlay').hidden = false;
    refreshSettingsForm();
  }
  function closeSettings() {
    $('settings-overlay').hidden = true;
  }
  $('settings-gear').addEventListener('click', openSettings);
  $('settings-close').addEventListener('click', closeSettings);
  $('settings-overlay').addEventListener('click', (ev) => {
    if (ev.target === $('settings-overlay')) closeSettings();
  });
  document.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape' && !$('settings-overlay').hidden) closeSettings();
  });
  $('settings-reload').addEventListener('click', refreshSettingsForm);
  $('settings-save').addEventListener('click', async () => {
    const patch = buildPatchFromForm();
    setSettingsStatus('saving…', false);
    try {
      const r = await fetch('/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patch),
      });
      const result = await r.json();
      if (!r.ok || result.error) {
        setSettingsStatus('error: ' + (result.error || r.status), true);
        return;
      }
      const applied = (result.applied || []).join(', ') || '(no live changes)';
      const deferred = (result.deferred || []).join(', ');
      let msg = 'saved. live: ' + applied;
      if (deferred) msg += '. deferred until restart: ' + deferred;
      setSettingsStatus(msg, false);
    } catch (e) {
      setSettingsStatus('error: ' + e.message, true);
    }
  });
})();
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    # Suppress per-request access logs — the 2 Hz poll would spam the console.
    def log_message(self, format, *args):
        return

    def handle_one_request(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        # Swallow socket-close races: when the dashboard reloads or closes a
        # tab mid-response, the 2 Hz poll otherwise spams a traceback.
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path == "/" or self.path.startswith("/?"):
            body = _INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/status":
            try:
                payload = self.server.snapshot_provider()  # type: ignore[attr-defined]
            except Exception as e:
                self._send_json(500, {"error": str(e)})
                return
            self._send_json(200, payload)
            return
        if self.path == "/config":
            provider = getattr(self.server, "config_provider", None)
            if provider is None:
                self._send_json(404, {"error": "config not available"})
                return
            try:
                self._send_json(200, provider())
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            return
        self.send_error(404)

    def do_PUT(self):  # noqa: N802
        if self.path == "/config":
            setter = getattr(self.server, "config_setter", None)
            if setter is None:
                self._send_json(404, {"error": "config update not available"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                new_data = json.loads(raw.decode("utf-8"))
            except Exception as e:
                self._send_json(400, {"error": f"bad JSON: {e}"})
                return
            try:
                result = setter(new_data)
            except Exception as e:
                self._send_json(500, {"error": str(e)})
                return
            self._send_json(200, {"ok": True, **(result or {})})
            return
        self.send_error(404)


class _Server(ThreadingHTTPServer):
    """Slight subclass that holds the snapshot/config callables for the handler."""

    daemon_threads = True

    def __init__(self, addr, handler, snapshot_provider, config_provider=None,
                 config_setter=None):
        super().__init__(addr, handler)
        self.snapshot_provider = snapshot_provider
        self.config_provider = config_provider
        self.config_setter = config_setter


class StatusServer:
    """HTTP status server bound to a local port.

    The dashboard reads from ``snapshot_provider``, a zero-arg callable
    returning a JSON-serializable dict. Optional ``config_provider`` and
    ``config_setter`` enable the ``/config`` GET/PUT endpoints used by the
    settings page in Phase 6.
    """

    def __init__(
        self,
        snapshot_provider: Callable[[], dict],
        config_provider: Optional[Callable[[], dict]] = None,
        config_setter: Optional[Callable[[dict], Optional[dict]]] = None,
    ):
        self._snapshot_provider = snapshot_provider
        self._config_provider = config_provider
        self._config_setter = config_setter
        self._server: Optional[_Server] = None
        self._thread: Optional[threading.Thread] = None

    def start(self, host: str = "127.0.0.1", port: int = 0) -> tuple[str, int]:
        self._server = _Server(
            (host, port), _Handler, self._snapshot_provider,
            config_provider=self._config_provider,
            config_setter=self._config_setter,
        )
        bound_host, bound_port = self._server.server_address[:2]
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="StatusServer", daemon=True
        )
        self._thread.start()
        return bound_host, bound_port

    def stop(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
