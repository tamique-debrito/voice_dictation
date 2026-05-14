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
import os
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

  /* Staleness chip: a small inline indicator next to the title that shows
     how old the displayed data is. Replaces the old binary "disconnected"
     banner that flipped on a single failed fetch. We only ever blank panels
     once data is truly long-stale (see #stale-banner). */
  #staleness {
    display: none;
    margin-left: 8px;
    padding: 1px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 500;
    vertical-align: middle;
  }
  #staleness.live  { display: none; }
  #staleness.soft  { display: inline-block; background: #3a3d45; color: var(--muted); }
  #staleness.warn  { display: inline-block; background: #6a5a1a; color: #f0e2a5; }
  #staleness.alert { display: inline-block; background: #6a3a1a; color: #ffd2a8; }
  #stale-banner {
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
  .timeline-controls {
    padding: 6px 12px;
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    align-items: center;
    border-bottom: 1px solid #1b1d23;
    background: var(--panel-2);
    font-size: 11px;
  }
  .timeline-controls .filter-label { color: var(--muted); }
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
  /* Stream chip colors are chosen NOT to overlap with any audio_window
     reason color (forced=orange, silence=blue, max=pink, drops=red/grey)
     or the press/marker color. Stream identity is conveyed by row
     position + shape, not by per-bar color. */
  .chip.stream-fast.on { background: #607d8b; color: #fff; border-color: #607d8b; }
  .chip.stream-hq.on   { background: #26a69a; color: #fff; border-color: #26a69a; }
  .chip-sep { display: inline-block; width: 1px; height: 14px; background: #3a3d45;
              vertical-align: middle; margin: 0 4px; }
  .filter-label { color: var(--muted); font-size: 11px; margin-right: 2px; }
  .legend-swatch { display: inline-block; width: 12px; height: 10px; vertical-align: middle;
                   margin-right: 2px; border-radius: 2px; }
  .legend-lbl { color: var(--muted); font-size: 10px; margin-right: 6px; }
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
  table.events tr.kind-debug_flag td.kind { color: #ff5252; font-weight: 800; }
  table.events tr.kind-debug_flag { background: rgba(255, 82, 82, 0.08); }

  /* Timeline body lays out a collapsible label gutter to the LEFT and
     the SVG to the right. The gutter labels each y-band with plain text;
     stream identity in the bars themselves is conveyed by row position
     plus the HQ-only teal-dotted-top shape signature — no per-stream
     coloring in the gutter, because the audio_window band's bars can
     take many different colors based on reason. */
  .timeline-row {
    display: flex;
    align-items: stretch;
    background: var(--panel-2);
  }
  #timeline-labels {
    flex: 0 0 120px;
    background: var(--panel);
    border-right: 1px solid #1b1d23;
    position: relative;
    height: 220px;
    font-size: 10px;
    color: var(--muted);
    transition: flex-basis 0.15s ease;
  }
  #timeline-labels.collapsed { flex-basis: 18px; }
  #timeline-labels.collapsed .row-label { display: none; }
  .row-label {
    position: absolute;
    left: 6px;
    right: 4px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    white-space: nowrap;
    /* line-height: 1 + explicit font-size so the label's text center
       sits exactly font-size/2 below its `top`. Each label's `top:` is
       then computed as (bar_center − font_size/2) so the text center
       lines up with the bar center in the SVG to the right. Default
       line-height (~1.2) would drift labels down ~1px per row,
       accumulating into a visible misalignment by the time you reach
       the segment / paste bands. */
    line-height: 1;
    font-size: 9px;
  }
  .row-label.group {
    font-weight: 700;
    color: #b6bac1;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 7px;
  }
  .row-label.sub {
    padding-left: 12px;
  }
  .press-tri-down {
    display: inline-block; width: 0; height: 0;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 6px solid #6abf4b; vertical-align: middle; margin-right: 2px;
  }
  .press-tri-down.end { border-top-color: #d64545; }
  #timeline-labels-toggle {
    position: absolute;
    top: 2px;
    right: 2px;
    width: 14px;
    height: 14px;
    background: #3a3d45;
    color: var(--muted);
    border-radius: 3px;
    text-align: center;
    line-height: 14px;
    cursor: pointer;
    user-select: none;
    font-size: 10px;
  }
  #timeline-labels-toggle:hover { background: #4a4d55; }
  #timeline-svg {
    flex: 1 1 auto;
    background: var(--panel-2);
    display: block;
    height: 220px;
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
  /* Swatch variants — shapes that mirror what the SVG actually draws,
     so legend ≈ what you see in the timeline. */
  #timeline-legend .swatch-rect   { width: 14px; height: 6px; border-radius: 1px; }
  #timeline-legend .swatch-stick  { width: 1px; height: 10px; }
  #timeline-legend .swatch-circle { width: 8px; height: 8px; border-radius: 50%; }
  #timeline-legend .swatch-tri {
    width: 0; height: 0; background: none;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 6px solid transparent;
    border-radius: 0;
  }
  #timeline-legend .swatch-tri.tri-green { border-top-color: #6abf4b; }
  #timeline-legend .swatch-tri.tri-red   { border-top-color: #d64545; }
  #timeline-legend .swatch-flag {
    width: auto; height: auto; background: none !important; border-radius: 0;
    font-size: 12px; line-height: 1;
  }
</style>
</head>
<body>
<header>
  <div>
    <span class="title">🎤 Persistent Monitoring</span>
    <span id="staleness"></span>
    <span class="meta" id="meta"></span>
  </div>
  <div>capture: <span id="capture" class="pill passive">passive</span></div>
  <div>marker:  <span id="marker" class="pill none">none</span></div>
  <div><span id="mute" class="pill muted" style="display:none">MUTED</span></div>
  <div class="meta" id="uptime"></div>
</header>
<div id="stale-banner">⚠ disconnected from monitoring service</div>
<div id="copy-toast">copied</div>
<div id="replay-audio-wrap" style="display:none; padding:6px 10px; background:#1b1f24; border-bottom:1px solid #2a2f36; display:flex; align-items:center; gap:10px;">
  <span style="font-size:11px; color:#9aa4ad;">🔊 session audio</span>
  <audio id="replay-audio" controls preload="auto" style="flex:1; height:32px;"></audio>
  <span id="replay-audio-status" style="font-size:11px; color:#6f7780;"></span>
</div>
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
        <span class="chip on" data-kind="debug_flag">🚩 debug_flag</span>
      </div>
      <div id="events-table" class="copyable" title="click to copy all visible events"><div class="empty">no events yet</div></div>
    </div>
  </section>
  <section id="timeline-section" class="collapsible" data-state="expanded">
    <h2><span class="chev">▸</span>Debug timeline</h2>
    <div class="body" id="timeline-body">
      <div id="timeline-legend">
        <span title="Transcribed audio window: horizontal bar spans the [start, end] audio time of one finalized segment. Hover the bar for the text."><span class="swatch swatch-rect" style="background:#6abf4b"></span>segment bar</span>
        <span title="Individual word from a segment, anchored at the word's start time. Useful for spotting per-word timing within a segment."><span class="swatch swatch-stick" style="background:#7e57c2"></span>word tick</span>
        <span title="Bare double-tap action key press: green = first tap (start), red = second tap (end). Each tap is logged whether or not it triggered an action."><span class="swatch swatch-tri tri-green"></span><span class="swatch swatch-tri tri-red"></span>press (start / end)</span>
        <span title="Marker action: opened or closed a recording/aside marker (or a custom marker type from settings)."><span class="swatch swatch-circle" style="background:#f0a04b"></span>marker</span>
        <span title="Cursor-position capture: records the focused app cursor location at the moment of a clipboard-window press, used to land the paste at the right spot."><span class="swatch swatch-circle" style="background:#c792ea"></span>cursor capture</span>
        <span title="Paste event: text typed into the focused app. Bar spans the press window; hover the bar for the pasted preview."><span class="swatch swatch-rect" style="background:#d64545"></span>paste</span>
        <span title="Debug flag: high-visibility bookmark stamped by double-tapping the debug-flag key (default 'e'). Use it to mark moments you want to revisit in post-session replay."><span class="swatch swatch-flag" style="color:#ff5252">🚩</span>debug flag</span>
        <span class="muted" id="timeline-window-label" title="Visible time range of the timeline (drag the SVG to pan; use the zoom controls to change span)."></span>
      </div>
      <div class="timeline-controls">
        <label class="filter-label">stream:</label>
        <span class="chip on stream-fast" data-stream="fast">fast</span>
        <span class="chip on stream-hq" data-stream="hq">hq</span>
        <span class="chip-sep"></span>
        <label class="filter-label">audio_window:</label>
        <span class="legend-swatch" style="background:#5fa8d3" title="VAD found a silence boundary → window cut and accepted"></span><span class="legend-lbl" title="VAD found a silence boundary → window cut and accepted">silence-cut</span>
        <span class="legend-swatch" style="background:#ec407a" title="Hit the max_window_seconds cap (no silence found in time) → window force-cut"></span><span class="legend-lbl" title="Hit the max_window_seconds cap (no silence found in time) → window force-cut">max_window</span>
        <span class="legend-swatch" style="background:#ffa726" title="Aggregator flushed early because a clipboard-window end-press requested an immediate flush"></span><span class="legend-lbl" title="Aggregator flushed early because a clipboard-window end-press requested an immediate flush">forced (end-press flush)</span>
        <span class="legend-swatch" style="background:#3a3d45;border-top:2px solid #ff5252" title="Window dropped because voiced_ms or voiced_frac fell below the VAD gating thresholds — not enough speech to bother transcribing"></span><span class="legend-lbl" title="Window dropped because voiced_ms or voiced_frac fell below the VAD gating thresholds">⚠S dropped (silent gate)</span>
        <span class="legend-swatch" style="background:#d64545;border-top:2px solid #ff5252" title="Window dropped because the transcriber's input queue was full — the transcriber is falling behind realtime (typically HQ during cold-start or backlog)"></span><span class="legend-lbl" title="Window dropped because the transcriber's input queue was full">⛔Q dropped (queue full)</span>
      </div>
      <div class="timeline-row">
        <div id="timeline-labels">
          <span id="timeline-labels-toggle" title="hide/show labels">‹</span>
          <!-- Each label's `top` = (bar-center y) - (font_size/2).
               With line-height:1 and explicit font sizes that gives
               text-center == bar-center.
               Bar geometry in the SVG (height: 220px, viewBox y match):
                 win  fast 20-28  (center 24) → font 9 → top 20
                 win  hq   30-38  (center 34) → top 30
                 seg  fast 46-54  (center 50) → top 46
                 seg  hq   66-74  (center 70) → top 66
                 paste    95-115  (center 105) → top 101
                 marker   y=140 circle r=4    → top 136
                 cursor   y=166 circle r=3    → top 162
               Group headers (font 7) live in the gaps between bands. -->
          <span class="row-label" style="top: 4px"><span class="press-tri-down"></span><span class="press-tri-down end"></span> Press</span>
          <span class="row-label group" style="top: 12px">Windows</span>
          <span class="row-label sub"   style="top: 20px">Fast</span>
          <span class="row-label sub"   style="top: 30px">HQ</span>
          <span class="row-label group" style="top: 39px">Segments</span>
          <span class="row-label sub"   style="top: 46px">Fast</span>
          <span class="row-label sub"   style="top: 66px">HQ</span>
          <span class="row-label"       style="top: 101px">Paste</span>
          <span class="row-label"       style="top: 136px">Marker</span>
          <span class="row-label"       style="top: 162px">Cursor capture</span>
        </div>
        <svg id="timeline-svg" viewBox="0 0 1000 220" preserveAspectRatio="none"></svg>
      </div>
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
        <label class="checkbox">
          <input type="checkbox" data-path="persistent.save_audio">
          Save per-chunk audio (chunk_NNN.wav) for annotation / fine-tune
          <span class="deferred-note">(applies on next launch)</span>
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
              <th style="width:60px" title="Flag = single-point marker (start+end at same time). Off = open/close interval.">Flag</th>
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
    const parts = [];
    parts.push(
      `<span title="Session id">${escape(s.session_id)}</span>` +
      ` · <span title="Fast-stream model running on this device">` +
      `${escape(s.model)} (${escape(s.device)})</span>` +
      ` · <span title="Number of transcript chunks finalized this session">` +
      `${s.chunk_count} chunk(s)</span>`
    );
    if (s.hq) {
      const now = s.now_seconds || 0;
      const edge = s.hq.leading_edge_seconds || 0;
      const lag = Math.max(0, now - edge);
      const model = s.hq.model.replace(/^faster-whisper:/, '');
      parts.push(
        `<span title="High-quality stream model and how far behind realtime its leading edge is (smaller = healthier; spikes during cold-start or under load)">` +
        `HQ:${escape(model)} (lag ${lag.toFixed(1)}s)</span>`
      );
    }
    // Per-stream drop counters. Only render when something has actually
    // been dropped so the header stays clean in healthy sessions. Each
    // counter gets a tooltip so the q/s suffix is self-explanatory.
    if (s.stream_stats) {
      for (const label of Object.keys(s.stream_stats)) {
        const st = s.stream_stats[label];
        const q = st.dropped_queue_full || 0;
        const sil = st.dropped_silent || 0;
        if (q + sil > 0) {
          const tip = (
            `Audio windows dropped on the ${label} stream. ` +
            `q = ${q}: dropped because the transcriber input queue was full (transcriber falling behind realtime). ` +
            `s = ${sil}: dropped by the VAD silent-gate (voiced_ms / voiced_frac below threshold — not enough speech to bother transcribing).`
          );
          parts.push(
            `<span title="${escape(tip)}">⚠ ${escape(label)} dropped ${q}q+${sil}s</span>`
          );
        }
      }
    }
    $('meta').innerHTML = parts.join(' · ');
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
  const EVENT_KINDS = ['audio_window','segment','press','marker','cursor_capture','paste','mute','debug_flag'];
  const enabledKinds = new Set(EVENT_KINDS);
  // Stream filter — events tagged with data.stream get filtered by this.
  // Events without a stream tag (press, marker, paste, debug_flag) are
  // unaffected by stream filtering and always render when their kind is on.
  const STREAMS = ['fast','hq'];
  const enabledStreams = new Set(STREAMS);
  document.querySelectorAll('.chip[data-kind]').forEach(c => {
    c.addEventListener('click', () => {
      const k = c.dataset.kind;
      if (enabledKinds.has(k)) { enabledKinds.delete(k); c.classList.remove('on'); }
      else { enabledKinds.add(k); c.classList.add('on'); }
    });
  });
  document.querySelectorAll('.chip[data-stream]').forEach(c => {
    c.addEventListener('click', () => {
      const s = c.dataset.stream;
      if (enabledStreams.has(s)) { enabledStreams.delete(s); c.classList.remove('on'); }
      else { enabledStreams.add(s); c.classList.add('on'); }
    });
  });
  // Timeline label gutter collapse/expand.
  const _labelsToggle = document.getElementById('timeline-labels-toggle');
  if (_labelsToggle) {
    _labelsToggle.addEventListener('click', () => {
      const gutter = document.getElementById('timeline-labels');
      const collapsed = gutter.classList.toggle('collapsed');
      _labelsToggle.textContent = collapsed ? '›' : '‹';
    });
  }
  function passesStreamFilter(e) {
    const st = e.data && e.data.stream;
    if (!st) return true;
    return enabledStreams.has(st);
  }

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
    const filtered = events
      .filter(e => enabledKinds.has(e.kind) && passesStreamFilter(e))
      .slice(-300);
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

    // Sub-row y-coordinates per stream for stream-tagged events.
    // Layout: audio fast/hq stacked, then segment fast (bar + words below),
    // then segment hq (bar + words below), then paste/marker/cursor unchanged.
    const ROWS = {
      audio_window: { fast: 20, hq: 30, h: 8 },
      segment:      { fast: 46, hq: 66, h: 8 },
      word:         { fast: 55, hq: 75, h: 5 },
    };
    // Lane separators only — band labels live in the HTML side panel to
    // the left, so they don't visually compete with the bars themselves.
    parts.push(`<line x1="0" y1="42" x2="${W}" y2="42" stroke="#1b1d23" stroke-width="0.5"/>`);
    parts.push(`<line x1="0" y1="62" x2="${W}" y2="62" stroke="#1b1d23" stroke-width="0.5" stroke-dasharray="2,3"/>`);

    for (const e of events) {
      const d = e.data || {};
      const stream = d.stream === 'hq' ? 'hq' : 'fast';
      // Apply stream filter for stream-tagged events.
      if (d.stream && !enabledStreams.has(stream)) continue;
      if (e.kind === 'audio_window' && d.start != null && d.end != null) {
        const x0 = x(d.start), x1 = Math.max(x(d.end), x0+1);
        const yRow = ROWS.audio_window[stream];
        const hRow = ROWS.audio_window.h;
        const isDrop = d.reason && d.reason.startsWith('dropped');
        const fill = d.reason === 'forced' ? '#ffa726'
                   : d.reason === 'silence' ? '#5fa8d3'
                   : d.reason === 'max_window' ? '#ec407a'
                   : d.reason === 'dropped_silent' ? '#3a3d45'
                   : d.reason === 'dropped_queue_full' ? '#d64545'
                   : '#5fa8d3';
        const opacity = isDrop ? 0.6 : 0.85;
        const title = escape(summarize(e));
        // Dropped windows: solid bar with a thick red top stroke (visible
        // even at 1-pixel bar width) and a contrast-on-dark glyph anchored
        // ABOVE the bar so very narrow bars don't get visually swallowed
        // by their own marker. Avoid diagonal cross-strokes — they look
        // like a filled box at small widths.
        parts.push(`<rect x="${x0}" y="${yRow}" width="${x1-x0}" height="${hRow}" fill="${fill}" opacity="${opacity}"><title>${title}</title></rect>`);
        if (stream === 'hq' && !isDrop) {
          // Teal dotted top stripe distinguishes HQ bars from fast bars
          // without changing the reason color. Skipped on drops because
          // the red top stroke is already a stronger differentiator.
          parts.push(`<line x1="${x0}" y1="${yRow+0.5}" x2="${x1}" y2="${yRow+0.5}" stroke="#26a69a" stroke-width="1" stroke-dasharray="2,2"/>`);
        }
        if (isDrop) {
          parts.push(`<line x1="${x0}" y1="${yRow-0.5}" x2="${x1}" y2="${yRow-0.5}" stroke="#ff5252" stroke-width="2"><title>${title}</title></line>`);
          const sym = d.reason === 'dropped_queue_full' ? '⛔Q' : '⚠S';
          // Place the glyph slightly ABOVE the bar and ALWAYS at the bar's
          // x-start, regardless of bar width, so even a 2-pixel bar is
          // unambiguously labeled.
          parts.push(`<text x="${x0}" y="${yRow-2}" fill="#ff5252" font-size="10" font-weight="bold"><title>${title}</title>${sym}</text>`);
        }
      } else if (e.kind === 'segment' && d.start != null && d.end != null) {
        const x0 = x(d.start), x1 = Math.max(x(d.end), x0+1);
        const yRow = ROWS.segment[stream];
        const hRow = ROWS.segment.h;
        // Same green for both streams — stream identity is conveyed by
        // row position. HQ rect gets a thin dotted top border so the
        // bar shape itself is also distinct, not just its location.
        parts.push(`<rect x="${x0}" y="${yRow}" width="${x1-x0}" height="${hRow}" fill="#6abf4b" opacity="0.7"><title>${escape(summarize(e))}</title></rect>`);
        if (stream === 'hq') {
          parts.push(`<line x1="${x0}" y1="${yRow+0.5}" x2="${x1}" y2="${yRow+0.5}" stroke="#26a69a" stroke-width="1" stroke-dasharray="2,2"/>`);
        }
        const wordY = ROWS.word[stream];
        for (const w of (d.words||[])) {
          const wx = x(w.s);
          parts.push(`<line x1="${wx}" y1="${wordY}" x2="${wx}" y2="${wordY+ROWS.word.h}" stroke="#7e57c2" stroke-width="1"><title>${escape(w.text)} [${w.s.toFixed(2)}–${w.e.toFixed(2)}] p=${w.p}</title></line>`);
        }
      } else if (e.kind === 'press') {
        const xx = x(d.session_time);
        const stroke = d.phase === 'start' ? '#6abf4b' : d.phase === 'end' ? '#d64545' : '#f0a04b';
        const ttl = escape(summarize(e));
        // Distinct shape signature: a downward triangle ▼ anchored at the
        // very top, plus a dashed line spanning the timeline. The
        // triangle is the unambiguous "press" icon — it can't be confused
        // with a forced/silence/max audio_window bar.
        parts.push(`<polygon points="${xx-4},2 ${xx+4},2 ${xx},10" fill="${stroke}"><title>${ttl}</title></polygon>`);
        parts.push(`<line x1="${xx}" y1="10" x2="${xx}" y2="${H-15}" stroke="${stroke}" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.7"><title>${ttl}</title></line>`);
        parts.push(`<text x="${xx+6}" y="18" fill="${stroke}" font-size="10">${d.key}/${d.phase}</text>`);
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
      } else if (e.kind === 'debug_flag') {
        // Tall red line spanning the full timeline height + a 🚩 glyph
        // anchored at the top so flagged moments are unmistakable at any
        // zoom level.
        const xx = x(e.ts);
        const title = escape(summarize(e));
        parts.push(`<line x1="${xx}" y1="0" x2="${xx}" y2="${H}" stroke="#ff5252" stroke-width="2"><title>${title}</title></line>`);
        parts.push(`<text x="${xx-6}" y="11" font-size="13"><title>${title}</title>🚩</text>`);
      }
    }
    svg.innerHTML = parts.join('');
  }

  // -----------------------------------------------------------------
  // Replay audio sync
  // -----------------------------------------------------------------
  // When the replayed session was recorded with --save-audio, each chunk
  // has a corresponding chunk_NNN.wav. The status payload's `audio.chunks`
  // is the sorted list of {idx, t_start, t_end, duration_s}. As the replay
  // clock (now_seconds) advances, we point an <audio> element at whichever
  // chunk currently covers `now_seconds`, seek inside that chunk to
  // `now - chunk.t_start`, and set playbackRate to replay.speed so the
  // audio advances at the same rate as the timeline.
  //
  // Between chunks (silence regions in the original recording) we pause
  // the element; on the next chunk's onset we load the new src and play.
  // We avoid fighting the user: explicit pauses are honored until they
  // click play again, and we only force-resync currentTime when drift
  // exceeds 0.4 s (otherwise we let the audio's own clock advance).
  const audioEl = $('replay-audio');
  const audioWrap = $('replay-audio-wrap');
  const audioStatus = $('replay-audio-status');
  let currentAudioChunkIdx = null;
  let userPaused = false;
  if (audioEl) {
    audioEl.addEventListener('pause', () => {
      // Distinguish user-initiated pause from our own programmatic
      // pause (e.g. silence gap). If the replay is still running and
      // we are inside a chunk, the user paused; honor that until play.
      if (!audioEl.dataset.programmaticPause) userPaused = true;
      audioEl.dataset.programmaticPause = '';
    });
    audioEl.addEventListener('play', () => { userPaused = false; });
  }

  function findChunkAt(chunks, t) {
    // chunks sorted by t_start. Linear scan is fine — typical sessions
    // have <500 chunks. Returns the chunk covering t, else null.
    for (const c of chunks) {
      if (t >= c.t_start && t <= c.t_end) return c;
      if (c.t_start > t) return null;  // sorted, no later chunk can cover t
    }
    return null;
  }

  function syncReplayAudio(s) {
    if (!audioEl) return;
    const replay = s.replay;
    const audio = s.audio;
    if (!replay || !replay.mode || !audio || !audio.available
        || !audio.chunks || audio.chunks.length === 0) {
      audioWrap.style.display = 'none';
      if (!audioEl.paused) {
        audioEl.dataset.programmaticPause = '1';
        audioEl.pause();
      }
      return;
    }
    audioWrap.style.display = 'flex';
    const now = Number(s.now_seconds || 0);
    const chunk = findChunkAt(audio.chunks, now);
    const speed = Number(replay.speed || 1.0);

    if (chunk === null) {
      // Between chunks — silence period. Pause but keep src loaded.
      audioStatus.textContent = 'silence';
      if (!audioEl.paused) {
        audioEl.dataset.programmaticPause = '1';
        audioEl.pause();
      }
      return;
    }

    const targetSrc = '/audio/' + chunk.idx;
    if (currentAudioChunkIdx !== chunk.idx) {
      // Load new chunk. Setting .src triggers reload; once metadata is
      // ready we seek and play.
      currentAudioChunkIdx = chunk.idx;
      audioEl.src = targetSrc;
      audioEl.playbackRate = Math.max(0.0625, Math.min(16, speed));
      audioStatus.textContent = 'chunk ' + chunk.idx;
      const onReady = () => {
        audioEl.removeEventListener('loadedmetadata', onReady);
        const pos = Math.max(0, now - chunk.t_start);
        try { audioEl.currentTime = pos; } catch (e) { /* not seekable yet */ }
        if (!userPaused) {
          audioEl.play().catch(() => { /* autoplay blocked — user must click ▶ */ });
        }
      };
      audioEl.addEventListener('loadedmetadata', onReady);
      return;
    }

    // Same chunk as last tick — keep playing, just nudge rate and check drift.
    if (Math.abs(audioEl.playbackRate - speed) > 0.001) {
      audioEl.playbackRate = Math.max(0.0625, Math.min(16, speed));
    }
    const expected = Math.max(0, now - chunk.t_start);
    if (!isNaN(audioEl.currentTime) && Math.abs(audioEl.currentTime - expected) > 0.4) {
      try { audioEl.currentTime = expected; } catch (e) { /* ignore */ }
    }
    if (audioEl.paused && !userPaused) {
      audioEl.play().catch(() => {});
    }
    audioStatus.textContent = 'chunk ' + chunk.idx
      + ' @ ' + expected.toFixed(1) + 's / ' + chunk.duration_s.toFixed(1) + 's';
  }

  // Track when we last got a successful snapshot. The staleness chip is
  // rendered off this timestamp by a separate 250ms tick that runs
  // independently of the fetch loop, so a single slow/failed poll never
  // flips a binary banner — the UI only complains when data is actually old.
  let lastSnapshotClientTime = Date.now();
  let pollInflight = false;

  async function poll() {
    // Drop the AbortController — with HTTP/1.1 keep-alive and a
    // server-side snapshot cache, every /status response is microseconds.
    // Aborting mid-flight was what polluted the browser's per-origin socket
    // pool and produced the "(canceled)" cascade in the first place.
    if (pollInflight) return;  // coalesce overlapping ticks (e.g. focus + interval)
    pollInflight = true;
    try {
      const r = await fetch('/status', { cache: 'no-store' });
      if (!r.ok) throw new Error('status ' + r.status);
      const s = await r.json();
      lastSnapshotClientTime = Date.now();
      $('stale-banner').style.display = 'none';
      renderHeader(s);
      renderPastes(s.paste_log);
      renderTranscript(s);
      renderEvents(s.debug_events || []);
      renderTimeline(s.debug_events || [], s.now_seconds || 0);
      syncReplayAudio(s);
    } catch (e) {
      console.warn('poll failed:', e);
    } finally {
      pollInflight = false;
    }
  }

  function renderStaleness() {
    const ageMs = Date.now() - lastSnapshotClientTime;
    const ageS = ageMs / 1000;
    const el = $('staleness');
    const banner = $('stale-banner');
    if (ageS < 1.5) {
      el.className = 'live';
      el.textContent = '';
      banner.style.display = 'none';
    } else if (ageS < 5) {
      el.className = 'soft';
      el.textContent = 'updated ' + Math.round(ageS) + 's ago';
      banner.style.display = 'none';
    } else if (ageS < 30) {
      el.className = 'warn';
      el.textContent = 'stale ' + Math.round(ageS) + 's';
      banner.style.display = 'none';
    } else if (ageS < 120) {
      el.className = 'alert';
      el.textContent = 'reconnecting… ' + Math.round(ageS) + 's';
      banner.style.display = 'none';
    } else {
      el.className = 'alert';
      el.textContent = 'disconnected ' + Math.round(ageS) + 's';
      banner.style.display = 'block';
    }
  }

  poll();
  setInterval(poll, 500);
  setInterval(renderStaleness, 250);
  // Backgrounded tabs throttle setInterval to ~1/min on macOS; kick a fresh
  // poll on refocus so the staleness chip clears within ~one tick instead of
  // waiting up to a minute. Cheap and harmless when not throttled.
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
  }
  function buildPatchFromForm() {
    const patch = {};
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
      `<input type="checkbox" class="m-flag" title="single-point marker"${m.flag ? ' checked' : ''}></td>` +
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
      const flag = tr.querySelector('.m-flag').checked;
      if (key && type) out.push({ key, type, description, flag });
    });
    return out;
  }
  $('markers-add').addEventListener('click', () => {
    $('markers-tbody').appendChild(buildMarkerRow({ key: '', type: '', description: '', flag: false }));
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
    # HTTP/1.1 + Content-Length on every response lets the browser keep one
    # TCP socket open across all polls. Under HTTP/1.0 (the BaseHTTPRequestHandler
    # default) every 500 ms poll opened a fresh connection, and any aborted
    # fetch left a half-closed socket polluting the browser's per-origin pool —
    # the root cause of the "(canceled)" cascades.
    protocol_version = "HTTP/1.1"

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
        self._send_bytes(code, "application/json; charset=utf-8", body)

    def _send_bytes(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path == "/" or self.path.startswith("/?"):
            self._send_bytes(200, "text/html; charset=utf-8", _INDEX_HTML.encode("utf-8"))
            return
        if self.path == "/status":
            # Serve from the background snapshot cache — never call
            # snapshot_provider() on the request thread, never take any app
            # lock. Guarantees microsecond response regardless of in-process
            # contention with the recorder/transcriber.
            cache = self.server.snapshot_cache  # type: ignore[attr-defined]
            body, err = cache.get_bytes()
            if err is not None:
                self._send_json(500, {"error": err})
                return
            self._send_bytes(200, "application/json; charset=utf-8", body)
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
        if self.path.startswith("/audio/"):
            self._serve_audio()
            return
        self.send_error(404)

    def _serve_audio(self) -> None:
        """GET /audio/<idx> — serve chunk_<idx>.wav from server.audio_dir.

        Supports HTTP Range requests (single-range only) so the browser's
        <audio> element can scrub mid-file. Without Range the whole file
        is sent inline."""
        audio_dir = getattr(self.server, "audio_dir", None)
        if not audio_dir:
            self.send_error(404)
            return
        # Path: /audio/<idx>(.wav)?  Accept either form.
        suffix = self.path[len("/audio/"):]
        if suffix.endswith(".wav"):
            suffix = suffix[:-4]
        try:
            idx = int(suffix)
        except ValueError:
            self.send_error(404)
            return
        # Guard against escapes; idx is now an int so it can't traverse.
        wav_path = os.path.join(audio_dir, f"chunk_{idx:03d}.wav")
        if not os.path.isfile(wav_path):
            self.send_error(404)
            return
        try:
            size = os.path.getsize(wav_path)
        except OSError:
            self.send_error(500)
            return

        range_header = self.headers.get("Range")
        start = 0
        end = size - 1
        is_partial = False
        if range_header and range_header.startswith("bytes="):
            spec = range_header[len("bytes="):].split(",", 1)[0].strip()
            try:
                a, b = spec.split("-", 1)
                if a:
                    start = int(a)
                if b:
                    end = int(b)
                if start < 0 or end >= size or start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                is_partial = True
            except ValueError:
                self.send_error(400)
                return

        length = end - start + 1
        code = 206 if is_partial else 200
        self.send_response(code)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if is_partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            with open(wav_path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

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


class _SnapshotCache:
    """Background-pumped cache of the latest status snapshot, serialized to bytes.

    The pump thread calls ``snapshot_provider()`` at a fixed cadence and stores
    the JSON-encoded payload. HTTP request handlers serve ``get_bytes()`` in
    O(1) without ever invoking the provider — so a slow snapshot (e.g. brief
    lock contention with the recorder/transcriber) can never block or abort a
    client poll.
    """

    def __init__(self, snapshot_provider: Callable[[], dict], interval_s: float = 0.2):
        self._snapshot_provider = snapshot_provider
        self._interval_s = interval_s
        self._lock = threading.Lock()
        self._body: bytes = b'{"error":"snapshot not yet available"}'
        self._error: Optional[str] = "snapshot not yet available"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def get_bytes(self) -> tuple[bytes, Optional[str]]:
        with self._lock:
            return self._body, self._error

    def refresh_now(self) -> None:
        """Compute one snapshot synchronously and update the cache."""
        try:
            payload = self._snapshot_provider()
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            with self._lock:
                self._body = body
                self._error = None
        except Exception as e:  # noqa: BLE001
            with self._lock:
                self._error = str(e)

    def start(self) -> None:
        # Prime the cache synchronously so the very first client request after
        # server start has real data rather than the placeholder error.
        self.refresh_now()
        self._thread = threading.Thread(
            target=self._run, name="StatusSnapshotPump", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self.refresh_now()
            # Event.wait() returns True early if stop is signalled — gives a
            # responsive shutdown without polling.
            if self._stop.wait(self._interval_s):
                return


class _Server(ThreadingHTTPServer):
    """Slight subclass that holds the snapshot/config callables for the handler."""

    daemon_threads = True

    def __init__(self, addr, handler, snapshot_cache, config_provider=None,
                 config_setter=None, audio_dir=None):
        super().__init__(addr, handler)
        self.snapshot_cache = snapshot_cache
        self.config_provider = config_provider
        self.config_setter = config_setter
        # When set, /audio/<idx> serves chunk_<idx>.wav from this directory
        # with HTTP Range support (so HTML5 <audio> can scrub). None = the
        # route returns 404. Set by ``StatusServer`` constructor.
        self.audio_dir = audio_dir


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
        audio_dir: Optional[str] = None,
    ):
        self._snapshot_provider = snapshot_provider
        self._config_provider = config_provider
        self._config_setter = config_setter
        self._audio_dir = audio_dir
        self._server: Optional[_Server] = None
        self._thread: Optional[threading.Thread] = None
        self._cache: Optional[_SnapshotCache] = None

    def start(self, host: str = "127.0.0.1", port: int = 0) -> tuple[str, int]:
        self._cache = _SnapshotCache(self._snapshot_provider)
        self._cache.start()
        self._server = _Server(
            (host, port), _Handler, self._cache,
            config_provider=self._config_provider,
            config_setter=self._config_setter,
            audio_dir=self._audio_dir,
        )
        bound_host, bound_port = self._server.server_address[:2]
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="StatusServer", daemon=True
        )
        self._thread.start()
        return bound_host, bound_port

    def stop(self) -> None:
        if self._cache is not None:
            try:
                self._cache.stop()
            except Exception:
                pass
            self._cache = None
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
