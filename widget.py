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
  table.pastes tr { cursor: help; }
  table.pastes tr:hover td { background: var(--panel-2); }

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
<main>
  <section id="pastes-section" class="collapsible" data-state="collapsed">
    <h2><span class="chev">▸</span>Paste log</h2>
    <div class="body" id="pastes-body"><div class="empty">no pastes yet</div></div>
  </section>
  <section id="transcript-section" class="collapsible" data-state="collapsed">
    <h2><span class="chev">▸</span>Transcript (tail)</h2>
    <div class="body" id="transcript-body">
      <div id="transcript" class="empty">no audio yet</div>
    </div>
  </section>
</main>
<script>
(function() {
  const $ = (id) => document.getElementById(id);
  const escape = (s) => s.replace(/[&<>"']/g, c => (
    {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const MARKER_RE = /&lt;&lt;&lt;MARKER:([a-z_]+):(start|end)&gt;&gt;&gt;/g;

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

  function renderTranscript(text) {
    const el = $('transcript');
    const body = $('transcript-body');
    attachScrollWatcher(body);
    if (!text) {
      el.className = 'empty';
      el.textContent = 'no audio yet';
      return;
    }
    el.className = '';
    const escaped = escape(text);
    const html = escaped.replace(MARKER_RE, (m, type, kind) => {
      const cls = ['marker-pill', type].join(' ');
      return `<span class="${cls}" title="${type}:${kind}">${type}:${kind}</span>`;
    });
    el.innerHTML = html;
    pinToBottomIfStuck(body);
  }

  function renderPastes(pastes) {
    const body = $('pastes-body');
    attachScrollWatcher(body);
    if (!pastes || pastes.length === 0) {
      body.innerHTML = '<div class="empty">no pastes yet</div>';
      return;
    }
    // Server sends chronological (oldest first, newest last). Render as-is
    // so the tail is at the bottom of the scrolling region.
    const rows = pastes.map(p => {
      const labelCls = p.label === 'r' ? 'r' : 'aside';
      return `<tr title="${escape(p.full || '')}">` +
             `<td class="ts">${escape(p.ts)}</td>` +
             `<td class="label ${labelCls}">${escape(p.label)}</td>` +
             `<td class="preview">${escape(p.preview)}</td>` +
             `</tr>`;
    }).join('');
    body.innerHTML = `<table class="pastes"><tbody>${rows}</tbody></table>`;
    pinToBottomIfStuck(body);
  }

  function renderHeader(s) {
    $('meta').textContent =
      `${s.session_id} · ${s.model} (${s.device}) · ${s.chunk_count} chunk(s)`;
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

  async function poll() {
    try {
      const r = await fetch('/status', { cache: 'no-store' });
      if (!r.ok) throw new Error('status ' + r.status);
      const s = await r.json();
      $('disconnected').style.display = 'none';
      renderHeader(s);
      renderPastes(s.paste_log);
      renderTranscript(s.transcript_tail);
    } catch (e) {
      $('disconnected').style.display = 'block';
    }
  }
  poll();
  setInterval(poll, 500);
})();
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    # Suppress per-request access logs — the 2 Hz poll would spam the console.
    def log_message(self, format, *args):
        return

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
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
                return
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)


class _Server(ThreadingHTTPServer):
    """Slight subclass that holds the snapshot provider for the handler."""

    daemon_threads = True

    def __init__(self, addr, handler, snapshot_provider):
        super().__init__(addr, handler)
        self.snapshot_provider = snapshot_provider


class StatusServer:
    """HTTP status server bound to a local port.

    The dashboard reads from ``snapshot_provider``, a zero-arg callable
    returning a JSON-serializable dict.
    """

    def __init__(self, snapshot_provider: Callable[[], dict]):
        self._snapshot_provider = snapshot_provider
        self._server: Optional[_Server] = None
        self._thread: Optional[threading.Thread] = None

    def start(self, host: str = "127.0.0.1", port: int = 0) -> tuple[str, int]:
        self._server = _Server((host, port), _Handler, self._snapshot_provider)
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
