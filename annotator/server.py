"""HTTP server for offline annotation of one or more session directories.

Standalone — does NOT share infrastructure with the live ``WidgetServer``.
A session is fully static once recorded, so the annotator just needs to
read files, serve audio with HTTP Range support, and append to
``annotations.jsonl``.

Two modes:
  - Single session: ``AnnotatorServer(session_dir)`` — backwards-compatible
    with the original API. Legacy routes ``/annotation/...`` still work.
  - Multi-session: ``AnnotatorServer.from_sessions([...])`` or
    ``AnnotatorServer.from_root(root_dir)`` — exposes a session picker in
    the UI and routes scoped under ``/s/<sid>/...``.

Routes (canonical)
------------------
``GET  /``                                → annotation HTML page
``GET  /sessions``                        → list of loaded sessions
``GET  /s/<sid>/annotation/chunks``       → per-chunk view (segments split
                                            on word midpoints at chunk
                                            boundaries)
``GET  /s/<sid>/annotation/audio/<idx>``  → ``audio/chunk_NNNN.wav`` with
                                            ``Range`` support
``GET  /s/<sid>/annotation/state``        → resolved last-write-wins state
``PUT  /s/<sid>/annotation/<idx>/<seg_id>`` → append a segment annotation
``PUT  /s/<sid>/annotation/<idx>``        → append a chunk-level annotation
                                            (reject / unreject)

Legacy routes (single-session only) — ``/annotation/...`` without a
session prefix, proxied to the lone session for backwards-compat.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

from .loader import ChunkView, load_chunks
from .storage import AnnotationStore


logger = logging.getLogger(__name__)


_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


@dataclass
class _SessionCtx:
    sid: str
    session_dir: Path
    audio_dir: Path
    store: AnnotationStore
    chunks: list[ChunkView] = field(default_factory=list)


class AnnotatorServer:
    def __init__(
        self,
        session_dir: str | Path | None = None,
        *,
        port: int = 0,
        sessions: Optional[list[str | Path]] = None,
    ) -> None:
        """Single-session: pass ``session_dir``. Multi-session: pass
        ``sessions=[...]`` (or use ``from_root``).  Passing both is an
        error.
        """
        if session_dir is not None and sessions is not None:
            raise ValueError("pass either session_dir or sessions, not both")
        self._port = port
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._sessions: dict[str, _SessionCtx] = {}
        # Track whether we were constructed in single-session mode so we
        # can keep the legacy ``/annotation/...`` routes alive.
        self._single_mode = session_dir is not None
        if session_dir is not None:
            ctx = _build_ctx(session_dir)
            self._sessions[ctx.sid] = ctx
        else:
            for s in (sessions or []):
                ctx = _build_ctx(s)
                if ctx.sid in self._sessions:
                    # Defensive — distinct dirs with the same basename.
                    suffix = 2
                    while f"{ctx.sid}_{suffix}" in self._sessions:
                        suffix += 1
                    ctx.sid = f"{ctx.sid}_{suffix}"
                self._sessions[ctx.sid] = ctx
        if not self._sessions:
            raise ValueError("AnnotatorServer requires at least one session")

    @classmethod
    def from_root(cls, root: str | Path, *, port: int = 0) -> "AnnotatorServer":
        """Discover every direct child of ``root`` that has an
        ``audio/manifest.json``."""
        root_p = Path(root).resolve()
        if not root_p.is_dir():
            raise ValueError(f"not a directory: {root_p}")
        found: list[Path] = []
        for child in sorted(root_p.iterdir()):
            if child.is_dir() and (child / "audio" / "manifest.json").exists():
                found.append(child)
        if not found:
            raise ValueError(
                f"no sessions under {root_p} "
                "(looking for <dir>/audio/manifest.json)"
            )
        return cls(sessions=found, port=port)

    # ------------------------------------------------------------------

    @property
    def actual_port(self) -> int:
        if self._httpd is None:
            return 0
        return self._httpd.server_address[1]

    def _default_sid(self) -> Optional[str]:
        """Return the lone session id if in single-session mode."""
        if not self._single_mode:
            return None
        return next(iter(self._sessions))

    def start(self) -> None:
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_kw) -> None:
                pass

            def _send_json(self, body, code: int = 200) -> None:
                data = json.dumps(body).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_html(self, html: str, code: int = 200) -> None:
                data = html.encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _read_body(self) -> dict:
                length = int(self.headers.get("Content-Length", "0") or "0")
                if not length:
                    return {}
                raw = self.rfile.read(length)
                try:
                    return json.loads(raw.decode("utf-8"))
                except Exception:
                    return {}

            def _lookup_ctx(self, sid: Optional[str]) -> Optional[_SessionCtx]:
                if sid is None:
                    sid = outer._default_sid()
                if sid is None:
                    return None
                return outer._sessions.get(sid)

            # --- GET --------------------------------------------------

            def do_GET(self) -> None:
                p = self.path
                if p == "/" or p == "/annotate":
                    self._send_html(_INDEX_HTML)
                    return
                if p == "/sessions":
                    self._send_json(outer._sessions_payload())
                    return

                # /s/<sid>/annotation/...
                m = re.match(r"^/s/([^/]+)(/annotation/.*)$", p)
                if m:
                    sid, rest = m.group(1), m.group(2)
                    self._dispatch_get_annotation(sid, rest)
                    return

                # Legacy /annotation/... (single-session only)
                if p.startswith("/annotation/"):
                    self._dispatch_get_annotation(None, p)
                    return

                self.send_error(404)

            def _dispatch_get_annotation(
                self, sid: Optional[str], rest: str,
            ) -> None:
                ctx = self._lookup_ctx(sid)
                if ctx is None:
                    self.send_error(404)
                    return
                if rest == "/annotation/chunks":
                    self._send_json(outer._chunks_payload(ctx))
                    return
                if rest == "/annotation/state":
                    self._send_json(ctx.store.resolved_state())
                    return
                m2 = re.match(r"^/annotation/audio/(\d+)$", rest)
                if m2:
                    outer._serve_audio(self, ctx, int(m2.group(1)))
                    return
                self.send_error(404)

            # --- PUT --------------------------------------------------

            def do_PUT(self) -> None:
                p = unquote(self.path)

                # /s/<sid>/annotation/...
                m = re.match(r"^/s/([^/]+)(/annotation/.*)$", p)
                if m:
                    sid, rest = m.group(1), m.group(2)
                    self._dispatch_put_annotation(sid, rest)
                    return

                # Legacy /annotation/...
                if p.startswith("/annotation/"):
                    self._dispatch_put_annotation(None, p)
                    return

                self.send_error(404)

            def _dispatch_put_annotation(
                self, sid: Optional[str], rest: str,
            ) -> None:
                ctx = self._lookup_ctx(sid)
                if ctx is None:
                    self.send_error(404)
                    return
                m = re.match(r"^/annotation/(\d+)/(.+)$", rest)
                if m:
                    chunk_idx = int(m.group(1))
                    seg_id = m.group(2)
                    body = self._read_body()
                    try:
                        ann = ctx.store.append_segment(
                            chunk_idx=chunk_idx,
                            segment_id=seg_id,
                            text=body.get("text", ""),
                            status=body.get("status", "edited"),
                            source_text_fast=body.get("source_text_fast"),
                            source_text_hq=body.get("source_text_hq"),
                            notes=body.get("notes"),
                        )
                    except ValueError as e:
                        self._send_json({"error": str(e)}, code=400)
                        return
                    self._send_json({"ok": True, "annotation": ann.to_dict()})
                    return
                m = re.match(r"^/annotation/(\d+)$", rest)
                if m:
                    chunk_idx = int(m.group(1))
                    body = self._read_body()
                    try:
                        ann = ctx.store.append_chunk(
                            chunk_idx=chunk_idx,
                            status=body.get("status", "rejected"),
                            notes=body.get("notes"),
                        )
                    except ValueError as e:
                        self._send_json({"error": str(e)}, code=400)
                        return
                    self._send_json({"ok": True, "annotation": ann.to_dict()})
                    return
                self.send_error(404)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", self._port), _Handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="annotator-http", daemon=True,
        )
        self._thread.start()
        logger.info(
            "AnnotatorServer started on http://127.0.0.1:%d/  sessions=%d "
            "single=%s", self.actual_port, len(self._sessions), self._single_mode,
        )

    def shutdown(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    # ------------------------------------------------------------------

    def _sessions_payload(self) -> dict:
        items = []
        for ctx in self._sessions.values():
            state = ctx.store.resolved_state()
            annotated = len(state["segments"])
            rejected = len(state["rejected_chunks"])
            total_segments = sum(len(c.segments) for c in ctx.chunks)
            items.append({
                "id": ctx.sid,
                "name": ctx.session_dir.name,
                "path": str(ctx.session_dir),
                "chunks": len(ctx.chunks),
                "segments": total_segments,
                "annotated_segments": annotated,
                "rejected_chunks": rejected,
            })
        return {
            "sessions": items,
            "single_mode": self._single_mode,
        }

    def _chunks_payload(self, ctx: _SessionCtx) -> dict:
        return {
            "session_id": ctx.sid,
            "session_dir": str(ctx.session_dir),
            "chunks": [c.to_dict() for c in ctx.chunks],
        }

    def _serve_audio(
        self, handler: BaseHTTPRequestHandler, ctx: _SessionCtx, idx: int,
    ) -> None:
        if idx < 0 or idx >= len(ctx.chunks):
            handler.send_error(404)
            return
        wav = ctx.audio_dir / ctx.chunks[idx].file
        if not wav.exists():
            handler.send_error(404)
            return
        size = wav.stat().st_size
        rng = handler.headers.get("Range", "")
        start, end = 0, size - 1
        partial = False
        if rng:
            m = _RANGE_RE.match(rng)
            if m:
                s, e = m.group(1), m.group(2)
                if s:
                    start = int(s)
                if e:
                    end = int(e)
                if start >= size or end >= size or start > end:
                    handler.send_response(416)
                    handler.send_header("Content-Range", f"bytes */{size}")
                    handler.end_headers()
                    return
                partial = True
        length = end - start + 1
        handler.send_response(206 if partial else 200)
        handler.send_header("Content-Type", "audio/wav")
        handler.send_header("Accept-Ranges", "bytes")
        handler.send_header("Content-Length", str(length))
        if partial:
            handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        handler.end_headers()
        with wav.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                buf = f.read(min(64 * 1024, remaining))
                if not buf:
                    break
                try:
                    handler.wfile.write(buf)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(buf)


def _build_ctx(session_dir: str | Path) -> _SessionCtx:
    p = Path(session_dir).resolve()
    return _SessionCtx(
        sid=p.name,
        session_dir=p,
        audio_dir=p / "audio",
        store=AnnotationStore(p),
        chunks=load_chunks(p),
    )


# ---------------------------------------------------------------------------
# HTML + JS
# ---------------------------------------------------------------------------

_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>voice_dictation — annotate</title>
<style>
:root {
  --bg: #1b1d23; --fg: #e6e6e6; --muted: #888; --accent: #6cf; --boundary: #d29b4a;
  --rejected: #444; --modified: #d4c14a; --hq: #8e5fd4; --fast: #6a7080;
  --row: #1f232b; --row-hi: #2c3340;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: var(--bg); color: var(--fg);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 13px; height: 100%; }
#app { display: flex; flex-direction: column; height: 100vh; }
#topbar {
  display: flex; align-items: center; gap: 12px;
  padding: 6px 12px; border-bottom: 1px solid #333; background: #14171c;
  flex-shrink: 0;
}
#topbar label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }
#session-select {
  background: #1b1d23; color: var(--fg); border: 1px solid #3a4050;
  padding: 3px 8px; border-radius: 3px; font-size: 12px; font-family: inherit;
  max-width: 60%;
}
#session-meta { color: var(--muted); font-size: 11px; font-family: monospace; }
#body { flex: 1; display: flex; overflow: hidden; min-height: 0; }
#picker { width: 220px; border-right: 1px solid #333; overflow-y: auto; flex-shrink: 0; }
#picker .row {
  padding: 6px 10px; border-bottom: 1px solid #2a2e36; cursor: pointer;
}
#picker .row:hover { background: var(--row-hi); }
#picker .row.active { background: var(--row-hi); }
#picker .row.rejected { color: var(--rejected); text-decoration: line-through; }
#picker .meta { font-size: 11px; color: var(--muted); }
#main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
#audio-bar { padding: 6px 12px; border-bottom: 1px solid #333;
  display: flex; align-items: center; gap: 12px; }
#audio-bar audio { flex: 1; height: 28px; }
#chunk-info { color: var(--muted); font-size: 11px; font-family: monospace; }
#segments-wrap { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
#segments-header, .seg {
  display: grid;
  grid-template-columns: 80px 4px 1fr 1fr 20px;
  align-items: stretch; gap: 0;
}
#segments-header {
  font-size: 11px; color: var(--muted); text-transform: uppercase;
  letter-spacing: 0.04em; background: #14171c; border-bottom: 1px solid #333;
  flex-shrink: 0;
}
#segments-header > div { padding: 6px 8px; border-right: 1px solid #2a2e36; }
#segments-header > div:last-child { border-right: none; }
#segments { flex: 1; overflow-y: auto; }
.seg {
  border-bottom: 1px solid #2a2e36;
  background: var(--row);
}
.seg:hover { background: var(--row-hi); }
.seg > .ts, .seg > .source, .seg > .edit-cell, .seg > .status-dot {
  border-right: 1px solid #2a2e36;
}
.seg > .status-dot { border-right: none; }
.seg .ts {
  font-family: monospace; font-size: 11px; color: var(--accent);
  cursor: pointer; padding: 8px 8px; line-height: 18px;
  user-select: none; white-space: nowrap;
}
.seg .ts:hover { text-decoration: underline; }
.seg .accent {
  background: var(--fast);
}
.seg.hq .accent { background: var(--hq); }
.seg.boundary .accent { background: var(--boundary); }
.seg .source {
  padding: 8px 10px; color: #b8b8b8; font-size: 13px; line-height: 18px;
  white-space: pre-wrap; word-break: break-word;
}
.seg .source del {
  color: #d97777; background: #3a1818; text-decoration: line-through;
  padding: 0 1px; border-radius: 2px;
}
.seg .source ins {
  color: #8de28d; background: #1a3a1a; text-decoration: underline;
  padding: 0 1px; border-radius: 2px;
}
.seg .source .word { cursor: pointer; }
.seg .source .word:hover { background: #2c3340; }
.seg .source del .word:hover { background: #4a1f1f; }
.seg .edit-cell { padding: 6px 8px; }
.seg .edit {
  width: 100%; padding: 4px 8px; line-height: 18px;
  background: #14171c; color: var(--fg);
  border: 1px solid #3a4050; border-radius: 3px;
  font-family: inherit; font-size: 13px;
  resize: none; outline: none; overflow: hidden;
  white-space: pre-wrap; word-break: break-word;
  display: block; box-sizing: border-box;
}
.seg .edit:focus { border-color: var(--accent); background: #11141a; }
.seg.modified .edit { background: #2a2814; border-color: var(--modified); }
.seg.modified .ts::after { content: " ●"; color: var(--modified); }
.seg .status-dot {
  font-size: 14px; line-height: 26px; text-align: center; user-select: none;
}
.seg[data-status="edited"] .status-dot::before { content: "✎"; color: var(--modified); }
.seg[data-status="accepted_fast"] .status-dot::before { content: "✓"; color: #6c6; }
.seg[data-status="accepted_hq"]   .status-dot::before { content: "✓"; color: var(--hq); }
.empty { color: var(--muted); padding: 20px; text-align: center; }
#chunk-actions { padding: 6px 12px; border-top: 1px solid #333;
  display: flex; gap: 8px; align-items: center; font-size: 12px; color: var(--muted); }
button {
  background: #2a2e36; color: var(--fg); border: 1px solid #3a3f4a;
  padding: 3px 8px; border-radius: 3px; font-size: 12px; cursor: pointer;
}
button:hover { background: #3a3f4a; }
button.danger { background: #c54; color: #fff; }
.toast {
  position: fixed; bottom: 16px; right: 16px;
  background: #2a2e36; color: var(--fg); padding: 6px 12px;
  border-radius: 4px; opacity: 0; transition: opacity 200ms; font-size: 12px;
}
.toast.show { opacity: 1; }
</style>
</head>
<body>
<div id="app">
  <div id="topbar">
    <label for="session-select">session</label>
    <select id="session-select"></select>
    <span id="session-meta"></span>
  </div>
  <div id="body">
    <div id="picker"></div>
    <div id="main">
      <div id="audio-bar">
        <audio id="player" controls preload="metadata"></audio>
        <span id="chunk-info"></span>
      </div>
      <div id="segments-wrap">
        <div id="segments-header">
          <div>time</div><div></div><div>source (canonical)</div><div>annotation (editable)</div><div></div>
        </div>
        <div id="segments"><div class="empty">select a chunk</div></div>
      </div>
      <div id="chunk-actions">
        <button id="reject-btn" class="danger">reject chunk (r)</button>
        <span>keys: ↑/↓ chunk · [/] session · space play/pause · click ts to seek · edits auto-save on blur · r reject</span>
      </div>
    </div>
  </div>
</div>
<div id="toast" class="toast"></div>
<script>__BOOT__</script>
</body>
</html>
"""


_ANNOTATE_JS = r"""
const state = {
  sessions: [],         // [{id, name, chunks, segments, annotated_segments, ...}]
  sessionIdx: -1,
  chunks: [],
  active: -1,
  resolved: {segments: {}, rejected_chunks: []},
};

function $(sel) { return document.querySelector(sel); }
function toast(msg) {
  const el = $("#toast"); el.textContent = msg; el.classList.add("show");
  clearTimeout(window._tt); window._tt = setTimeout(()=>el.classList.remove("show"), 1200);
}
function esc(s) {
  return (s ?? "").toString().replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"})[c]);
}
function fmtMs(ms) {
  const s = ms / 1000;
  if (s < 60) return s.toFixed(1) + "s";
  const m = Math.floor(s/60), r = s - m*60;
  return m + ":" + r.toFixed(1).padStart(4, "0");
}

function currentSession() {
  return state.sessions[state.sessionIdx];
}
function apiBase() {
  const s = currentSession();
  return s ? `/s/${encodeURIComponent(s.id)}` : "";
}

async function loadSessions() {
  const r = await fetch("/sessions").then(r=>r.json());
  state.sessions = r.sessions;
  const sel = $("#session-select");
  sel.innerHTML = "";
  for (let i = 0; i < state.sessions.length; i++) {
    const s = state.sessions[i];
    const opt = document.createElement("option");
    opt.value = i;
    const annotated = s.annotated_segments || 0;
    const total = s.segments || 0;
    const pct = total > 0 ? Math.round(100 * annotated / total) : 0;
    opt.textContent = `${s.name}  (${s.chunks} chunks · ${annotated}/${total} segs · ${pct}%)`;
    sel.appendChild(opt);
  }
  sel.value = "0";
  state.sessionIdx = 0;
  sel.onchange = () => {
    state.sessionIdx = parseInt(sel.value, 10);
    loadSession();
  };
  // Hide picker dropdown if only one session.
  if (state.sessions.length <= 1) {
    $("#topbar").style.display = "none";
  }
}

async function loadSession() {
  const s = currentSession();
  if (!s) return;
  $("#session-meta").textContent = s.path;
  const [chunks, state_] = await Promise.all([
    fetch(`${apiBase()}/annotation/chunks`).then(r=>r.json()),
    fetch(`${apiBase()}/annotation/state`).then(r=>r.json()),
  ]);
  state.chunks = chunks.chunks;
  state.resolved = state_;
  state.active = -1;
  renderPicker();
  if (state.chunks.length) selectChunk(0);
}

function chunkStatus(c) {
  if (state.resolved.rejected_chunks.includes(c.chunk_idx)) return "rejected";
  let any = false, all = c.segments.length > 0;
  for (const s of c.segments) {
    const k = c.chunk_idx + "|" + s.segment_id;
    if (state.resolved.segments[k]) any = true;
    else all = false;
  }
  if (all) return "complete";
  if (any) return "partial";
  return "untouched";
}

function renderPicker() {
  const el = $("#picker"); el.innerHTML = "";
  for (const c of state.chunks) {
    const status = chunkStatus(c);
    const row = document.createElement("div");
    row.className = "row" + (c.chunk_idx === state.active ? " active" : "") +
      (status === "rejected" ? " rejected" : "");
    const dur = (c.duration_ms / 1000).toFixed(1);
    row.innerHTML = `<div>chunk ${c.chunk_idx} · ${dur}s</div>
      <div class="meta">${c.segments.length} segs · ${status}</div>`;
    row.onclick = () => selectChunk(c.chunk_idx);
    el.appendChild(row);
  }
}

function selectChunk(idx) {
  state.active = idx;
  const c = state.chunks[idx];
  if (!c) return;
  $("#player").src = `${apiBase()}/annotation/audio/${idx}`;
  $("#chunk-info").textContent =
    `${c.file} · ${(c.duration_ms/1000).toFixed(1)}s · ` +
    `[${c.start_accepted_ms}–${c.end_accepted_ms}] ms`;
  renderPicker();
  renderSegments();
}

function autoSize(ta) {
  ta.style.height = "auto";
  const h = Math.max(ta.scrollHeight, 26);
  ta.style.height = h + "px";
}

// Tokenize edit text into bare words (no whitespace) for LCS matching.
function editWords(s) {
  const m = (s || "").match(/\S+/g);
  return m || [];
}

// Normalize for compare: strip non-alphanumerics so "Hello," matches "hello".
function wnorm(s) {
  return (s || "").toLowerCase().replace(/[^a-z0-9]+/g, "");
}

// Word-level diff over source `words` ({text, start_accepted_ms}) vs the
// raw edited string, returning ops with timestamps attached to source-side
// words so the renderer can build clickable spans.
//   ops: ["=", text, ms]   kept source word (clickable)
//        ["-", text, ms]   deleted source word (still clickable)
//        ["+", text]       inserted edit-side word (no timestamp)
function wordDiffWithTs(sourceWords, editText) {
  const a = sourceWords.map(w => ({raw: (w.text || "").trim(),
                                   key: wnorm(w.text), ms: w.start_accepted_ms}));
  const b = editWords(editText).map(t => ({raw: t, key: wnorm(t)}));
  const m = a.length, n = b.length;
  const dp = Array(m + 1);
  for (let i = 0; i <= m; i++) dp[i] = new Int32Array(n + 1);
  for (let i = m - 1; i >= 0; i--) {
    for (let j = n - 1; j >= 0; j--) {
      if (a[i].key && a[i].key === b[j].key) dp[i][j] = dp[i + 1][j + 1] + 1;
      else dp[i][j] = Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const ops = [];
  let i = 0, j = 0;
  while (i < m && j < n) {
    if (a[i].key && a[i].key === b[j].key) {
      // Show the original source casing/punctuation, not the edit's.
      ops.push(["=", a[i].raw, a[i].ms]);
      i++; j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      ops.push(["-", a[i].raw, a[i].ms]); i++;
    } else {
      ops.push(["+", b[j].raw]); j++;
    }
  }
  while (i < m) { ops.push(["-", a[i].raw, a[i].ms]); i++; }
  while (j < n) { ops.push(["+", b[j].raw]); j++; }
  return ops;
}

function renderDiffClickable(sourceWords, editText) {
  // No source words → fall back to plain escaped source text (no link).
  if (!sourceWords || !sourceWords.length) return esc(editText || "");
  const ops = wordDiffWithTs(sourceWords, editText || "");
  const parts = [];
  let prev = null;       // "=" | "-" | "+" | null
  for (const op of ops) {
    const kind = op[0];
    // Open/close <del>/<ins> wrappers when op kind changes.
    if (kind !== prev) {
      if (prev === "-") parts.push("</del>");
      else if (prev === "+") parts.push("</ins>");
      if (kind === "-") parts.push("<del>");
      else if (kind === "+") parts.push("<ins>");
    }
    if (kind === "+") {
      parts.push(" " + esc(op[1]));
    } else {
      // = or -: clickable source word with its timestamp.
      const ms = op[2];
      parts.push(' <span class="word" data-ms="' + ms + '">' +
                 esc(op[1]) + '</span>');
    }
    prev = kind;
  }
  if (prev === "-") parts.push("</del>");
  else if (prev === "+") parts.push("</ins>");
  // Strip the leading space introduced by the first token.
  let out = parts.join("");
  if (out.startsWith(" ")) out = out.slice(1);
  // Spaces immediately after an opening tag look weird; tidy them.
  out = out.replace(/(<(?:del|ins)>) /g, "$1");
  return out;
}

function renderSegments() {
  const el = $("#segments"); el.innerHTML = "";
  const c = state.chunks[state.active];
  if (!c) { el.innerHTML = '<div class="empty">select a chunk</div>'; return; }
  if (!c.segments.length) {
    el.innerHTML = '<div class="empty">no segments in this chunk</div>'; return;
  }
  for (const s of c.segments) {
    const key = c.chunk_idx + "|" + s.segment_id;
    const ann = state.resolved.segments[key];
    const status = ann ? ann.status : "";
    const currentText = (ann ? ann.text : s.text) || "";
    const modified = currentText.trim() !== (s.text || "").trim();

    const div = document.createElement("div");
    div.className = "seg " + s.stream +
      (s.boundary ? " boundary" : "") +
      (modified ? " modified" : "");
    if (status) div.dataset.status = status;
    div.dataset.segId = s.segment_id;
    div.dataset.srcText = s.text || "";

    const startSec = (s.start_accepted_ms - c.start_accepted_ms) / 1000;
    div.innerHTML = `
      <div class="ts" title="seek to ${startSec.toFixed(2)}s${s.boundary ? ' · boundary:' + s.boundary : ''}">${fmtMs(s.start_accepted_ms - c.start_accepted_ms)}</div>
      <div class="accent"></div>
      <div class="source">${renderDiffClickable(s.words || [], currentText)}</div>
      <div class="edit-cell"><textarea class="edit" rows="1" spellcheck="false">${esc(currentText)}</textarea></div>
      <div class="status-dot"></div>`;
    const ts = div.querySelector(".ts");
    ts.onclick = () => playSegment(s);
    const ta = div.querySelector(".edit");
    const sourceCell = div.querySelector(".source");
    ta.addEventListener("input", () => {
      autoSize(ta);
      const isModified = ta.value.trim() !== (s.text || "").trim();
      div.classList.toggle("modified", isModified);
      sourceCell.innerHTML = renderDiffClickable(s.words || [], ta.value);
    });
    ta.addEventListener("blur", () => maybeSaveOnBlur(s, div, ta));
    el.appendChild(div);
    autoSize(ta);
  }
}

async function maybeSaveOnBlur(s, div, ta) {
  const c = state.chunks[state.active];
  const text = ta.value;
  const key = c.chunk_idx + "|" + s.segment_id;
  const prev = state.resolved.segments[key];
  const prevText = prev ? prev.text : null;
  if (prev == null && text.trim() === (s.text || "").trim()) return;
  if (prev != null && text === prevText) return;
  await saveSegment(s, "edited", div, text);
}

async function saveSegment(s, status, div, textOverride) {
  const c = state.chunks[state.active];
  const ta = div.querySelector(".edit");
  const text = textOverride !== undefined ? textOverride : ta.value;
  const body = {
    text, status,
    source_text_fast: s.stream === "fast" ? s.text : null,
    source_text_hq:   s.stream === "hq"   ? s.text : null,
  };
  const r = await fetch(
    `${apiBase()}/annotation/${c.chunk_idx}/${encodeURIComponent(s.segment_id)}`,
    {method: "PUT", headers: {"Content-Type": "application/json"},
     body: JSON.stringify(body)}
  );
  const out = await r.json();
  if (!r.ok) { toast("save failed: " + (out.error || r.status)); return; }
  state.resolved.segments[c.chunk_idx + "|" + s.segment_id] = out.annotation;
  div.dataset.status = out.annotation.status;
  renderPicker();
  toast("saved");
}

function playSegment(s) {
  const c = state.chunks[state.active];
  const p = $("#player");
  p.currentTime = (s.start_accepted_ms - c.start_accepted_ms) / 1000;
  p.play();
}

async function toggleReject() {
  const c = state.chunks[state.active]; if (!c) return;
  const currentlyRejected = state.resolved.rejected_chunks.includes(c.chunk_idx);
  const status = currentlyRejected ? "unrejected" : "rejected";
  const r = await fetch(`${apiBase()}/annotation/${c.chunk_idx}`,
    {method: "PUT", headers: {"Content-Type": "application/json"},
     body: JSON.stringify({status})});
  if (!r.ok) { toast("reject toggle failed"); return; }
  if (currentlyRejected) {
    state.resolved.rejected_chunks = state.resolved.rejected_chunks.filter(x => x !== c.chunk_idx);
  } else {
    state.resolved.rejected_chunks.push(c.chunk_idx);
  }
  renderPicker();
  toast(status);
}

function switchSession(delta) {
  if (state.sessions.length <= 1) return;
  let i = state.sessionIdx + delta;
  if (i < 0) i = state.sessions.length - 1;
  if (i >= state.sessions.length) i = 0;
  state.sessionIdx = i;
  $("#session-select").value = String(i);
  loadSession();
}

$("#reject-btn").onclick = toggleReject;

// Delegated click → seek to a source word's timestamp.
$("#segments").addEventListener("click", (e) => {
  const w = e.target.closest(".word");
  if (!w) return;
  const ms = parseInt(w.dataset.ms, 10);
  if (!isFinite(ms)) return;
  const c = state.chunks[state.active]; if (!c) return;
  const p = $("#player");
  p.currentTime = Math.max(0, (ms - c.start_accepted_ms) / 1000);
  p.play();
});

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT" ||
      e.target.tagName === "SELECT") return;
  if (e.key === "ArrowDown") { selectChunk(Math.min(state.active+1, state.chunks.length-1)); e.preventDefault(); }
  else if (e.key === "ArrowUp") { selectChunk(Math.max(state.active-1, 0)); e.preventDefault(); }
  else if (e.key === "r") { toggleReject(); e.preventDefault(); }
  else if (e.key === "[") { switchSession(-1); e.preventDefault(); }
  else if (e.key === "]") { switchSession(+1); e.preventDefault(); }
  else if (e.key === " ") {
    const p = $("#player"); if (p.paused) p.play(); else p.pause(); e.preventDefault();
  }
});

(async () => {
  await loadSessions();
  await loadSession();
})();
"""


_INDEX_HTML = _INDEX_HTML.replace("__BOOT__", _ANNOTATE_JS)
