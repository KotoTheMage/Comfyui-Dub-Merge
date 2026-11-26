/**
 * ComfyUI Dub Merge - browser timeline editor.
 *
 * Adds a "Load Timeline" button to the Merge Translated Takes node.  The
 * button fetches the node's data bundle (manifest.json + WAVs written by
 * the merge into ComfyUI's temp dir, served over /dubmerge/timeline/<id>)
 * and opens a scrollable left-to-right, two-track timeline:
 *
 *   - top track: the separated original voice, as-is (not editable);
 *   - bottom track: the dub pieces placed exactly as the merge plays them -
 *     short clips at their timestamps (draggable to retime, resizable at
 *     both edges: pulling an edge adds front/end silence); full-pass takes
 *     show as "(pass)" at their real windows (read-only - the merge reads
 *     them at absolute offsets);
 *   - overlapping dubs queue one after the other, exactly like the node's
 *     queue_overlaps option: a piece overlapped by the previous one starts
 *     right after it (after its audible phrase, or its declared span when it
 *     carries an edit) and shows a dashed border + "⇥" marker;
 *   - the spans where the merge will silence the original are overlaid on
 *     the voice track, so the exact mute region is visible while editing;
 *     spans under pieces marked AMBIENT (alt+click a piece, green tint +
 *     "amb" badge) keep the original playing instead - at the node's
 *     ambient_db level in the preview, as gated bed + original in the merge -
 *     so leaked crowd / rain / room tone continues under those dubs;
 *   - everything can be played together with the original for comparison;
 *   - second markers along the ruler.
 *
 * "Apply" writes the drawn adjustments into the node's timeline_edits
 * widget (JSON); the next merge run silences the original exactly across
 * the edited spans and applies the per-piece ambient fill to the marked
 * pieces.  Re-opening the editor RESTORES the saved edits: it prefers the
 * widget's current JSON and falls back to the copy embedded in the last
 * merge run's bundle (manifest.timeline_edits) when the widget comes up
 * empty.  Every JSON shape the node accepts is understood (the editor's
 * {"version":1,"edits":[...]}, a bare list, or an {id: {...}} map), and a
 * value that cannot be applied is reported in the status line instead of
 * being silently ignored.  Apply re-saves the restored edits unchanged -
 * pressing Apply after a plain re-open can no longer wipe them; Reset
 * (or double-clicking pieces) is what clears edits.  If the button is
 * never pressed (or nothing applied) the node executes exactly as
 * without the editor.
 */
import { app } from "../../scripts/app.js";

const MIN_SPAN = 0.05;          // shortest editable piece span (s)
const RULER_H = 30;
const VOICE_H = 110;
const DUBS_H = 130;
// the edit protocol this editor speaks; the merge node's python stamps the
// version it understands into the manifest (edit_protocol).  The browser JS
// hot-reloads on refresh, the python only loads at ComfyUI startup - a pack
// update without a restart leaves an OLD python serving a NEW editor, and
// that python silently drops the current edit entries (merges unedited).
// 3 = the edit anchors reference the STRIPPED piece buffers (the merge
// strips before applying the edits, edited pieces included); an older python
// would revert an edited piece's buffer to the full take and misplace the
// sound by the baked head.
const EDIT_PROTOCOL = 3;
/* Snap - batch-align every placed dub to the voice track.  Each piece's
   SOUND start (audioStart + snd border) is pulled to the closest voice
   onset within SNAP_WIN seconds; pieces with no onset in reach (silence
   only, or continuous speech spanning the whole window) stay untouched. */
const SNAP_WIN = 0.5;       // search radius, seconds
const SNAP_MIN_GAP = 0.08;  // silence that makes a voice onset a real start
const EDGE_PX = 9;              // resize-handle width on a selected piece

const CSS = `
.dmt-overlay { position: fixed; inset: 0; z-index: 99999; background: rgba(0,0,0,.55);
  display: flex; align-items: center; justify-content: center; }
.dmt-panel { position: relative;
  background: #17181c; color: #d8dae0; border: 1px solid #34363d;
  border-radius: 10px; box-shadow: 0 12px 48px rgba(0,0,0,.6); width: min(1240px, 94vw);
  max-height: 94vh; max-height: 94dvh;
  display: flex; flex-direction: column; overflow-x: hidden; overflow-y: auto;
  font: 13px/1.4 -apple-system, "Segoe UI", Roboto, sans-serif; }
.dmt-header { display: flex; gap: 6px; align-items: center; padding: 8px 10px;
  border-bottom: 1px solid #26282e; flex-wrap: wrap; }
.dmt-title { font-weight: 600; margin-right: 8px; white-space: nowrap; }
.dmt-sub { color: #8f93a0; font-size: 11px; margin-right: 10px; white-space: nowrap; }
.dmt-btn { background: #26282e; color: #d8dae0; border: 1px solid #3a3d45;
  border-radius: 6px; padding: 4px 10px; cursor: pointer; font: inherit; }
.dmt-btn:hover { background: #32353d; }
.dmt-btn.primary { background: #2d5a8a; border-color: #3d74ad; }
.dmt-btn.primary:hover { background: #38699f; }
.dmt-btn.warn { color: #ff8a8a; }
.dmt-btn.danger { background: #7a2a2a; border-color: #a04040; color: #ffd8d8; }
.dmt-btn.danger:hover { background: #8f3232; }
/* Reset confirmation - the button erases every saved edit at once, so it
   asks first (Yes/Cancel), sits on the far left of the header away from
   Apply, and defaults focus to Cancel so a stray Enter does not confirm */
.dmt-confirm { position: absolute; inset: 0; z-index: 20;
  background: rgba(0,0,0,.45); display: flex; align-items: center;
  justify-content: center; }
.dmt-confirm-box { background: #202227; border: 1px solid #3a3d45;
  border-radius: 8px; padding: 14px 16px; max-width: 460px;
  box-shadow: 0 8px 32px rgba(0,0,0,.55); }
.dmt-confirm-msg { margin-bottom: 12px; color: #e2e4ea; }
.dmt-confirm-row { display: flex; justify-content: flex-end; gap: 8px; }
.dmt-chk { display: inline-flex; align-items: center; gap: 4px; color: #a9adb8;
  font-size: 12px; user-select: none; cursor: pointer; }
.dmt-scroll { flex: 1 1 auto; min-height: 0; overflow: auto;
  overflow-anchor: none; background: #101114; cursor: default; }
.dmt-content { position: relative; }
/* rows are pinned to the scrollport with an explicit transform (renderAll)
   - position: sticky left here quantizes subpixel scroll offsets and drifts
   under browser page zoom / HiDPI, which showed up as the rows walking
   apart across zoom in/out cycles */
.dmt-content canvas { display: block; position: absolute; left: 0; top: 0;
  will-change: transform; }
/* the body caps at the viewport so the dialog never outgrows it: header
   and footer keep their space, the track area shrinks (scrolling inside)
   instead of pushing the controls offscreen */
.dmt-body { position: relative; display: flex; flex-direction: column;
  overflow: hidden; min-height: 180px; flex: 0 1 auto;
  max-height: min(46vh, 420px); max-height: min(46dvh, 420px); }
.dmt-float { position: absolute; inset: 0; pointer-events: none; }
.dmt-footer { display: flex; justify-content: space-between; gap: 10px;
  padding: 6px 10px; border-top: 1px solid #26282e; color: #8f93a0; font-size: 11px; }
.dmt-status { color: #d8b96a; }
.dmt-hint { text-align: right; }
`;

let styleInjected = false;
function injectStyles() {
    if (styleInjected) return;
    const tag = document.createElement("style");
    tag.textContent = CSS;
    document.head.appendChild(tag);
    styleInjected = true;
}

function fmtClock(t) {          // seconds -> m:ss.d (h: prefixed past an hour)
    const ms = Math.max(0, Math.round(t * 1000));
    const h = Math.floor(ms / 3600000), m = Math.floor(ms / 60000) % 60;
    const s = Math.floor(ms / 1000) % 60, d = Math.floor(ms / 100) % 10;
    const base = (h ? h + ":" + String(m).padStart(2, "0") : String(m)) + ":" +
        String(s).padStart(2, "0");
    return base + "." + d;
}

// ---------------------------------------------------------------------------
async function fetchBundle(node) {
    const res = await fetch("/dubmerge/timeline/" + node.id, { cache: "no-store" });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.error || ("HTTP " + res.status));
    return body;
}

// the most recently opened timeline overlay - it owns the Space hotkey
let activeTimeline = null;

async function openTimeline(node) {
    injectStyles();
    const manifest = await fetchBundle(node);
    // a manifest without (or below) the protocol this editor speaks is
    // served by an older python: the pack was copied in but ComfyUI was
    // never restarted.  That python drops the current edit entries - warn
    // loudly instead of letting the merge silently ignore the user's edits.
    const stalePython = manifest.edit_protocol == null ||
        manifest.edit_protocol < EDIT_PROTOCOL;
    const editsWidget = (node.widgets || []).find((w) => w.name === "timeline_edits");
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) throw new Error("Web Audio is not available in this browser");
    const ac = new AC();

    async function getWav(name) {
        const r = await fetch("/dubmerge/timeline/" + node.id + "/file/" +
            encodeURIComponent(name), { cache: "no-store" });
        if (!r.ok) throw new Error("failed to fetch " + name);
        return ac.decodeAudioData(await r.arrayBuffer());
    }

    const voiceBuf = await getWav(manifest.voice);
    const pieces = [];
    for (const p of manifest.pieces) {
        const fileBuf = await getWav(p.file);
        // placement exactly as the merge plays it: whole-track takes are
        // full passes read at absolute track offsets (file sample i sounds
        // at track i), everything else is a segment clip placed at its
        // window start.  Older caches (no placement fields) fall back to
        // place-clip semantics - the next merge re-classifies anyway.
        const layout = p.layout || "place";
        const aS = Math.max(0, (p.a_ms != null ? p.a_ms : p.start_ms) / 1000);
        const bS = p.b_ms != null ? p.b_ms / 1000 : aS + fileBuf.duration;
        let audioS = p.audio_ms != null ? p.audio_ms / 1000 : aS;
        let buf = fileBuf;
        if (layout === "overlay") {
            // the piece is exactly the slice the merge plays, at the window
            const srC = fileBuf.sampleRate;
            const i0 = Math.max(0, Math.min(fileBuf.length, Math.round(aS * srC)));
            const i1 = Math.max(i0, Math.min(fileBuf.length, Math.round(bS * srC)));
            const sl = ac.createBuffer(1, Math.max(1, i1 - i0), srC);
            sl.getChannelData(0).set(fileBuf.getChannelData(0).subarray(i0, i1));
            buf = sl;
            audioS = aS;
        }
        pieces.push({
            id: String(p.id),
            take: p.take,
            filename: p.filename || "",
            layout,
            samples: buf.getChannelData(0),
            buf,
            start: aS,
            // audio_ms may sit BEFORE the window start: a left-edge drag cut
            // the file's baked head (window starts inside the buffer) - take
            // it as drawn, or the piece desyncs from the merge on reopen
            audioStart: layout === "overlay" ? aS : audioS,
            end: Math.max(aS + MIN_SPAN, bS),
            engineQueued: !!p.queued,
            amb: false,          // alt+click mark: original continues under it
            init: null,          // filled after saved edits are applied
            engineInit: null,    // the bundle's own geometry (reset baseline)
            hadSavedEdit: false, // restored from the saved JSON (Apply re-emits)
            edited: false,
            userEdited: false,   // set by a real user action (drag / saved edit)
        });
    }
    // the bundle's geometry is the "no user edit" baseline: double-click
    // reset (and Reset) must go back to THIS, not to a restored saved edit
    for (const p of pieces) {
        p.engineInit = { start: p.start, audioStart: p.audioStart, end: p.end };
    }
    // restore the saved adjustments: prefer the widget's current JSON and
    // fall back to the copy the last merge run embedded in the bundle when
    // the widget comes up empty (stale graph state, wipe accidents)
    const widgetRaw = editsWidget ? String(editsWidget.value == null ? "" : editsWidget.value) : "";
    const editsFromWidget = widgetRaw.trim() !== "";
    const restored = applySavedEdits(pieces, editsFromWidget
        ? widgetRaw : String(manifest.timeline_edits || ""));
    // mirror the node's queue_overlaps: pieces the engine would queue (an
    // earlier dub's audio runs past this one's timestamp) load already
    // queued, so the editor shows the layout the merge produces
    const qw = (node.widgets || []).find((w) => w.name === "queue_overlaps");
    const sw = (node.widgets || []).find((w) => w.name === "silence_db");
    const aw = (node.widgets || []).find((w) => w.name === "ambient_db");
    if (activeTimeline) activeTimeline.close();   // one timeline at a time
    const editor = new TimelineEditor(node, manifest, ac, voiceBuf, pieces,
        editsWidget, {
            queueOverlaps: qw ? !!qw.value : true,
            silenceDb: sw && Number.isFinite(Number(sw.value)) ? Number(sw.value) : -50,
            ambientDb: aw && Number.isFinite(Number(aw.value)) ? Number(aw.value) : -12,
            restored,
            editsFromWidget,
            stalePython,
        });
    const queuedAtLoad = editor.queuePass(false).shifted.length;
    for (const p of pieces) p.init = { start: p.start, audioStart: p.audioStart, end: p.end };

    editor.show(queuedAtLoad);
}

// ---------------------------------------------------------------------------
// Saved-edit restore + re-emission - pure functions, mirrored verbatim by
// the regression harness in scripts/repro_timeline_restore.mjs (the code
// between the @dubmerge-pure markers is extracted and eval'd there).
// ---------------------------------------------------------------------------

/* @dubmerge-pure-start */
function normalizeSavedEdits(raw) {
    // Accepts every shape the node's parse_timeline_edits accepts:
    //   {"version": 1, "edits": [{id, ...}, ...]}  (the editor's Apply form)
    //   [{id, ...}, ...]                           (a bare list)
    //   {"<piece id>": {...}, ...}                 (an {id: {...}} map)
    // Returns {list, error}; malformed input never throws.
    if (raw == null || String(raw).trim() === "") return { list: [], error: null };
    let data;
    try { data = JSON.parse(String(raw)); }
    catch (err) {
        return { list: [], error: "not valid JSON (" +
            (err && err.message ? err.message : err) + ")" };
    }
    if (Array.isArray(data)) return { list: data, error: null };
    if (data && Array.isArray(data.edits)) return { list: data.edits, error: null };
    if (data && typeof data === "object") {
        const list = [];
        for (const k of Object.keys(data)) {
            const v = data[k];
            if (v && typeof v === "object") list.push(Object.assign({ id: k }, v));
        }
        return { list, error: null };
    }
    return { list: [], error: "expected the editor's JSON ({\"version\":1,\"edits\":[...]}), a list or an {id: {...}} map, got " +
        (data === null ? "null" : typeof data) };
}

function applySavedEdits(pieces, raw) {
    // Restores the saved adjustments onto freshly-built pieces.  Returns a
    // stats object so nothing is ever silently ignored: {entries, restored,
    // marks, stale, overlay, error}.  Geometry entries set hadSavedEdit so
    // Apply re-emits them unchanged - a plain re-open followed by Apply can
    // no longer wipe the saved edits.
    const stats = { entries: 0, restored: 0, marks: 0, stale: 0, overlay: 0,
        error: null };
    const norm = normalizeSavedEdits(raw);
    stats.error = norm.error;
    if (norm.error) return stats;
    const byId = new Map(pieces.map((p) => [p.id, p]));
    for (const e of norm.list) {
        stats.entries++;
        const p = e ? byId.get(String(e.id)) : null;
        if (!p) { stats.stale++; continue; }
        if (p.layout === "overlay") { stats.overlay++; continue; }
        if (e.start_ms == null) {
            // mark-only entry (no geometry): just the ambient flag
            if (e.amb) { p.amb = true; stats.marks++; }
            continue;
        }
        p.start = Math.max(0, (e.start_ms || 0) / 1000);
        // audio_ms may sit before the window start (a left-edge drag cut the
        // file's baked head) - restore exactly what was saved
        p.audioStart = e.audio_ms != null ? e.audio_ms / 1000 : p.start;
        // the window may end before the piece audio does (an edge drag cut
        // the file's baked tail silence) - restore exactly what was saved
        p.end = Math.max(p.start + MIN_SPAN, (e.end_ms || 0) / 1000);
        p.amb = !!e.amb;
        p.edited = true;
        p.userEdited = true;
        p.hadSavedEdit = true;
        stats.restored++;
    }
    return stats;
}

function pieceChangedPure(p) {
    const i = p.init;
    if (!i) return false;
    return Math.abs(p.start - i.start) > 1e-6 ||
           Math.abs(p.audioStart - i.audioStart) > 1e-6 ||
           Math.abs(p.end - i.end) > 1e-6;
}

function buildEditsJson(pieces) {
    // Serializes the pieces' adjustments exactly like the old in-class
    // applyEdits did - except that pieces restored from the saved JSON are
    // always re-emitted (hadSavedEdit), so Apply can no longer silently
    // drop or wipe them.  Engine-derived queue positions are still never
    // written (only real user edits + restored edits go out).
    const edits = [];
    for (const p of pieces) {
        if (p.layout === "overlay") continue;   // merge ignores these
        const changed = (p.userEdited && pieceChangedPure(p)) || p.hadSavedEdit;
        if (changed) {
            const e = {
                id: p.id,
                start_ms: Math.round(p.start * 1000),
                end_ms: Math.round(p.end * 1000),
                audio_ms: Math.round(p.audioStart * 1000),
            };
            if (p.amb) e.amb = true;
            edits.push(e);
        } else if (p.amb) {
            edits.push({ id: p.id, amb: true });
        }
    }
    return edits.length ? JSON.stringify({ version: 1, edits }) : "";
}
/* @dubmerge-pure-end */

// ---------------------------------------------------------------------------
class TimelineEditor {
    constructor(node, manifest, ac, voiceBuf, pieces, editsWidget, opts = {}) {
        this.node = node;
        this.manifest = manifest;
        this.ac = ac;
        this.voiceBuf = voiceBuf;
        this.voice = voiceBuf.getChannelData(0);
        this.pieces = pieces;
        this.editsWidget = editsWidget || null;
        // saved-edit restore outcome (from openTimeline): drives the status
        // line so ignored edits are never silent
        this.restored = opts.restored || null;
        this.stalePython = !!opts.stalePython;
        this.editsFromWidget = opts.editsFromWidget !== false;
        this.queueOverlaps = opts.queueOverlaps !== false;
        const db = Number(opts.silenceDb);
        this.thresh = Math.pow(10, (Number.isFinite(db) ? db : -50.0) / 20);
        // level of the original span under pieces marked ambient (alt+click),
        // mirroring the node's ambient_db widget - the merge mixes the gated
        // bed at its own level plus the replaced stem at this gain there
        const adb = Number(opts.ambientDb);
        this.ambDb = Number.isFinite(adb) ? adb : -12.0;
        this.ambGain = Math.pow(10, this.ambDb / 20);
        this.pps = 100;                       // pixels per second (zoom)
        this.playheadT = 0;
        this.playing = null;                  // {sources, voiceGain, dubsGain, raf}
        this.selected = null;
        this.dragging = null;
        this.muteVoice = false;
        this.muteDubs = false;
        this.muteUnderWindows = true;
        this.follow = true;
    }

    t0() { return this.scrollEl.scrollLeft / this.pps; }
    cssW() { return this.scrollEl.clientWidth; }
    trackDur() { return this.manifest.duration_ms / 1000; }
    duration() {                          // editable span: track + any dragged tail
        return Math.max(this.trackDur(), ...this.pieces.map((p) => p.end), 1) + 1;
    }

    show(queuedAtLoad) {
        const wrap = document.createElement("div");
        wrap.className = "dmt-overlay";
        wrap.innerHTML = `
        <div class="dmt-panel">
          <div class="dmt-header">
            <span class="dmt-title">Dub timeline</span>
            <span class="dmt-sub"></span>
            <button class="dmt-btn play">▶ Play</button>
            <button class="dmt-btn zoom-out">−</button>
            <button class="dmt-btn zoom-in">+</button>
            <button class="dmt-btn fit">Fit</button>
            <label class="dmt-chk"><input type="checkbox" class="f-follow" checked>follow</label>
            <label class="dmt-chk"><input type="checkbox" class="f-mv">mute voice</label>
            <label class="dmt-chk"><input type="checkbox" class="f-md">mute dubs</label>
            <label class="dmt-chk"><input type="checkbox" class="f-mw" checked>silence voice under dubs</label>
            <button class="dmt-btn snap">Snap</button>
            <button class="dmt-btn reset warn">Reset</button>
            <span style="flex:1"></span>
            <button class="dmt-btn apply primary">Apply to node</button>
            <button class="dmt-btn close">✕</button>
          </div>
          <div class="dmt-body">
            <div class="dmt-scroll">
              <div class="dmt-content">
                <canvas class="cv-ruler" height="${RULER_H}"></canvas>
                <canvas class="cv-voice" height="${VOICE_H}"></canvas>
                <canvas class="cv-dubs" height="${DUBS_H}"></canvas>
              </div>
            </div>
            <canvas class="dmt-float"></canvas>
          </div>
          <div class="dmt-footer">
            <span class="dmt-status"></span>
            <span class="dmt-hint">space = play/stop · alt+click piece = mark ambient ·
              drag = retime · drag edge = add front/end silence ·
              overlapping dubs queue one after the other ·
              ctrl+wheel = zoom · click ruler = seek · double-click piece = reset</span>
          </div>
          <div class="dmt-confirm" style="display:none">
            <div class="dmt-confirm-box">
              <div class="dmt-confirm-msg"></div>
              <div class="dmt-confirm-row">
                <button class="dmt-btn ccancel">Cancel</button>
                <button class="dmt-btn cyes danger">Yes, reset</button>
              </div>
            </div>
          </div>
        </div>`;
        document.body.appendChild(wrap);
        this.wrap = wrap;
        this.scrollEl = wrap.querySelector(".dmt-scroll");
        this.contentEl = wrap.querySelector(".dmt-content");
        this.cvRuler = wrap.querySelector(".cv-ruler");
        this.cvVoice = wrap.querySelector(".cv-voice");
        this.cvDubs = wrap.querySelector(".cv-dubs");
        this.cvFloat = wrap.querySelector(".dmt-float");
        this.subEl = wrap.querySelector(".dmt-sub");
        this.statusEl = wrap.querySelector(".dmt-status");
        this.subEl.textContent =
            this.pieces.length + " dub piece(s) · track " + fmtClock(this.trackDur()) +
            " · " + (this.manifest.sep_method || "voice");

        wrap.querySelector(".close").onclick = () => this.close();
        wrap.addEventListener("pointerdown", (e) => {
            if (e.target === wrap) this.close();
        });
        wrap.querySelector(".play").onclick = () => this.togglePlay();
        wrap.querySelector(".zoom-in").onclick = () => this.zoom(1.3, this.cssW() / 2);
        wrap.querySelector(".zoom-out").onclick = () => this.zoom(1 / 1.3, this.cssW() / 2);
        wrap.querySelector(".fit").onclick = () => this.fit();
        wrap.querySelector(".f-follow").onchange = (e) => { this.follow = e.target.checked; };
        wrap.querySelector(".f-mv").onchange = (e) => {
            this.muteVoice = e.target.checked;
            if (this.playing) this.restartAt(this.playheadT);
        };
        wrap.querySelector(".f-md").onchange = (e) => {
            this.muteDubs = e.target.checked;
            if (this.playing) this.restartAt(this.playheadT);
        };
        wrap.querySelector(".f-mw").onchange = (e) => {
            this.muteUnderWindows = e.target.checked;
            if (this.playing) this.restartAt(this.playheadT);
        };
        wrap.querySelector(".apply").onclick = () => this.applyEdits();
            wrap.querySelector(".snap").onclick = () => this.confirmSnap();
            wrap.querySelector(".reset").onclick = () => this.confirmReset();
            // Reset confirmation: Yes wipes, Cancel (or Esc, or clicking the
            // backdrop) leaves everything untouched.  Cancel is focused by
            // default so a stray Enter cannot confirm the erase.
            this.confirmEl = wrap.querySelector(".dmt-confirm");
            this.confirmEl.onclick = (e) => {
                if (e.target === this.confirmEl) this.hideConfirm(true);
            };
            wrap.querySelector(".ccancel").onclick = () => this.hideConfirm(true);
            wrap.querySelector(".cyes").onclick = () => {
                // dispatch on the dialog that asked: snap Yes runs the
                // batch alignment, reset Yes wipes - confirmAction is read
                // BEFORE hideConfirm clears it
                const action = this.confirmAction;
                this.hideConfirm(false);
                if (action === "snap") this.snapAll();
                else this.resetEdits();
            };

        this.scrollEl.addEventListener("scroll", () => this.renderAll());
        this.scrollEl.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });
        // own ctrl+wheel on the whole dialog: the browser page zoom must
        // never fire while the timeline is open - at high zoom the panel
        // outgrows the viewport and clips its own header and voice track
        // with no scrollbar to get them back.  Over the track area the
        // gesture timeline-zooms (onWheel above); anywhere else it is
        // swallowed so the page zoom cannot creep in either.
        wrap.addEventListener("wheel", (e) => {
            if (!e.ctrlKey && !e.metaKey) return;
            e.preventDefault();
        }, { passive: false });
        // window-level capture ownership of ctrl+wheel while the dialog is
        // open: capture listeners on window/document run BEFORE anything on
        // this overlay, so ComfyUI/litegraph (or an extension) could eat or
        // re-dispatch the gesture before the timeline ever sees it - seen as
        // ctrl+scroll-down never zooming out while the graph zoomed in
        // behind the backdrop instead.  This listener runs first and keeps
        // the gesture for the dialog (plain wheels pass through untouched).
        window.addEventListener("wheel", this._onWinWheel = (e) => {
            if (!this.wrap || !this.wrap.contains(e.target)) return;
            if (!e.ctrlKey && !e.metaKey) return;
            e.preventDefault();
            e.stopImmediatePropagation();
            if (!this.scrollEl.contains(e.target)) return;
            const r = this.scrollEl.getBoundingClientRect();
            this.zoom(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX - r.left);
        }, { passive: false, capture: true });
        this.cvRuler.addEventListener("pointerdown", (e) => this.seekFromEvent(e));
        this.cvVoice.addEventListener("pointerdown", (e) => this.seekFromEvent(e));
        this.bindDubsEvents();
        window.addEventListener("resize", this._onResize = () => this.resize());
        activeTimeline = this;
        // capture phase: Space must be taken before ComfyUI's canvas
        // handlers see it (it pans the graph when held)
        document.addEventListener("keydown", this._onKey = (e) => {
            if (e.key === "Escape") {
                // Escape while the Reset confirmation is up cancels the
                // dialog - it must not close the whole timeline underneath
                // a prompt that asks whether to erase hours of work
                if (this.confirmEl && this.confirmEl.style.display !== "none") {
                    this.hideConfirm(true);
                    return;
                }
                this.close();
                return;
            }
            if (e.code !== "Space" || e.repeat ||
                e.altKey || e.ctrlKey || e.metaKey) return;
            if (activeTimeline !== this) return;
            const t = e.target;
            if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" ||
                      t.tagName === "SELECT" || t.isContentEditable)) return;
            e.preventDefault();
            e.stopImmediatePropagation();
            this.togglePlay();
        }, true);

        this.resize();
        this.fitIfNeeded();
        this.renderAll();
        // what happened to the saved timeline_edits on load - spelled out,
        // never silently ignored
        const rs = this.restored;
        let status = "";
        if (this.stalePython) {
            status = "WARNING: the merge node's python is OLDER than this " +
                "editor (the pack was updated but ComfyUI was not " +
                "restarted) - the merge is dropping timeline edits, so " +
                "trims/moves will not reach the audio.  Fully restart " +
                "ComfyUI, then hard-refresh the browser.  ";
            console.warn("[dub-merge] manifest edit_protocol " +
                (this.manifest.edit_protocol == null ? "missing" :
                    this.manifest.edit_protocol) + " < " + EDIT_PROTOCOL +
                " - the running python predates the editor's edit protocol");
        }
        if (rs && rs.error) {
            status = "timeline_edits could NOT be applied: " + rs.error +
                " - showing the unedited timeline; the widget value is kept " +
                "(fix it by hand, or press Reset to clear it)";
        } else if (rs && (rs.restored || rs.marks)) {
            status = "Restored " + rs.restored + " saved edit(s)" +
                (rs.marks ? " + " + rs.marks + " ambient mark(s)" : "") +
                " from " + (this.editsFromWidget
                    ? "the timeline_edits widget"
                    : "the last merge run (the widget was empty)") +
                " - edit more or Apply to re-save";
            if (rs.stale) {
                status += " · " + rs.stale + " skipped (piece id(s) no longer match - takes changed?)";
            }
            if (rs.overlay) {
                status += " · " + rs.overlay + " skipped (full passes)";
            }
        } else if (rs && rs.entries) {
            status = "timeline_edits is set but none of its " + rs.entries +
                " edit(s) matched a piece - the takes changed since it was " +
                "saved; the widget value is kept (Reset clears it)";
        }
        if (queuedAtLoad) {
            status += (status ? " · " : "") + queuedAtLoad +
                " overlapped piece(s) queued one after the other (queue overlaps)";
        }
        this.setStatus(status);
    }

    close() {
        this.stopPlayback();
        window.removeEventListener("wheel", this._onWinWheel, true);
        window.removeEventListener("resize", this._onResize);
        document.removeEventListener("keydown", this._onKey, true);
        if (activeTimeline === this) activeTimeline = null;
        this.ac.close().catch(() => {});
        this.wrap.remove();
    }

    setStatus(text) { this.statusEl.textContent = text; }

    // -- layout / rendering --------------------------------------------------
    resize() {
        const w = this.cssW(), dpr = window.devicePixelRatio || 1;
        // the rows stack vertically and each canvas paints its own row from
        // ITS y=0, so every canvas must sit at its row offset - with all
        // three at top:0 the opaque dubs row painted over the voice row and
        // the ruler, hiding the original track at any zoom level
        const tops = [0, RULER_H, RULER_H + VOICE_H];
        let ri = 0;
        for (const cv of [this.cvRuler, this.cvVoice, this.cvDubs]) {
            const cssH = parseInt(cv.getAttribute("height"), 10);
            cv.style.width = w + "px";
            cv.style.height = cssH + "px";
            cv.style.top = tops[ri++] + "px";
            cv.width = Math.round(w * dpr);
            cv.height = Math.round(cssH * dpr);
        }
        this.cvFloat.style.width = w + "px";
        this.cvFloat.style.height = (RULER_H + VOICE_H + DUBS_H) + "px";
        this.cvFloat.width = Math.round(w * dpr);
        this.cvFloat.height =
            Math.round((RULER_H + VOICE_H + DUBS_H) * dpr);
        this.contentEl.style.width =
            Math.max(this.duration() * this.pps, w) + "px";
        // the rows are absolutely positioned - give the content its height
        this.contentEl.style.height = (RULER_H + VOICE_H + DUBS_H) + "px";
        this.renderAll();
    }

    fitIfNeeded() {
        if (this.duration() * this.pps < this.cssW() * 0.9) this.fit();
    }

    fit() {
        this.pps = Math.max(2, (this.cssW() - 20) / this.duration());
        this.resize();
        this.scrollEl.scrollLeft = 0;
        // render once more in the same task: resize() drew with the old
        // scroll offset, and the scroll event that would correct it only
        // fires later - never paint the intermediate frame
        this.renderAll();
    }

    zoom(factor, anchorX) {
        const t = this.t0() + anchorX / this.pps;
        this.pps = Math.min(800, Math.max(2, this.pps * factor));
        this.resize();
        this.scrollEl.scrollLeft = Math.max(0, t * this.pps - anchorX);
        // same-task re-render at the final offset (see fit)
        this.renderAll();
    }

    onWheel(e) {
        if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            const rect = this.scrollEl.getBoundingClientRect();
            this.zoom(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX - rect.left);
        } else if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
            e.preventDefault();
            this.scrollEl.scrollLeft += e.deltaY;
        }
    }

    renderAll() {
        // pin the row canvases to the scrollport: they are absolutely
        // positioned at the content's left edge and follow the scroll offset
        // explicitly, so ruler, voice and dubs can never separate
        const pin = "translate3d(" + this.scrollEl.scrollLeft + "px,0,0)";
        for (const cv of [this.cvRuler, this.cvVoice, this.cvDubs]) {
            cv.style.transform = pin;
        }
        this.drawRuler();
        this.drawVoice();
        this.drawDubs();
        this.drawFloat();
    }

    ctx2d(cv) {
        const dpr = window.devicePixelRatio || 1;
        const g = cv.getContext("2d");
        g.setTransform(dpr, 0, 0, dpr, 0, 0);
        return g;
    }

    // min/max envelope of samples between two times, drawn as 1px columns
    drawWave(g, samples, sr, tStart, tEnd, x0, x1, yMid, amp, color) {
        const i0 = Math.max(0, Math.floor(tStart * sr));
        const i1 = Math.min(samples.length, Math.ceil(tEnd * sr));
        if (i1 <= i0 || x1 <= x0) return;
        g.fillStyle = color;
        const cols = Math.max(1, Math.floor(x1 - x0));
        const per = (i1 - i0) / cols;
        for (let c = 0; c < cols; c++) {
            const a = i0 + Math.floor(c * per);
            const b = Math.max(a + 1, i0 + Math.floor((c + 1) * per));
            let lo = 1, hi = -1;
            for (let i = a; i < b; i++) {
                const v = samples[i];
                if (v < lo) lo = v;
                if (v > hi) hi = v;
            }
            if (hi < lo) continue;
            g.fillRect(x0 + c, yMid - hi * amp, 1, Math.max(1, (hi - lo) * amp));
        }
    }

    voicePlan() {
        // sorted, non-overlapping [start, end, amb] spans where the original
        // is silenced; amb = a dub piece marked ambient keeps the original
        // playing under it (at the ambient_db level, like the merge).  When
        // two spans overlap, the earlier one is clipped at the later one's
        // start - the same truncation the merge uses without queueing.
        const ws = this.pieces.filter((p) => !this.isPass(p))
            .map((p) => [p.start, p.end, !!p.amb])
            .sort((a, b) => a[0] - b[0]);
        const out = [];
        for (const w of ws) {
            const last = out[out.length - 1];
            if (last && w[0] < last[1] - 1e-9 && w[2] !== last[2]) {
                last[1] = Math.max(last[0], w[0]);   // earlier take clipped
            }
            if (last && w[0] <= last[1] + 1e-9 && w[2] === last[2]) {
                last[1] = Math.max(last[1], w[1]);
            } else if (w[1] > w[0]) {
                out.push(w.slice());
            }
        }
        return out;
    }

    voiceWindows() {
        // sorted, merged [start, end] spans where the original is silenced
        const out = [];
        for (const [a, b] of this.voicePlan()) {
            if (out.length && a <= out[out.length - 1][1] + 1e-9) {
                out[out.length - 1][1] = Math.max(out[out.length - 1][1], b);
            } else out.push([a, b]);
        }
        return out;
    }

    drawRuler() {
        const g = this.ctx2d(this.cvRuler), w = this.cssW(), t0 = this.t0();
        g.fillStyle = "#1b1d22"; g.fillRect(0, 0, w, RULER_H);
        const steps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
        const step = steps.find((s) => s * this.pps >= 64) || 600;
        const sub = step / 5;
        g.font = "10px monospace"; g.textBaseline = "top";
        const tEnd = t0 + w / this.pps;
        for (let t = Math.floor(t0 / sub) * sub; t <= tEnd + 1e-9; t += sub) {
            const x = (t - t0) * this.pps;
            const major = Math.abs(t / step - Math.round(t / step)) < 1e-6;
            g.strokeStyle = major ? "#4a4e59" : "#2c2f37";
            g.beginPath();
            g.moveTo(x, major ? 8 : RULER_H - 12);
            g.lineTo(x, RULER_H); g.stroke();
            if (major) {
                g.fillStyle = "#9aa0ad";
                g.fillText(fmtClock(t), x + 3, 5);
            }
        }
        g.strokeStyle = "#34363d";
        g.beginPath(); g.moveTo(0, RULER_H - 0.5); g.lineTo(w, RULER_H - 0.5); g.stroke();
    }

    drawVoice() {
        const g = this.ctx2d(this.cvVoice), w = this.cssW(), t0 = this.t0();
        const t1 = t0 + w / this.pps;
        g.fillStyle = "#14161a"; g.fillRect(0, 0, w, VOICE_H);
        g.font = "10px sans-serif"; g.fillStyle = "#20242c";
        g.fillText("original voice (as-is)", 8, 6);
        const mid = VOICE_H / 2;
        this.drawWave(g, this.voice, this.voiceSr(), t0, t1, 0, w, mid,
            VOICE_H * 0.44, "#3fbf9f");
        // exact spans where the merge silences the original; spans under
        // ambient-marked pieces keep it at the ambient_db level instead
        g.font = "10px sans-serif";
        for (const [a, b, marked] of this.voicePlan()) {
            const x0 = (a - t0) * this.pps, x1 = (b - t0) * this.pps;
            if (x1 < 0 || x0 > w) continue;
            g.fillStyle = marked ? "rgba(103, 201, 116, 0.20)"
                                 : "rgba(255, 93, 93, 0.20)";
            g.fillRect(Math.max(0, x0), 0,
                Math.min(w, x1) - Math.max(0, x0), VOICE_H);
            g.strokeStyle = marked ? "rgba(103, 201, 116, 0.55)"
                                   : "rgba(255, 93, 93, 0.55)";
            g.strokeRect(Math.max(0, x0) + 0.5, 0.5,
                Math.min(w, x1) - Math.max(0, x0) - 1, VOICE_H - 1);
            if (marked && x1 - x0 > 34) {
                // green strip: the original continues here (ambience bed)
                g.fillStyle = "rgba(103, 201, 116, 0.9)";
                g.fillRect(Math.max(0, x0), VOICE_H - 4,
                    Math.min(w, x1) - Math.max(0, x0), 3);
                g.fillStyle = "#9fdca6";
                g.fillText("amb " + (this.ambDb >= 0 ? "+" : "") +
                    this.ambDb.toFixed(0) + " dB", Math.max(0, x0) + 4,
                    VOICE_H - 18);
            }
        }
    }

    voiceSr() { return this.voiceBuf.sampleRate; }

    drawDubs() {
        const g = this.ctx2d(this.cvDubs), w = this.cssW(), t0 = this.t0();
        const t1 = t0 + w / this.pps;
        g.fillStyle = "#14161a"; g.fillRect(0, 0, w, DUBS_H);
        g.font = "10px sans-serif"; g.fillStyle = "#20242c";
        g.fillText("dub pieces (drag / resize)", 8, 6);
        const yTop = 20, h = DUBS_H - yTop - 8, mid = yTop + h / 2;
        const dp = this.dragging ? this.dragging.piece : null;
        for (const p of this.pieces) {
            const xA = (p.start - t0) * this.pps;
            const xB = (p.end - t0) * this.pps;
            const xS = (p.audioStart - t0) * this.pps;
            // audio actually played is bounded by the piece span on BOTH
            // sides - the window can start inside the buffer (a left-edge
            // drag cut the file's baked head) and end before the buffer ends
            // (tail cut / audible handover of a queued-over piece)
            const aPlay = Math.max(p.start, p.audioStart);
            const bPlay = Math.min(p.end, p.audioStart + p.buf.duration);
            const playLen = Math.max(0, bPlay - p.audioStart);
            const xE = xS + playLen * this.pps;
            if (xB < 0 || xA > w) continue;
            // pads: silence added in front / at the end by dragging
            g.fillStyle = "rgba(111, 179, 255, 0.08)";
            if (xS - xA > 0) g.fillRect(xA, yTop, xS - xA, h);
            if (xB - xE > 0) g.fillRect(xE, yTop, xB - xE, h);
            g.strokeStyle = "rgba(111, 179, 255, 0.25)";
            for (const [px0, px1] of [[xA, xS], [xE, xB]]) {
                if (px1 - px0 > 2) g.strokeRect(px0 + 0.5, yTop + 0.5, px1 - px0 - 1, h - 1);
            }
            // piece body + its audio waveform, clipped to the played span
            g.fillStyle = p.amb ? "#1f2c22" : "#1e2c3c";
            g.fillRect(xA, yTop, xB - xA, h);
            const aW = Math.max(t0, aPlay), bW = Math.min(t1, bPlay);
            if (bW > aW) {
                this.drawWave(g, p.samples, p.buf.sampleRate, aW, bW,
                    Math.max(0, (aW - t0) * this.pps),
                    Math.min(w, (bW - t0) * this.pps), mid, h * 0.42,
                    p.amb ? "#8fd694" : "#6fb3ff");
            }
            g.lineWidth = 1;
            if (p.queuedAfter) g.setLineDash([5, 3]);
            g.strokeStyle = p.amb ? "#67c974"
                : (p.edited ? "#d8b96a" : "#39404d");
            g.strokeRect(xA + 0.5, yTop + 0.5, xB - xA - 1, h - 1);
            g.setLineDash([]);
            // while dragging: pieces the dropped piece would overlap (the
            // queue pass resolves them on drop) light up red
            if (dp && p !== dp && p.start < dp.end && dp.start < p.end) {
                g.strokeStyle = "rgba(255, 93, 93, 0.95)";
                g.lineWidth = 2;
                g.strokeRect(xA + 0.5, yTop + 0.5, xB - xA - 1, h - 1);
                g.lineWidth = 1;
            }
            g.fillStyle = "#c9cedb";
            g.font = "10px sans-serif";
            g.fillText("take " + p.take +
                    (p.layout === "overlay" ? " (pass)" : "") +
                    (p.queuedAfter ? " ⇥" : "") + (p.edited ? " ✎" : "") +
                    (p.amb ? " amb" : ""),
                xA + 4, yTop + 4, Math.max(20, xB - xA - 8));
            if (this.selected === p) {
                g.strokeStyle = "#ffd166";
                g.lineWidth = 2;
                g.strokeRect(xA, yTop, xB - xA, h);
                g.lineWidth = 1;
                if (p.layout !== "overlay") {
                    g.fillStyle = "#ffd166";
                    g.fillRect(xA - 3, mid - 12, 6, 24);
                    g.fillRect(xB - 3, mid - 12, 6, 24);
                }
            }
        }
    }

    drawFloat() {
        const g = this.ctx2d(this.cvFloat), w = this.cssW();
        const H = RULER_H + VOICE_H + DUBS_H;
        g.clearRect(0, 0, w, H);
        const x = (this.playheadT - this.t0()) * this.pps;
        if (x < -20 || x > w + 20) return;
        g.strokeStyle = "#ff5d5d";
        g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke();
        g.font = "10px monospace";
        const label = fmtClock(this.playheadT);
        g.fillStyle = "#ff5d5d";
        g.fillRect(x + 2, 2, g.measureText(label).width + 8, 14);
        g.fillStyle = "#17181c";
        g.fillText(label, x + 6, 5);
    }

    // -- dubs canvas interaction ---------------------------------------------
    bindDubsEvents() {
        const cv = this.cvDubs;
        cv.style.touchAction = "none";
        cv.addEventListener("pointerdown", (e) => {
            const rect = cv.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const t = this.t0() + x / this.pps;
            const hit = this.hitPiece(t, x);
            if (!hit) { this.selected = null; this.renderAll(); return; }
            this.selected = hit.piece;
            if (e.altKey) {
                // alt+click toggles the ambient mark: the original span
                // (leaked crowd / rain / room tone) keeps playing under the
                // piece - at the ambient_db level in the preview, as the
                // gated bed + original-at-ambient_db in the merge
                if (hit.piece.layout === "overlay") {
                    this.setStatus("full passes cannot be marked ambient - " +
                        "the pass IS the original audio");
                } else {
                    hit.piece.amb = !hit.piece.amb;
                    this.setStatus(hit.piece.amb
                        ? "take " + hit.piece.take + " marked ambient: the " +
                          "original span plays under it at " +
                          (this.ambDb >= 0 ? "+" : "") + this.ambDb.toFixed(1) +
                          " dB (leaked crowd/rain continues) - press Apply to " +
                          "keep the mark"
                        : "take " + hit.piece.take +
                          " ambient mark removed (original muted under it)");
                    if (this.playing) this.restartAt(this.playheadT);
                }
                this.renderAll();
                return;
            }
            if (hit.piece.layout === "overlay") {
                // full passes are read at absolute offsets - the merge
                // ignores their edits, so there is nothing to drag here
                this.dragging = null;
                this.setStatus("take " + hit.piece.take +
                    " is a full pass (read at absolute offsets) - retiming " +
                    "it requires a short clip take");
                this.renderAll();
                return;
            }
            this.dragging = { mode: hit.mode, piece: hit.piece,
                grabDt: t - hit.piece.start };
            cv.setPointerCapture(e.pointerId);
            this.renderAll();
        });
        cv.addEventListener("pointermove", (e) => {
            const rect = cv.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const t = this.t0() + x / this.pps;
            if (!this.dragging) {
                const hit = this.hitPiece(t, x);
                cv.style.cursor = hit
                    ? (hit.piece.layout === "overlay" ? "default"
                        : hit.mode === "body" ? "grab" : "ew-resize") : "default";
                return;
            }
            const d = this.dragging, p = d.piece;
            if (d.mode === "body") {
                // move audio + span together (retime the dub)
                const ns = Math.max(0, t - d.grabDt);
                const delta = ns - p.start;
                p.start += delta; p.audioStart += delta; p.end += delta;
            } else if (d.mode === "left") {
                // pulling left adds front silence; pulling right consumes it
                // - down to the SOUND border: the file's baked lead silence
                // is cut, the sound itself never is
                const snd = this.sndBorders(p);
                p.start = Math.max(0,
                    Math.min(t, p.audioStart + (snd ? snd.start : 0),
                        p.end - MIN_SPAN));
            } else {
                // pulling right adds end silence; pulling left consumes it -
                // down to the SOUND border: the file's baked tail silence is
                // cut, the sound itself never is.  (The old floor was the
                // file end - pieces drawn shorter than their file, e.g.
                // windows clipped at the next take or handed over at the
                // audible end, snapped UP to the file border on the first
                // touch and could never be dragged back.)
                const snd = this.sndBorders(p);
                p.end = Math.max(
                    snd ? p.audioStart + snd.end
                        : p.audioStart + p.buf.duration,
                    p.start + MIN_SPAN, t);
            }
            this.contentEl.style.width =
                Math.max(this.duration() * this.pps, this.cssW()) + "px";
            this.renderAll();
        });
        const finish = () => {
            if (!this.dragging) return;
            const p = this.dragging.piece;
            this.dragging = null;
            // a bare click (select without moving) must not un-edit a piece
            // restored from the saved JSON: the widget still carries its
            // edit, and occupancyOf must keep queueing by the declared end
            // exactly like the merge does
            p.edited = p.edited || this.pieceChanged(p);
            p.userEdited = p.edited;
            // resolve overlaps: pieces the drop left on top of another are
            // queued one after the other, like the merge's queue_overlaps
            const { shifted, clamped } = this.queuePass(true);
            this.contentEl.style.width =
                Math.max(this.duration() * this.pps, this.cssW()) + "px";
            let msg = "take " + p.take + ": original silenced " +
                fmtClock(p.start) + " - " + fmtClock(p.end) +
                ", piece audio at " + fmtClock(p.audioStart);
            for (const q of shifted) {
                if (q.queuedAfter) {
                    msg += " · take " + q.take + " queued after take " +
                        q.queuedAfter.prev.take +
                        " (+" + fmtClock(q.queuedAfter.shift) + ")";
                }
            }
            for (const c of clamped) {
                msg += " · take " + c.take +
                    " span trimmed to its audible end (handover)";
            }
            this.setStatus(msg + " - not applied yet (press Apply to node)");
            this.renderAll();
        };
        cv.addEventListener("pointerup", finish);
        cv.addEventListener("pointercancel", finish);
        cv.addEventListener("dblclick", (e) => {
            const rect = cv.getBoundingClientRect();
            const hit = this.hitPiece(this.t0() + (e.clientX - rect.left) / this.pps,
                e.clientX - rect.left);
            if (!hit || hit.piece.layout === "overlay") return;
            const p = hit.piece;
            const had = p.hadSavedEdit;
            // reset to the bundle's own geometry (engineInit), not to a
            // restored saved edit - double-clicking a restored piece is how
            // one saved edit is dropped
            Object.assign(p, p.engineInit || p.init);
            p.edited = false;
            p.userEdited = false;
            p.hadSavedEdit = false;
            this.queuePass(true);
            this.contentEl.style.width =
                Math.max(this.duration() * this.pps, this.cssW()) + "px";
            this.setStatus("take " + p.take + " reset to its timestamp" +
                (had ? " - its saved edit will no longer be applied" : ""));
            this.renderAll();
        });
    }

    hitPiece(t, x) {
        for (let i = this.pieces.length - 1; i >= 0; i--) {
            const p = this.pieces[i];
            if (t < p.start || t > p.end) continue;
            const xA = (p.start - this.t0()) * this.pps;
            const xB = (p.end - this.t0()) * this.pps;
            const near = this.selected === p ? EDGE_PX : 5;
            if (x - xA < near) return { piece: p, mode: "left" };
            if (xB - x < near) return { piece: p, mode: "right" };
            return { piece: p, mode: "body" };
        }
        return null;
    }

    seekFromEvent(e) {
        const rect = this.scrollEl.getBoundingClientRect();
        const t = Math.max(0, this.t0() + (e.clientX - rect.left) / this.pps);
        this.playheadT = t;
        if (this.playing) this.restartAt(t);
        else this.drawFloat();
    }

    // -- playback --------------------------------------------------------------
    togglePlay() {
        if (this.playing) this.stopPlayback();
        else this.play();
    }

    restartAt(t) {
        this.stopPlayback();
        this.playheadT = t;
        this.play();
    }

    // voice segments [start, end, ambient] to schedule for a play from
    // startT, in order.  Segments that end at/before the playhead are
    // dropped (their audio is already past - scheduling one would ask Web
    // Audio for a negative duration, which throws), everything that
    // straddles the playhead starts at the playhead.
    voiceSchedulePlan(startT, dur) {
        const EPS = 0.005, out = [];
        if (this.muteVoice) return out;
        if (!this.muteUnderWindows) {
            if (dur - startT >= EPS) out.push([startT, dur, false]);
            return out;
        }
        let cur = Math.max(0, startT);
        for (const [a, b, marked] of this.voicePlan()) {
            if (a > cur && Math.min(a, dur) - cur >= EPS) {
                out.push([cur, Math.min(a, dur), false]);
            }
            if (marked) {
                const s = Math.max(a, startT), e = Math.min(b, dur);
                if (e - s >= EPS) out.push([s, e, true]);
            }
            cur = Math.max(cur, b);
        }
        if (dur - cur >= EPS) out.push([cur, dur, false]);
        return out;
    }

    play() {
        // never let a scheduling error brick the controls: without this
        // wrapper, one thrown RangeError inside _play() left this.playing
        // null and the Play button + Space hotkey silently re-threw on
        // every later press
        try {
            this._play();
        } catch (err) {
            this.stopPlayback();
            this.setStatus("Playback failed: " +
                (err && err.message ? err.message : err) +
                " - playback state reset");
        }
    }

    _play() {
        const ac = this.ac;
        ac.resume().catch(() => {});
        const sources = [];
        const voiceGain = ac.createGain();
        const dubsGain = ac.createGain();
        voiceGain.gain.value = this.muteVoice ? 0 : 1;
        dubsGain.gain.value = this.muteDubs ? 0 : 1;
        voiceGain.connect(ac.destination);
        dubsGain.connect(ac.destination);
        const startT = this.playheadT;
        const now = ac.currentTime + 0.03;
        const dur = this.duration();

        // schedule voice[a:b] in sync with the timeline
        const scheduleVoice = (a, b, dest) => {
            a = Math.max(0, Math.min(a, this.voiceBuf.duration));
            b = Math.max(a, Math.min(b, this.voiceBuf.duration));
            const off = Math.max(a, startT);        // absolute position in buffer
            if (b - off < 0.005) return;   // nothing audible left of the playhead
            const src = ac.createBufferSource();
            src.buffer = this.voiceBuf;
            src.connect(dest || voiceGain);
            src.start(now + Math.max(0, off - startT), off, b - off);
            sources.push(src);
        };

        // the original plays everywhere EXCEPT the dub windows - exactly
        // what the merge will do - except under pieces marked ambient:
        // there it continues at the ambient_db level (the merge adds the
        // gated bed at its own level on top, which the editor cannot
        // rebuild)
        const vplan = this.voiceSchedulePlan(startT, dur);
        let ambGain = null;
        if (vplan.some((v) => v[2])) {
            ambGain = ac.createGain();
            ambGain.gain.value = this.ambGain;
            ambGain.connect(ac.destination);
            this.playingAmb = ambGain;
        }
        for (const [a, b, marked] of vplan) {
            scheduleVoice(a, b, marked ? ambGain : null);
        }
        if (!this.muteDubs) {
            for (const p of this.pieces) {
                // the merge plays the piece only inside its window [start,
                // end] - edge drags can cut the file's baked head and/or
                // tail, so preview that window, not the whole buffer
                const a0 = Math.max(p.start, p.audioStart);
                const b0 = Math.min(p.end, p.audioStart + p.buf.duration);
                if (b0 <= startT || b0 - a0 < 0.005) continue;
                const src = ac.createBufferSource();
                src.buffer = p.buf;
                src.connect(dubsGain);
                const a = Math.max(a0, startT);
                const off = a - p.audioStart;   // >= 0
                src.start(now + Math.max(0, a - startT), off, b0 - a);
                sources.push(src);
            }
        }
        const startAC = ac.currentTime;
        const tick = () => {
            if (!this.playing) return;
            this.playheadT = startT + (this.ac.currentTime - startAC);
            if (this.follow) {
                const x = (this.playheadT - this.t0()) * this.pps;
                if (x > this.cssW() * 0.6 || x < 0) {
                    this.scrollEl.scrollLeft =
                        this.playheadT * this.pps - this.cssW() * 0.3;
                }
            }
            this.drawFloat();
            this.playing.raf = requestAnimationFrame(tick);
        };
        this.playing = { sources, voiceGain, dubsGain, raf: 0 };
        tick();
        this.wrap.querySelector(".play").textContent = "■ Stop";
    }

    stopPlayback() {
        if (!this.playing) return;
        cancelAnimationFrame(this.playing.raf);
        for (const s of this.playing.sources) {
            try { s.stop(); } catch { /* already stopped */ }
        }
        try {
            this.playing.voiceGain.disconnect();
            this.playing.dubsGain.disconnect();
        } catch { /* noop */ }
        if (this.playingAmb) {
            try { this.playingAmb.disconnect(); } catch { /* noop */ }
            this.playingAmb = null;
        }
        this.playing = null;
        const btn = this.wrap.querySelector(".play");
        if (btn) btn.textContent = "▶ Play";
    }

    // -- queuing (mirrors the merge's queue_overlaps pass) ---------------------
    isPass(p) {
        // overlay full pass: the engine reads it at absolute track offsets,
        // hands over at the next timestamp - never queued, never queues
        return p.layout === "overlay";
    }

    audibleEnd(p) {
        // index just past the last sample above the silence threshold - the
        // same handover the engine uses (trailing file silence is stripped,
        // so a TTS tail cannot push queued dialogue rightward)
        const s = p.samples;
        for (let i = s.length - 1; i >= 0; i--) {
            if (Math.abs(s[i]) > this.thresh) {
                return (i + 1) / p.buf.sampleRate;
            }
        }
        return 0;
    }

    sndBorders(p) {
        // sound borders inside the piece buffer - first/last sample above
        // the silence threshold, in seconds relative to audioStart.  The
        // edge drags clamp against these (not the file extent), so the
        // baked lead/tail silence of a dub file can be cut all the way to
        // the sound but the sound itself never is.  Cached per piece;
        // null when the buffer is silent (clamps fall back to the file
        // extent).
        if (p._snd === undefined) {
            const s = p.samples;
            let a = -1, b = -1;
            for (let i = 0; i < s.length; i++) {
                if (Math.abs(s[i]) > this.thresh) {
                    if (a < 0) a = i;
                    b = i;
                }
            }
            p._snd = a < 0 ? null
                : { start: a / p.buf.sampleRate,
                    end: (b + 1) / p.buf.sampleRate };
        }
        return p._snd;
    }

    occupancyOf(p) {
        // explicit spans (pieces with an edit) queue by their declared end,
        // plain clips by their audible end - exactly like the merge.  The
        // declared end only owns the handover where it actually differs
        // from the file extent: a tail cut queues by the drawn end, silence
        // added past the file queues by the drawn end, but a window ending
        // at the file border hands over at the audible end - the baked TTS
        // tail must not push the queue (same rule, same 2 ms tolerance).
        if (p.edited || (p.init && this.pieceChanged(p))) {
            const fileEnd = p.audioStart + p.buf.duration;
            return Math.abs(p.end - fileEnd) > 2e-3
                ? p.end : p.audioStart + this.audibleEnd(p);
        }
        return p.audioStart + this.audibleEnd(p);
    }

    queuePass(mark) {
        // walk pieces in timestamp order; a piece overlapped by the previous
        // one is pushed to start right after it (the chain cascades).  The
        // pass only moves pieces for DISPLAY - the merge re-derives the same
        // chain from the timestamps, so Apply never writes these derived
        // positions (only real user edits are written).
        const shifted = [], clamped = [];
        if (!this.queueOverlaps) return { shifted, clamped };
        const EPS = 1e-4;
        // pieces the merge itself queued keep their marker through passes
        for (const p of this.pieces) {
            p.queuedAfter = p.engineQueued ? { fromEngine: true } : null;
        }
        for (let i = 1; i < this.pieces.length; i++) {
            const prev = this.pieces[i - 1], p = this.pieces[i];
            if (this.isPass(prev) || this.isPass(p)) continue;
            const prevEdited = prev.edited || (prev.init && this.pieceChanged(prev));
            const occ = this.occupancyOf(prev);
            if (occ > p.start + EPS) {
                if (!prevEdited && prev.end > occ + EPS) {
                    // handover at the audible end: the previous piece's
                    // silent tail belongs to the queued piece now - end its
                    // drawn span there, or the queued piece is drawn on top
                    // of the tail and the chain reads as one big pile
                    prev.end = occ;
                    clamped.push(prev);
                }
                const shift = occ - p.start;
                p.start += shift; p.audioStart += shift; p.end += shift;
                p.queuedAfter = { prev, at: occ, shift };
                shifted.push(p);
            }
        }
        return { shifted, clamped };
    }

    // -- edits -----------------------------------------------------------------
    pieceChanged(p) {
        return pieceChangedPure(p);
    }

    applyEdits() {
        if (!this.editsWidget) {
            this.setStatus("timeline_edits widget not found on the node");
            return;
        }
        // only pieces the user actually changed are written - the merge's
        // own queue pass re-derives every queued position from the
        // timestamps, so writing derived positions here would pin them to
        // declared-span semantics and desync the handover by the TTS tail.
        // Pieces restored from the saved JSON go out unchanged
        // (hadSavedEdit), so pressing Apply after a plain re-open can no
        // longer wipe the saved edits.  Ambient marks that were never
        // dragged go out as mark-only entries (no geometry), so they cannot
        // pin anything either.
        const value = buildEditsJson(this.pieces);
        const cur = String(this.editsWidget.value == null ? "" : this.editsWidget.value);
        if (!value && cur.trim()) {
            // nothing to write but the widget holds data - never wipe it
            // silently; Reset is the explicit way to clear
            this.setStatus("No edits drawn - kept the existing timeline_edits " +
                "value (use Reset to clear it)");
            return;
        }
        this.editsWidget.value = value;
        if (this.editsWidget.callback) this.editsWidget.callback(this.editsWidget.value);
        app.graph.setDirtyCanvas(true, true);
        let nGeo = 0, nMark = 0;
        try {
            const d = JSON.parse(value || "{}");
            const list = Array.isArray(d.edits) ? d.edits : [];
            for (const e of list) {
                if (e && e.start_ms != null) nGeo++; else nMark++;
            }
        } catch { /* cannot happen (we just built it) */ }
        this.setStatus(value
            ? "Applied " + nGeo + " edit(s)" +
              (nMark ? " (" + nMark + " ambient mark(s))" : "") +
              " to the node - run the merge to hear them"
            : "No edits - node will merge without adjustments");
        this.drawDubs();
    }

    confirmReset() {
        // Reset erases EVERY piece adjustment and the widget value at once
        // (hours of timeline work are one mis-click away), so it asks
        // first.  Counts what would actually be discarded - the adjustments
        // Apply would write (piece edits + ambient marks) plus the raw
        // widget value, matched or not.  Nothing to erase -> no dialog.
        const ej = buildEditsJson(this.pieces);   // module fn, not a method
        let n = 0;
        try {
            n = ej ? (JSON.parse(ej).edits || []).length : 0;
        } catch (err) { n = 0; }
        const raw = this.editsWidget
            ? String(this.editsWidget.value == null ? "" : this.editsWidget.value)
            : "";
        if (n === 0 && raw.trim() === "") { this.resetEdits(); return; }
        this.confirmAction = "reset";
        this.confirmEl.querySelector(".cyes").textContent = "Yes, reset";
        this.confirmEl.querySelector(".dmt-confirm-msg").textContent =
            n === 0
                ? "Reset clears the timeline_edits widget value (no matched " +
                  "piece edits). This cannot be undone."
                : "Reset discards " +
                  (n === 1 ? "1 saved adjustment" : n + " saved adjustments") +
                  " and the timeline_edits widget value. This cannot be undone.";
        this.confirmEl.style.display = "flex";
        this.confirmEl.querySelector(".ccancel").focus();
    }

    confirmSnap() {
        // Snap batch-retimes EVERY placed dub at once (the whole point is
        // skipping the manual drag-per-piece work), so it asks first - the
        // same Yes/Cancel dialog Reset uses, Cancel focused by default so a
        // stray Enter cannot confirm.  Overlay passes are read at absolute
        // sample offsets and cannot be retimed, so they are not offered.
        let n = 0;
        for (const p of this.pieces) {
            if (p.layout !== "overlay") n++;
        }
        if (n === 0) {
            this.setStatus("Nothing to snap - the timeline has no dub pieces");
            return;
        }
        this.confirmAction = "snap";
        this.confirmEl.querySelector(".cyes").textContent = "Yes, snap";
        this.confirmEl.querySelector(".dmt-confirm-msg").textContent =
            "Snap moves all " +
            (n === 1 ? "dub piece" : n + " dub pieces") +
            " to the closest voice-track onset within " +
            Math.round(SNAP_WIN * 1000) + " ms of their sound start" +
            " (pieces with no onset in reach stay put).  " +
            "Review the result, then Apply to node to save.";
        this.confirmEl.style.display = "flex";
        this.confirmEl.querySelector(".ccancel").focus();
    }

    voiceOnsets() {
        // starts of voiced regions in the voice track (channel 0), in track
        // seconds.  A region only counts when at least SNAP_MIN_GAP of
        // silence precedes it - or it opens the track - so micro-dips
        // inside continuous speech are not "starting points"; a window that
        // sits in the middle of one long sound finds no onset and the dub
        // stays where it is.  Cached: the voice buffer never changes while
        // the editor is open.
        if (this._onsets) return this._onsets;
        const v = this.voice, sr = this.voiceSr();
        const minGap = Math.max(1, Math.round(SNAP_MIN_GAP * sr));
        const out = [];
        let silenceRun = 0;
        for (let i = 0; i < v.length; i++) {
            if (Math.abs(v[i]) > this.thresh) {
                // i is the first voiced sample after the silence run - an
                // onset when the gap before it is a real one, or when it
                // opens the track (i === 0, nothing preceded)
                if (i === 0 || silenceRun >= minGap) {
                    out.push(i / sr);
                }
                silenceRun = 0;
            } else {
                silenceRun++;
            }
        }
        this._onsets = out;
        return out;
    }

    snapAll() {
        // pull each placed dub's sound start to the closest voice onset
        // within SNAP_WIN; the whole piece moves (start, audioStart, end
        // together - the same retime a body drag does), trims and pads stay
        // exactly as drawn.  A piece with no onset in reach (only silence
        // in the window, or speech continuous across it) stays untouched.
        const onsets = this.voiceOnsets();
        let moved = 0, aligned = 0, kept = 0, candidates = 0;
        for (const p of this.pieces) {
            if (p.layout === "overlay") continue;
            candidates++;
            const snd = this.sndBorders(p);
            if (!snd) {               // fully silent buffer - nothing to align
                kept++;
                continue;
            }
            const soundStart = p.audioStart + snd.start;
            // onsets are sorted - binary-search into the window
            let lo = 0, hi = onsets.length;
            while (lo < hi) {
                const m = (lo + hi) >> 1;
                if (onsets[m] < soundStart - SNAP_WIN) lo = m + 1;
                else hi = m;
            }
            let best = null, bestD = Infinity;
            for (let k = lo; k < onsets.length &&
                    onsets[k] <= soundStart + SNAP_WIN; k++) {
                const d = Math.abs(onsets[k] - soundStart);
                if (d < bestD) { bestD = d; best = onsets[k]; }
            }
            if (best == null) {
                kept++;
                continue;
            }
            if (bestD < 1e-6) {       // already sitting on the onset
                aligned++;
                continue;
            }
            const delta = Math.max(best - soundStart, -p.start);
            p.start += delta;
            p.audioStart += delta;
            p.end += delta;
            p.edited = true;
            p.userEdited = true;
            moved++;
        }
        this.queuePass(true);
        this.renderAll();
        const parts = [];
        if (moved) parts.push(moved + " of " + candidates + " dub piece(s) " +
            "moved to voice onsets");
        if (aligned) parts.push(aligned + " already on a voice onset");
        if (kept) parts.push(kept + " kept in place (no voice onset within " +
            Math.round(SNAP_WIN * 1000) + " ms)");
        this.setStatus("Snap: " + (parts.join(", ") || "nothing to do") +
            " - Apply to node to save");
        return { moved, aligned, kept, candidates };
    }

    hideConfirm(cancelled) {
        this.confirmEl.style.display = "none";
        if (cancelled) {
            this.setStatus(this.confirmAction === "snap"
                ? "Snap cancelled - your dubs are untouched"
                : "Reset cancelled - your edits are untouched");
        }
        this.confirmAction = null;
    }

    resetEdits() {
        for (const p of this.pieces) Object.assign(p, p.engineInit || p.init);
        for (const p of this.pieces) {
            p.edited = false;
            p.userEdited = false;
            p.hadSavedEdit = false;
            p.amb = false;      // the widget is wiped - marks go too
        }
        this.queuePass(true);
        if (this.editsWidget) this.editsWidget.value = "";
        app.graph.setDirtyCanvas(true, true);
        this.setStatus("Edits cleared - node will merge without adjustments");
        this.renderAll();
    }
}

app.registerExtension({
    name: "zai.DubMergeTimeline",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData || nodeData.name !== "DubMergeTakes") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated
                ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;
            const btn = node.addWidget("button", "🎬 Load Timeline", null, () => {
                openTimeline(node).catch((err) => {
                    alert("Dub timeline: " + (err && err.message ? err.message : err));
                });
            });
            btn.serialize = false;
            return r;
        };
    },
});
