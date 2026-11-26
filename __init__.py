"""ComfyUI Dub Merge.

Merge translated take files (HH.MM.SS.mmm_TTTTT.flac) into a full track.
The highest take number per timestamp wins; superseded lower takes are
moved into a "fail" folder inside the takes folder on every real merge
(move_fails, on by default) so the source folder never needs manual
cleanup.
With the default voice-isolation mode the original is first split into
voice + background (Demucs ML separation when available, ffmpeg centre
extraction otherwise, or an externally supplied background track); the dubs
(highest take number wins) replace the voice track only at their timestamps
and the untouched background is added back, so the background never shifts
or ducks under the dubbed parts.  queue_overlaps queues explosive
back-to-back dialogue: a take whose audio runs past the next take's start
pushes that take to begin right after it ends, so no dubbed phrase is
truncated and no two dubs overlap.  Dub pieces marked ambient in the
timeline editor (alt+click) keep the non-speech part of the replaced stem
(rain, ambience, music bed) playing under their windows - extracted with an
adaptive spectral noise-floor gate, guided by the separated background stem
in voice-isolation mode - plus the original span itself mixed under the dub
at the ambient_db level, so leaked crowd/rain continues under the dub and
continuous background content stays at a constant level instead of ducking
under every phrase, without voice remnants echoing the dub.  Loud slow
ambience events
(a passing train, a rain swell, a distant crowd) are recognised as ambience
by their slow envelope and smooth spectrum and kept at full level,
preserving the immersive atmosphere.  background_only (for ambience-heavy
scenes) goes one step further: the separated voice stem is discarded
entirely and the dubs are merged with the background alone, so ambience
that leaked into the voice stem can never duck under the dubs.

Timeline editor: the merge node has a Load Timeline button (browser UI).
Run the merge once, press the button and a scrollable two-track timeline
opens - the separated original voice on top, the dub pieces at their
timestamps below, marked with second ticks.  Pieces are draggable and
resizable (adding front/end silence), the spans where the original will be
silenced are overlaid on the voice track, and voice + dubs can be played
together for comparison.  Apply writes the adjustments into the node's
timeline_edits widget; without edits the node executes exactly as without
the editor.

Nodes:
  - Load Media (Video/Audio)
  - Merge Translated Takes

Save/preview the results with ComfyUI's core nodes: Save Audio (or Preview
Audio) for the merged audio, Save Text (or the native Preview as Text node)
for the report.
"""

import os
import re

# Allow dropping ffmpeg/ffprobe executables into <pack>/bin on Windows.
_PKG_BIN = os.path.join(os.path.dirname(__file__), "bin")
if os.path.isdir(_PKG_BIN):
    _path = os.environ.get("PATH", "")
    if _PKG_BIN not in _path:
        os.environ["PATH"] = _PKG_BIN + os.pathsep + _path

# Browser-side code (the Load Timeline button / timeline editor UI).
WEB_DIRECTORY = "./web"

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception:  # pragma: no cover - surface a readable install error
    import traceback
    print("[ComfyUI Dub Merge] Failed to import nodes:", flush=True)
    traceback.print_exc()
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}


def _cache_dir_for(node_id):
    """Timeline cache folder for one node instance (temp/dubmerge_timeline/<id>)."""
    import folder_paths
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(node_id))
    return os.path.join(folder_paths.get_temp_directory(), "dubmerge_timeline", safe)


def _install_timeline_routes():
    """Serve the timeline editor's data bundle over HTTP.

    GET /dubmerge/timeline/<node_id>           -> manifest.json (or 404 with a
                                                  helpful error before the first run)
    GET /dubmerge/timeline/<node_id>/file/<n>  -> voice.wav / piece_<id>.wav
    """
    try:
        from server import PromptServer  # noqa: F401
        from aiohttp import web
        routes = PromptServer.instance.routes
    except Exception:
        return  # headless / no server - the button simply won't have data

    @routes.get("/dubmerge/timeline/{node_id}")
    async def dubmerge_timeline_manifest(request):
        node_id = request.match_info.get("node_id", "")
        path = os.path.join(_cache_dir_for(node_id), "manifest.json")
        if not node_id or not os.path.isfile(path):
            return web.json_response(
                {"error": "No timeline data yet - run the Merge Translated Takes "
                          "node once (mode = merge) first, then press Load Timeline."},
                status=404)
        return web.FileResponse(path, headers={"Cache-Control": "no-store"})

    @routes.get("/dubmerge/timeline/{node_id}/file/{name}")
    async def dubmerge_timeline_file(request):
        node_id = request.match_info.get("node_id", "")
        name = request.match_info.get("name", "")
        if not node_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or ".." in name:
            return web.json_response({"error": "bad request"}, status=400)
        path = os.path.join(_cache_dir_for(node_id), name)
        if not os.path.isfile(path):
            return web.json_response({"error": "not found"}, status=404)
        return web.FileResponse(path, headers={"Cache-Control": "no-store"})


try:
    _install_timeline_routes()
except Exception:  # pragma: no cover - never block ComfyUI startup on UI extras
    pass


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
