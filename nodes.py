"""ComfyUI Dub Merge - node definitions.

Nodes:
  DubLoadMedia  - load the full track from a video or audio file (ffmpeg).
  DubMergeTakes - replace the original voice with the translated take audio
                  (voice/background isolation with Demucs ML, ffmpeg centre
                  extraction, or an externally supplied background track) and
                  output the merged AUDIO plus a text report.

Save/preview the results with ComfyUI's core nodes: Save Audio (or Preview
Audio) for the merged audio, Save Text (or any string preview node) for the
report.
"""

import os
import re

import numpy as np
import torch

import folder_paths

from . import dub_core


# --------------------------------------------------------------------------
# Load Media (Video/Audio)
# --------------------------------------------------------------------------

class DubLoadMedia:
    """Load a full track from a video file (mp4/mkv/webm/mov/...) or an audio
    file (wav/flac/mp3/m4a/...), or from an http(s) URL.  ffmpeg is used for
    anything soundfile cannot read directly."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "media_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to the full track: a video (mp4/mkv/webm/mov/avi/...) "
                               "or audio (wav/flac/mp3/m4a/...) file, or an http(s) URL. "
                               "Absolute path, or relative to ComfyUI's input directory.",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "load"
    CATEGORY = "audio/dubbing"

    def load(self, media_path):
        path = dub_core.resolve_media_path(media_path, folder_paths.get_input_directory())
        data, sr = dub_core.load_audio_file(path)
        waveform = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32)).unsqueeze(0)
        return ({"waveform": waveform, "sample_rate": int(sr)},)

    @classmethod
    def IS_CHANGED(cls, media_path):
        try:
            p = dub_core.resolve_media_path(media_path, folder_paths.get_input_directory())
        except Exception:
            return float("NaN")
        if p.startswith(("http://", "https://")):
            return float("NaN")
        try:
            st = os.stat(p)
            return "%s|%d|%d" % (p, st.st_mtime_ns, st.st_size)
        except OSError:
            return float("NaN")


# --------------------------------------------------------------------------
# Merge Translated Takes
# --------------------------------------------------------------------------

class DubMergeTakes:
    """Replace the original voice with the translated takes; keep the background.

    The original is first split into voice + background, the dubs are merged
    into the voice track only and the background is re-added unchanged - so
    the background cannot shift, duck or restart under the dubbed parts.
    Use VOICE-ONLY dub takes in this mode.

    Where the background comes from (isolation_engine):
      auto   - Demucs ML separation when the `demucs` package is installed
               (pip install demucs), otherwise ffmpeg centre extraction.
      demucs - neural vocals/background split; the background is the sum of
               the non-vocal stems and is passed through untouched.  Use this
               (or an external background) when the WHOLE mix ducks under the
               dubs - that happens with mono-like mixes where everything,
               ambience included, is centred and the centre extractor would
               mute it all.
      center - ffmpeg mid/side extraction + exact complement background.
               Exact, but every centred element (ambience, centred music)
               sits in the voice track and is muted under the dubs.
    Alternatively connect your own no-vocals/background track (UVR, demucs
    CLI, DAW export) to the optional `background_audio` input - it is used
    as-is (voice = track - background, sample-exact).

    Timeline editor: the node has a Load Timeline button (browser UI).  Run
    the merge once, press the button and a scrollable two-track timeline
    opens: the separated original voice on top (as-is) and the dub pieces at
    their timestamps below, marked with second ticks.  Pieces are draggable
    (retime the dub) and resizable at both edges (pulling an edge adds
    front/end silence and widens the muted span on the voice track, shown
    as an overlay), and everything can be played together with the original
    for comparison.  Apply writes the adjustments into the timeline_edits
    widget; the next merge silences the original exactly across the edited
    spans.  Without edits (button never pressed / nothing applied) the node
    executes exactly as without the editor.

    queue_overlaps queues overlapping dubs instead of truncating them: when
    a take's audio runs past the next take's start time, the next take is
    shifted to begin right after the current one's audible phrase ends, so
    explosive back-to-back dialogue plays in full, one line after the other.
    Trailing file silence is stripped from the handover - TTS dub files often
    end with several hundred ms of baked-in silence, which would otherwise
    push every queued line rightward and desync the dialogue more and more.

    Dub pieces marked ambient in the timeline editor (alt+click a piece,
    then Apply; ambient_db sets the blend level) keep the non-speech part of
    the replaced stem (rain, ambience, music bed) playing under their
    windows, so continuous background content that the separation placed
    inside the voice stem stays at a constant level instead of ducking under
    every phrase, and the original span itself is mixed under the dub at
    ambient_db so leaked crowd/rain continues while the dub stays dominant.
    Unmarked pieces mute the replaced stem completely.  The gate's noise
    floor is adaptive (it follows the recurring quiet level of every
    time/frequency bin instead of one global value, so dense dialogue no
    longer inflates it), and in voice-isolation mode the separated
    background stem guides the gate as a per-bin reference - voice remnants
    that sit at or below the ambience level (fricatives, breath, soft
    endings), which a plain level gate cannot catch, are suppressed too
    instead of echoing the dub from the background.  A sustained-ambience
    keep additionally recognises loud but slowly varying, spectrally flat
    events (a passing train, a rain swell, a distant crowd) that a plain
    level gate would suppress like speech - by their slow envelope (they
    stay within a few dB of their recent level for many seconds, while
    speech spikes syllable by syllable) and by their smooth spectrum
    (speech is a harmonic comb) - and keeps them at full level, so the
    immersive atmosphere survives the dub windows.

    background_only is for ambience-heavy scenes: the separated voice stem is
    discarded completely and the dubs are merged with the background alone.
    The subtle ambience that leaks into the voice stem otherwise keeps
    ducking under the dubs (the ambient fill rebuilds only an approximation
    of it, and the level difference is very audible), while here the leak is
    gone for good and the background plays at a constant level from start to
    finish.  The original voice is removed as completely as the separation
    removes it - pair it with the demucs engine or a background_audio input
    for the cleanest split.  Needs separation = voice_isolation.

    separation = full_mix replaces the whole mix instead (legacy behaviour).

    Take files: HH.MM.SS.mmm_TTTTT.flac - the highest take number per
    timestamp wins.  Output: merged AUDIO + a text report listing every
    replaced piece with its timestamps.

    Route the outputs into ComfyUI's core nodes: Save Audio / Preview Audio
    for the audio, Save Text (or the native Preview as Text node) for the
    report.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "takes_folder": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Folder with the translated take files (00.01.52.612_00001.flac). "
                               "Absolute path, or relative to ComfyUI's input directory.",
                }),
                "extensions": ("STRING", {
                    "default": ".flac",
                    "tooltip": "File extensions treated as takes. Default '.flac'. "
                               "Example: '.flac .wav' to widen.",
                }),
                "recursive": ("BOOLEAN", {"default": False, "tooltip": "Also scan subfolders."}),
                "timestamp_mode": (["start", "end"], {
                    "default": "start",
                    "tooltip": "start: filename time = where the dubbed phrase starts; the "
                               "window runs until the next take's start (or track end). "
                               "end: filename time = where the phrase ends; the window is "
                               "end minus take duration (for segment-length takes).",
                }),
                "window_mode": (["until_next", "auto_trim"], {
                    "default": "until_next",
                    "tooltip": "until_next: original is muted from each take's start until the "
                               "next take's start (full-track dub passes). auto_trim: detect the "
                               "actual dubbed phrase inside that span by silence threshold "
                               "(takes that are silent outside the phrase); when the phrase ends "
                               "the original voice is held silent for tail_silence_ms (if set) and "
                               "then crossfades back in across the tail pad.",
                }),
                "separation": (["voice_isolation", "full_mix"], {
                    "default": "voice_isolation",
                    "tooltip": "voice_isolation (recommended): the original is first split "
                               "into voice + background; dubs are merged into the voice track "
                               "only and the background is re-added unchanged, so it never "
                               "shifts or ducks under the dub. Use VOICE-ONLY dub takes with "
                               "this mode. full_mix: legacy - dubs replace the entire mix "
                               "including the background.",
                }),
                "isolation_engine": (["auto", "demucs", "center"], {
                    "default": "auto",
                    "tooltip": "How the voice/background split is made. auto: Demucs ML "
                               "separation when the demucs package is installed, otherwise "
                               "ffmpeg centre extraction. demucs: neural vocals/background "
                               "split - use this when the WHOLE mix (background included) "
                               "ducks under the dubs, which happens on mono-like mixes "
                               "where the ambience is also centred; the background stem is "
                               "passed through untouched. center: ffmpeg mid/side "
                               "extraction - sample-exact, but every centred element "
                               "(ambience, centred music) is muted under the dubs, so it "
                               "only fits mixes with a wide stereo background. Or connect "
                               "your own background to the background_audio input.",
                }),
                "voice_low_hz": ("FLOAT", {
                    "default": 100.0, "min": 0.0, "max": 2000.0, "step": 10.0,
                    "tooltip": "center engine only: low edge of the voice band; content below "
                               "it (bass, rumble) always stays in the background. 0 = no "
                               "low cut. Ignored by the demucs engine and the background "
                               "input.",
                }),
                "voice_high_hz": ("FLOAT", {
                    "default": 12000.0, "min": 1000.0, "max": 24000.0, "step": 100.0,
                    "tooltip": "center engine only: high edge of the voice band; content above "
                               "it (cymbals, air) always stays in the background. Raise to "
                               "about sample_rate/2 to disable the high cut.",
                }),
                "demucs_model": (["htdemucs", "htdemucs_ft", "mdx_extra_q"], {
                    "default": "htdemucs",
                    "tooltip": "demucs engine only: separation model. htdemucs = general "
                               "purpose. htdemucs_ft = fine-tuned, better quality, slower. "
                               "mdx_extra_q = alternative network, often strong on speech. "
                               "Model weights are downloaded on first use (~80-160 MB).",
                }),
                "fade_ms": ("INT", {
                    "default": 10, "min": 0, "max": 200, "step": 1,
                    "tooltip": "Equal-gain crossfade at window edges, between contiguous takes, "
                               "and across the auto_trim tail pad (0 = hard cuts).",
                }),
                "gain_db": ("FLOAT", {
                    "default": 0.0, "min": -24.0, "max": 24.0, "step": 0.1,
                    "tooltip": "Volume of the dubbed pieces only, applied to the take "
                               "audio before merging (0 dB = no change; positive = "
                               "louder dubs, negative = quieter).  The original voice "
                               "and the background are untouched, so this cannot clip "
                               "them; the report shows the value on its Settings line.",
                }),
                "silence_db": ("FLOAT", {
                    "default": -50.0, "min": -96.0, "max": -6.0, "step": 0.5,
                    "tooltip": "Silence threshold (level below this counts as silence): "
                               "auto_trim uses it to detect the dubbed phrase inside "
                               "the window.",
                }),
                "tail_pad_ms": ("FLOAT", {
                    "default": 200.0, "min": 0.0, "max": 2000.0, "step": 10.0,
                    "tooltip": "auto_trim / extend: padding kept after the last audible sample. "
                               "In auto_trim the original crossfades back in across this tail "
                               "(equal-power curve), starting after the tail_silence hold if one "
                               "is set - raise it for a longer, smoother blend.",
                }),
                "overflow": (["extend", "truncate", "skip"], {
                    "default": "extend",
                    "tooltip": "When take content passes the end of the track: extend the "
                               "output, truncate at track end, or skip that take.",
                }),
                "mode": (["merge", "report_only"], {
                    "default": "merge",
                    "tooltip": "report_only is a dry run: returns the original audio "
                               "unchanged together with the report.",
                }),
                "queue_overlaps": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Queue overlapping dubs: when a take's audio (start time + "
                               "duration) runs past the next take's start time, the next take "
                               "is shifted to begin right after the current one's audible "
                               "phrase ends - explosive back-to-back dialogue is neither "
                               "truncated nor overlapped.  Trailing file silence is stripped "
                               "from the handover: TTS dub files often end with several "
                               "hundred ms of baked-in silence, which would otherwise push "
                               "every queued line rightward and desync the dialogue more "
                               "and more.  Explicit start-end filename windows queue by "
                               "their declared span; full-track passes keep the legacy "
                               "handover at the next timestamp. Turn off to restore the old "
                               "behaviour (the earlier take is clipped at the next take's "
                               "start).",
                }),
                "ambient_db": ("FLOAT", {
                    "default": -12.0, "min": -30.0, "max": 6.0, "step": 0.5,
                    "tooltip": "Level (dB, relative to the original) at which the "
                               "original span plays under dub pieces MARKED ambient "
                               "with alt+click in the timeline editor (Apply writes "
                               "the marks into timeline_edits; unmarked pieces mute "
                               "the replaced stem completely).  A marked piece plays "
                               "dub + gated non-speech bed + the original span at "
                               "this level, so leaked crowd / rain / room tone that "
                               "the gate cannot rebuild - and a faint ghost of the "
                               "original voice - continue under the dub while it "
                               "stays dominant.  0 = the original plays under the "
                               "dub untouched (use for pure-ambience spans); -30 = "
                               "barely audible; default -12.  The gated bed itself "
                               "always plays at its own recorded level under marked "
                               "windows so the background never pumps.",
                }),
                "background_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "For ambience-heavy scenes: after the voice/background "
                               "split, discard the separated voice stem completely and "
                               "merge the dubs with the background alone.  The subtle "
                               "ambience that leaks into the voice stem is otherwise "
                               "muted under every dub and comes back between phrases "
                               "(the ambient fill can only rebuild an approximation of it), "
                               "so the leak audibly ducks with each phrase and sounds "
                               "distracting and unprofessional; here it is removed for "
                               "good and the background plays at a constant level from "
                               "start to finish.  The original voice is suppressed as "
                               "completely as the separation manages - pair with the "
                               "demucs engine or a background_audio input for the "
                               "cleanest split.  Needs separation = voice_isolation "
                               "(ignored with full_mix).",
                }),
                "timeline_edits": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Adjustments from the timeline editor (JSON, written by "
                               "the editor's Apply button - do not edit by hand).  "
                               "Empty = no adjustments: the merge runs exactly as "
                               "without the editor.  Each edit records where the "
                               "original voice is silenced (start_ms - end_ms) and "
                               "where the dub piece's audio plays (audio_ms) for one "
                               "piece, as drawn on the two-track timeline - plus "
                               "\"amb\": true for pieces marked ambient "
                               "(alt+click), which may also appear as a mark-only "
                               "entry without any geometry.",
                }),
                "move_fails": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keeps the takes folder clean automatically: after a "
                               "successful merge, every superseded take file - a lower "
                               "take number at the same timestamp (e.g. 00014 when "
                               "00015 exists; the highest take number wins) - is MOVED "
                               "into a 'fail' folder inside the takes folder, so only "
                               "the winning take per timestamp remains and no manual "
                               "cleanup is needed.  When the TTS pipeline later "
                               "regenerates a higher take, the previous winner moves "
                               "to fail/ on the next run the same way.  Idempotent, "
                               "never runs on dry runs (mode = plan only), never "
                               "overwrites (a regenerated filename gets a _2/_3 "
                               "suffix), and quarantined files are ignored by every "
                               "later scan so they can never re-enter a merge.  Turn "
                               "off to keep superseded files in place (they are "
                               "listed in the report as before).",
                }),
                "strip_silence": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Strip the baked-in leading AND trailing silence from every "
                               "dub clip (TTS files often carry 100-500 ms of quiet at both "
                               "ends): each placed clip is trimmed to its actual phrase at "
                               "the silence_db threshold, so the dub starts right at its "
                               "timestamp instead of late, and the timeline editor's pieces "
                               "sit at the sound.  A small guard pad (max(20 ms, fade_ms)) "
                               "is kept before the first word and the tail pad "
                               "(tail_pad_ms) after the last one for the crossfades; the "
                               "editor's edge drags can still cut those pads all the way "
                               "to the sound borders, or drag the edges back out into "
                               "silence.  Full-track passes and pieces placed by the "
                               "timeline editor are never touched.",
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
            "optional": {
                "background_audio": ("AUDIO", {
                    "tooltip": "Optional: supply the background track yourself (no-vocals stem "
                               "from UVR / demucs CLI / a DAW export). It is used as-is, so the "
                               "background can never duck: voice = track - background is "
                               "computed sample-exactly. Auto-resampled and channel-matched; "
                               "only used when separation = voice_isolation. Overrides the "
                               "isolation_engine widget.",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "report")
    FUNCTION = "merge"
    CATEGORY = "audio/dubbing"

    def merge(self, audio, takes_folder, extensions, recursive, timestamp_mode,
              window_mode, separation, isolation_engine, voice_low_hz,
              voice_high_hz, demucs_model, fade_ms, gain_db, silence_db,
              tail_pad_ms, overflow, mode, queue_overlaps=True, ambient_db=-12.0,
              background_only=False, timeline_edits="", move_fails=True,
              strip_silence=True, background_audio=None, unique_id=None):
        waveform = audio["waveform"]
        sr = int(audio["sample_rate"])
        np_in = waveform.detach().cpu().numpy().astype(np.float32)

        try:
            folder = dub_core.resolve_folder(takes_folder, folder_paths.get_input_directory())
        except ValueError:
            folder = ""

        # data bundle for the browser timeline editor, written on every real
        # merge run and served to the Load Timeline button via HTTP routes
        cache_dir = None
        if unique_id:
            try:
                cache_dir = os.path.join(
                    folder_paths.get_temp_directory(), "dubmerge_timeline",
                    re.sub(r"[^A-Za-z0-9_.-]", "_", str(unique_id)))
            except Exception:
                cache_dir = None

        bg_np = None
        if background_audio is not None and separation == "voice_isolation":
            bw = background_audio["waveform"].detach().cpu().numpy().astype(np.float32)
            if bw.ndim == 3:
                bw = bw[0]
            elif bw.ndim == 1:
                bw = bw[None, :]
            bsr = int(background_audio["sample_rate"])
            if bsr != sr:
                bw = dub_core.resample_np(bw, bsr, sr)
            ch = np_in.shape[-2] if np_in.ndim == 3 else np_in.shape[0]
            if bw.shape[0] != ch:
                bw = dub_core.match_channels(bw, ch)
            bg_np = np.ascontiguousarray(bw, dtype=np.float32)

        merged, report, _stats = dub_core.merge_pipeline(
            np_in, sr,
            takes_folder=folder,
            extensions=extensions,
            recursive=bool(recursive),
            timestamp_mode=timestamp_mode,
            window_mode=window_mode,
            separation=separation,
            isolation_engine=isolation_engine,
            demucs_model=demucs_model,
            voice_low_hz=float(voice_low_hz),
            voice_high_hz=float(voice_high_hz),
            background_waveform=bg_np,
            fade_ms=float(fade_ms),
            gain_db=float(gain_db),
            silence_db=float(silence_db),
            tail_pad_ms=float(tail_pad_ms),
            queue_overlaps=bool(queue_overlaps),
            ambient_db=float(ambient_db),
            background_only=bool(background_only),
            timeline_edits=str(timeline_edits or ""),
            move_fails=bool(move_fails),
            strip_silence=bool(strip_silence),
            timeline_cache_dir=cache_dir,
            overflow=overflow,
            apply=(mode == "merge"),
        )
        print(report, flush=True)
        out_audio = {
            "waveform": torch.from_numpy(np.ascontiguousarray(merged, dtype=np.float32)),
            "sample_rate": sr,
        }
        return (out_audio, report)

    @classmethod
    def IS_CHANGED(cls, audio, takes_folder, extensions, recursive, timestamp_mode,
                   window_mode, separation, isolation_engine, voice_low_hz,
                   voice_high_hz, demucs_model, fade_ms, gain_db, silence_db,
                   tail_pad_ms, overflow, mode, queue_overlaps=True, ambient_db=-12.0,
                   background_only=False, timeline_edits="", move_fails=True,
                   background_audio=None, unique_id=None):
        import hashlib
        h = hashlib.sha256()
        h.update(repr((takes_folder, extensions, recursive, timestamp_mode, window_mode,
                       separation, isolation_engine, voice_low_hz, voice_high_hz,
                       demucs_model, fade_ms, gain_db, silence_db, tail_pad_ms,
                       overflow, mode, queue_overlaps,
                       ambient_db, background_only, timeline_edits, move_fails)).encode("utf-8"))
        try:
            folder = dub_core.resolve_folder(takes_folder, folder_paths.get_input_directory())
        except Exception:
            folder = None
        if folder and os.path.isdir(folder):
            for root, dirs, files in os.walk(folder):
                dirs.sort()
                for name in sorted(files):
                    p = os.path.join(root, name)
                    try:
                        st = os.stat(p)
                        h.update(("%s|%d|%d;" % (p, st.st_mtime_ns, st.st_size))
                                 .encode("utf-8", "replace"))
                    except OSError:
                        pass
        return h.hexdigest()


# --------------------------------------------------------------------------
# mappings
# --------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "DubLoadMedia": DubLoadMedia,
    "DubMergeTakes": DubMergeTakes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DubLoadMedia": "Load Media (Video/Audio)",
    "DubMergeTakes": "Merge Translated Takes",
}
