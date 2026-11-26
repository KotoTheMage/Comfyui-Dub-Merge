"""ComfyUI Dub Merge - core engine.

Filename parsing, take selection, audio IO, window planning, merging and
reporting for dubbed-audio replacement.  Pure numpy (no torch) so it can be
unit-tested outside ComfyUI; nodes.py bridges torch tensors.

Take file naming convention (dub passes):

    HH.MM.SS.mmm_TTTTT.flac
    e.g.  00.01.52.612_00001.flac

- The timestamp marks where the dubbed phrase starts in the full track
  (or where it ends, with timestamp_mode="end").
- "_TTTTT" is the take number.  For one timestamp the HIGHEST take number
  wins ("last good take"); all lower takes are superseded and - on a real
  merge with move_fails on (the node default) - moved into a "fail" folder
  inside the takes folder, so only the winner per timestamp remains.
- Optional explicit windows are also recognised:
      start-end_TTTTT.flac   e.g.  00.01.52.612-00.01.58.300_00002.flac

Take files are expected to be full passes of the track with the translated
phrase recorded/placed at the timestamped position ("full-track dubs").
The original track is muted from each take's start and the take audio is
used at the same sample positions (positional overlay).  Short segment
clips are detected and placed at the window start automatically
(placement mode).

Voice isolation (separation="voice_isolation"): before merging, the original
is split into a voice track and a background track.  The dubs are merged into
the voice track only and the background is added back afterwards unchanged,
so the background cannot shift, duck or restart under the dubbed parts.
Dub takes should then contain voice only.

Post-phrase silence (tail_silence_ms, auto_trim): when the original phrase is
LONGER than the dub, the tail of the original voice (e.g. a leftover
"...des ka" after a shorter translated line) would fade back in right after
the dub ends.  tail_silence_ms keeps the original VOICE muted for that extra
time after the dubbed phrase's last audible sample; the original voice then
fades back in across the tail pad.  Only the voice track is silenced - the
background is never affected.

The background can come from four places (isolation_engine widget / optional
background_audio input):

- "auto" (node default): Demucs ML separation when the optional `demucs`
  package is installed, otherwise ffmpeg centre (mid/side) extraction.
- "demucs": neural source separation (htdemucs) - the vocals stem is the
  voice, the sum of the other stems is the background and is passed through
  untouched.  Works on mono-like mixes where centre extraction would mute
  the whole centred mix under the dubs.
- "center": ffmpeg mid extraction ((L+R)/2, band-limited); the background is
  the exact complement, background = track - voice.  Best when the background
  is stereo-wide; on mono-like mixes everything centred (ambience, music)
  sits in the voice track and would be muted under the dubs.
- external `background_audio` node input: the caller supplies the background
  (e.g. the no-vocals stem from UVR / demucs CLI / a DAW export); voice =
  track - background is computed sample-exactly.

Ambient fill (per piece, marked in the timeline editor): continuous
background content that ended up INSIDE the replaced stem - centred rain or
music with the centre engine, ambience bleeding into the demucs vocals stem -
is muted under every dub window and comes back between windows, so the
background audibly pumps up and down with each phrase (ducking rain, breathing
music).  The non-speech part of the replaced stem is extracted with an
adaptive spectral noise-floor gate (extract_ambient_stem) and keeps playing
under the windows of dub pieces the user MARKED ambient (alt+click in the
browser timeline editor; the marks travel in timeline_edits, unmarked pieces
mute the replaced stem completely), so rain/ambience/music stay at a constant
level and only the original speech is removed.  On top of the gate, the
marked window also mixes the replaced stem itself at ambient_db (default
-12 dB) - the "original + dub at different levels" blend: the leaked
ambience the gate cannot rebuild (crowd chatter, scene texture) and a faint
ghost of the original voice continue under the dub, which stays dominant.
In voice-isolation mode the separated background stem additionally guides
the gate (per-bin reference mask), which also removes the soft/fricative
speech remnants that sit at or below the ambience level - without it the
fill would echo the original voice under the dubs.  A sustained-ambience
keep recognises loud but slowly varying, spectrally flat events (a train
passing by, a rain swell, a distant crowd) that sit far above the gate's
noise floor - a plain level gate would suppress them like speech (measured
up to -50 dB) - and keeps them at full level; speech is excluded by its fast
syllabic envelope and harmonic-comb spectrum, so the voice suppression is
unchanged.  Outside the marked windows the output is untouched.

Background-only mode (background_only, voice-isolation only): the separated
voice stem is discarded entirely - the dubs are overlaid on silence and the
untouched background is the only original content in the output.  Ambience
that leaked into the voice stem (and that even the ambient fill can only
rebuild approximately) never mutes under the dubs and never comes back
between them, so nothing ducks; the price is that the original voice is
removed only as completely as the separation removes it.  Meant for
ambience-heavy scenes where a ducking leak is more distracting than an
imperfectly suppressed voice.  The ambient fill has nothing left to refill in
this mode and is skipped.
"""

import os
import random
import re
import json
import shutil
import subprocess

import numpy as np

__all__ = [
    "parse_take_filename", "scan_take_files", "select_best_takes",
    "load_audio_file", "ffmpeg_decode", "resample_np", "match_channels",
    "separate_voice_background", "separate_demucs", "DemucsUnavailable",
    "extract_ambient_stem",
    "resolve_media_path", "resolve_folder", "prepare_merge", "apply_merge",
    "merge_pipeline", "build_report", "fmt_clock", "fmt_seconds",
    "move_superseded_takes", "FAIL_DIR_NAME",
    "AUDIO_EXTS", "VIDEO_EXTS", "TIMESTAMP_RE",
]

AUDIO_EXTS = {
    ".flac", ".wav", ".wave", ".aif", ".aiff", ".aifc",
    ".mp3", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".wma", ".caf",
}
VIDEO_EXTS = {
    ".mp4", ".m4v", ".mkv", ".webm", ".mov", ".avi", ".flv",
    ".ts", ".mts", ".m2ts", ".mpg", ".mpeg", ".wmv", ".3gp",
}

# HH.MM.SS.mmm [_- HH.MM.SS.mmm] [_-] TTTTT
_TS = r"(?P<h{0}>[0-9]{{1,2}})\.(?P<m{0}>[0-9]{{1,2}})\.(?P<s{0}>[0-9]{{1,2}})\.(?P<ms{0}>[0-9]{{1,3}})"
TIMESTAMP_RE = re.compile(
    _TS.format("1") + r"(?:[-_ ]" + _TS.format("2") + r")?[_ ](?P<take>[0-9]+)"
)


# --------------------------------------------------------------------------
# filename parsing
# --------------------------------------------------------------------------

def _to_ms(h, m, s, ms_str):
    """'612' -> 612 ms, '61' -> 610 ms, '6' -> 600 ms (fractional reading)."""
    ms = int(ms_str.ljust(3, "0"))
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + ms


def parse_take_filename(path):
    """Parse 'HH.MM.SS.mmm_TTTTT' (optionally 'start-end_TTTTT') from a path.

    Returns dict(path, filename, start_ms, end_ms, take) or None when the
    stem does not contain a valid timestamp_take pattern.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    m = TIMESTAMP_RE.search(stem)
    if not m:
        return None
    gd = m.groupdict()
    start_ms = _to_ms(gd["h1"], gd["m1"], gd["s1"], gd["ms1"])
    end_ms = _to_ms(gd["h2"], gd["m2"], gd["s2"], gd["ms2"]) if gd.get("h2") else None
    return {
        "path": path,
        "filename": os.path.basename(path),
        "start_ms": start_ms,
        "end_ms": end_ms,
        "take": int(gd["take"]),
    }


def fmt_clock(ms):
    ms = int(round(ms))
    h, rem = divmod(max(0, ms), 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, msec = divmod(rem, 1000)
    return "%02d:%02d:%02d.%03d" % (h, m, s, msec)


def fmt_seconds(sec):
    return "%.3fs" % float(sec)


def ms_to_samples(ms, sr):
    return int((int(ms) * sr + 500) // 1000)


# --------------------------------------------------------------------------
# folder scanning + take selection
# --------------------------------------------------------------------------

def _norm_extensions(extensions):
    if not extensions:
        return set()
    exts = set()
    for item in re.split(r"[,;\s]+", str(extensions).strip()):
        item = item.strip().lstrip("*").lstrip(".").lower()
        if item:
            exts.add("." + item)
    return exts


def _register_file(path, out, exts):
    name = os.path.basename(path)
    if name.startswith("."):  # hidden / temp files
        return
    ext = os.path.splitext(name)[1].lower()
    if ext not in exts:
        out["ignored"].append(path)
        return
    out["audio_files"].append(path)
    info = parse_take_filename(path)
    if info is None:
        out["unmatched"].append(path)
    else:
        out["parsed"].append(info)


def scan_take_files(folder, extensions=".flac", recursive=False):
    """Scan a folder for take files.  Returns a scan-info dict.

    The fail folder (<folder>/fail - where superseded takes are quarantined,
    see move_superseded_takes) is never scanned: files parked there are out
    of the running, no matter how take-like their names still look."""
    exts = _norm_extensions(extensions) or {".flac"}
    out = {
        "folder": folder, "extensions": str(extensions), "recursive": bool(recursive),
        "audio_files": [], "parsed": [], "unmatched": [], "ignored": [],
        "error": None, "losers": [], "n_groups": 0,
    }
    fail_dir = os.path.join(os.path.abspath(folder), FAIL_DIR_NAME)
    if recursive:
        for root, dirs, files in os.walk(folder):
            # prune only the fail folder directly inside the scan root - a
            # subfolder that merely happens to be called "fail" elsewhere
            # stays a normal folder
            dirs[:] = [d for d in dirs
                       if os.path.abspath(os.path.join(root, d)) != fail_dir]
            for name in files:
                _register_file(os.path.join(root, name), out, exts)
    else:
        try:
            names = sorted(os.listdir(folder))
        except OSError as exc:
            out["error"] = "cannot list folder: %s" % exc
            return out
        for name in names:
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                _register_file(path, out, exts)
    out["parsed"].sort(key=lambda d: (d["start_ms"], d["take"], d["filename"]))
    return out


FAIL_DIR_NAME = "fail"


def move_superseded_takes(losers, folder, warnings=None,
                          fail_name=FAIL_DIR_NAME):
    """Quarantine superseded takes: move every loser file into
    <folder>/fail so the takes folder keeps only the winning take per
    timestamp (the highest number).  `losers` is the (loser_info,
    winner_info) list from select_best_takes; winner_info is unused today
    but kept in the signature for reporting.

    Never raises: a file that cannot be moved (missing, locked, no
    permission) is skipped with a warning and the merge proceeds.  A name
    collision inside the fail folder (the TTS pipeline regenerated the same
    filename) is resolved with a numeric suffix (_2, _3, ...) instead of
    overwriting.  The fail folder itself is pruned from every later scan,
    so quarantined files can never re-enter the merge.

    Returns (moved_names, fail_dir, warnings); moved_names are the archive
    names (post-suffix), fail_dir is None when nothing was moved.
    """
    def _warn(msg):
        if warnings is not None:
            warnings.append(msg)

    losers = [lw for lw in (losers or [])
              if isinstance(lw, tuple) and lw and isinstance(lw[0], dict)]
    if not losers:
        return [], None, []
    base = os.path.abspath(str(folder))
    if not os.path.isdir(base):
        _warn("fail folder: takes folder not found - nothing moved")
        return [], None, []
    fail_dir = os.path.join(base, fail_name)
    moved, warns = [], []
    for loser, _win in losers:
        src = loser.get("path")
        name = loser.get("filename") or (os.path.basename(src) if src else None)
        if not src or not name or not os.path.isfile(src):
            warns.append("fail folder: %s not found - left where it is"
                         % (name or src or "<unknown>"))
            continue
        try:
            os.makedirs(fail_dir, exist_ok=True)
        except OSError as exc:
            warns.append("fail folder: cannot create %s (%s) - %s left in place"
                         % (fail_dir, exc, name))
            continue
        dst = os.path.join(fail_dir, name)
        if os.path.exists(dst):
            stem, ext = os.path.splitext(name)
            n = 2
            while os.path.exists(os.path.join(fail_dir, "%s_%d%s" % (stem, n, ext))):
                n += 1
            dst = os.path.join(fail_dir, "%s_%d%s" % (stem, n, ext))
        try:
            shutil.move(src, dst)
            moved.append(os.path.basename(dst))
        except (OSError, shutil.Error) as exc:
            warns.append("fail folder: could not move %s (%s)" % (name, exc))
    return moved, (fail_dir if moved else None), warns


def select_best_takes(parsed):
    """Group parsed take files by start time; highest take number wins.

    Returns (winners_sorted_by_start, losers) where losers is a list of
    (loser_info, winner_info) tuples.
    """
    groups = {}
    for info in parsed:
        groups.setdefault(info["start_ms"], []).append(info)
    winners, losers = [], []
    for start_ms in sorted(groups):
        group = sorted(groups[start_ms], key=lambda d: (d["take"], d["filename"]))
        win = group[-1]
        winners.append(win)
        for loser in group[:-1]:
            losers.append((loser, win))
    winners.sort(key=lambda d: d["start_ms"])
    return winners, losers


# --------------------------------------------------------------------------
# audio IO
# --------------------------------------------------------------------------

def ffmpeg_decode(path, timeout=600.0):
    """Decode any media (video/URL/exotic audio) to float32 [C, T] via ffmpeg."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg was not found in PATH. Install ffmpeg (or drop ffmpeg/ffprobe "
            "executables into the 'bin' folder of this node pack) to decode "
            "video or compressed audio sources."
        )
    sr, ch = 48000, 2
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,channels", "-of", "json", str(path)],
            capture_output=True, timeout=30.0)
        if probe.returncode == 0 and probe.stdout:
            info = json.loads(probe.stdout.decode("utf-8", "replace"))
            streams = info.get("streams") or []
            if streams:
                sr = int(streams[0].get("sample_rate", sr))
                ch = int(streams[0].get("channels", ch))
    except Exception:
        pass
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-sn", "-dn",
           "-ac", str(ch), "-ar", str(sr), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace")[-400:]
        raise RuntimeError("ffmpeg failed to decode %s: %s" % (path, err))
    raw = np.frombuffer(proc.stdout, dtype=np.float32)
    if ch > 1:
        usable = (raw.size // ch) * ch
        data = raw[:usable].reshape(-1, ch).T.copy()
    else:
        data = raw.reshape(1, -1).copy()
    if data.shape[1] == 0:
        raise RuntimeError("ffmpeg decoded no audio from %s (missing audio stream?)" % path)
    return data, int(sr)


def load_audio_file(path):
    """Load any audio/video/URL into float32 [C, T] + sample rate.

    Order: soundfile (fast, flac/wav/ogg/mp3) -> torchaudio -> ffmpeg.
    """
    p = str(path)
    if p.startswith("http://") or p.startswith("https://"):
        return ffmpeg_decode(p)
    try:
        import soundfile as sf
        data, sr = sf.read(p, dtype="float32", always_2d=True)
        return np.ascontiguousarray(data.T), int(sr)
    except Exception:
        pass
    try:
        import torchaudio
        wav, sr = torchaudio.load(p)
        return np.ascontiguousarray(wav.numpy(), dtype=np.float32), int(sr)
    except Exception:
        pass
    return ffmpeg_decode(p)


def resample_np(data, sr_in, sr_out):
    """Resample float32 [C, T].  torchaudio when available, linear interp fallback."""
    if int(sr_in) == int(sr_out) or data.shape[-1] == 0:
        return np.asarray(data, dtype=np.float32)
    try:
        import torch
        import torchaudio
        t = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))
        t = torchaudio.functional.resample(t, int(sr_in), int(sr_out))
        return np.ascontiguousarray(t.numpy(), dtype=np.float32)
    except Exception:
        n_in = data.shape[-1]
        n_out = max(1, int(round(n_in * sr_out / float(sr_in))))
        if n_in == 1:
            return np.repeat(data, n_out, axis=-1).astype(np.float32)
        x_in = np.linspace(0.0, n_in - 1.0, n_in, dtype=np.float64)
        x_out = np.linspace(0.0, n_in - 1.0, n_out, dtype=np.float64)
        out = np.empty((data.shape[0], n_out), dtype=np.float32)
        for c in range(data.shape[0]):
            out[c] = np.interp(x_out, x_in, data[c]).astype(np.float32)
        return out


def match_channels(data, target_ch):
    """Match channel count: mono duplicated, multi down-mixed by grouped mean."""
    ch = data.shape[0]
    if ch == target_ch:
        return data
    if ch == 1:
        return np.repeat(data, target_ch, axis=0)
    if target_ch == 1:
        return np.mean(data, axis=0, keepdims=True)
    if ch > target_ch:
        groups = np.array_split(np.arange(ch), target_ch)
        return np.stack([data[list(g)].mean(axis=0) for g in groups])
    idx = np.arange(target_ch) % ch
    return data[idx].copy()


# --------------------------------------------------------------------------
# path resolution
# --------------------------------------------------------------------------

def _is_windows_abs(p):
    return os.name == "nt" and re.match(r"^[A-Za-z]:", p) is not None


def resolve_media_path(path, input_dir=""):
    """Resolve a media path: absolute, relative to ComfyUI input dir, or URL."""
    p = str(path).strip().strip('"').strip("'")
    if not p:
        raise ValueError("media_path is empty")
    if p.startswith("http://") or p.startswith("https://"):
        return p
    if os.path.isabs(p) or _is_windows_abs(p):
        if os.path.exists(p):
            return p
        raise FileNotFoundError("media file not found: %s" % p)
    candidates = []
    if input_dir:
        candidates.append(os.path.join(input_dir, p))
    candidates.append(os.path.join(os.getcwd(), p))
    candidates.append(p)
    for cand in candidates:
        if cand and os.path.exists(cand):
            return os.path.abspath(cand)
    raise FileNotFoundError(
        "media file not found: %s (searched ComfyUI input directory and working directory)" % p)


def resolve_folder(path, input_dir=""):
    """Resolve a takes folder path; returns the raw string when it does not
    exist so prepare_merge can report a readable error."""
    p = str(path).strip().strip('"').strip("'")
    if not p:
        raise ValueError("takes_folder is empty")
    if os.path.isabs(p) or _is_windows_abs(p):
        return p
    candidates = []
    if input_dir:
        candidates.append(os.path.join(input_dir, p))
    candidates.append(os.path.join(os.getcwd(), p))
    candidates.append(p)
    for cand in candidates:
        if cand and os.path.isdir(cand):
            return os.path.abspath(cand)
    return p


# --------------------------------------------------------------------------
# voice / background separation (voice isolation)
# --------------------------------------------------------------------------

def _scipy_signal():
    """Return the scipy.signal module, or None when scipy is unavailable."""
    try:
        import scipy.signal  # noqa: F401
        return scipy.signal
    except Exception:
        return None


def _zero_phase_bandpass(data, sr, low_hz, high_hz):
    """Zero-phase band limiting (forward-backward butterworth via scipy).

    Zero-phase matters here: the background is computed as track - voice, so
    any time/phase shift introduced by the filter would leave a smeared copy
    of the original voice in the background.  Forward-backward filtering
    cancels the phase exactly; only the magnitude response (|H|^2) remains.

    Returns a new [C, T] float32 array, or None when scipy is unavailable.
    Falls back to the unfiltered input for signals too short to pad.
    """
    ss = _scipy_signal()
    if ss is None:
        return None
    nyq = sr * 0.5
    lo = max(0.0, min(float(low_hz), nyq * 0.98))
    hi = min(max(float(high_hz), lo + 1.0), nyq * 0.98)
    if lo <= 0.0 and hi >= nyq * 0.98:
        return np.asarray(data, dtype=np.float32)
    if data.shape[-1] < 64:  # too short for forward-backward padding
        return np.asarray(data, dtype=np.float32)
    if lo <= 0.0:
        sos = ss.butter(4, hi, btype="lowpass", fs=sr, output="sos")
    elif hi >= nyq * 0.98:
        sos = ss.butter(4, lo, btype="highpass", fs=sr, output="sos")
    else:
        sos = ss.butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
    try:
        out = ss.sosfiltfilt(sos, data, axis=-1)
    except ValueError:
        return np.asarray(data, dtype=np.float32)
    return np.ascontiguousarray(out, dtype=np.float32)


def separate_voice_background(track, sr, low_hz=100.0, high_hz=12000.0):
    """Split a [C, T] mix into (voice, background, method) with
    voice + background == track, sample-exact.

    The voice is extracted with ffmpeg: for stereo, the pan filter folds the
    two channels into the mid signal ((L+R)/2 in both channels), which
    carries centred dialogue; mono/multichannel input is band-limited as-is.
    The extracted mid signal is band-limited to [low_hz, high_hz] -
    zero-phase (scipy sosfiltfilt) when scipy is available, otherwise with
    ffmpeg's highpass/lowpass IIR filters inside the same ffmpeg run.  Both
    channels of the extracted voice are bit-identical, so the background
    keeps the full side (stereo width) signal.

    The background is the exact complement: background = track - voice.  It
    is re-added unchanged after the dub merge, so the background cannot
    shift, duck or restart across the dubbed windows, and outside those
    windows the output reconstructs the input exactly.

    Raises RuntimeError when ffmpeg is missing or fails.
    """
    data = np.ascontiguousarray(np.asarray(track, dtype=np.float32))
    C, T = data.shape[0], data.shape[1]
    if T == 0:
        return data.copy(), data.copy(), "empty track"
    nyq = sr * 0.5
    lo = max(0.0, float(low_hz))
    hi = min(max(float(high_hz), lo + 1.0), nyq * 0.995)
    want_band = lo > 0.0 or hi < nyq * 0.99
    use_scipy = want_band and (_scipy_signal() is not None)

    parts = []
    if C == 2:
        parts.append("pan=stereo|c0=0.5*c0+0.5*c1|c1=0.5*c0+0.5*c1")
    if want_band and not use_scipy:
        if lo > 0.0:
            parts.append("highpass=f=%.1f" % lo)
        if hi < nyq * 0.99:
            parts.append("lowpass=f=%.1f" % hi)
    filters = ",".join(parts)

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "voice isolation needs ffmpeg in PATH - install ffmpeg (or drop "
            "ffmpeg/ffprobe into this pack's bin/ folder), or set the "
            "separation widget to full_mix")
    cmd = ["ffmpeg", "-v", "error",
           "-f", "f32le", "-ar", str(int(sr)), "-ac", str(C), "-i", "pipe:0",
           "-vn"]
    if filters:
        cmd += ["-af", filters]
    cmd += ["-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]
    try:
        proc = subprocess.run(cmd, input=np.ascontiguousarray(data.T).tobytes(),
                              capture_output=True, timeout=600.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("ffmpeg voice separation failed: %s" % exc)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace")[-300:]
        raise RuntimeError("ffmpeg voice separation failed: %s" % err)
    raw = np.frombuffer(proc.stdout, dtype=np.float32)
    usable = (raw.size // C) * C
    if usable < C:
        raise RuntimeError("ffmpeg voice separation produced no audio")
    voice = raw[:usable].reshape(-1, C).T.copy()
    if voice.shape[1] < T:
        voice = np.pad(voice, ((0, 0), (0, T - voice.shape[1])))
    elif voice.shape[1] > T:
        voice = voice[:, :T]
    if use_scipy:
        filtered = _zero_phase_bandpass(voice, sr, lo, hi)
        if filtered is not None:
            voice = filtered
    voice = np.ascontiguousarray(voice, dtype=np.float32)
    background = np.ascontiguousarray(data - voice)

    if C == 2:
        mode = "ffmpeg center (mid) extraction"
    else:
        mode = "ffmpeg band extraction (%d ch, no center fold-down)" % C
    if want_band and use_scipy:
        method = "%s, zero-phase band %.0f-%.0f Hz (scipy sosfiltfilt)" % (mode, lo, hi)
    elif want_band:
        method = "%s, IIR band %.0f-%.0f Hz (ffmpeg)" % (mode, lo, hi)
    else:
        method = mode + ", full range"
    return voice, background, method


class DemucsUnavailable(RuntimeError):
    """Raised when the Demucs ML separator is missing, broken or unloaded."""


def _align_len(data, T):
    """Pad/trim a [C, T'] float32 array to exactly T samples."""
    cur = data.shape[1]
    if cur == T:
        return np.ascontiguousarray(data, dtype=np.float32)
    if cur < T:
        return np.pad(data, ((0, 0), (0, T - cur))).astype(np.float32)
    return np.ascontiguousarray(data[:, :T], dtype=np.float32)


def separate_demucs(track, sr, model_name="htdemucs", shifts=1, overlap=0.25):
    """Split a [C, T] mix into (voice, background, method) with Demucs.

    Real (neural) source separation: the model returns a vocals stem plus the
    remaining stems; the voice is the vocals stem and the background is the
    sum of the other stems ("no vocals"), passed through untouched.  Unlike
    centre extraction this also works on mono-like mixes, where the whole
    centred mix would otherwise be muted under the dubs.

    Requires the optional `demucs` package (pip install demucs); the model
    weights (~80 MB for htdemucs) are downloaded on first use.  Raises
    DemucsUnavailable when the package or model cannot be used.

    Mono input is duplicated to stereo for the model and the stems are
    averaged back to mono; >2-channel input is rejected (use the centre
    engine or an external background input instead).  The audio is resampled
    to the model rate and back as needed; output length equals the input
    length.  voice + background reconstruct the input up to the model's
    residual (small, but not sample-exact like the centre engine).  Runs are
    reproducible: demucs's internal random shift is seeded per call.
    """
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except Exception as exc:  # ImportError or a broken demucs install
        raise DemucsUnavailable("demucs package not importable (%s) - "
                                "install it with: pip install demucs" % exc)

    data = np.ascontiguousarray(np.asarray(track, dtype=np.float32))
    C, T = data.shape[0], data.shape[1]
    if T == 0:
        return data.copy(), data.copy(), "empty track"
    if C not in (1, 2):
        raise DemucsUnavailable(
            "demucs engine needs mono or stereo input, got %d channels - "
            "use the centre engine or supply a background_audio input" % C)

    try:
        model = get_model(model_name)
    except Exception as exc:
        raise DemucsUnavailable("could not load demucs model %r (%s) - the "
                                "weights are downloaded on first use, check "
                                "your network" % (model_name, exc))
    model_sr = int(model.samplerate)
    v_idx = list(model.sources).index("vocals")

    wav = data
    mono = C == 1
    if mono:
        wav = np.repeat(wav, 2, axis=0)
    if model_sr != sr:
        wav = resample_np(wav, sr, model_sr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
        try:
            # demucs's "shift trick" draws its internal time offset from the
            # global, unseeded `random` module (apply_model: offset =
            # random.randint(0, 0.5s)), which makes every run separate a
            # randomly shifted copy.  Seed it temporarily so a given track
            # always yields the same stems - reproducible runs, stable tests.
            rng_state = random.getstate()
            random.seed(20240)
            try:
                stems = apply_model(model, xt[None], device=device,
                                    shifts=int(shifts), split=True,
                                    overlap=float(overlap), progress=True)[0]
            finally:
                random.setstate(rng_state)
        except Exception as exc:
            raise DemucsUnavailable("demucs separation failed: %s" % exc)
        stems = stems.detach().cpu().numpy().astype(np.float32)

    voice = stems[v_idx]
    background = stems.sum(axis=0) - stems[v_idx]
    if mono:
        voice = voice.mean(axis=0, keepdims=True)
        background = background.mean(axis=0, keepdims=True)
    if model_sr != sr:
        voice = resample_np(voice, model_sr, sr)
        background = resample_np(background, model_sr, sr)

    voice = _align_len(voice, T)
    background = _align_len(background, T)

    bits = ["Demucs %s (vocals stem vs sum of the other stems)" % model_name]
    if model_sr != sr:
        bits.append("resampled %d <-> %d Hz" % (sr, model_sr))
    if mono:
        bits.append("mono duplicated to stereo")
    bits.append("CPU" if not torch.cuda.is_available() else "GPU")
    return voice, background, ", ".join(bits)


# --------------------------------------------------------------------------
# ambient (non-speech) fill for the dub windows
# --------------------------------------------------------------------------

def _periodic_hann(n):
    """Periodic Hann window (COLA-exact at hop = n/2: two adjacent windows
    sum to exactly 1.0, so the sqrt-Hann analysis+synthesis pair needs no
    WOLA normalisation buffer)."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def _smoothstep(x):
    """Hermite smoothstep on a clipped [0, 1] input (float32 in/out)."""
    x = np.clip(x, np.float32(0.0), np.float32(1.0))
    return x * x * (np.float32(3.0) - np.float32(2.0) * x)


# Sustained-ambience keep: a loud event whose per-bin level stays within a
# few dB of its recent typical (coarse p25, symmetric ~11 s) or recent peak
# (trailing ~11 s) level is sustained ambience (train pass-by, rain swell,
# distant crowd) - not speech.  Two-sided keep window on the ENERGY ratio
# r = baseline/current: full keep inside [-2.6, +2.6] dB, tapering to zero
# at -8 dB below (a speech spike) / +8 dB above (an unrelated swell).
_SUSTAIN_GC = 8          # sub-blocks (~0.27 s) per coarse block (~2.2 s)
_SUSTAIN_WIN = 5         # coarse blocks per baseline window (~10.9 s)
_SUSTAIN_LO = 0.16       # energy ratio where the keep starts (~-8 dB)
_SUSTAIN_HI = 0.55       # energy ratio where the keep is full (~-2.6 dB)
# Per-bin voice guard: speech is a harmonic comb - tooth bins stand far
# above the median of their +/-_PEAK_W-bin neighbourhood, while every kind
# of ambience (rain, train rolloff, crowd, fricatives) sits within ~2.2x of
# it (measured p95: speech F0 240 = 58x, F0 110 = 11.5x, F0 80 = 5.1x;
# rain/train/crowd/music 2.1-2.2x).  Bins peakier than _PEAK_HI are never
# sustained ambience; the band [_PEAK_LO, _PEAK_HI] tapers so the noise's
# own spectral texture (p95 ~2.2) is kept nearly untouched.
_PEAK_W = 12             # +/- bins for the local median (~+/-280 Hz @48k)
_PEAK_LO = 1.8           # peakiness at/below which a bin is fully keepable
_PEAK_HI = 2.8           # peakiness at/above which a bin is never kept


def extract_ambient_stem(data, sr, gamma=4.0, mask_floor=0.02, block=512,
                         reference=None, ref_gamma=5.0, ref_floor=0.02,
                         ref_min_explained=0.35, floor_bias=2.5,
                         stats_out=None):
    """Non-speech part (rain / ambience / music bed) of a [C, T] stem.

    Why this exists: continuous background content that ended up inside the
    stem the dubs replace (centred rain or music with the centre engine,
    ambience bleeding into the demucs vocals stem) is muted under every dub
    window and comes back between windows - the background audibly pumps up
    and down with each phrase.  This extracts exactly that content so it can
    keep playing under the dubs: a spectral gate keeps every time/frequency
    bin whose energy sits at/below the local noise floor (rain, ambience,
    music bed) and suppresses bins far above it (speech).  A dub window then
    carries dub + this ambient stem instead of dub alone, and the rain level
    stays constant.

    numpy-only STFT/OLA: sqrt-Hann analysis + synthesis windows at hop =
    n_fft // 2 (constant overlap-add).  One shared mask for all channels, so
    the kept background keeps its stereo image.  Masking runs in fixed-size
    blocks (results are independent of the block size - each block carries
    two context frames so every mask that reaches the output is computed
    from fully interior neighbours).

    The noise floor is ADAPTIVE: per-bin 25th-percentile magnitudes over
    ~0.27 s sub-blocks, then a neighbourhood minimum over +/-6 sub-blocks
    (~3 s) and a short smoothing, scaled by floor_bias, bounded above by the
    per-bin median over the quietest sub-blocks of the whole track.  It
    therefore follows the RECURRING quiet level of every bin.  A single
    global static floor was inflated by speech on dialogue-dense tracks (the
    "quietest frames" still contained speech), which let a large part of the
    original voice through - the fill then echoed the original voice under
    the dubs.  The adaptive floor stays at the ambience level through dense
    dialogue, and the within-block percentile + global bound keep it above
    the ambience's own Rayleigh fluctuations, so the gate does not breathe
    with the noise it is meant to preserve.

    SUSTAINED-AMBIENCE KEEP: the level gate alone has ~+20 dB of headroom
    over the recurring quiet level, and louder events - a train passing by,
    a rain swell, a distant crowd - ducked by up to -50 dB under the dub
    windows (neural separators push a loud event into the vocals stem, so
    the level gate sees it as speech).  Loud ambience is distinguished from
    speech by TIME STRUCTURE and SPECTRUM, not by level: per-bin coarse
    (~2.2 s) levels that stay within a few dB of their recent typical or
    recent peak level over the last ~11 s are sustained (slowly varying),
    while speech spikes 6-20 dB over its own baseline syllable by syllable;
    and speech is a harmonic comb - tooth bins stand far above the median of
    their spectral neighbourhood, while every ambience (rain, train rolloff,
    crowd, even fricatives) sits within ~2.2x of it.  Bins passing BOTH
    tests are kept at full level regardless of loudness; everything else
    goes through the unchanged level/reference path.  The keep is therefore
    immune to dialogue density (no level can pass the comb test) and works
    with or without a background reference.

    reference (optional): the separated background stem ([C, T], same
    length).  A level gate cannot catch voice components that sit at/below
    the ambience level (fricatives, breath, soft speech endings) - but the
    background stem knows where the ambience IS.  When the reference
    explains the stem's quiet content (ambience bleeding into a demucs
    vocals stem also lives in the background stem; the centre engine's
    diffuse ambience also lives in the side signal), a second per-bin
    keep-mask is applied: bins far above the reference's level are voice the
    separator put into this stem and are suppressed, while bins explained by
    the reference (the ambience itself) are kept.  When the reference does
    NOT explain the quiet content (e.g. perfectly centred ambience with the
    centre engine - the side signal is empty there), it is ignored and the
    level gate alone remains.  stats_out (optional dict) receives
    "reference_used", "reference_explained" and "sustain_used".

    gamma (>= 1) is the headroom before suppression starts: bins up to
    gamma x the floor are kept untouched (gamma = 4 = +6 dB, above the
    natural crest factor of diffuse noise), speech ~20 dB over the floor
    ends up ~34 dB down.  mask_floor caps the suppression depth.  Returns a
    [C, T] float32 array; zeros when the input is too short for an STFT.
    """
    data = np.ascontiguousarray(np.asarray(data, dtype=np.float32))
    if data.ndim == 1:
        data = data[None, :]
    C, T = data.shape[0], data.shape[1]
    out = np.zeros((C, T), dtype=np.float32)
    if T == 0:
        return out
    n_fft = 2048 if sr < 64000 else 4096
    if T < n_fft:
        return out
    hop = n_fft // 2
    win = np.sqrt(_periodic_hann(n_fft)).astype(np.float32)
    offset = n_fft // 2          # front pad: every real sample is then
    n_frames = 3 + T // hop      # covered by exactly two full windows
    freqs = n_fft // 2 + 1
    eps = np.float32(1e-10)

    def _span(a, b):
        """Virtual padded-signal slice [C, b-a], zero-filled outside data."""
        s = np.zeros((C, b - a), dtype=np.float32)
        lo = max(a - offset, 0)
        hi = min(b - offset, T)
        if hi > lo:
            s[:, lo - (a - offset): hi - (a - offset)] = data[:, lo:hi]
        return s

    # optional ambience reference (background stem), padded/trimmed to T
    R = None
    if reference is not None:
        R = np.asarray(reference, dtype=np.float32)
        if R.ndim == 1:
            R = R[None, :]
        if R.shape[1] < T:
            pad = np.zeros((R.shape[0], T), dtype=np.float32)
            pad[:, :R.shape[1]] = R
            R = pad
        elif R.shape[1] > T:
            R = np.ascontiguousarray(R[:, :T])
        if R.shape[1] != T or T == 0:
            R = None

    def _frames_mag(arr, a, b):
        """Mono magnitude frames [nf, F] of arr ([Cr, T]) over the virtual
        padded sample span [a, b): nf = (b - a - n_fft) // hop + 1 frames."""
        s = np.zeros((arr.shape[0], b - a), dtype=np.float32)
        lo = max(a - offset, 0)
        hi = min(b - offset, T)
        if hi > lo:
            s[:, lo - (a - offset): hi - (a - offset)] = arr[:, lo:hi]
        frv = np.lib.stride_tricks.sliding_window_view(
            s, n_fft, axis=1)[:, ::hop]
        X = np.fft.rfft(frv * win[None, None, :], axis=-1)
        return np.sqrt((X.real ** 2 + X.imag ** 2).mean(axis=0)).astype(np.float32)

    # ---- pass 1: adaptive per-bin noise floor (+ optional reference) ------
    # Per-bin ~p25 magnitudes over ~0.27 s sub-blocks, then a neighbourhood
    # MINIMUM over +/-6 sub-blocks (~3 s) and a 5-tap smoothing, scaled by
    # floor_bias and bounded by the global quiet-level median.  The floor
    # follows the RECURRING quiet level of every bin: it stays at the
    # ambience level through dense dialogue (where a single global static
    # floor was inflated by speech and let the voice through into the fill),
    # while the within-block percentile keeps it above the ambience's own
    # Rayleigh dips, so the gate does not thin the noise it keeps.
    L = int(round(0.27 * sr / hop))
    L = max(5, min(33, L))
    if L % 2 == 0:
        L += 1                       # odd -> exact median via one partition
    step = 4 * L
    # global sub-block edges; a trailing orphan (< 3 frames, i.e. the window-
    # attenuated tail frames) is absorbed into the previous block, otherwise
    # its mean would collapse the floor at the very end of the track
    edges = list(range(0, n_frames, L)) + [n_frames]
    while len(edges) > 2 and edges[-1] - edges[-2] < 3:
        del edges[-2]
    q_chunks = []
    h_chunks = []
    rq_chunks = []
    for f0 in range(0, n_frames, step):
        c1 = min(n_frames, f0 + step)
        S = _frames_mag(data, f0 * hop, (c1 - 1) * hop + n_fft)
        if R is not None:
            Rm = _frames_mag(R, f0 * hop, (c1 - 1) * hop + n_fft)
        bounds = [f0] + [e for e in edges if f0 < e < c1] + [c1]
        kq = max(1, L // 4)          # ~p25 within a full sub-block
        kh = max(1, L // 2)          # ~p50 within a full sub-block
        for j in range(len(bounds) - 1):
            b0, b1 = bounds[j] - f0, bounds[j + 1] - f0
            blk = S[b0:b1]
            if blk.shape[0] >= 3:
                k = min(kq, blk.shape[0] - 1)
                q_chunks.append(np.partition(blk, k, axis=0)[k])
                k2 = min(kh, blk.shape[0] - 1)
                h_chunks.append(np.partition(blk, k2, axis=0)[k2])
            else:
                q_chunks.append(blk.mean(axis=0))
                h_chunks.append(blk.mean(axis=0))
            if R is not None:
                rblk = Rm[b0:b1]
                if rblk.shape[0] >= 3:
                    k = min(kq, rblk.shape[0] - 1)
                    rq_chunks.append(np.partition(rblk, k, axis=0)[k])
                else:
                    rq_chunks.append(rblk.mean(axis=0))
    B = np.asarray(q_chunks, dtype=np.float32)                # [n_sub, F]
    H = np.asarray(h_chunks, dtype=np.float32)                # [n_sub, F]
    edges_arr = np.asarray(edges, dtype=np.int64)
    pad = 6
    # +inf padding: edge windows take the min over the AVAILABLE sub-blocks
    # only (repeating the edge sub-blocks would drag the floor down at the
    # track ends and thin the ambience there)
    Bp = np.concatenate([np.full((pad, B.shape[1]), np.inf, dtype=np.float32),
                         B,
                         np.full((pad, B.shape[1]), np.inf, dtype=np.float32)], axis=0)
    nb = np.lib.stride_tricks.sliding_window_view(Bp, 2 * pad + 1, axis=0)
    # low quantile (~p20) of the neighbouring sub-block values instead of a
    # strict minimum: it still lands on speech-free sub-blocks in dense
    # dialogue, but does not chase the noise's own deepest statistical dips
    # (a strict min thinned stationary rain peaks audibly)
    nb = np.sort(nb, axis=-1)
    floor_curve = nb[..., 2].astype(np.float32)              # [n_sub, F]
    # global bound: per-bin median over the quietest sub-blocks of the whole
    # track.  A long phrase (> the local window) would otherwise inflate the
    # local floor; the global quiet level caps it (and a short local window
    # keeps the floor responsive when the ambience itself changes).
    eBg = np.mean(B.astype(np.float64) ** 2, axis=1)
    nqg = max(4, int(0.40 * eBg.size))
    quietg = np.argsort(eBg)[:nqg]
    gfloor = np.median(B[quietg].astype(np.float64), axis=0).astype(np.float32)
    # density-robust bound: on very dialogue-dense material the "quietest
    # 40%" sub-blocks still contain speech, the cap inflates to the speech
    # level and the original voice passes the gate.  The gap-sensitive
    # sub-block p50 (H) drops to the ambience level in every sub-block a
    # pause touches, so its track-wide minimum is a safe cap at any speech
    # density (for pure ambience it is the noise's own p50 dip, still far
    # above the Rayleigh tail that would thin the ambience peaks).
    np.minimum(gfloor, H.min(axis=0), out=gfloor)
    np.minimum(floor_curve, gfloor[None, :], out=floor_curve)
    sp2 = 2
    Cp = np.concatenate([np.repeat(floor_curve[:1], sp2, axis=0), floor_curve,
                         np.repeat(floor_curve[-1:], sp2, axis=0)], axis=0)
    cs = np.zeros((Cp.shape[0] + 1, Cp.shape[1]), dtype=np.float32)
    np.cumsum(Cp, axis=0, out=cs[1:])
    floor_curve = ((cs[5:] - cs[:-5]) * np.float32(1.0 / 5.0)).astype(np.float32)
    floor_curve *= np.float32(floor_bias)

    # ---- sustained-ambience baselines (pass 1, global) ---------------------
    # Coarse (~2.2 s) per-bin levels: p50 of the sub-block p50 magnitudes
    # (the p50 is gap-sensitive: a <=0.3 s pause inside a sub-block pulls it
    # to the ambience level, which is what separates close speech - it
    # pauses - from multi-talker babble - it never fully drops).
    # From them: Ls_lo = ~p25 over a symmetric +/-2 coarse window (the recent
    # QUIET level - steady ambience) and Ls_hi = max over the trailing 5
    # coarse blocks (the recent PEAK level - plateaus, swells).  A bin whose
    # current level sits within a few dB of either is sustained ambience.
    # Global arrays + edge padding -> the keep does not depend on the block
    # size; disabled on very short inputs (< 3 coarse blocks of history).
    n_subblk = H.shape[0]
    n_coarse = (n_subblk + _SUSTAIN_GC - 1) // _SUSTAIN_GC
    sustain = None
    if n_coarse >= 3:
        padc = -n_subblk % _SUSTAIN_GC
        if padc:
            Hp = np.concatenate(
                [H, np.full((padc, H.shape[1]), np.nan, np.float32)], axis=0)
        else:
            Hp = H
        Hc = np.nanmedian(
            Hp.reshape(n_coarse, _SUSTAIN_GC, H.shape[1]),
            axis=1).astype(np.float32)                        # [n_coarse, F]
        wp = _SUSTAIN_WIN - 1
        ws = _SUSTAIN_WIN // 2
        Hp2 = np.concatenate(
            [np.full((wp, H.shape[1]), -np.inf, np.float32), Hc], axis=0)
        Ls_hi = np.lib.stride_tricks.sliding_window_view(
            Hp2, _SUSTAIN_WIN, axis=0).max(axis=-1)           # trailing max
        Hp3 = np.concatenate(
            [np.full((ws, H.shape[1]), np.inf, np.float32), Hc,
             np.full((ws, H.shape[1]), np.inf, np.float32)], axis=0)
        Ls_lo = np.sort(np.lib.stride_tricks.sliding_window_view(
            Hp3, _SUSTAIN_WIN, axis=0), axis=-1)[..., 1]      # ~p25 of window
        coarse_edges = np.arange(0, n_subblk, _SUSTAIN_GC, dtype=np.int64)
        sustain = (np.ascontiguousarray(Ls_hi, np.float32),
                   np.ascontiguousarray(Ls_lo, np.float32),
                   coarse_edges, n_coarse)
    sustain_share = 0.0

    # background-stem reference: per-bin gain k + explained-energy test.
    # k[f] = median stem magnitude / median reference magnitude over the
    # quietest sub-blocks (the speech-free moments), so k*|R| estimates the
    # ambience level in THIS stem.  The reference is only used when it
    # explains a meaningful share of that quiet energy (demucs bleed, or the
    # centre engine's diffuse side signal) - never when it is empty there.
    use_ref = False
    k_ref = None
    explained = 0.0
    if R is not None and float(ref_gamma) > 1.0:
        RB = np.asarray(rq_chunks, dtype=np.float32)
        eB = np.mean(B.astype(np.float64) ** 2, axis=1)
        nq = max(4, int(0.40 * eB.size))
        quiet = np.argsort(eB)[:nq]
        Sq = np.median(B[quiet].astype(np.float64), axis=0)
        Rq = np.median(RB[quiet].astype(np.float64), axis=0)
        k_ref = np.clip(Sq / (Rq + 1e-9), 0.0, 4.0).astype(np.float32)
        explained = float(np.sum((k_ref * Rq) ** 2) / (np.sum(Sq ** 2) + 1e-18))
        use_ref = bool(explained >= float(ref_min_explained)
                       and np.any(k_ref > 1e-3))
    if isinstance(stats_out, dict):
        stats_out["reference_used"] = use_ref
        stats_out["reference_explained"] = explained
        stats_out["sustain_used"] = False

    # ---- pass 2: masked ISTFT in blocks ------------------------------------
    g = np.float32(max(1.0, float(gamma)))
    keep = np.float32(min(1.0, max(0.0, float(mask_floor))))
    g_r = np.float32(max(1.0, float(ref_gamma)))
    keep_r = np.float32(min(1.0, max(0.0, float(ref_floor))))
    n_sub = floor_curve.shape[0]
    acc = np.zeros((C, ((n_frames - 1) * hop + n_fft)), dtype=np.float32)
    for f0 in range(0, n_frames, block):
        # Two context frames per side: the 3-tap smoothings reach two frames
        # out, so with +/-2 context every mask that reaches an OLA frame is
        # computed from fully interior neighbours - the output does not
        # depend on the block size.
        c0 = max(0, f0 - 2)
        c1 = min(n_frames, f0 + block + 2)
        span = _span(c0 * hop, (c1 - 1) * hop + n_fft)
        frv = np.lib.stride_tricks.sliding_window_view(
            span, n_fft, axis=1)[:, ::hop]               # [C, c1-c0, n_fft]
        X = np.fft.rfft(frv * win[None, None, :], axis=-1)
        S = np.sqrt((X.real ** 2 + X.imag ** 2).mean(axis=0)).astype(np.float32)
        Sp = np.concatenate([S[:1], S, S[-1:]], axis=0)
        S_sm = ((Sp[:-2] + Sp[1:-1] + Sp[2:]) / 3.0).astype(np.float32)
        # per-frame floor: the sub-block curve is looked up by frame index
        # (searchsorted over the global sub-block edges), so the masks never
        # depend on the block size
        fidx = np.clip(np.searchsorted(edges_arr, np.arange(c0, c1),
                                       side="right") - 1, 0, n_sub - 1)
        fc = floor_curve[fidx]
        mask = g * (fc * fc) / (S_sm * S_sm + eps)
        np.clip(mask, 0.0, 1.0, out=mask)
        np.maximum(mask, keep, out=mask)
        if use_ref:
            Rm = _frames_mag(R, c0 * hop, (c1 - 1) * hop + n_fft)
            Rp = np.concatenate([Rm[:1], Rm, Rm[-1:]], axis=0)
            R_sm = ((Rp[:-2] + Rp[1:-1] + Rp[2:]) / 3.0).astype(np.float32)
            mref = g_r * (k_ref[None, :] * R_sm) ** 2 / (S_sm * S_sm + eps)
            np.clip(mref, 0.0, 1.0, out=mref)
            np.maximum(mref, keep_r, out=mref)
            mask *= mref
        if sustain is not None:
            # sustained-ambience keep: loud but SLOWLY varying, spectrally
            # flat bins stay at full level (train pass-by, rain swell, crowd)
            Ls_hiv, Ls_lov, cedges, ncs = sustain
            cidx = np.clip(np.searchsorted(cedges, fidx, side="right") - 1,
                           0, ncs - 1)
            den = S_sm * S_sm + eps
            with np.errstate(divide="ignore", invalid="ignore"):
                r_hi = (Ls_hiv[cidx] * Ls_hiv[cidx]) / den
                r_lo = (Ls_lov[cidx] * Ls_lov[cidx]) / den
                # two-sided keep window on the baseline/current energy ratio
                def _side(x):
                    return _smoothstep((x - np.float32(_SUSTAIN_LO))
                                       / np.float32(_SUSTAIN_HI - _SUSTAIN_LO))
                # a bin far above its baseline is a speech spike, one far
                # below is a noise dip - only COMPARABLE levels are kept
                w = np.maximum(
                    _side(r_hi) * _side(np.float32(1.0) / r_hi),
                    _side(r_lo) * _side(np.float32(1.0) / r_lo))
            # per-bin voice guard: speech comb teeth stand far above the
            # median of their spectral neighbourhood; ambience never does
            Spk = np.concatenate(
                [np.repeat(S_sm[:, :1], _PEAK_W, axis=1), S_sm,
                 np.repeat(S_sm[:, -1:], _PEAK_W, axis=1)], axis=1)
            pmed = np.median(np.lib.stride_tricks.sliding_window_view(
                Spk, 2 * _PEAK_W + 1, axis=1), axis=-1)
            pk = S_sm / (pmed + eps)
            w *= np.clip((np.float32(_PEAK_HI) - pk)
                         / np.float32(_PEAK_HI - _PEAK_LO),
                         np.float32(0.0), np.float32(1.0))
            np.clip(w, 0.0, 1.0, out=w)
            np.nan_to_num(w, copy=False)
            mask = (np.float32(1.0) - w) * mask + w
            # energy-weighted keep share of this block (skirt bins around a
            # comb carry little energy, so the share stays low for speech)
            den = float((S_sm.astype(np.float64) ** 2).sum()) + 1e-20
            sustain_share = max(
                sustain_share,
                float((w.astype(np.float64)
                       * (S_sm.astype(np.float64) ** 2)).sum()) / den)
        Mp = np.concatenate([mask[:1], mask, mask[-1:]], axis=0)
        mask = ((Mp[:-2] + Mp[1:-1] + Mp[2:]) / 3.0).astype(np.float32)
        np.clip(mask, 0.0, 1.0, out=mask)
        Xamb = X * mask[None, :, :]
        frames_amb = np.fft.irfft(Xamb, n=n_fft, axis=-1).astype(np.float32)
        frames_amb *= win[None, None, :]
        # OLA via the hop = n/2 parity tiling (frames of one parity tile
        # their sample range contiguously; sqrt-Hann^2 = Hann is COLA-1.0,
        # so no normalisation buffer is needed).  Only this block's CENTRAL
        # frames are added: the c0/f0+block context frames exist solely for
        # the 3-tap mask smoothing and are central frames of the neighbouring
        # blocks - adding them here too would double them (+6 dB each block
        # boundary).
        j_lo = f0 - c0                      # 0 or 1 context frame to skip
        j_hi = min(c1 - c0, f0 + block - c0)
        n_sub = j_hi - j_lo
        if n_sub > 0:
            base = c0 + j_lo                # global frame index of sub[0]
            sub = frames_amb[:, j_lo:j_hi]
            # global-even frames start at base*hop (base even) or base*hop+hop
            # (base odd); global-odd frames take the other offset.
            par = base % 2
            pe = sub[:, par::2].reshape(C, -1)
            po = sub[:, (par ^ 1)::2].reshape(C, -1)
            pe_at, po_at = (0, hop) if par == 0 else (hop, 0)
            seg = acc[:, base * hop: base * hop + (n_sub - 1) * hop + n_fft]
            w = min(pe.shape[1], seg.shape[1] - pe_at)
            if w:
                seg[:, pe_at:pe_at + w] += pe[:, :w]
            w2 = min(po.shape[1], seg.shape[1] - po_at)
            if w2:
                seg[:, po_at:po_at + w2] += po[:, :w2]
    out[:, :] = acc[:, offset:offset + T]
    if isinstance(stats_out, dict):
        stats_out["sustain_used"] = bool(sustain_share > 0.2)
        stats_out["sustain_share"] = float(sustain_share)
    return np.nan_to_num(out, copy=False).astype(np.float32)


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------

def _max_abs_per_sample(region):
    """Peak-abs over channels, without materialising a [C, T] temp copy."""
    amax = np.abs(region[0])
    for c in range(1, region.shape[0]):
        np.maximum(amax, np.abs(region[c]), out=amax)
    return amax


def _content_end(data, thresh, tail_smp):
    """Index just past the last audible sample (+ tail padding). 0 when silent."""
    amax = _max_abs_per_sample(data)
    above = np.flatnonzero(amax > thresh)
    if above.size == 0:
        return 0
    return min(data.shape[1], int(above[-1]) + 1 + int(tail_smp))

# --------------------------------------------------------------------------
# merge engine
# --------------------------------------------------------------------------

def _decide_layout(take_len, window_len, track_len, start_smp=0):
    """'overlay' = take is a full pass covering the whole track, read it at
    the same sample offsets.  'place' = take is a segment clip: place its
    start at the window start.

    Everything shorter than the track is a segment clip.  TTS dub files with
    baked-in lead silence are often longer than their (early) timestamp, and
    reading them at absolute offsets cut the phrase out of the clip entirely
    - the merge played the file's silent tail instead of the phrase, so whole
    stretches of dubs vanished from the output."""
    if take_len >= track_len:
        return "overlay"
    return "place"


def _slice_padded(data, start, end):
    """data[:, start:end] with zero padding when the take is shorter."""
    ch = data.shape[0]
    if start >= data.shape[1]:
        return np.zeros((ch, end - start), dtype=np.float32)
    if end <= data.shape[1]:
        return data[:, start:end]
    out = np.zeros((ch, end - start), dtype=np.float32)
    out[:, : data.shape[1] - start] = data[:, start:]
    return out


def _slice_window(data, rel_start, rel_end):
    """data[:, rel_start:rel_end] padded with zeros on BOTH sides.

    Used by timeline-edited placements: the piece audio may begin after the
    window start (silence added in front) and end before the window end
    (silence added at the back)."""
    ch, L = data.shape[0], data.shape[1]
    out = np.zeros((ch, rel_end - rel_start), dtype=np.float32)
    s0, s1 = max(0, rel_start), min(L, rel_end)
    if s1 > s0:
        out[:, s0 - rel_start : s1 - rel_start] = data[:, s0:s1]
    return out


def edit_id(info):
    """Stable id of a take for the timeline editor: timestamp + take number."""
    return "%s_%s" % (info["start_ms"], info["take"])


def parse_timeline_edits(raw, warnings=None):
    """Normalise timeline-editor adjustments.

    Accepts None, '' , a JSON string, an already-parsed dict
    ({id: {start_ms, end_ms, audio_ms}}) or the widget's JSON shape
    ({"version": 1, "edits": [{id, ...}, ...]}; a bare list of edit objects
    is accepted too) and returns {id: {...}} with int values.  The window
    [start_ms, end_ms] is where the original voice is silenced; the piece
    audio plays from audio_ms (>= start_ms: dragging the piece right adds
    front silence, dragging the right edge adds end silence).  An entry may
    additionally carry "amb": true (the piece was marked ambient with
    alt+click) and may consist of ONLY that flag - a mark without any
    geometry, which flags the piece for the per-piece ambient fill without
    pinning its position.

    Malformed input NEVER raises: it returns {} (or skips the offending
    entry) and appends a reason to `warnings` so the report tells the user
    what was ignored.  A broken widget value must not kill the merge - in
    particular, workflows saved before the timeline editor carry a stale
    scalar (True / 500.0) in the timeline_edits slot after ComfyUI's
    positional widget mapping."""
    def _warn(msg):
        if warnings is not None:
            warnings.append("timeline_edits ignored: " + msg)

    if raw is None:
        return {}
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        try:
            raw = json.loads(raw)
        except ValueError as exc:
            _warn("not valid JSON (%s) - merging without adjustments" % exc)
            return {}
    if isinstance(raw, (bool, int, float)):
        _warn("expected the editor's JSON, got %r - a stale value from an "
              "older workflow in the timeline_edits slot (the widget list "
              "changed in this version); merging without adjustments. "
              "Fix: set the timeline_edits widget to empty / re-create the "
              "node." % (raw,))
        return {}
    entries = None
    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict) and isinstance(raw.get("edits"), list):
        entries = raw["edits"]
    if entries is not None:
        # widget JSON shape: {"version": 1, "edits": [{id, start_ms, ...}, ...]}
        out = {}
        for e in entries:
            if not isinstance(e, dict) or "id" not in e:
                _warn("entry missing 'id' skipped: %r" % (e,))
                continue
            pid = str(e["id"])
            vals = _parse_edit_ms(pid, e, warnings)
            if vals is not None:
                out[pid] = vals
        return out
    if not isinstance(raw, dict):
        _warn("expected a JSON object, got %s - merging without adjustments"
              % type(raw).__name__)
        return {}
    out = {}
    for pid, e in raw.items():
        vals = _parse_edit_ms(pid, e, warnings)
        if vals is not None:
            out[str(pid)] = vals
    return out


def _parse_edit_ms(pid, e, warnings=None):
    """Parse one edit entry; returns None (with a warning) instead of raising.

    An entry is either a geometry edit ({start_ms, end_ms, audio_ms},
    optionally plus "amb": true) or an ambient mark alone ({"amb": true}).
    """
    if not isinstance(e, dict):
        if warnings is not None:
            warnings.append("timeline_edits ignored: entry %s is not an object" % pid)
        return None
    vals = {}
    if any(k in e for k in ("start_ms", "end_ms", "audio_ms")):
        try:
            vals = {k: int(round(float(e[k])))
                    for k in ("start_ms", "end_ms", "audio_ms")}
        except (KeyError, TypeError, ValueError):
            if warnings is not None:
                warnings.append("timeline_edits ignored: entry %s has invalid ms "
                                "values - merging that piece without adjustments" % pid)
            return None
        if vals["end_ms"] <= vals["start_ms"]:
            if warnings is not None:
                warnings.append("timeline_edits ignored: entry %s has "
                                "end_ms <= start_ms - merging that piece without "
                                "adjustments" % pid)
            return None
    elif not e.get("amb"):
        if warnings is not None:
            warnings.append("timeline_edits ignored: entry %s has neither ms "
                            "values nor an ambient mark - merging that piece "
                            "without adjustments" % pid)
        return None
    if e.get("amb"):
        vals["amb"] = True
    return vals


def write_timeline_cache(cache_dir, sr, track_len, segments, voice_stem,
                         sep_method, cache_sr=22050, edits_raw=""):
    """Write the browser timeline editor's data bundle into cache_dir.

    The bundle is a manifest.json plus 16-bit mono WAVs (downsampled to
    cache_sr): the separated voice stem (or the full mix when there is no
    voice stem) and one file per winning dub piece (post-gain audio as it
    will play in the merge).  `edits_raw` (the node's timeline_edits widget
    value as consumed by this run) is embedded in the manifest so the
    editor can restore the saved edits even when the widget itself comes
    up empty.  Returns the manifest dict.  Raises on I/O errors; the
    caller turns those into merge warnings."""
    import soundfile as sf

    os.makedirs(cache_dir, exist_ok=True)
    cs = int(min(int(cache_sr), int(sr)))
    keep = set()

    def _mono16(data, src_sr, path):
        mono = match_channels(np.asarray(data, dtype=np.float32), 1)
        wav = resample_np(np.ascontiguousarray(mono), int(src_sr), cs)
        sf.write(path, np.ascontiguousarray(wav[0]), cs, subtype="PCM_16")

    voice_path = os.path.join(cache_dir, "voice.wav")
    _mono16(voice_stem, sr, voice_path)
    keep.add("voice.wav")

    pieces = []
    for seg in segments:
        data = seg["data"]
        if data.shape[1] == 0 or seg["b"] <= seg["a"]:
            continue
        pid = edit_id(seg["info"])
        name = "piece_%s.wav" % pid
        _mono16(data, sr, os.path.join(cache_dir, name))
        keep.add(name)
        # exact placement as the merge plays it: the final window (after
        # queue_overlaps) and where the file's first sample lands.  place
        # clips play from the window start (or the edited audio_ms); overlay
        # full passes are read at absolute offsets, so their sample 0 is
        # track position 0 and edits cannot move them.
        layout = seg.get("layout") or "place"
        a_ms = int(round(seg["a"] * 1000.0 / sr))
        if seg.get("edit_place_at") is not None and layout == "place":
            audio_ms = int(round(seg["edit_place_at"] * 1000.0 / sr))
        elif layout == "place":
            audio_ms = a_ms
        else:
            audio_ms = 0
        pieces.append({
            "id": pid,
            "take": seg["info"]["take"],
            "filename": seg["info"]["filename"],
            "start_ms": int(seg["info"]["start_ms"]),
            "dur_ms": int(round(data.shape[1] * 1000.0 / sr)),
            "layout": layout,
            "a_ms": a_ms,
            "b_ms": int(round(seg["b"] * 1000.0 / sr)),
            "audio_ms": audio_ms,
            "queued": "queued_shift" in seg,
            "file": name,
        })

    manifest = {
        "version": 1,
        # edit protocol this python speaks: 2 = edit entries carry audio_ms
        # (the piece's placement anchor, which may sit BEFORE start_ms after
        # a left-edge trim); 3 = the anchors reference the STRIPPED piece
        # buffers (strip_silence runs before the edit overrides and strips
        # edited pieces too, so an edit can no longer revert a previously
        # stripped buffer to the full take).  A python older than its own
        # editor drops entries it does not understand and merges unedited -
        # the editor reads this field and tells the user to restart ComfyUI
        # instead of failing mysteriously (the JS hot-reloads on refresh;
        # the python does not).
        "edit_protocol": 3,
        "cache_sr": cs,
        "duration_ms": int(round(track_len * 1000.0 / sr)),
        "sep_method": sep_method or "",
        "voice": "voice.wav",
        # the timeline_edits value this run consumed (raw widget JSON), so
        # re-opening the editor restores the edits even when the widget
        # itself comes up empty (stale graph state, wipe accidents)
        "timeline_edits": str(edits_raw or ""),
        "pieces": pieces,
    }
    # drop pieces from earlier runs whose takes are gone now
    for name in os.listdir(cache_dir):
        if name.startswith("piece_") and name.endswith(".wav") and name not in keep:
            try:
                os.remove(os.path.join(cache_dir, name))
            except OSError:
                pass
    with open(os.path.join(cache_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    return manifest


def prepare_merge(track_len, sr, ch_count, takes_folder, *, extensions=".flac",
                  recursive=False, timestamp_mode="start", window_mode="until_next",
                  silence_db=-50.0, tail_pad_ms=200.0, tail_silence_ms=0.0,
                  queue_overlaps=False, overflow="extend", gain_db=0.0,
                  fade_ms=10.0, trim_lead=False, strip_silence=False,
                  timeline_edits=None,
                  ambient_db=-12.0):
    """Scan the folder, pick winning takes, load + conform them and plan windows.

    Returns (segments, scan_info, warnings).  Each segment dict holds:
      info (parsed filename), data [C, L] float32 conformed to track sr/ch,
      a, b (window in samples), explicit_window, layout, contiguous_next, notes.
    auto_trim segments additionally carry audible_end (just past the take's
    last audible sample) and, when tail_silence_ms > 0, hold_end (audible_end
    + the post-phrase silence) - the original voice fades back in only after
    hold_end (see _tail_xfade_start).
    Segments moved by queue_overlaps carry queued_shift (samples shifted)
    and a matching note; scan["queued"] counts them.

    queue_overlaps: when a take's audio runs past the next take's timestamp,
    the next take is queued - shifted to start right after the current one's
    audio ends - instead of being truncated by it (explosive back-to-back
    dialogue).  Segment clips (place layout) queue after their AUDIBLE end:
    trailing file silence is stripped, so a TTS tail baked into the dub file
    cannot push the following phrase rightward (queued dialogue would drift
    out of sync more with every line).  Explicit filename windows and
    timeline-edited pieces queue by their declared span, keeping their drawn
    pads (the piece audio moves with the shifted window).  Full-track passes
    keep the legacy handover at the next timestamp.

    trim_lead: pre-process each placed dub clip - leading silence (TTS clips
    often start with 100-500 ms of baked-in quiet) is cut so the voice starts
    right at the phrase timestamp instead of late.  A guard pad of
    max(20 ms, fade_ms) of quiet is kept ahead of the first audible sample so
    the window-start crossfade and soft word onsets are not shaved.  Applies
    to place-layout clips only: full-track passes are read at absolute sample
    offsets and are never shifted.  With timestamp_mode=end the phrase end
    stays anchored at its timestamp - the window start moves up by the
    trimmed amount instead.  The silence_db threshold decides what counts as
    silence (measured on the gained take audio, same as auto_trim).

    strip_silence: the same lead trim plus the trailing half - the file's
    baked tail silence is cut too, down to the tail pad (tail_pad_ms).  Dub
    pieces then start and end at their actual sound (plus the quality pads),
    the editor shows the real phrase extent, and the user can still drag the
    piece edges back into silence or cut the pads to the sound borders.
    Same placement rules as trim_lead (place-layout, non-edited clips only;
    end-anchored windows keep their phrase-end alignment).

    timeline_edits: adjustments drawn in the browser timeline editor, as
    {id: {start_ms, end_ms, audio_ms}} (see parse_timeline_edits; an entry
    may also be an ambient mark alone, {"amb": true}, or carry "amb" in
    addition to the geometry - pieces marked ambient get the per-piece
    ambient fill at ambient_db).  An edit
    turns the take into an explicit window: the original voice is silenced
    exactly across [start_ms, end_ms] and the piece audio plays from
    audio_ms - front/end silence comes from where the piece was dragged.
    Edited takes skip trim_lead, auto_trim and queueing (explicit windows
    queue by their declared span).  Full-track passes cannot be re-placed;
    their edits are ignored with a note.
    """
    if not takes_folder or not str(takes_folder).strip():
        scan = scan_take_files("", extensions, recursive)
        scan["error"] = "takes folder is empty - set the takes_folder widget"
        scan["folder"] = ""
        return [], scan, []
    folder = str(takes_folder)
    if not os.path.isdir(folder):
        scan = scan_take_files(folder, extensions, recursive)
        scan["error"] = "takes folder not found: %s" % folder
        return [], scan, []

    scan = scan_take_files(folder, extensions, recursive)
    winners, losers = select_best_takes(scan["parsed"])
    scan["losers"] = losers
    scan["n_groups"] = len(set(i["start_ms"] for i in scan["parsed"]))

    warnings = []
    gain = 10.0 ** (gain_db / 20.0)
    thresh = 10.0 ** (silence_db / 20.0)
    tail_smp = max(0, int(round(sr * tail_pad_ms / 1000.0)))
    silence_smp = max(0, int(round(sr * tail_silence_ms / 1000.0)))

    segments = []
    for win in winners:
        try:
            data, sr_seg = load_audio_file(win["path"])
        except Exception as exc:
            warnings.append("%s: could not load (%s); skipped" % (win["filename"], exc))
            continue
        if sr_seg != sr:
            data = resample_np(data, sr_seg, sr)
        data = np.ascontiguousarray(match_channels(np.asarray(data, dtype=np.float32), ch_count))
        if gain_db:
            data = data * np.float32(gain)
        if data.shape[1] == 0:
            warnings.append("%s: file contains no audio; skipped" % win["filename"])
            continue
        segments.append({"info": win, "data": data, "notes": []})
    segments.sort(key=lambda s: (s["info"]["start_ms"], s["info"]["take"]))
    n = len(segments)

    # ---- provisional windows ------------------------------------------------
    for i, seg in enumerate(segments):
        info, data = seg["info"], seg["data"]
        take_len = data.shape[1]
        ts_smp = ms_to_samples(info["start_ms"], sr)
        if info["end_ms"] is not None:
            a, b = ts_smp, ms_to_samples(info["end_ms"], sr)
            if b < a:
                a, b = b, a
                seg["notes"].append("filename window reversed; start/end swapped")
            seg["explicit_window"] = True
        elif timestamp_mode == "end":
            b = ts_smp
            a = b - take_len
            if a < 0:
                a = 0
                seg["notes"].append("end-mode window starts before track; clamped to 0")
            seg["explicit_window"] = True
            seg["end_anchored"] = True   # phrase END is the placement anchor
        else:  # timestamp_mode == "start"
            a = ts_smp
            if i < n - 1:
                b = ms_to_samples(segments[i + 1]["info"]["start_ms"], sr)
            else:
                b = int(track_len)
                if overflow == "extend":
                    ce = _content_end(data, thresh, tail_smp)
                    if ce > b:
                        b = int(ce)
                        seg["notes"].append(
                            "take content runs %s past track end; output extended"
                            % fmt_seconds((ce - track_len) / sr))
                elif overflow == "truncate":
                    ce = _content_end(data, thresh, tail_smp)
                    if ce > track_len:
                        seg["notes"].append(
                            "take content runs %s past track end; dropped (overflow=truncate)"
                            % fmt_seconds((ce - track_len) / sr))
            seg["explicit_window"] = False
        seg["a"], seg["b"] = int(a), int(b)
        seg["layout"] = _decide_layout(take_len, seg["b"] - seg["a"], track_len, a)

    # ---- strip leading / trailing silence (voice starts at the timestamp) --
    # TTS dub clips often carry baked-in quiet at BOTH ends: leading silence
    # makes a clip placed at its phrase timestamp start speaking late (the
    # dub drifts out of sync), and trailing silence inflates the drawn piece
    # and the editor's file border.  strip_silence cuts both from every
    # placed clip, down to quality pads: a guard of max(20 ms, fade_ms) is
    # kept ahead of the first audible sample (the window-start crossfade
    # lives there and soft word onsets must not be shaved) and the tail pad
    # (tail_pad_ms) after the last one (the merge's tail crossfade room, the
    # same pad auto_trim and overflow=extend work with).  The clip's audio
    # then starts right at its phrase timestamp and the editor's piece
    # borders sit at the sound - the user can still drag the edges back into
    # silence, or cut the remaining pads all the way to the sound borders.
    # The pass runs BEFORE the timeline-edit overrides, and EDITED pieces are
    # stripped too: the editor's saved anchors (audio_ms) are always drawn
    # against the piece buffers the previous merge shipped - stripped files -
    # so the buffers must be stripped again before those anchors are applied.
    # Skipping edited pieces here would revert their buffers to the full
    # take the moment any edit landed on a previously-stripped piece: the
    # baked head would play as leading silence and the drawn end (saved
    # against the stripped sound border) would cut the phrase by the same
    # head - the dub drifts late and truncates at both ends, and every
    # reopen shows it again.  Stripping is idempotent within the guard, so
    # re-running it on already-stripped audio is a no-op.  Full-track passes
    # are read at absolute sample offsets and are never touched.
    # End-anchored windows (timestamp_mode=end) keep their phrase-end
    # alignment: the window start moves up by the trimmed lead (timeline
    # edits, when present, are applied afterwards and overwrite the
    # geometry).  The threshold is silence_db on the gained take audio, the
    # same silence definition that auto_trim and the queue handover use.
    # trim_lead (API-only, kept for backward compatibility) does the lead
    # half alone.
    scan["lead_trim_count"] = 0
    scan["lead_trim_smp"] = 0
    scan["tail_strip_smp"] = 0
    scan["strip_count"] = 0
    do_lead = strip_silence or trim_lead
    if do_lead or strip_silence:
        guard = max(int(round(sr * 0.020)), int(round(sr * fade_ms / 1000.0)))
        for seg in segments:
            if seg.get("layout") != "place":
                continue
            data = seg["data"]
            above = np.flatnonzero(_max_abs_per_sample(data) > thresh)
            if above.size == 0:
                continue
            lead_cut = 0
            if do_lead and int(above[0]) > guard:
                lead_cut = int(above[0]) - guard
            tail_cut = 0
            if strip_silence:
                tail_cut = max(0, data.shape[1] -
                               (int(above[-1]) + 1 + tail_smp))
            if lead_cut <= 0 and tail_cut <= 0:
                continue
            seg["data"] = np.ascontiguousarray(
                data[:, lead_cut:data.shape[1] - tail_cut])
            if seg.get("end_anchored") and lead_cut > 0:
                seg["a"] += lead_cut     # keep the phrase end at its timestamp
            if lead_cut > 0:
                scan["lead_trim_count"] += 1
                scan["lead_trim_smp"] += lead_cut
                seg["notes"].append(
                    "leading silence %s stripped%s: voice now starts %s after "
                    "the phrase timestamp (guard pad)"
                    % (fmt_seconds(lead_cut / sr),
                       ", window start moved up" if seg.get("end_anchored")
                       else "",
                       fmt_seconds(guard / sr)))
            if tail_cut > 0:
                scan["tail_strip_smp"] += tail_cut
                seg["notes"].append(
                    "trailing silence %s stripped (tail pad %s kept)"
                    % (fmt_seconds(tail_cut / sr),
                       fmt_seconds(tail_smp / sr)))
            scan["strip_count"] += 1

    # ---- timeline editor overrides ------------------------------------------
    # Adjustments drawn in the browser timeline editor (see
    # parse_timeline_edits): the window [start_ms, end_ms] is where the
    # original voice is silenced and the piece audio plays from audio_ms -
    # front silence when the piece was dragged right / its left edge pulled
    # right, end silence when its right edge was pulled right.  Edited takes
    # become explicit windows placed at an arbitrary offset; everything
    # downstream (queue handover, fades, ambient fill) works on them as on
    # declared spans.
    scan["edited"] = 0
    edits = parse_timeline_edits(timeline_edits, warnings)
    if edits:
        for seg in segments:
            e = edits.get(edit_id(seg["info"]))
            if e is None:
                continue
            if seg.get("layout") != "place":
                seg["notes"].append("timeline edit ignored: full-track pass is "
                                    "read at absolute sample offsets")
                continue
            seg["amb"] = bool(e.get("amb"))
            if "start_ms" not in e:
                # ambient mark alone: flag the piece for the per-piece fill,
                # leave its geometry (and the queue handover) untouched
                seg["notes"].append(
                    "timeline edit: marked ambient - the original span plays "
                    "under the dub at %+.1f dB" % float(ambient_db))
                continue
            a = ms_to_samples(e["start_ms"], sr)
            b = ms_to_samples(e["end_ms"], sr)
            pa = ms_to_samples(e["audio_ms"], sr)
            # the window may start AFTER the piece audio (a left-edge drag
            # cut the file's baked head) or end before the audio ends (tail
            # cut) - _slice_window renders both exactly as drawn.  Only
            # bound the placement to the window end, so a malformed entry
            # cannot place the audio outside the muted span.
            pa = min(pa, b)
            seg["a"], seg["b"] = int(a), int(b)
            seg["edit_place_at"] = int(pa)
            seg["explicit_window"] = True
            seg["edited"] = True
            scan["edited"] += 1
            seg["notes"].append(
                "timeline edit: original silenced %s - %s, piece audio plays "
                "from %s%s"
                % (fmt_clock(e["start_ms"]), fmt_clock(e["end_ms"]),
                   fmt_clock(e["audio_ms"]),
                   ", ambient" if e.get("amb") else ""))

    # ---- queue overlapping takes (explosive dialogue) -----------------------
    # When a dub's audio runs past the next take's timestamp, the next take
    # would cut it off mid-phrase and the two dubs would overlap.  Queue
    # instead: the next take starts right after the current one's audio
    # ends.  Segment clips (place layout) queue after their AUDIBLE end -
    # trailing file silence (TTS tails) is stripped so queued dialogue stays
    # in sync instead of drifting rightward by the tail of every file;
    # timeline-edited clips queue by their drawn end only where it actually
    # differs from the file extent (a real tail cut, or deliberate silence
    # drawn past the file) and at their audible end when the window still
    # ends at the file border; full-track passes hand over at the next
    # timestamp exactly as before.
    scan["queued"] = 0
    if queue_overlaps:
        # Emptiness is judged BEFORE any shifts: window ends are only
        # repaired after this pass, so a take shifted a moment ago legitimately
        # has b < a here and that must not read as "empty window".
        empty = [seg["b"] <= seg["a"] for seg in segments]
        for i in range(1, len(segments)):
            prev, seg = segments[i - 1], segments[i]
            if empty[i - 1]:
                continue                      # empty window occupies nothing
            if seg.get("layout") == "overlay":
                # a full pass is read at absolute sample offsets - shifting
                # its window would only desync (or silence) its audio
                continue
            prev_stripped = 0
            if prev["explicit_window"]:
                occ = prev["b"]               # declared replace span
                pa = prev.get("edit_place_at")
                if pa is not None:
                    # The declared end owns the handover only where it
                    # actually differs from the file extent: a drawn end
                    # inside the audio is a real tail trim (the drawn end
                    # wins), one past the file is deliberate silence (the
                    # drawn delay wins) - but a window that still ends at
                    # the file's own border (an edit that never touched the
                    # tail, e.g. a left-edge trim) must hand over at the
                    # AUDIBLE end like an unedited clip, or the file's
                    # baked TTS tail silence pushes every queued phrase
                    # rightward and the chain drifts out of the cluster.
                    file_end = int(pa) + int(prev["data"].shape[1])
                    if abs(prev["b"] - file_end) <= max(
                            1, int(round(sr * 0.002))):
                        occ = int(pa) + _content_end(prev["data"], thresh, 0)
            elif prev.get("layout") == "place":
                # Queue after the clip's AUDIBLE end: TTS dub files often end
                # with several hundred ms of baked-in silence - counting it
                # would push every queued phrase rightward and desync the
                # dialogue more with every line.
                ce = _content_end(prev["data"], thresh, 0)
                occ = prev["a"] + ce
                prev_stripped = int(prev["data"].shape[1]) - int(ce)
            else:
                continue                      # full-track pass: legacy handover
            if occ > seg["a"]:
                old_a = int(seg["a"])
                shift = int(occ - seg["a"])
                seg["a"] = old_a + shift
                if seg["explicit_window"]:
                    seg["b"] += shift         # move the declared span wholesale
                if seg.get("edit_place_at") is not None:
                    seg["edit_place_at"] += shift   # audio moves with the span
                scan["queued"] += 1
                seg["queued_shift"] = shift
                seg["notes"].append(
                    "overlapped %s - queued after it: start shifted %s -> %s (+%s)"
                    % (prev["info"]["filename"], fmt_clock(old_a * 1000.0 / sr),
                       fmt_clock(seg["a"] * 1000.0 / sr), fmt_seconds(shift / sr)))
                if prev_stripped > max(1, int(round(sr * 0.002))):
                    prev["notes"].append(
                        "trailing file silence %s stripped: the next dub queues "
                        "right after the phrase, not after the file"
                        % fmt_seconds(prev_stripped / sr))
        # Window ends follow the queued schedule: each replaced span now runs
        # until the next (possibly shifted) take starts, so a queued-over
        # clip plays to its full length instead of being truncated.  A queued
        # take can land at (or past) an immovable full pass's window start -
        # its replaced span must never collapse below its own audible phrase,
        # or the queued dub would be dropped from the merge entirely.
        for i, seg in enumerate(segments):
            if seg["explicit_window"] or i + 1 >= len(segments):
                continue
            new_b = int(segments[i + 1]["a"])
            floor = int(seg["a"] + _content_end(seg["data"], thresh, 0))
            if new_b < floor:
                new_b = floor
            if new_b > seg["b"]:
                seg["b"] = new_b

    # Last take: a placed clip can run past the track end.  With
    # overflow=extend the output extends to the clip's audible end
    # (its content indexes are clip-relative, unlike a full pass).
    if (segments and overflow == "extend" and not segments[-1]["explicit_window"]
            and segments[-1].get("layout") == "place"):
        last = segments[-1]
        ce = last["a"] + _content_end(last["data"], thresh, tail_smp)
        if ce > last["b"]:
            last["b"] = int(ce)
            last["notes"].append("take content runs %s past track end; output extended"
                                 % fmt_seconds((ce - track_len) / sr))

    # ---- auto trim (silent-bed takes) ---------------------------------------
    if window_mode == "auto_trim":
        for i, seg in enumerate(segments):
            if seg["explicit_window"]:
                continue
            data = seg["data"]
            cap = seg["b"]
            if i == n - 1 and overflow == "extend":
                cap = max(cap, data.shape[1])
            a = seg["a"]
            span = cap - a
            if span <= 0:
                seg["notes"].append("empty window; nothing replaced")
                seg["b"] = seg["a"]
                continue
            if seg.get("layout") == "place":
                # Short take placed at the window start: trim on the take's
                # own samples, not at absolute track positions.
                src = data[:, :min(data.shape[1], span)]
            else:
                src = data[:, a:min(cap, data.shape[1])]
            if src.shape[1] == 0:
                seg["notes"].append("no take audio inside window; nothing replaced")
                seg["b"] = seg["a"]
                continue
            amax = _max_abs_per_sample(src)
            above = np.flatnonzero(amax > thresh)
            if above.size == 0:
                seg["notes"].append("take silent in window (< %.1f dB); nothing replaced" % silence_db)
                seg["b"] = seg["a"]
            else:
                # above indexes src; window-relative position = index for both
                # layouts (overlay src starts at a, placed src at the window).
                audible_end = a + int(above[-1]) + 1
                seg["audible_end"] = int(audible_end)  # phrase ends here
                hold_end = audible_end + silence_smp   # post-phrase voice silence
                if silence_smp > 0:
                    seg["hold_end"] = int(hold_end)
                new_b = min(hold_end + tail_smp, cap)
                new_b = max(new_b, a + 1)
                if new_b < seg["b"]:
                    note = "auto-trimmed window from %s to %s" % (
                        fmt_seconds((seg["b"] - a) / sr), fmt_seconds((new_b - a) / sr))
                    if new_b > audible_end:
                        note += "; original fades back in across the tail pad"
                    seg["notes"].append(note)
                if silence_smp > 0:
                    if hold_end < new_b:
                        seg["notes"].append(
                            "voice held silent %g ms after the phrase to cover leftover "
                            "original speech (background unaffected)" % tail_silence_ms)
                    else:
                        seg["notes"].append("tail silence pad clipped at the window end")
                seg["b"] = int(new_b)

    # ---- overflow policy ----------------------------------------------------
    kept = []
    for seg in segments:
        a, b = seg["a"], seg["b"]
        if b <= a:
            kept.append(seg)
            continue
        if a >= track_len:
            if overflow == "extend":
                kept.append(seg)
            else:
                warnings.append("%s: starts at %s past track end; skipped"
                                % (seg["info"]["filename"], fmt_clock(seg["info"]["start_ms"])))
            continue
        if b > track_len:
            if overflow == "truncate":
                seg["notes"].append("window truncated at track end (lost %s)"
                                    % fmt_seconds((b - track_len) / sr))
                seg["b"] = int(track_len)
            elif overflow == "skip":
                warnings.append("%s: window passes track end; skipped (overflow=skip)"
                                % seg["info"]["filename"])
                continue
            else:
                if not any("output extended" in note for note in seg["notes"]):
                    seg["notes"].append("window passes track end by %s; output extended"
                                        % fmt_seconds((b - track_len) / sr))
        kept.append(seg)
    segments = kept

    # ---- sort, clip overlaps, contiguity ------------------------------------
    segments.sort(key=lambda s: (s["a"], s["info"]["take"]))
    for i, seg in enumerate(segments):
        if i + 1 < len(segments):
            nxt = segments[i + 1]
            if nxt["a"] < seg["b"]:
                if seg.get("queued_shift"):
                    # the queue schedule deliberately runs this replaced span
                    # past the next take's original start (e.g. a take queued
                    # up against an immovable full pass) - clipping it here
                    # would cut the queued phrase short, or drop it entirely
                    seg["notes"].append(
                        "queued span kept past next take's start (%s overlap)"
                        % fmt_seconds((seg["b"] - nxt["a"]) / sr))
                else:
                    seg["notes"].append("window overlapped next segment; clipped %s"
                                        % fmt_seconds((seg["b"] - nxt["a"]) / sr))
                    seg["b"] = int(nxt["a"])
            seg["contiguous_next"] = bool(nxt["a"] == seg["b"])
        else:
            seg["contiguous_next"] = False

    # ---- take-length sanity notes -------------------------------------------
    check_len = ms_to_samples(50, sr)
    for seg in segments:
        a, b, data = seg["a"], seg["b"], seg["data"]
        if b <= a:
            continue
        take_len = data.shape[1]
        smooth_tail = seg.get("audible_end") is not None
        if seg.get("layout") == "place":
            if take_len < (b - a) and not seg.get("edited"):
                if smooth_tail:
                    seg["notes"].append("take shorter than window by %s; original fades back in across the tail"
                                        % fmt_seconds(((b - a) - take_len) / sr))
                else:
                    seg["notes"].append("take shorter than window by %s; tail stays silent"
                                        % fmt_seconds(((b - a) - take_len) / sr))
            continue
        if b > take_len:
            if smooth_tail:
                seg["notes"].append("take audio ends %s before window end; original fades back in across the tail"
                                    % fmt_seconds((b - take_len) / sr))
            else:
                seg["notes"].append("take audio ends %s before window end; tail stays silent"
                                    % fmt_seconds((b - take_len) / sr))
        elif b < take_len and not seg.get("contiguous_next"):
            # Window truncates the take before its file ends: only then is
            # active audio at the boundary meaningful (a music bed would fire
            # on every window; a natural file end is not a cut).
            check = data[:, max(a, b - check_len):b]
            if check.shape[1] > 0 and float(np.max(np.abs(check))) > thresh:
                seg["notes"].append("take audio still active at window end (phrase may be cut)")

    out_len = int(track_len)
    for seg in segments:
        out_len = max(out_len, seg["b"])
    scan["out_len"] = out_len
    return segments, scan, warnings


def _tail_xfade_start(seg, a, b, fade_smp):
    """Absolute sample where the take -> original tail crossfade starts.

    auto_trim remembers where the take's audible content ends ("audible_end").
    When a post-phrase silence is requested (tail_silence_ms > 0), the
    segment carries hold_end = audible_end + silence: the original VOICE
    stays muted that much longer, covering the leftover tail of a longer
    original phrase (e.g. a residual "...des ka" after a shorter dub), and
    the crossfade back into the original starts only at hold_end.

    The tail pad that follows is near-silent take audio, so instead of keeping
    the original muted to the window end (which audible listeners hear as a
    dead-air stutter), the original fades back in across that tail while the
    take fades out - an equal-power crossfade into the original (sqrt curves:
    constant summed power, so the original is clearly audible early in the
    fade instead of dipping).

    Returns the absolute start sample, or None to fall back to the plain
    window-end fade.  Suppressed for contiguous windows (those crossfade into
    the next take, not into the original) and when fade_smp == 0 (hard cuts).
    """
    if fade_smp <= 0 or seg.get("contiguous_next") or b - a < 2:
        return None
    e = seg.get("hold_end", seg.get("audible_end"))
    if e is None:
        return None
    e = min(max(int(e), a + fade_smp), b)
    if b - e < 2:
        return None
    return e


def _build_orig_env(out_len, segments, fade_smp):
    """Mute envelope for the original track: 1.0 outside windows, 0.0 inside,
    with equal-gain fades at edges.  Contiguous windows stay muted through the
    shared boundary so takes crossfade into each other, not back into the
    original.  auto_trim windows fade the original back in across the take's
    silent tail pad with an equal-power (sqrt) curve instead of holding the
    mute until the window end; when the segment carries a post-phrase silence
    (hold_end), the mute extends through that hold first, covering leftover
    original speech after a shorter dub."""
    env = np.ones(out_len, dtype=np.float32)
    for i, seg in enumerate(segments):
        a = max(0, seg["a"])
        b = min(out_len, seg["b"])
        if b - a < 1:
            continue
        f = min(fade_smp, (b - a) // 2)
        env[a:b] = 0.0
        if f > 0:
            prev_b = segments[i - 1]["b"] if i > 0 else None
            if prev_b is None or prev_b < a:
                env[a:a + f] = np.linspace(1.0, 0.0, f, dtype=np.float32)
        if not seg.get("contiguous_next"):
            ts = _tail_xfade_start(seg, a, b, f)
            if ts is not None:
                # Equal-power fade-in: sqrt(r) reaches -6 dB at 25% of the
                # fade (vs 50% for linear), so the original is audible early.
                r = np.linspace(0.0, 1.0, b - ts, dtype=np.float32)
                env[ts:b] = np.sqrt(r)
            elif f > 0:
                env[b - f:b] = np.linspace(0.0, 1.0, f, dtype=np.float32)
    return env


def _apply_take(out, seg, fade_smp, out_len):
    """Add one take's contribution into `out` (in-place, overlap-add)."""
    a = max(0, seg["a"])
    b = min(out_len, seg["b"])
    if b <= a:
        return
    data = seg["data"]
    f = min(fade_smp, (b - a) // 2)
    place_at = seg.get("edit_place_at")
    if place_at is not None:
        # Timeline-edited placement: the piece audio plays from place_at,
        # front/end window padding comes from _slice_window's zero pads.
        body = _slice_window(data, a - place_at, b - place_at)
    elif seg.get("layout") == "place":
        body = _slice_padded(data, 0, b - a)
    else:
        body = _slice_padded(data, a, b)
    env = np.ones(b - a, dtype=np.float32)
    if f > 0:
        env[:f] = np.linspace(0.0, 1.0, f, dtype=np.float32)
    ts = _tail_xfade_start(seg, a, b, f)
    if ts is not None:
        # Equal-power crossfade back into the original across the take's
        # silent tail pad (mirrors the sqrt rise in _build_orig_env;
        # sqrt(1-r) + sqrt(r) keeps summed power constant).
        r = np.linspace(0.0, 1.0, b - ts, dtype=np.float32)
        env[ts - a:] = np.sqrt(1.0 - r)
    elif f > 0 and not seg.get("contiguous_next"):
        env[b - a - f:] = np.linspace(1.0, 0.0, f, dtype=np.float32)
    out[:, a:b] += body * env[None, :]
    if seg.get("contiguous_next") and f > 0:
        # Crossfade into the next take: this take keeps playing (fading out)
        # for f samples past the shared boundary.
        ext_end = min(out_len, b + f)
        if ext_end > b:
            if place_at is not None:
                ext = _slice_window(data, b - place_at, ext_end - place_at)
            elif seg.get("layout") == "place":
                ext = _slice_padded(data, b - a, (b - a) + (ext_end - b))
            else:
                ext = _slice_padded(data, b, ext_end)
            ramp = np.linspace(1.0, 0.0, ext_end - b, dtype=np.float32)
            out[:, b:ext_end] += ext * ramp[None, :]


def _ambient_mask(out_len, segments, fade_smp):
    """Mute complement for the AMBIENT-MARKED windows only (per-piece fill).

    Returns float32 [out_len] that is 1.0 where a marked piece's window
    silences the original and 0.0 everywhere else, or None when no piece is
    marked.  Built from the same envelope shapes as _build_orig_env (edge
    fades, tail crossfades, hold extensions), so the fill crossfades into
    the original exactly where the mute does.  The contiguity flags are
    re-checked against MARKED neighbours: two contiguous marked windows keep
    their fill through the shared boundary, but a marked window followed by
    an unmarked one fades its fill out at the boundary instead of cutting.
    """
    idx = [i for i, s in enumerate(segments) if s.get("amb")]
    if not idx:
        return None
    proxies = []
    for i in idx:
        s = dict(segments[i])
        nxt = segments[i + 1] if i + 1 < len(segments) else None
        s["contiguous_next"] = bool(
            s.get("contiguous_next") and nxt is not None
            and nxt.get("amb") and int(nxt["a"]) == int(s["b"]))
        proxies.append(s)
    return 1.0 - _build_orig_env(out_len, proxies, fade_smp)


def apply_merge(track_batch, sr, segments, fade_ms, fill_wave=None,
                marked_segments=None, raw_wave=None, raw_gain=0.0):
    """Apply prepared segments to a [B, C, T] track batch.

    fill_wave: optional [C, T'] ambient stem (extract_ambient_stem) that is
    added wherever the original is muted (envelope 1 - mute), so continuous
    background content that lives inside the replaced stem (rain, ambience,
    music bed) keeps playing at a constant level under the dubs.  At the
    window edges the two envelopes are complementary, so the ambient part of
    the original crossfades into its extracted copy without a level step.

    marked_segments: the subset of `segments` whose pieces were marked
    ambient in the timeline editor.  When given, the fill applies ONLY under
    those windows (mask from _ambient_mask); other dub windows mute the
    replaced stem completely.  When None, the fill applies under every
    window (legacy whole-track behaviour, kept for direct API callers).

    raw_wave ([C, T]) + raw_gain: with a marked mask, the replaced stem
    itself is additionally mixed under the marked windows at raw_gain
    (linear), the "original + dub at different dB levels" blend - the leaked
    ambience the gate cannot rebuild (crowd chatter, scene-specific texture)
    and a faint ghost of the original voice continue under the dub, which
    stays dominant.  Added BEFORE the takes, so a marked window carries
    dub + bed + raw*gain.

    Returns (merged [B, C, out_len], out_len).  The input batch is not modified.
    """
    wf = np.asarray(track_batch, dtype=np.float32)
    if wf.ndim == 2:
        wf = wf[None, ...]
    B, C, T = wf.shape
    out_len = int(T)
    for seg in segments:
        out_len = max(out_len, seg["b"])
    fade_smp = max(0, int(round(sr * fade_ms / 1000.0)))
    env = _build_orig_env(out_len, segments, fade_smp)
    base = np.zeros((C, out_len), dtype=np.float32)
    base[:, :T] = wf[0]
    base *= env[None, :]
    if fill_wave is not None:
        fill = np.asarray(fill_wave, dtype=np.float32)
        if fill.ndim == 1:
            fill = fill[None, :]
        fp = np.zeros((C, out_len), dtype=np.float32)
        n = min(fill.shape[1], out_len)
        fp[:, :n] = fill[:, :n]
        if marked_segments is None:
            base += fp * (1.0 - env[None, :])
        else:
            mask = _ambient_mask(out_len, segments, fade_smp)
            if mask is not None:
                base += fp * mask[None, :]
            if raw_wave is not None and raw_gain > 0.0:
                raw = np.asarray(raw_wave, dtype=np.float32)
                if raw.ndim == 1:
                    raw = raw[None, :]
                rp = np.zeros((C, out_len), dtype=np.float32)
                n = min(raw.shape[1], out_len)
                rp[:, :n] = raw[:, :n]
                base += rp * mask[None, :] * np.float32(raw_gain)
    out = np.empty((B, C, out_len), dtype=np.float32)
    for bi in range(B):
        ob = base.copy()
        for seg in segments:
            _apply_take(ob, seg, fade_smp, out_len)
        out[bi] = ob
    return out, out_len


# --------------------------------------------------------------------------
# report + pipeline
# --------------------------------------------------------------------------

def build_report(scan, segments, warnings, sr, ch_count, track_len, out_len, *,
                 fade_ms, gain_db, window_mode, timestamp_mode, overflow,
                 applied, n_batches=1, separation="full_mix",
                 voice_low_hz=100.0, voice_high_hz=12000.0, sep_method=None,
                 tail_pad_ms=200.0, tail_silence_ms=0.0, queue_overlaps=False,
                 ambient_pieces=0, ambient_db=-12.0, ambient_fill_ref=False,
                 background_only=False, trim_lead=False,
                 strip_silence=False,
                 fails_moved=0, fail_dir=None, fail_enabled=False):
    L = []
    L.append("==== Dub Merge Report ====")
    L.append("Track        : %d ch, %d Hz, %s (%d samples)"
             % (ch_count, sr, fmt_clock(track_len * 1000.0 / sr), track_len))
    if n_batches > 1:
        L.append("Batch items  : %d (merge applied to every batch item)" % n_batches)
    L.append("Takes folder : %s  [recursive: %s, extensions: %s]"
             % (scan.get("folder") or "(not set)",
                "on" if scan.get("recursive") else "off", scan.get("extensions")))
    L.append("Settings     : timestamp=%s, window=%s, fade=%g ms, tail_pad=%g ms, "
             "tail_silence=%g ms, gain=%+g dB, overflow=%s, queue=%s, "
             "strip=%s, mode=%s"
             % (timestamp_mode, window_mode, fade_ms, tail_pad_ms, tail_silence_ms,
                gain_db, overflow, "on" if queue_overlaps else "off",
                "lead+tail" if strip_silence else ("lead" if trim_lead else "off"),
                "merge" if applied else "REPORT ONLY (dry run)"))
    if separation == "voice_isolation":
        L.append("Voice isolation: ON - dubs replace only the voice; the background is "
                 "separated first and re-added unchanged")
        if sep_method:
            L.append("                 separation: %s" % sep_method)
        if background_only:
            L.append("Background only: ON - the separated voice stem is discarded; the dubs "
                     "are merged with the background alone, so ambience leaked into the "
                     "voice stem cannot duck under the dubs (for ambience-heavy scenes)")
    else:
        L.append("Voice isolation: off (dubs replace the full mix)")
    if applied:
        if background_only:
            if ambient_pieces:
                L.append("Ambient fill : %d piece(s) marked ambient but background-only "
                         "mode never mutes the background - nothing to refill"
                         % ambient_pieces)
            else:
                L.append("Ambient fill : not used - background-only mode never mutes the "
                         "background, there is no replaced stem to refill")
        elif ambient_pieces:
            L.append("Ambient fill : %d dub piece(s) marked ambient (alt+click in the "
                     "timeline editor) - the gated non-speech bed keeps playing under "
                     "their windows and the original span is mixed under the dub at "
                     "%+.1f dB, so leaked crowd/rain/room tone continues while the dub "
                     "stays dominant (the original voice ghosts in at that level)"
                     % (ambient_pieces, ambient_db)
                     + (" (background-referenced voice suppression)"
                        if ambient_fill_ref else ""))
        else:
            L.append("Ambient fill : off - no pieces marked ambient (alt+click a dub "
                     "piece in the timeline editor to mark one)")
    if scan.get("error"):
        L.append("")
        L.append("ERROR: %s" % scan["error"])
        L.append("")
        L.append("Summary: 0 pieces replaced.")
        return "\n".join(L)

    L.append("Found        : %d audio file(s) -> %d take name(s) parsed, %d unmatched, %d ignored"
             % (len(scan.get("audio_files", [])), len(scan.get("parsed", [])),
                len(scan.get("unmatched", [])), len(scan.get("ignored", []))))
    n_used = sum(1 for s in segments if s["b"] > s["a"])
    L.append("Take groups  : %d | pieces used: %d | superseded (lower take): %d"
             % (scan.get("n_groups", 0), n_used, len(scan.get("losers", []))))
    n_queued = scan.get("queued", 0)
    if n_queued:
        L.append("Queued takes : %d started late, right after the previous dub's phrase "
                 "ended (trailing file silence stripped - no truncation, no overlap)"
                 % n_queued)
    if strip_silence or trim_lead:
        n_lead = scan.get("lead_trim_count", 0)
        n_tail = scan.get("tail_strip_smp", 0)
        if n_lead or n_tail:
            msg = ("Silence strip : %d take(s), %s of leading silence removed "
                   "(voice starts at the phrase timestamp)"
                   % (scan.get("strip_count", 0),
                      fmt_seconds(scan.get("lead_trim_smp", 0) / sr)))
            if strip_silence:
                msg += ", %s of trailing silence removed (tail pad kept)" % fmt_seconds(n_tail / sr)
            L.append(msg)
        else:
            L.append("Silence strip : on - no take had silence to remove")
    n_edits = scan.get("edited", 0)
    if n_edits:
        L.append("Timeline edits: %d dub piece(s) placed by the timeline editor "
                 "(original silenced exactly across the edited spans)" % n_edits)

    if segments:
        L.append("")
        L.append("Replaced pieces: %d" % n_used)
        for idx, seg in enumerate(segments, 1):
            a, b = seg["a"], seg["b"]
            if b > a:
                L.append("  #%d  %s -> %s  (%s)  take %-5d  %s"
                         % (idx, fmt_clock(a * 1000.0 / sr), fmt_clock(b * 1000.0 / sr),
                            fmt_seconds((b - a) / sr), seg["info"]["take"], seg["info"]["filename"]))
            else:
                L.append("  #%d  (nothing replaced)  take %-5d  %s"
                         % (idx, seg["info"]["take"], seg["info"]["filename"]))
            for note in seg["notes"]:
                L.append("      - %s" % note)
    else:
        L.append("")
        L.append("Replaced pieces: 0")

    losers = scan.get("losers", [])
    if losers:
        L.append("")
        L.append("Superseded takes (%d):" % len(losers))
        for loser, win in losers[:50]:
            L.append("  %s  (take %d < %d at %s)"
                     % (loser["filename"], loser["take"], win["take"], fmt_clock(win["start_ms"])))
        if len(losers) > 50:
            L.append("  ... and %d more" % (len(losers) - 50))
        if fail_dir:
            L.append("  -> %d superseded file(s) moved to the fail folder: %s"
                     % (fails_moved, fail_dir))
        elif not applied:
            L.append("  -> dry run - superseded files are moved to <takes folder>/fail "
                     "only on a real merge")
        elif not fail_enabled:
            L.append("  -> move_fails is off - superseded files left in place")

    unmatched = scan.get("unmatched", [])
    ignored = scan.get("ignored", [])
    if unmatched:
        L.append("")
        L.append("Files without a valid HH.MM.SS.mmm_TTTTT name (%d):" % len(unmatched))
        for p in unmatched[:10]:
            L.append("  %s" % os.path.basename(p))
        if len(unmatched) > 10:
            L.append("  ... and %d more" % (len(unmatched) - 10))
    if ignored:
        L.append("")
        L.append("Files ignored by the extension filter (%d):" % len(ignored))
        for p in ignored[:10]:
            L.append("  %s" % os.path.basename(p))
        if len(ignored) > 10:
            L.append("  ... and %d more" % (len(ignored) - 10))

    if warnings:
        L.append("")
        L.append("Warnings (%d):" % len(warnings))
        for w in warnings:
            L.append("  - %s" % w)

    replaced_total = sum(s["b"] - s["a"] for s in segments if s["b"] > s["a"])
    L.append("")
    if out_len > track_len:
        L.append("Summary: %d pieces replaced (%s total). Output %s (extended from %s)."
                 % (n_used, fmt_seconds(replaced_total / sr),
                    fmt_clock(out_len * 1000.0 / sr), fmt_clock(track_len * 1000.0 / sr)))
    else:
        L.append("Summary: %d pieces replaced (%s total). Output %s."
                 % (n_used, fmt_seconds(replaced_total / sr), fmt_clock(out_len * 1000.0 / sr)))
    return "\n".join(L)


def merge_pipeline(waveform_np, sr, *, takes_folder, extensions=".flac", recursive=False,
                   timestamp_mode="start", window_mode="until_next", fade_ms=10.0,
                   gain_db=0.0, silence_db=-50.0, tail_pad_ms=200.0, tail_silence_ms=0.0,
                   queue_overlaps=False, overflow="extend",
                   separation="full_mix",
                   isolation_engine="center", demucs_model="htdemucs",
                   voice_low_hz=100.0, voice_high_hz=12000.0,
                   background_waveform=None, background_only=False,
                   trim_lead=False, strip_silence=False, timeline_edits=None,
                   timeline_cache_dir=None, ambient_db=-12.0, move_fails=False,
                   apply=True):
    """One-call merge: plan windows from `takes_folder` and apply to the track.

    waveform_np: float32 [B, C, T] (or [C, T]).  Returns (merged, report, stats).

    separation="voice_isolation" splits the track into voice + background
    first: the dubs are merged into the voice track only and the background
    is re-added unchanged, so it cannot shift under the dubbed parts (dub
    takes should then be voice-only).  The engine picks the splitter:
    isolation_engine="demucs" uses neural separation (raises RuntimeError
    when demucs is missing), "center" uses the ffmpeg mid/side extractor,
    "auto" tries demucs and falls back to centre extraction (with a report
    warning).  background_waveform ([C, T] float32, already resampled/channel-
    matched by the caller) overrides both: voice = track - background,
    sample-exact.  "full_mix" merges into the full mix (legacy behaviour).

    tail_silence_ms (auto_trim only) keeps the original VOICE muted that many
    extra milliseconds after each dubbed phrase ends, covering the leftover
    tail of a longer original phrase (e.g. a residual "...des ka" after a
    shorter dub); the original voice then fades back in across the tail pad.
    Only the voice/merge track is silenced - with voice isolation the
    background is never affected.

    queue_overlaps queues overlapping dubs instead of truncating them: when
    a take's audio (segment clip audible length - trailing file silence is
    stripped - or an explicit filename window's declared span) runs past the
    next take's start time, the next take is shifted to start right after
    the current one's phrase ends.  Full-track passes keep the legacy
    handover at the next timestamp.

    move_fails quarantines superseded takes: after a successful merge
    (apply=True) every losing take file (a lower take number at the same
    timestamp - e.g. 00014 when 00015 exists) is MOVED into a "fail" folder
    inside the takes folder, so the folder keeps only the winner per
    timestamp and never needs manual cleanup.  Idempotent (later runs find
    nothing to move), collision-safe (a regenerated filename gets a _2/_3
    suffix), and the fail folder is pruned from every scan, so quarantined
    files can never re-enter a merge.  Set False to keep the legacy
    behaviour (superseded files stay in place, listed in the report).

    trim_lead pre-processes placed dub clips before the merge: leading
    silence (TTS clips often start with 100-500 ms of baked-in quiet) is cut
    so the voice starts right at the phrase timestamp, with a guard pad of
    max(20 ms, fade_ms) kept ahead of the first audible sample.  Placed
    clips only - full-track passes are read at absolute sample offsets and
    are never shifted; timestamp_mode=end keeps the phrase-end anchor by
    moving the window start up instead.  The silence_db threshold defines
    the silence (measured on the gained take audio).

    strip_silence does the same lead trim AND cuts the baked trailing
    silence of every placed dub clip, down to the tail pad (tail_pad_ms):
    pieces start and end at their actual sound, and the timeline editor's
    piece borders sit at the phrase (the edges can still be dragged back
    into silence).  Same rules as trim_lead (place-layout, non-edited clips
    only).

    ambient_db sets the level (dB, relative to the original) at which the
    replaced stem is mixed under AMBIENT-MARKED dub pieces - pieces the user
    marked with alt+click in the browser timeline editor (the marks ride in
    timeline_edits as per-piece "amb" flags; there is no whole-track switch
    any more).  For a marked piece the merge plays dub + gated non-speech
    bed + replaced stem at ambient_db: the leaked crowd/rain/room tone the
    gate cannot rebuild (its texture is speech-like and gets suppressed)
    and a faint ghost of the original voice continue under the dub, so the
    scene ambience never stops while the dub stays dominant (-12 dB default;
    0 = the original plays under the dub untouched).  Unmarked pieces mute
    the replaced stem completely (legacy).  The gated bed is extracted once
    when at least one piece is marked; outside the marked windows the
    output is untouched.

    background_only (voice_isolation only) discards the separated voice stem
    completely: the dubs are overlaid on silence and the untouched background
    is the only original content mixed underneath them.  Ambience that leaked
    into the voice stem then never mutes under the dubs and never comes back
    between them (ambient fill can only rebuild an approximation of it), so
    nothing ducks - meant for ambience-heavy scenes, at the price that the
    original voice is removed only as completely as the separation removes
    it.  The ambient fill has nothing left to refill in this mode and is
    skipped; with separation="full_mix" the option is ignored with a warning.

    API defaults keep the pure numpy behaviour (full_mix / center,
    queue_overlaps off, no ambient marks, trim_lead off, no timeline edits);
    the ComfyUI node defaults to voice_isolation + auto + queue_overlaps on,
    no timeline edits unless the browser timeline editor wrote adjustments
    or ambient marks into the timeline_edits widget.

    timeline_edits carries adjustments drawn in the browser timeline editor
    (opened with the node's Load Timeline button): each edited dub piece
    becomes an explicit window - the original voice is silenced exactly
    where the piece was drawn and the piece audio plays from the position
    it was dragged to.  With no edits (or without pressing the button) the
    merge runs exactly as without the editor.  timeline_cache_dir, when
    given on a real merge run, receives the editor's data bundle (voice
    stem + dub pieces as mono WAVs + manifest.json) which the button serves
    to the browser.
    """
    wf = np.asarray(waveform_np, dtype=np.float32)
    if wf.ndim == 2:
        wf = wf[None, ...]
    B, C, T = wf.shape
    segments, scan, warnings = prepare_merge(
        T, sr, C, takes_folder, extensions=extensions, recursive=recursive,
        timestamp_mode=timestamp_mode, window_mode=window_mode, silence_db=silence_db,
        tail_pad_ms=tail_pad_ms, tail_silence_ms=tail_silence_ms,
        queue_overlaps=queue_overlaps, overflow=overflow, gain_db=gain_db,
        fade_ms=fade_ms, trim_lead=trim_lead, strip_silence=strip_silence,
        timeline_edits=timeline_edits, ambient_db=ambient_db)
    sep_method = None
    background = None
    voice = None
    marked = [s for s in segments if s.get("amb")]
    if background_only and separation != "voice_isolation":
        warnings.append("background_only needs separation = voice_isolation (full_mix "
                        "has no separated background to merge against) - option ignored")
        background_only = False
    if apply and separation == "voice_isolation":
        if background_waveform is not None:
            bg = np.asarray(background_waveform, dtype=np.float32)
            if bg.ndim == 1:
                bg = bg[None, :]
            if bg.shape[0] != C:
                bg = match_channels(bg, C)
            if bg.shape[1] < T:
                warnings.append("background input is shorter than the track "
                                "(%d vs %d samples) - padded with silence"
                                % (bg.shape[1], T))
                pad = np.zeros((C, T), dtype=np.float32)
                pad[:, :bg.shape[1]] = bg
                bg = pad
            elif bg.shape[1] > T:
                bg = bg[:, :T]
            voice = np.ascontiguousarray(wf[0] - bg)
            background = np.ascontiguousarray(bg)
            sep_method = "external background input (voice = track - background)"
        elif isolation_engine in ("auto", "demucs"):
            try:
                voice, background, sep_method = separate_demucs(
                    wf[0], sr, demucs_model)
            except DemucsUnavailable as exc:
                if isolation_engine == "demucs":
                    raise RuntimeError(
                        "voice isolation engine 'demucs' is not available: %s"
                        % exc) from exc
                warnings.append("voice isolation: %s - fell back to ffmpeg "
                                "centre extraction" % exc)
                voice, background, sep_method = separate_voice_background(
                    wf[0], sr, voice_low_hz, voice_high_hz)
                sep_method += " [demucs unavailable - centre fallback]"
        else:
            voice, background, sep_method = separate_voice_background(
                wf[0], sr, voice_low_hz, voice_high_hz)
        if background_only:
            # The voice stem is discarded completely: the dubs are overlaid on
            # silence and the untouched background is added back below, so
            # ambience that leaked into the voice stem can never duck under
            # the dubs (it is gone for good) and the original voice leaves
            # with the stem.
            voice = None
            merge_src = np.zeros((B, C, T), dtype=np.float32)
        elif B > 1:
            merge_src = np.repeat(voice[None, ...], B, axis=0)
        else:
            merge_src = voice[None, ...]
    else:
        if apply and background_waveform is not None:
            warnings.append("background_audio input is only used with "
                            "separation = voice_isolation - it was ignored")
        merge_src = wf
    if apply:
        fill_wave = None
        fill_stats = {}
        if marked and not background_only:
            # the separated background stem is the ambience reference for the
            # fill's voice suppression (ignored automatically when it does
            # not explain the quiet content, e.g. centred ambience)
            ref = background if separation == "voice_isolation" else None
            fill_wave = extract_ambient_stem(merge_src[0], sr, reference=ref,
                                             stats_out=fill_stats)
        merged, out_len = apply_merge(
            merge_src, sr, segments, fade_ms, fill_wave=fill_wave,
            marked_segments=marked or None,
            raw_wave=merge_src[0] if (marked and not background_only) else None,
            raw_gain=10.0 ** (float(ambient_db) / 20.0))
        if background is not None:
            if background.shape[1] < out_len:
                pad = np.zeros((C, out_len), dtype=np.float32)
                pad[:, : background.shape[1]] = background
                background = pad
            merged += background[None, :, :]
    else:
        fill_stats = {}
        merged, out_len = wf, int(T)
    timeline_manifest = None
    if timeline_cache_dir and apply:
        # Bundle for the browser timeline editor (button in the node UI):
        # the separated voice stem - or the full mix when there is none -
        # plus every winning dub piece, as small mono WAVs + a manifest.
        # background_only discards the voice stem, so the mix stands in.
        try:
            timeline_manifest = write_timeline_cache(
                timeline_cache_dir, sr, T, segments,
                voice if voice is not None else wf[0],
                sep_method or separation,
                edits_raw=str(timeline_edits or ""))
        except Exception as exc:
            warnings.append("timeline editor cache could not be written: %s" % exc)

    # quarantine superseded takes: after a successful real merge, move every
    # lower-take file into <takes folder>/fail so the source folder keeps
    # only the winning take per timestamp.  Never on dry runs (apply=False)
    # and never when move_fails is off; failures to move become warnings.
    moved_fails, fail_dir = [], None
    if apply and move_fails and scan.get("losers"):
        moved_fails, fail_dir, _fw = move_superseded_takes(
            scan["losers"], takes_folder, warnings=warnings)
    report = build_report(scan, segments, warnings, sr, C, T, out_len,
                          fade_ms=fade_ms, gain_db=gain_db, window_mode=window_mode,
                          timestamp_mode=timestamp_mode, overflow=overflow,
                          applied=apply, n_batches=B,
                          separation=separation, voice_low_hz=voice_low_hz,
                          voice_high_hz=voice_high_hz, sep_method=sep_method,
                          tail_pad_ms=tail_pad_ms, tail_silence_ms=tail_silence_ms,
                          strip_silence=strip_silence,
                          queue_overlaps=queue_overlaps,
                          ambient_pieces=len(marked),
                          ambient_db=float(ambient_db),
                          ambient_fill_ref=bool(fill_stats.get("reference_used")),
                          background_only=bool(background_only),
                          trim_lead=bool(trim_lead),
                          fails_moved=len(moved_fails),
                          fail_dir=fail_dir,
                          fail_enabled=bool(move_fails))
    n_used = sum(1 for s in segments if s["b"] > s["a"])
    stats = {
        "pieces": n_used,
        "superseded": len(scan.get("losers", [])),
        "queued": scan.get("queued", 0),
        "out_len": out_len,
        "track_len": T,
        "replaced_seconds": sum((s["b"] - s["a"]) / sr for s in segments if s["b"] > s["a"]),
        "separation": separation,
        "sep_method": sep_method,
        "ambient_fill": bool(marked and apply and not background_only),
        "ambient_pieces": len(marked),
        "ambient_db": float(ambient_db),
        "ambient_fill_ref": bool(fill_stats.get("reference_used")),
        "fails_moved": len(moved_fails),
        "fail_dir": fail_dir,
        "background_only": bool(background_only and apply),
        "lead_trimmed": int(scan.get("lead_trim_count", 0)),
        "tail_stripped_ms": int(round(scan.get("tail_strip_smp", 0) * 1000.0 / sr)),
        "timeline_edited": int(scan.get("edited", 0)),
    }
    return merged, report, stats

