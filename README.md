# ComfyUI Dub Merge

Merge **translated take files** into a **full track** (movie audio, podcast,
music bed...) directly inside ComfyUI.  With the default **voice isolation**
the original is first split into **voice + background** (Demucs ML
separation when installed, ffmpeg centre extraction otherwise, or your own
background track); the winning translated take replaces only the **voice**
at each take's timestamp and the untouched **background** is added back
afterwards - so the background never shifts, ducks or restarts under the
dubbed parts.  For ambience-heavy scenes, `background_only` goes one step
further: the separated voice stem is discarded entirely and the dubs are
merged with the background alone, so ambience that leaked into it can never
duck under the dubs.

Take files follow this naming convention:

```
HH.MM.SS.mmm_TTTTT.flac
00.01.52.612_00001.flac
\_________ __________/\__/
     hour.min.sec.ms    take number
```

- The timestamp marks where the dubbed phrase **starts** in the full track
  (switchable to *end* via a widget).
- `_TTTTT` is the take number.  When several takes exist for one timestamp,
  the **highest take number wins** ("last good take"); lower takes are
  superseded and listed in the report - and moved into a **`fail` folder**
  inside the takes folder on every real merge (`move_fails`, on by
  default), so only the winning take per timestamp remains and no manual
  cleanup is needed.  See *Superseded takes are quarantined automatically*
  below.
- **Whole-track takes** (files about as long as the track) are treated as
  **full passes**: the translated phrase is recorded at the timestamped
  position and the take audio is read at the same sample positions as the
  original.  **Everything shorter than the track is a segment clip** (the
  typical TTS dub file) and is **placed at its timestamp** - its phrase
  plays exactly at the timestamp, regardless of baked-in lead silence
  (which is trimmed).  The merge keeps every clip's phrase intact: clips
  that would run into the next dub are queued one after the other (see
  *Queueing overlapping dubs* below).
  In voice-isolation mode the takes should contain **voice only** (no music
  or ambience) - see *Voice isolation* below.
- Explicit windows are also understood automatically:
  `00.01.52.612-00.01.58.300_00002.flac` replaces exactly that span.

### Superseded takes are quarantined automatically (`move_fails`, on by default)

Every merge run re-selects the winner per timestamp (the highest take
number) and - on a real merge, not a `report_only` dry run - **moves every
losing file into `<takes folder>/fail`**, keeping the original filename.
The result is a takes folder that contains exactly one take per timestamp:
the one the merge actually used.  Nothing is deleted; "fail" only means
"not selected" (a bad take your TTS pipeline superseded, or an older
attempt).

Behaviour in detail:

- **Idempotent** - a second run finds no losers and moves nothing; the
  folder is stable from the first merge on.
- **Self-healing** - when your TTS pipeline later drops a higher take
  (e.g. `_00016` next to the `_00015` that won last time), the next merge
  switches to `_00016` and the previous winner moves into `fail/` exactly
  like any other loser.
- **Never overwrites** - if `fail/` already contains a file with the same
  name (pipeline regenerated identical names), the moved file gets a
  `_2`/`_3` suffix instead of clobbering it.
- **Quarantined files stay out of every later scan** - the `fail` folder
  directly inside the takes folder is pruned from scanning (recursive mode
  included), so a parked file can never sneak back into a merge, however
  take-like its name still is.
- **Failures are warnings, not errors** - a file that is missing or locked
  is skipped with a report warning; the merge itself is never affected.
- Turn `move_fails` **off** to keep superseded files in place (the report
  lists them as before); the stats carry `fails_moved` and the `fail` path
  either way.

## Nodes

| Node | What it does |
|------|--------------|
| **Load Media (Video/Audio)** | Loads the full track from a video (`mp4 mkv webm mov avi flv ts mpg wmv m4v...`), an audio file (`wav flac mp3 ogg opus m4a aac...`) or an `http(s)://` URL. ffmpeg handles everything soundfile cannot. |
| **Merge Translated Takes** | Scans a folder for take files, picks the highest take per timestamp, splits voice from background (Demucs ML / ffmpeg centre / your own background track), merges the dubs into the voice track and re-adds the background. Outputs the **merged AUDIO** and a **text report** (how many pieces were replaced and at what timestamps, superseded takes, warnings). |

Works with any AUDIO source as input - native `LoadAudio`, VHS `LoadVideo`,
or the included **Load Media** node.

## Saving and previewing the results

This pack ships no save node - use ComfyUI's core nodes:

- **Save Audio** (core) - saves the merged audio as FLAC; `Save Audio (MP3)`
  and `Save Audio (Opus)` for compressed formats; **Preview Audio** for
  listen-only. The saved/previewed audio shows up with a player in the node.
- **Save Text** (core) - writes the report string to a `.txt` file (also
  `.csv` / `.md` / `.json`). To just read the report inline, use the native
  **Preview as Text** node instead - the report is also printed to the
  ComfyUI console either way.

The bundled example workflow is wired exactly this way:
`audio` -> **Save Audio**, `report` -> **Save Text**.

## Installation

1. Copy / clone this folder into `ComfyUI/custom_nodes/comfyui-dub-merge`.
2. `pip install -r requirements.txt` (numpy + soundfile; torch/torchaudio come
   with ComfyUI).
3. **ffmpeg** must be available on PATH for video sources and m4a/aac
   decode.  On Windows you can instead copy `ffmpeg.exe` + `ffprobe.exe`
   into this pack's `bin/` folder - it is added to PATH automatically.
4. Restart ComfyUI.  The nodes appear under **audio > dubbing**.

An example graph is in `examples/` (`workflow_ui.json` for drag-and-drop,
`workflow_api.json` for API use).

## Usage

1. **Load Media (Video/Audio)** - point `media_path` at the full movie/audio
   (absolute path, or relative to ComfyUI's `input/` folder, or a URL).
2. **Merge Translated Takes** - connect the audio, set `takes_folder` to the
   folder with your translated `HH.MM.SS.mmm_TTTTT.flac` files.
3. **Save / preview the result** - connect the merge node's `audio` output to
   the core **Save Audio** node (or **Preview Audio** to just listen), and the
   `report` output to the core **Save Text** node (or a string-preview node).

## Voice isolation (default)

Replacing the full mix under a dub makes the background shift slightly at
every window edge - even a sample-accurate mute/unmute of the whole mix is
audible because music and ambience restart.  Voice isolation removes that:

```
original track ──> voice track  (+ dubs merged here)
      │                              │
      └──> background track ─────────┴──> final mix  (background constant)
```

1. **Separate** the original into a voice track and a background track.
2. **Merge the dubs into the voice track** - all window modes, crossfades
   and the auto_trim tail work exactly as before, but now they hand over
   from dub to *original voice* instead of dub to *full mix*.
3. **Re-add the background unchanged** - it cannot shift, dip or restart
   under the dubbed parts, no matter what happens inside the voice track.

Because the dub replaces the voice only, **use voice-only dub takes**
(record the translation without music/ambience).

### Separation engines (`isolation_engine`)

| Engine | How the split is made | Best for |
|--------|------------------------|----------|
| `auto` (default) | Demucs ML separation when the `demucs` package is installed, otherwise ffmpeg centre extraction (a note appears in the report when it falls back). | Everyone. |
| `demucs` | Neural source separation (`htdemucs`): the **vocals stem** is the voice, the sum of the other stems is the background and is passed through untouched. Requires `pip install demucs` (model weights are downloaded on first use, ~80-160 MB). | **Mono-like mixes** - movie/TV mixes where the music and ambience are also centred. Centre extraction would mute the whole centred mix under the dubs; Demucs does not care about panning. |
| `center` | ffmpeg mid extraction (`(L+R)/2`, band-limited to `voice_low_hz` - `voice_high_hz`, zero-phase). The background is the exact complement `original - voice`, so outside the dubbed windows the output reconstructs the input **sample-exactly**. | Mixes whose background is stereo-wide (hard-panned or wide ambience) and where the voice is dead-centre. Fast, no extra deps. |

**Why the engine matters:** if the whole original (background included)
seems to duck under the dubs, your mix is mono-like - everything, ambience
included, sits in the centre, so the centre extractor folds it all into the
"voice" track and mutes it. Switch to `demucs` (or feed an external
background) and only the voice is replaced while the background keeps
running at constant level.

**External background input:** for maximum control, connect any
no-vocals/background track (from UVR, the demucs CLI, a DAW export...) to
the optional `background_audio` input.  It is used as-is - `voice =
original - background` is computed sample-exactly, so the background can
never duck, and you can even clean it up first.  It overrides the
`isolation_engine` widget and is auto-resampled/channel-matched.

Notes: with the demucs engine the original voice is *rebuilt* by the model,
so outside the dubbed windows the output is the model's reconstruction
(transparent for practical purposes, not sample-exact).  With the centre
engine, background elements that share the voice band and are dead-centre
used to be the only content ducked under the dubs - the ambient fill (below)
now keeps even that content running under the dub pieces you mark.  Set
`separation = full_mix` to
turn isolation off entirely (dubs replace the entire mix - legacy
behaviour).

### Ambient fill (per piece, marked in the timeline editor)

Even with a perfect separation, some continuous background always ends up
**inside the stem the dubs replace**: rain, room tone and centred music with
the `center` engine, a little ambience bleeding into the vocals stem with
`demucs`.  That content was muted under every dubbed window and came back
between windows - so rain *pumped* up and down with every phrase ("the
background is constantly ducking").

The fill is **per dub piece, not a global switch**: open the timeline editor
(`Load Timeline`), **alt+click a dub piece** to mark it ambient (green tint
and an `amb` badge; the silenced span on the voice track turns green with an
`amb ±N dB` label), press **Apply to node** and run the merge.  Under a
marked piece the merge plays

- the **gated non-speech bed** - the non-speech part of the replaced stem,
  extracted with an **adaptive spectral noise-floor gate**, keeps playing
  under the piece's window, so the rain/ambience/music bed stays at a
  constant level while only the original speech is removed (at the window
  edges the fill crossfades into the original's own ambient content with
  complementary envelopes - no level step); and
- the **original span itself at the `ambient_db` level** (default -12 dB) -
  your "original + dub at different levels" blend: the leaked crowd
  chatter / scene texture the gate cannot rebuild (it is speech-like and
  gets suppressed) and a faint ghost of the original voice continue under
  the dub, which stays dominant.  `ambient_db = 0` mixes the original under
  the dub untouched (best for pure-ambience spans), -30 dB is barely
  audible.

Unmarked pieces mute the replaced stem completely (legacy behaviour), so
you fix exactly the scenes where the leak matters - a crowd/distant
ambience that leaked into the voice track, a rain scene - without running
the fill (and its faint voice ghost) under dialogue dubs that do not need
it.  Outside the marked windows the output is untouched, exactly as
before.  The preview in the timeline editor plays the original at
`ambient_db` under marked pieces, so you can hear the blend before running
the merge (the editor cannot rebuild the gated bed - the merge adds it on
top).

Two things keep the bed free of voice remnants (the fill would otherwise
*echo* the original voice under the dubs):

- **Adaptive noise floor** - the gate's floor follows the *recurring quiet
  level* of every time/frequency bin (low-percentile sub-block profiles,
  neighbourhood minimum over a few seconds, bounded by the track's global
  quiet level) instead of one static global value.  A static floor is
  inflated by speech on dialogue-dense tracks - the "quietest frames" still
  contain speech - and lets a large part of the original voice through.
- **Background-referenced suppression** (voice isolation only) - a level
  gate can never catch voice that sits at or below the ambience level
  (fricatives, breath, soft phrase endings).  The separated background stem
  knows where the ambience IS, so the gate applies a second per-bin keep
  mask: bins far above the background's level are voice the separator put
  into this stem and are suppressed, bins explained by the background (the
  ambience itself) are kept.  The reference is skipped automatically when
  it does not explain the quiet content (e.g. perfectly centred ambience
  with the `center` engine, where the side signal is empty).  Measured on
  dialogue-dense material: speech leakage in the fill drops from roughly
  -19 dB to roughly -28 dB relative to the speech.  For the best result
  install `demucs` (the `auto` engine uses it when available).
- **Sustained-ambience keep (loud slow events stay at full level)** - the
  level gate has ~+20 dB of headroom over the recurring quiet level, but a
  train passing by, a rain swell or a distant crowd can sit 25-35 dB above
  it - and neural separators push a loud event *into* the vocals stem as it
  gets louder - so a plain level gate mutes exactly those events under the
  dubs (measured up to -50 dB).  The gate therefore also recognises
  ambience by its TIME STRUCTURE and SPECTRUM, not its level: per-bin
  coarse (~2 s) levels that stay within a few dB of their recent typical or
  recent peak level over the last ~11 s are sustained (slowly varying),
  while speech spikes 6-20 dB over its own baseline syllable by syllable;
  and speech is a harmonic comb - tooth bins stand far above the median of
  their spectral neighbourhood, while every ambience (rain, train rolloff,
  crowd, even fricatives) sits within ~2.2x of it.  Bins passing both tests
  are kept at full level regardless of loudness; the voice-suppression path
  is unchanged, so no extra voice leaks in.  Measured: a +24 dB train
  pass-by ducking -50 dB under the dubs now holds within ~1 dB, a distant
  crowd within ~0 dB, dense narration is still suppressed.

- Works with every engine (`center`, `demucs`, external `background_audio`)
  and with both `until_next` and `auto_trim` windows, and in `full_mix`
  mode too (the bed is then extracted from the full mix).
- Marks travel in the `timeline_edits` widget with the geometry edits
  (`"amb": true`, also as a mark-only entry without geometry), so they
  survive save/load and apply on every later merge run.  The report lists
  the marked pieces on the `Ambient fill :` line (with a
  `background-referenced` note when the background stem guided the
  suppression); with no marks it says so and the merge behaves exactly
  like the legacy mute.

### Background-only mode (`background_only`, off by default)

The ambient fill can only *rebuild* the ambience that leaked into the
replaced stem, and the rebuild is never perfect - on ambience-heavy scenes
(rain, wind, crowds, traffic) the subtle leak still ducks a little under
every dubbed phrase and comes back between them, which reads as pumping and
sounds distracting and unprofessional.

`background_only` removes the problem at the source: after the voice /
background split, **the separated voice stem is discarded entirely** - the
dubs are overlaid on silence and the **untouched background stem is the only
original content in the output**.  The leaked ambience is gone for good, so
nothing can duck, and the background plays at a constant level from the
first to the last sample.

The trade-off: the original voice is now removed only as completely as the
separation removes it - whatever the separator leaves in the background stem
(a faint ghost of loud lines with the `center` engine on mono-like mixes)
stays audible.  For the cleanest split pair `background_only` with the
`demucs` engine or a `background_audio` input.  Use it for ambience-heavy
scenes; keep it off when the dialogue must vanish completely.

In this mode the ambient fill has nothing left to refill (the background is
never muted) and is skipped; the report shows `Background only: ON` and
says so (pieces marked ambient are listed as ignored there).  The option
needs `separation = voice_isolation`
- with `full_mix` it is ignored and a warning is added to the report.

### How windows are built

- `timestamp_mode = start` (default): the window for take *i* runs from its
  timestamp until the **next take's timestamp** (last take: until track end).
  This matches full-track dub passes: the take keeps playing (music, ambience)
  until the next translated phrase takes over.  Adjacent takes are
  **crossfaded into each other** across the shared boundary (equal-gain,
  `fade_ms`), so the original never clicks back in between.
- **Queueing overlapping dubs (`queue_overlaps`, on by default)**: in fast,
  explosive dialogue a dub's file often runs past the next take's timestamp -
  without queueing it would be **cut off mid-phrase** and the two dubs would
  overlap.  With queueing, the next take is shifted to start **right after
  the current one's audible phrase ends** (its window moves with it), so
  every phrase plays in full, one after the other, and the original voice
  stays muted in between.  Segment-clip takes queue by their **audible
  length**: trailing file silence is **auto-stripped** from the handover, so
  a TTS tail baked into the dub file (often ~0.5 s) cannot push the following
  phrase rightward - queued dialogue stays in sync instead of drifting later
  with every line.  Explicit `start-end` filename windows queue by their
  **declared span**; full-track passes keep the legacy handover at the next
  timestamp.  Shifts are listed in the report ("queued after ... start
  shifted ...", plus a "trailing file silence stripped" note).  Turn the
  widget off to restore the old clipping behaviour.
- `window_mode = auto_trim`: inside that span, the actual dubbed phrase is
  detected with the `silence_db` threshold and only the phrase (+ tail pads)
  is replaced - use this when takes are **silent outside the phrase** and you
  want the original audio preserved between phrases.  When the phrase ends,
  the original voice **crossfades back in across the tail pad** (equal-power
  sqrt curve over 200 ms by default, so the original is audible early in the
  fade and there is no silent gap or dip between the dubbed phrase and the
  original audio).  The replaced part applies to the **voice track only** -
  the background keeps running untouched.  Works for full-track passes and
  short segment clips alike (short clips are trimmed on their own samples).
- `timestamp_mode = end`: the timestamp is the phrase **end**; the window is
  `end - take duration` (for segment-length takes).
- **Timeline editor**: the node has a **Load Timeline** button - press it
  (after running the merge once) and a scrollable two-track timeline opens
  where you can drag and resize every dub piece and draw exactly where the
  original is silenced, before committing the timings to the merge.  See
  the *Timeline editor* section below.
- **Baked silence is stripped from every dub clip** (`strip_silence`, on by
  default): TTS files often carry 100-500 ms of quiet at both ends - the
  lead makes the dub start late, the tail inflates the piece.  Each placed
  clip is trimmed to its actual phrase at the `silence_db` threshold (a
  small guard pad before the first word and the `tail_pad_ms` pad after the
  last one are kept for the crossfades), so the dub starts right at its
  timestamp and the timeline editor's pieces sit at the sound.
- Sample rates are matched automatically (takes are resampled to the track
  rate); mono/stereo/channel differences are handled by duplicate/downmix.

## Timeline editor (visual sync)

The automatic timings are derived from the filenames; when a dub needs to be
nudged - TTS lead silence pushing the phrase late, an original phrase whose
tail must stay muted after a shorter dub, a line that should start a moment
earlier - open the timeline and set it by eye instead of guessing:

1. **Run the merge once** (mode = `merge`).  Every real run writes a data
   bundle into ComfyUI's temp dir: the separated original voice stem (or the
   full mix when there is none) plus every winning dub piece, downsampled to
   mono 22.05 kHz WAVs, one per node instance.
2. Press **Load Timeline** on the node.  A window opens with the **original
   voice on top** (as-is, not editable) and the **dub pieces at their
   timestamps below**, both over a ruler marked in **seconds** (scroll
   horizontally, ctrl+wheel or the +/− buttons to zoom, Fit for the whole
   track).  While the window is open, **Space toggles play/stop**
   (click the ruler or the waveforms first to set the playhead; Escape or
   ✕ closes the window and releases the key).
3. Edit the pieces:
   - **pieces are placed exactly as the merge plays them** - the bundle
     carries each piece's real window and audio position (after
     `queue_overlaps`), so dubs the merge queued already appear queued and
     the original-silence overlay is exact.  **Whole-track takes** (files
     about as long as the track - the merge reads them at absolute offsets)
     show as **"(pass)"** and are read-only in the editor; every shorter
     take is a segment clip and is draggable.  After updating the pack, run
     the merge once so the bundle carries the exact placement (older
     bundles fall back to clip semantics);
   - **drag a piece** to retime the dub - it moves as a whole;
   - **drag its left edge** to add or consume **front silence** (the piece
     audio stays put; only the replaced span grows leftward);
   - **drag its right edge** to add or consume **end silence** - the
     original voice stays muted after the dub ends, covering the leftover
     tail of a longer original phrase ("...des ka" after a shorter line);
   - **alt+click a piece to mark it ambient** - a green tint and an `amb`
     badge appear on the piece, and its span on the voice track turns green
     with an `amb ±N dB` label: instead of muting the original under that
     piece, the merge keeps the leaked crowd / rain / room tone running
     there (gated bed + the original span at the node's `ambient_db`
     level).  Use it for dubs over ambience-heavy spans - a distant crowd
     or rain scene that leaked into the voice track - where the dub
     otherwise sounds "on and off" against the background.  Click again to
     unmark; the mark survives Apply/save/load and is written as
     `"amb": true` (mark-only, without pinning the piece's geometry).
     Full passes cannot be marked (they ARE the original audio).
   - **edges drag with silence in both directions, all the way to the sound
     borders**: pulling an edge out adds front/end silence; pulling it back
     in consumes that silence and then cuts the file's own baked lead/tail
     silence, stopping exactly at the sound (the spoken audio itself is
     never cut).  Every position is reachable in both directions - extending
     a piece and then dragging the edge back restores the original border.
     The merge renders a cut piece exactly as drawn (a cut head plays
     nothing and the sound keeps its drawn position), and the preview plays
     the piece inside its window only - the timeline always sounds like the
     merge will.
   - double-click a piece to reset it to its timestamp (the ambient mark is
     kept - it is a separate toggle; the Reset button clears it together
     with everything else).
   - **zooming (ctrl+wheel or the +/− buttons) keeps the ruler and both
     tracks locked together**: the view stays anchored under the cursor and
     the rows cannot drift apart, no matter how often you zoom in and back
     out (rows are pinned explicitly and the scroll container ignores
     browser scroll-anchoring).
   - **ctrl+wheel is owned by the dialog**: the browser page zoom never
     fires while the editor is open (even over the header or the dark
     backdrop), so an accidental zoom cannot push the Apply button, the
     checkboxes or the original track offscreen; the dialog is also capped
     to the viewport and its track area scrolls internally on short
     windows instead of clipping the controls.  The gesture is taken at
     the window capture level, so no other script or extension can
     intercept it (ctrl+scroll-down always zooms out).
   - **overlapping dubs queue one after the other** - the editor mirrors the
     node's `queue_overlaps` option: a piece that would sit on top of the
     previous one is placed right after it instead (after the previous
     piece's audible phrase - trailing file silence is stripped - or after
     its whole drawn span when the previous piece carries an edit).  This
     applies at load time and after every drag: drop a piece onto another
     and it snaps behind it, while the pieces it would overlap light up red
     while you drag; a piece the merge itself queued shows a dashed border
     and a `⇥` marker.  Queue positions are re-derived by the merge from
     the timestamps, so the merge reproduces exactly what you see.  Full
     passes never queue (the
     engine hands them over at the next timestamp).  The editor reads the
     node's current `queue_overlaps` and `silence_db` widgets (the silence
     threshold for the audible-end handover); re-open the editor after
     changing them.
   The red overlay on the voice track shows **exactly where the original
   will be silenced** - it follows your edits live; spans under pieces
   marked ambient are green instead (the original continues there at the
   `ambient_db` level, mirrored by the preview).
4. Press **▶ Play** to hear the original voice and the dubs **together** at
   the current timings.  `silence voice under dubs` (default) previews the
   merge result - the voice plays only between the pieces' spans; turn it
   off to hear the raw overlap for sync comparison.  `mute voice` /
   `mute dubs` solo either side.
5. Press **Apply to node**: the drawn adjustments are written into the
   node's `timeline_edits` widget and saved with the workflow.  **Re-run the
   merge** to render them.  Every edited piece becomes an explicit window:
   the original is silenced exactly across the drawn span and the piece
   audio plays from the position it was dragged to; the report lists the
   edits ("Timeline edits: N dub piece(s)...").

### Saved edits survive re-opening

The editor **restores your saved adjustments every time it opens**: it
reads the node's `timeline_edits` widget and, when the widget comes up
empty, falls back to the copy the last merge run embedded in its bundle
(`manifest.json` carries the value it consumed).  Every JSON shape the
node accepts is understood - the editor's `{"version": 1, "edits": [...]}`,
a bare list, or an `{"<piece id>": {...}}` map - including JSON you edited
by hand.  Anything that cannot be applied (unparseable JSON, piece ids
that no longer match because the takes changed) is reported in the
editor's status line instead of being silently ignored, and the widget
value is always kept.

Pressing **Apply** after a plain re-open re-saves the restored edits
unchanged - it can no longer wipe them.  **Reset** (or double-clicking the
pieces) is what clears edits.  If the status line says the edits came
"from the last merge run (the widget was empty)", press Apply once to
write them back into the widget before saving the workflow.

If the button is never pressed - or nothing is applied - the node merges
exactly as without the editor.  The button needs a previous merge run: it
serves the last written bundle for that node instance.  Notes: full-track
passes are read at absolute sample offsets and cannot be re-placed by the
editor (edits on them are ignored with a report note); with
`timestamp_mode=end` a drawn span simply replaces the derived window; the
bundle is a preview/downsample (mono 22.05 kHz), the merge itself always
uses the full-quality take files.

> Migration: the old `tail_silence_ms` (global post-phrase hold) and
> `trim_lead` (automatic front-silence trim) widgets were **removed** from
> the node in favour of the editor - end silence is now drawn per piece
> (step 3, right edge) and lead silence is fixed by dragging the piece left
> until the onset sits on the beat.  Both remain available programmatically:
> `dub_core.merge_pipeline(..., tail_silence_ms=500)` /
> `merge_pipeline(..., trim_lead=True)` still work for API users.
> Workflows saved before this change carry a stale value (`True` / `500.0`)
> in the `timeline_edits` slot after ComfyUI's positional widget mapping;
> the node reports "timeline_edits ignored ... stale value" and merges
> exactly as before instead of erroring - set the widget to empty (or
> re-create the node) to silence the warning.

## Overflow (take content past the track end)

| `overflow` | Behaviour |
|------------|-----------|
| `extend` (default) | The output is padded so the take's tail is fully included (guarded against trailing silence). |
| `truncate` | Cut at track end; the loss is noted in the report. |
| `skip` | Leave that take out entirely; flagged in the report. |

### Options reference (Merge Translated Takes)

| Widget | Default | Meaning |
|--------|---------|---------|
| `takes_folder` | - | Folder with take files (absolute, or relative to `input/`). |
| `extensions` | `.flac` | Which extensions count as takes; e.g. `.flac .wav`. |
| `recursive` | off | Also scan subfolders. |
| `timestamp_mode` | `start` | Timestamp = phrase start or phrase end. |
| `window_mode` | `until_next` | `until_next` for full-track passes, `auto_trim` for silent-bed takes. |
| `separation` | `voice_isolation` | `voice_isolation` = dubs replace only the voice, background re-added unchanged (use voice-only dub takes). `full_mix` = dubs replace the entire mix (legacy). |
| `isolation_engine` | `auto` | How the voice/background split is made: `auto` = Demucs ML when installed else centre extraction, `demucs` = neural separation (requires `pip install demucs`), `center` = ffmpeg mid/side. Overridden by the `background_audio` input. |
| `demucs_model` | `htdemucs` | Model for the demucs engine: `htdemucs` (general), `htdemucs_ft` (better, slower), `mdx_extra_q` (alternative, often strong on speech). |
| `voice_low_hz` | 100 | `center` engine only: low edge of the voice band; content below (bass, rumble) always stays in the background. 0 = no low cut. |
| `voice_high_hz` | 12000 | `center` engine only: high edge of the voice band; content above (cymbals, air) always stays in the background. Raise to ~sample_rate/2 to disable. |
| `fade_ms` | 10 | Equal-gain crossfade at joins and across the `auto_trim` tail (0 = hard cut). |
| `gain_db` | 0 | **Volume of the dubbed pieces**, adjustable from -24 to +24 dB in 0.1 steps before merging. Applied to the take audio only, right after loading - the original voice and the background are untouched, so a quiet dub can be brought up (or an over-loud TTS line brought down) without affecting anything else. The report's Settings line shows the value used (`gain=+3.5 dB`). |
| `silence_db` | -50 | Silence threshold (level below this counts as silence): `auto_trim` uses it to detect the dubbed phrase inside the window. Measured on the take audio after `gain_db`. |
| `tail_pad_ms` | 200 | Padding after the last audible sample (`auto_trim` / `extend`); in `auto_trim` the original voice crossfades back in across this tail (equal-power). Raise for a longer blend, lower for a tighter splice. |
| `move_fails` | on | After a **successful merge**, every **superseded take file** - a lower take number at the same timestamp (e.g. `00.36.34.877_00014.flac` when `_00015` exists; the highest take number wins) - is **moved into a `fail` folder** inside the takes folder, so only the winning take per timestamp remains and no manual cleanup is needed.  When the TTS pipeline later regenerates a higher take, the previous winner is demoted into `fail/` on the next run the same way.  Idempotent; never runs on dry runs (`mode = report_only`); never overwrites (a regenerated filename gets a `_2`/`_3` suffix); the `fail` folder itself is pruned from every scan, so quarantined files can never re-enter a merge.  Off = superseded files stay in place, listed in the report (legacy). |
| `strip_silence` | on | **Strips the baked-in leading AND trailing silence** from every placed dub clip at the `silence_db` threshold: the dub starts right at its timestamp instead of late, and the timeline editor's pieces sit at the sound. A guard pad of max(20 ms, `fade_ms`) is kept before the first word and the `tail_pad_ms` pad after the last one for the crossfades; the editor's edge drags can still cut those pads all the way to the sound borders, or drag the edges back out into silence. Full-track passes and pieces placed by the timeline editor are never touched. Off = the files play exactly as recorded (legacy). |
| `queue_overlaps` | on | Queue overlapping dubs: when a take's audio runs past the next take's start time, the next take starts right after the current one's **audible phrase ends** (trailing file silence is auto-stripped, so TTS tails cannot desync the queued dialogue) instead of truncating it. Off = legacy behaviour (earlier take clipped at the next start). |
| `ambient_db` | -12.0 | Level (dB, relative to the original) at which the original span plays under dub pieces **marked ambient** (alt+click in the timeline editor, then Apply).  A marked piece plays dub + gated non-speech bed (rain, ambience, music bed - extracted with an adaptive spectral noise-floor gate, background-stem-referenced in voice-isolation mode; loud slow ambience events like a train, a rain swell or a distant crowd are recognised and kept at full level) **+ the original span at this level**, so leaked crowd/rain/room tone that the gate cannot rebuild continues under the dub while it stays dominant (the original voice ghosts in at this level).  0 = the original plays under the dub untouched (pure-ambience spans), -30 = barely audible.  Unmarked pieces mute the replaced stem completely (legacy). |
| `background_only` | off | Ambience-heavy scenes: after the split, **discard the separated voice stem** and merge the dubs with the **background alone**. The subtle ambience that leaks into the voice stem normally ducks under every dub (the ambient fill can only rebuild an approximation of it); here the leak is gone for good and the background plays at a constant level from start to finish. The original voice is removed only as completely as the separation manages - pair with the `demucs` engine or a `background_audio` input for the cleanest split. Needs `separation = voice_isolation` (ignored with `full_mix`); skips the ambient fill. |
| `timeline_edits` | (empty) | Adjustments from the **timeline editor** (JSON, written by its Apply button - do not edit by hand). Empty = no adjustments: the merge runs exactly as without the editor. Each edit records where the original voice is silenced (`start_ms` - `end_ms`) and where the dub piece's audio plays (`audio_ms`) for one piece, as drawn on the two-track timeline - plus `"amb": true` for pieces marked ambient, which may also appear as a mark-only entry without any geometry. |
| `overflow` | `extend` | extend / truncate / skip past the track end. |
| `mode` | `merge` | `report_only` = dry run: original audio + report only. |
| `background_audio` (input) | - | Optional AUDIO input: your own background track (UVR / demucs CLI / DAW export). Used as-is (`voice = original - background`, sample-exact), auto-resampled and channel-matched; only used with `separation = voice_isolation`. |

Engine-only options (programmatic `dub_core.merge_pipeline` use, no widget):
`tail_silence_ms` (post-phrase voice hold - its UI replacement is drawing end
silence on a piece in the timeline editor) and `trim_lead` (automatic
front-silence trim - its UI replacement is dragging the piece so the onset
sits on the timestamp).

## Report example

```
==== Dub Merge Report ====
Track        : 2 ch, 48000 Hz, 00:00:10.000 (480000 samples)
Takes folder : /data/episode12/takes  [recursive: off, extensions: .flac]
Settings     : timestamp=start, window=auto_trim, fade=10 ms, tail_pad=200 ms, tail_silence=0 ms, gain=+0 dB, overflow=extend, queue=on, lead_trim=off, mode=merge
Voice isolation: ON - dubs replace only the voice; the background is separated first and re-added unchanged
                 separation: Demucs htdemucs (vocals stem vs sum of the other stems), resampled 48000 <-> 44100 Hz, CPU
Ambient fill : 2 dub piece(s) marked ambient (alt+click in the timeline editor) - the gated non-speech bed keeps playing under their windows and the original span is mixed under the dub at -12.0 dB, so leaked crowd/rain/room tone continues while the dub stays dominant (the original voice ghosts in at that level) (background-referenced voice suppression)
Found        : 6 audio file(s) -> 5 take name(s) parsed, 1 unmatched, 1 ignored
Take groups  : 3 | pieces used: 3 | superseded (lower take): 2
Queued takes : 1 started late, right after the previous dub's phrase ended (trailing file silence stripped - no truncation, no overlap)
Timeline edits: 1 dub piece(s) placed by the timeline editor (original silenced exactly across the edited spans)

Replaced pieces: 3
  #1  00:00:01.500 -> 00:00:03.000  (1.500s)  take 2     00.00.01.500_00002.flac
      - timeline edit: original silenced 00:00:01.500 - 00:00:03.400, piece audio plays from 00:00:01.500
  #2  00:00:03.000 -> 00:00:04.700  (1.700s)  take 7     00.00.03.000_00007.flac
      - trailing file silence 0.300s stripped: the next dub queues right after the phrase, not after the file
  #3  00:00:04.700 -> 00:00:07.800  (3.100s)  take 1     00.00.04.500_00001.flac
      - overlapped 00.00.03.000_00007.flac - queued after it: start shifted 00:00:04.500 -> 00:00:04.700 (+0.200s)
      - auto-trimmed window from 5.300s to 3.100s; original fades back in across the tail pad

Superseded takes (2):
  00.00.01.500_00001.flac  (take 1 < 2 at 00:00:01.500)
  00.00.03.000_00003.flac  (take 3 < 7 at 00:00:03.000)
  -> 2 superseded file(s) moved to the fail folder: /input/dubs/fail

Files without a valid HH.MM.SS.mmm_TTTTT name (1):
  badname.flac

Summary: 3 pieces replaced (6.300s total). Output 00:00:10.000.
```

## Troubleshooting

- **"ffmpeg was not found in PATH"** - install ffmpeg (Windows: `winget install
  ffmpeg`) or copy `ffmpeg.exe`/`ffprobe.exe` into the pack's `bin/` folder.
- **"timeline_edits ignored ... stale value" warning** - your workflow was
  saved with the older widget set (before the timeline editor): after the
  widget list changed, ComfyUI's positional mapping drops the old
  `tail_silence_ms` / `trim_lead` values into the `timeline_edits` slot.
  The node merges without adjustments (never a hard error) and says so in
  the report.  Fix: clear the `timeline_edits` widget (or delete and re-add
  the Merge Translated Takes node) and re-run.  The editor's Apply button
  always writes valid JSON; a hand-broken value is likewise warned about
  and skipped, per entry.
- **Play button / Space dead after marking a piece ambient (older pack
  version)** - a scheduling bug sent Web Audio a negative duration once a
  green (ambient) span lay behind the playhead, and every Play / Space
  press silently re-threw.  Fixed: spans before the playhead are skipped,
  and any playback error now resets the play state and shows
  "Playback failed: ..." in the status line instead of bricking the
  controls.  Hard-refresh the browser (Ctrl+F5) after updating the pack.
- **Edits lost when re-opening the timeline (older pack version)** - this
  pack now restores the saved `timeline_edits` on every editor open (from
  the widget, or from the last merge run's bundle copy when the widget is
  empty), understands every JSON shape the node accepts, and Apply can no
  longer wipe the restored edits.  If you just updated the pack and still
  see the old behaviour, hard-refresh the browser (Ctrl+F5) - ComfyUI
  serves the editor script and browsers cache it aggressively.  Existing
  JSON in `timeline_edits` stays valid and is restored on the next open.
- **"0 pieces replaced" / unmatched files** - filenames must contain
  `HH.MM.SS.mmm_TTTTT` before the extension (a prefix is allowed).  Check the
  `extensions` widget (default `.flac` only) and that `takes_folder` points at
  the right directory.  Use `mode = report_only` to inspect the plan without
  rendering.
- **Output extended unexpectedly / too much replaced** - you are probably in
  `until_next` mode with takes that are silent outside the phrase; switch
  `window_mode` to `auto_trim`.
- **A dip or gap right after a dubbed phrase (auto_trim)** - the tail pad is
  an equal-power crossfade back into the original (200 ms by default).  Make
  the blend longer by raising `tail_pad_ms` (up to 2000 ms), or raise
  `silence_db` (e.g. -40) if breath/room noise is delaying the detected
  phrase end and pushing the fade later.
- **Leftover original voice after a short dub ("...des ka" still heard after
  the translated line ends)** - the original phrase is longer than the dub,
  so its tail fades back in right after the dub stops.  Open the **timeline
  editor** and drag the piece's **right edge** rightward: the original voice
  stays muted across the drawn span (the red overlay shows the exact mute
  region), then fades back in across `tail_pad_ms`.  Only the voice track is
  silenced, the background is never affected.  Keep the span short of the
  *next* original sentence; the pad is clipped at the next take's start so it
  can never mute the following dub's lead-in.  (API users: the old global
  `tail_silence_ms` engine option still works.)
- **Dubs start late / out of lip sync** - TTS dub files often begin with
  baked-in lead silence, so a clip placed at its timestamp starts speaking
  late.  Open the **timeline editor** and **drag the piece left** until its
  waveform onset sits at the beat (play it against the original with
  "silence voice under dubs" off for a raw sync comparison), then Apply and
  re-run.  The editor shows the piece's real audio, so no threshold guessing
  is involved.  (API users: `trim_lead=True` on `merge_pipeline` trims the
  lead automatically.)
- **Dubbed phrases cut off / overlapping in fast dialogue** - a dub's file
  runs past the next take's timestamp.  Keep `queue_overlaps` on (default):
  the next take is queued to start right after the current one's audible
  phrase ends, so no phrase is truncated and no dead air is inserted between
  lines (trailing file silence is stripped from the handover).  The report
  lists every shift ("queued after ... +Xs").  If you prefer the old hard
  cut, turn the widget off.
- **Original dialogue between phrases is muted (until_next)** - the window
  runs from each take to the next take, so everything in between (residue
  included) is silenced.  Switch to `auto_trim`, or draw the exact replaced
  spans in the timeline editor, to keep the original between phrases.
- **Background shifts / jumps under the dubs** - you are in
  `separation = full_mix`, which mutes and replaces the entire mix at every
  window.  Switch to `voice_isolation` (default): the voice is separated
  from the background first, dubs replace the voice only, and the
  background is added back unchanged - it cannot shift.  Remember to use
  voice-only dub takes with this mode.
- **The WHOLE mix ducks under the dubs (background included, original
  almost silenced, dub sounds like an empty room)** - you are using the
  `center` engine on a mono-like mix: everything centred (ambience and
  music included) lands in the "voice" track and is muted together with
  the dialogue.  Switch `isolation_engine` to `demucs` (install it with
  `pip install demucs`) or connect your own no-vocals track to the
  `background_audio` input - then only the voice is replaced and the
  background keeps running at constant level.
- **Rain / ambience / music pumps up and down with every phrase (constant
  background ducking, even with demucs or an external background)** -
  part of the continuous background ended up *inside* the replaced voice
  stem and was muted under every dub window.  Mark the affected dub pieces
  ambient in the timeline editor (alt+click, then Apply): the non-speech
  part of the replaced stem is extracted with an adaptive noise-floor gate
  (guided by the separated background stem) and keeps playing under those
  windows, plus the original span continues at the `ambient_db` level, so
  the background never stops.  The report's `Ambient fill :` line lists
  the marked pieces.  If you hear gating artefacts on unusual material,
  lower `ambient_db` and report the case.
- **Rain, a passing train or distant people get quieter under the dubs
  (ambience events ducking)** - the event is far louder than the gate's
  noise floor and was treated like speech.  The sustained-ambience keep
  (part of the ambient fill, applied to pieces marked ambient) recognises
  such events by their
  slow envelope and smooth spectrum and keeps them at full level.  If an
  event still ducks, it is most likely tonal/music-like (a harmonised
  swell, a synth pad) - tonal content is deliberately NOT kept, because it
  is indistinguishable from speech by spectrum; widen `fade_ms` a little
  to soften the window edges in that case.
- **"demucs engine is not available"** - install the package into ComfyUI's
  Python environment: `pip install demucs`.  The first run downloads the
  model weights (~80-160 MB).  On CPU expect roughly real-time separation
  speed for `htdemucs`; a GPU makes it much faster.  `auto` falls back to
  centre extraction automatically (with a warning in the report).
- **Faint original voice remains under a dub (voice_isolation)** - with the
  `center` engine, voice that is not dead-centre (wide reverb, hard-panned
  dialogue) stays partly in the background; narrow the band
  (`voice_low_hz` / `voice_high_hz`).  With the `demucs` engine, tune the
  model (`htdemucs_ft` is the fine-tuned variant) - or connect an external
  background where you control the split.
- **Music/ambience missing under a dub (voice_isolation)** - the dub take
  should contain voice only; the original background is re-added
  automatically.  If the take still carries a music bed, it will be doubled
  over the preserved background - re-export the takes voice-only.
- **Very long movies + demucs engine** - the separation runs the model over
  the whole track; on CPU expect a few times real-time.  For hour-long
  films, prefer a GPU or separate the background once with the demucs CLI /
  UVR and feed it to `background_audio` (the merge itself is then fast and
  the background is sample-exact).
- **New takes ignored on re-queue** - shouldn't happen: the node hashes the
  folder contents (`IS_CHANGED`) and re-runs when files change.
- **Milliseconds** - `.612` = 612 ms; shorter fields are read as fractions
  (`.61` = 610 ms, `.6` = 600 ms).
- **Very long movies** - audio is processed as 32-bit float in RAM
  (~1.4 GB per hour of stereo 48 kHz); close other memory-hungry nodes if you
  hit OOM.

## Notes on semantics

- Replacement is **positional**: take audio is read at the same sample offsets
  as the original.  Takes must be time-aligned with the track (same start,
  same speed).  If your takes drift, align them upstream first.
- Short takes switch into **placement** mode automatically (the take starts
  at the window start) whenever they cannot be read positionally: end mode,
  explicit `start-end` filenames, or a take that ends before the window's
  absolute start.  Full-length takes always stay in positional mode.
  `auto_trim` trims placed clips on their own samples, so short clips with
  leading/trailing silence are handled correctly.
- Timeline edits (`timeline_edits`) turn a placed clip into an **explicit
  window**: the original is silenced exactly across the drawn span and the
  piece audio plays from the position it was dragged to (front/end silence
  comes from where the edges were pulled).  Edited pieces skip `auto_trim`,
  automatic trimming and queue shifting - they queue by their declared span
  like explicit filename windows - and windows drawn over each other are
  clipped at the next piece's start, so the last word always wins.
- Queueing (`queue_overlaps`) applies to **segment clips** (queued by their
  audible length - trailing file silence is stripped, so a TTS tail baked
  into the file never shifts the next phrase) and explicit `start-end`
  windows (queued by their declared span); full-track passes are never
  queued - they hand over at the next timestamp exactly as before.  A queued
  take moves as a whole (its whole window shifts), and chained shifts
  cascade: each take starts after the previous one's audible phrase ends,
  with no extra gap.
- Ambient fill (pieces marked ambient in the timeline editor) applies
  wherever those pieces' windows mute the original: the dub window,
  `auto_trim` post-phrase holds and the gaps between queued takes that
  belong to the marked piece.  Outside the marked windows the output is
  bit-identical to a merge without marks (centre engine / external
  background) or the model's reconstruction (demucs).  The bed is extracted
  once per merge from the replaced stem and is deterministic - the same
  track always yields the same result.
