# Rechka CLI (Linux)

Offline Russian speech recognition for Linux: GigaAM v3 on sherpa-onnx —
the same engine and model as the Android app — driven entirely from the
command line and one TOML config file.

> **Status:** file transcription, microphone dictation and the
> IBus/Fcitx5 output engines are implemented. The IBus engine needs the
> distro PyGObject/IBus bindings (see *Input methods*); the daemon can
> run it under the distro python3, so the uv environment itself does not
> need PyGObject. Where no usable bindings exist the daemon says exactly
> that and keeps working without IM output.

## Install

With [uv](https://docs.astral.sh/uv/):

```bash
uv tool install rechka-cli        # from PyPI, once published
# or from a checkout:
uv sync                           # dev environment
uv run rechka --help
```

The GigaAM model (~326 MB) is **not** part of the package. Fetch it with:

```bash
rechka download    # into [model].dir from the config; verifies SHA-256
```

This pulls the four GigaAM files from the `model-gigaam-v3` GitHub
release (repository/tag configurable in `[download]`, overridable with
`--repo`/`--tag`) plus `silero_vad.onnx` for VAD segmentation into
`[vad].model` (default: next to the GigaAM files). Already-verified
files are skipped; `--force` re-downloads. `rechka check` re-verifies
names, sizes and SHA-256 at any time. Without the VAD model, file mode
falls back to fixed 30 s chunks.

Other system pieces: ffmpeg is needed only for non-WAV files (mp3, m4a,
video). Nothing else — recognition is fully offline and local.

## Configure

```bash
rechka config init     # writes a commented sample to ~/.config/rechka/rechka.toml
rechka config show     # effective configuration as JSON
rechka check           # config + model + detected IM framework
```

All settings live in that one file: model location/threads, VAD pause
length, output sinks, user dictionary (same `ключ=значение` format as the
Android app), IM framework selection, daemon socket. See
`config.example.toml`.

## Use

**Transcribe a file** (works now):

```bash
rechka transcribe lecture.mp3                 # transcript -> stdout (default)
rechka transcribe lecture.mp3 -o -            # same, explicitly
rechka transcribe lecture.mp3 -o ~/texts/     # -> ~/texts/lecture.txt
rechka transcribe lecture.mp3 -o result.txt   # -> this exact file (overwritten)
rechka transcribe lecture.mp3 -o ibus         # type/commit via the IBus engine
```

`-o`/`--output` decides where the transcript goes: `-` is stdout;
`ibus`/`fcitx5` (any case) selects the input-method engine; an existing
folder receives `<source name without extension>.txt` inside it; any
other value is treated as the output file path. Without `-o`, the
transcript goes to stdout.

Progress is off by default; `--show-progress` prints it to stderr
(even when the transcript goes to stdout). With VAD available, segments
follow speech pauses; output is emitted per segment as recognition
proceeds.

**Benchmark:**

```bash
rechka benchmark lecture.mp3 [--runs 3] [--show-text]
```

Reports model startup (load) time separately, then the full-pipeline
speed with Silero VAD segmentation and without it (fixed chunks) as
×-realtime and RTF, plus the time to first output (segmentation +
recognition up to the first text-producing segment). Best of N runs;
decode is measured once and excluded from the mode timings, and
segmentation is itemized in the output.

**Daemon + listening** (dictation):

```bash
rechka serve        # foreground daemon; loads nothing heavy until asked
rechka status       # state, mic backend, last dictated text, IM status
rechka toggle       # start listening / stop (bind this to a hotkey!)
rechka start        # start listening (loads the model on first use)
rechka stop         # stop; the trailing phrase is flushed, not lost
rechka shutdown
```

The daemon holds state and answers control commands over a Unix socket
(`$XDG_RUNTIME_DIR/rechka.sock`), so `toggle` is instant and the model
is loaded once — not per keypress. While listening, microphone audio is
VAD-segmented and transcribed per segment; every segment is committed
through the IM engine as soon as it is recognized (see below), and
`stop` flushes the final phrase. With IBus, `start`/`toggle` also
switches the system input to *Rechka* and `stop` switches back to the
input that was active before dictation, after that final phrase has
been committed; if you change input yourself while dictating, your
choice is left alone. `rechka status` reports this as
`im_switch_active`, `im_previous_engine` and `im_switch_detail`.

Microphone backends (`[audio] backend`, default `auto`): `sounddevice`
(the `uv sync --extra mic` extra — needs the system PortAudio library,
e.g. `libportaudio2`), then the subprocess recorders `pw-cat`
(PipeWire), `parec` (PulseAudio) and `arecord` (ALSA), all delivering
16 kHz mono.

**Autostart at login (systemd user unit):**

```bash
rechka systemd install
systemctl --user daemon-reload
systemctl --user enable --now rechka.service
```

The unit is a plain background user service: it starts asynchronously,
never blocks login, and if it crashes systemd restarts it without
affecting the rest of the session.

## Input methods (IBus primary, Fcitx5 secondary)

`rechka im detect` reports which framework the current session uses —
from `XMODIFIERS`/`GTK_IM_MODULE`/`QT_IM_MODULE`/`IBUS_ADDRESS`, falling
back to running processes. On `serve`, the daemon detects the framework
and starts **only the matching engine** (`im.framework = "auto"`; can be
pinned in config). Both engines are separate processes serving commit
requests on `$XDG_RUNTIME_DIR/rechka-engine-<framework>.sock`; the daemon
(and `transcribe -o ...`) are their clients.

**IBus (primary):** a real, transparent IBus engine (`ibus_engine.py`,
PyGObject). It never consumes keys, so normal typing is unaffected.
Dictated text is committed into the focused app whenever *Rechka* is
the active input source there. Text dictated while another source is
active is queued and inserted, in order, when *Rechka* next gains
focus; `rechka status` shows the queue state.

The engine needs the IBus typelib and PyGObject: Fedora `sudo dnf
install python3-gobject ibus-libs`, Debian/Ubuntu `sudo apt install
python3-gi gir1.2-ibus-1.0`. It does **not** need them inside the uv
venv: on `serve` the daemon prefers rechka's own Python but otherwise
runs the engine under the distro `/usr/bin/python3` (the engine only
needs the standard library from rechka itself). Installing
PyGObject into rechka's environment instead (`uv run --extra ibus
rechka serve`, or `--with PyGObject`) works too.

For GNOME to offer *Rechka* as an input source, install its IBus
component XML once:

```bash
rechka im install
systemctl --user daemon-reload
systemctl --user restart org.freedesktop.IBus.session.GNOME.service
# (logging out and back in works too; restart `rechka serve` afterwards
# if it was already running, and remove all this with `rechka im uninstall`)
```

`im install` writes `~/.local/share/ibus/component/rechka.xml`, a
small distro-python launcher it points at, and a systemd user drop-in
that adds the user component directory to `IBUS_COMPONENT_PATH` and
restarts the daemon with `--cache none` (IBus only scans
`/usr/share/ibus/component` by default, and Fedora's system registry
cache would otherwise shadow the user directory; `--system` installs
into the system directory instead). This file is what makes the engine
discoverable:
GNOME Settings lists engines from the daemon's XML-backed registry,
whereas the runtime registration the daemon performs on `serve` alone
only makes *Rechka* activatable for the session — it never appears in
Settings without the XML.

You do **not** need to add *Rechka* under Settings → Keyboard → Input
Sources: that list is only what the Super+Space switcher cycles, while
the daemon activates the engine directly over the IBus bus when
dictation starts and switches back when it stops. If you added *Rechka*
there earlier and do not want it in the switcher, remove it in Settings
or run `rechka im hide` (and `rechka im unhide` to bring it back). While
you dictate, the indicator still shows *Rechka* as the active source —
it genuinely is active then — and it leaves the cycle list alone the
rest of the time.

**Fcitx5 (secondary):** a native Fcitx5 addon would be a compiled C++
module in Fcitx5's addon directories — not shippable in a Python
package — so the Fcitx5 engine (`fcitx5_engine.py`) uses
virtual-keyboard typing instead: `ydotool` (Wayland) / `wtype`
(wlroots) / `xdotool` (X11), whichever the session offers. It commits
into whatever window is focused; if no typing tool is installed, commits
queue and are retried on the next segment.

`transcribe -o ibus|fcitx5` uses the same engines: it reuses the running
one, starts one if needed (and stops it afterwards if it started it),
and fails loudly rather than queueing text silently.

## Layout

```
src/rechka/
  cli.py          commands (serve/toggle/transcribe/check/config/im/systemd)
  config.py       TOML config + defaults (Android SharedPreferences -> file)
  model.py        GigaAM file names/sizes/SHA-256, verification
  transcriber.py  sherpa-onnx OfflineRecognizer (lazy native import)
  segmentation.py Silero VAD segmentation with fixed-chunk fallback
  audiofile.py    WAV (stdlib) + ffmpeg decoding to 16 kHz mono
  dictionary.py   user replacements, Android-compatible semantics
  outputs.py      stdout / file / IM sinks
  im.py           IBus vs Fcitx5 session detection + engine lifecycle/client
  ibus_engine.py  IBus engine process (PyGObject, transparent passthrough)
  fcitx5_engine.py Fcitx5-side typing engine (ydotool/wtype/xdotool)
  engine_server.py engine socket protocol + commit/focus queue bridge
  mic.py          microphone capture (sounddevice/pw-cat/parec/arecord)
  daemon.py       Unix-socket control protocol + state
  systemd.py      --user unit rendering/install
```

Linux is the primary target; platform specifics are isolated in
`paths.py` (XDG), `im.py` (framework detection) and `systemd.py`, so the
core (config, model, transcriber, segmentation, dictionary, outputs)
stays portable where it costs nothing.

## Roadmap

1. Dictation-time/word statistics parity with the Android app (optional).
2. Optional: stream partial (unfinalized) hypotheses into the engine as
   pre-edit text instead of segment-by-segment commits.
