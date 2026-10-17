# rechka-cli

Offline Russian speech recognition for Linux from the command line.

`rechka-cli` is the Linux companion to [**Rechka for Android**](https://github.com/dm1sh/rechka).
Both projects use the same fully local recognition stack — GigaAM v3,
sherpa-onnx and Silero VAD — so audio never leaves the machine. The Android
app provides a floating microphone, a voice keyboard and file transcription
on a phone; this project provides the desktop/terminal equivalents:

- transcribe audio and video files;
- dictate into the focused application through IBus or Fcitx5;
- run a small background daemon controlled by `rechka start`, `rechka stop`
  and `rechka toggle` (ideal for a global hotkey).

## Features

- **Fully offline.** Recognition runs locally with GigaAM v3 on sherpa-onnx.
- **File transcription** with Silero VAD segmentation, or fixed 30-second
  chunks when VAD is disabled/unavailable.
- **Microphone dictation** through a long-lived daemon, so the model is
  loaded once instead of on every hotkey press.
- **IBus-first input method support**, including automatic switching to the
  Rechka input while dictating and back afterwards. Fcitx5 is supported
  through a typing fallback.
- **A few VAD windows of pre-roll** are passed to the recognizer before each
  detected speech segment, without re-sending audio from the previous
  segment. This reduces swallowed phrase beginnings.
- **One TOML config file** for the model location, microphone, VAD pause,
  dictionary, input-method framework and daemon behaviour.
- **Output routing** to stdout, a file, a directory, or an input-method
  engine.
- **Benchmark command** reporting startup time, realtime factor and time to
  first output, with and without VAD.

## Requirements

- Linux.
- Python 3.11 or 3.12, managed with [uv](https://docs.astral.sh/uv/).
- `ffmpeg` only if you need to decode non-WAV media (MP3, M4A, MP4, …).
  16 kHz mono WAV files work without it.
- For microphone capture, one of:
  - the `mic` extra (`sounddevice`) plus the system PortAudio library
    (Debian/Ubuntu: `libportaudio2`; Fedora: `portaudio`);
  - PipeWire's `pw-cat`, PulseAudio's `parec`, or ALSA's `arecord`.
- For the IBus engine, the distro GI bindings: Fedora
  `python3-gobject` + `ibus-libs`, Debian/Ubuntu `python3-gi` +
  `gir1.2-ibus-1.0`. The daemon can run the engine under the distro
  `python3`, so these do not have to be installed into the uv environment.

## Install

From a checkout:

```bash
git clone https://github.com/dm1sh/rechka-cli.git
cd rechka-cli
uv sync
uv run rechka --help
```

Useful extras:

```bash
uv sync --extra mic   # PortAudio-based microphone backend
uv sync --extra ibus  # PyGObject for running the IBus engine in this venv
```

The GigaAM model is deliberately **not** part of the Python package
(~326 MB). Download it once:

```bash
uv run rechka download
uv run rechka check
```

`download` fetches the four GigaAM v3 files and `silero_vad.onnx` from the
GitHub release configured in `[download]` (default repository
`dm1sh/rechka`, tag `model-gigaam-v3`), verifies sizes and SHA-256, and
skips files that are already present and valid. See
`rechka download --help` for `--repo`, `--tag`, `--force` and `--no-vad`.

## Quick start

Transcribe a file to stdout:

```bash
uv run rechka transcribe lecture.mp3
```

Choose the output explicitly:

```bash
uv run rechka transcribe lecture.mp3 -o -              # stdout
uv run rechka transcribe lecture.mp3 -o result.txt     # exact file
uv run rechka transcribe lecture.mp3 -o ~/texts/       # ~/texts/lecture.txt
uv run rechka transcribe lecture.mp3 -o ibus           # commit via IBus
uv run rechka transcribe lecture.mp3 -o fcitx5         # commit via Fcitx5
```

`-o`/`--output` accepts:

| Value | Meaning |
| --- | --- |
| omitted or `-` | print to stdout |
| `ibus` or `fcitx5` (any case) | send to that input-method engine |
| an existing directory | write `<source-file-stem>.txt` inside it |
| any other path | write exactly that file |

Progress is hidden by default. Add `--show-progress` to print it to stderr;
the transcript itself still goes to the selected output.

## Dictation daemon

Start the daemon and bind `toggle` to a desktop hotkey:

```bash
uv run rechka serve       # foreground daemon
uv run rechka toggle      # start/stop listening
uv run rechka start       # start listening explicitly
uv run rechka stop        # stop; the trailing phrase is flushed
uv run rechka status      # daemon, microphone, model and IM status
uv run rechka shutdown    # ask the daemon to exit
```

The daemon listens on `$XDG_RUNTIME_DIR/rechka.sock`. While listening, it
segments microphone audio with VAD, transcribes each completed segment,
applies the optional user dictionary, and commits the text through the
selected input-method engine.

With IBus, `start` remembers the current input, switches the session to
Rechka, and `stop` switches back after the final phrase has been committed.
If you change the input yourself while dictating, your choice is left
alone. `rechka status` reports this in `im_switch_active`,
`im_previous_engine` and `im_switch_detail`.

### Autostart with systemd

Install the user unit and enable it at login:

```bash
uv run rechka systemd install
systemctl --user daemon-reload
systemctl --user enable --now rechka.service
```

The generated unit runs `rechka serve` as a plain background user service:
it does not block login, and failures are restarted by systemd without
affecting the rest of the session. Inspect it first with
`rechka systemd print`; regenerate it after upgrades with
`rechka systemd install`.

## Input methods

Detect what the current session uses:

```bash
uv run rechka im detect
```

On `serve`, the daemon starts only the engine matching the detected
framework (`[im] framework = "auto"` by default; it can be pinned to
`ibus`, `fcitx5` or `none`).

### IBus

Install the Rechka IBus component once so GNOME/IBus can discover it:

```bash
uv run rechka im install
systemctl --user daemon-reload
systemctl --user restart org.freedesktop.IBus.session.GNOME.service
# logging out and back in also works
```

You do **not** need to add Rechka in GNOME Settings → Keyboard → Input
Sources. That list only controls the Super+Space switcher; the daemon
activates the Rechka engine directly while dictating. If you added it
there earlier and do not want it in the switcher:

```bash
uv run rechka im hide     # remove from the GNOME switcher only
uv run rechka im unhide   # put it back
```

`rechka im uninstall` removes the installed IBus component. The IBus engine
is transparent to normal typing: dictated text is committed into the
focused application while Rechka is active; if another input is active,
text is queued in order and inserted when Rechka next gains focus.

### Fcitx5

A native Fcitx5 addon would be a compiled C++ module, so the Fcitx5 side
uses the available virtual-keyboard typing tool instead — `ydotool`,
`wtype` or `xdotool`, depending on the session. Text is committed into the
focused window; if no typing tool is available, commits are queued and
retried.

## Configuration

```bash
uv run rechka config init   # write a commented sample config
uv run rechka config path   # print the config path in use
uv run rechka config show   # effective configuration as JSON
uv run rechka check         # verify config, model files and IM detection
```

Default config location: `~/.config/rechka/rechka.toml` (or
`$RECHKA_CONFIG`, or `--config PATH`). The main sections are:

| Section | What it controls |
| --- | --- |
| `[model]` | GigaAM directory and recognizer thread count |
| `[audio]` | microphone device, sample rate and capture backend |
| `[vad]` | VAD enablement, model path, pause length, file chunking |
| `[text]` | user dictionary and the optional voice suffix |
| `[im]` | input-method framework: `auto`, `ibus`, `fcitx5`, `none` |
| `[daemon]` | whether the IM engine is started with the daemon |
| `[download]` | GitHub repository/tag for `rechka download` |

Unknown sections and keys are ignored with a warning; syntax and type
errors are reported. See `config.example.toml` for the annotated sample.

## Benchmark

```bash
uv run rechka benchmark lecture.mp3 --runs 3
uv run rechka benchmark lecture.mp3 --show-text
```

The report separates model startup from file decoding, then compares the
full pipeline with Silero VAD segmentation against fixed chunks: elapsed
time, ×-realtime ratio (and RTF), segment count, and time to first output.

## How it works

```text
file / microphone
    → decode or capture 16 kHz mono audio
    → Silero VAD segments (+ a few pre-roll windows, never re-sent)
    → GigaAM v3 recognition, one segment at a time
    → optional dictionary replacements
    → stdout / file / IBus / Fcitx5
```

The recognizer is created once per file transcription, and once per daemon
lifetime for dictation. VAD segment samples are copied out of sherpa-onnx
before the native segment is popped from the detector queue.

## Development

```bash
uv run pytest
uv build
```

Project layout:

```text
src/rechka/
  cli.py            command-line interface
  daemon.py         background daemon and Unix-socket protocol
  pipeline.py       file-transcription pipeline
  segmentation.py   Silero VAD / fixed segmentation and pre-roll
  transcriber.py    sherpa-onnx GigaAM recognizer
  mic.py            microphone backends
  im.py             framework detection, engine lifecycle, GNOME helpers
  ibus_engine.py    IBus engine process
  fcitx5_engine.py  Fcitx5 typing engine
  engine_server.py  engine socket protocol and commit queue
  outputs.py        stdout/file/IM sinks
  config.py         TOML configuration
  model.py          model file verification
  download.py       model download from GitHub releases
  dictionary.py     user text replacements
  systemd.py        systemd --user unit
```

## Troubleshooting

- **`rechka check` says the model is missing or invalid** — run
  `rechka download`, or point `[model].dir` at an existing GigaAM v3
  directory.
- **File mode uses fixed chunks** — the Silero VAD model was not found;
  check `[vad].model` or re-run `rechka download`.
- **No microphone backend** — install the `mic` extra plus PortAudio, or
  make sure `pw-cat`, `parec` or `arecord` is available.
- **Rechka is not offered by GNOME/IBus** — run `rechka im install`, then
  restart the GNOME IBus service or log out and back in. On Fedora the
  generated drop-in also forces `--cache none` so the user component is
  not shadowed by the system registry cache.
- **The IBus engine cannot load PyGObject in the uv venv** — install the
  distro packages (`python3-gobject`/`ibus-libs` on Fedora,
  `python3-gi`/`gir1.2-ibus-1.0` on Debian/Ubuntu); the daemon falls back
  to the distro `python3` for the engine, or use `uv sync --extra ibus`.

## Related projects

- [Rechka for Android](https://github.com/dm1sh/rechka) — the original
  mobile app: floating microphone, voice keyboard and offline file
  transcription.
- [GigaAM](https://github.com/salute-developers/GigaAM) — the speech
  recognition model family.
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — the ONNX runtime
  used for recognition and VAD.
- [Silero VAD](https://github.com/snakers4/silero-vad) — voice activity
  detection.

## License

MIT.
