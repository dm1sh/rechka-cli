"""Command-line interface.

Daemon control (the daemon holds model/mic state):
  rechka serve | toggle | start | stop | status | shutdown

One-shot work:
  rechka transcribe FILE [-o OUTPUT]
  rechka benchmark FILE          startup + VAD vs no-VAD pipeline speed
  rechka check [--fast]          verify config + model files
  rechka download                fetch model files from the GitHub release

Setup helpers:
  rechka config init|path|show
  rechka im detect               which IM framework this session uses
  rechka im install              install the IBus component XML so GNOME
                                 Settings can offer Rechka as input source
  rechka systemd print|install   autostart-at-login unit (systemd --user)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, paths, systemd
from .audiofile import AudioDecodeError
from .config import ConfigError, effective_summary, load_config
from .daemon import Daemon, DaemonUnavailableError, send_command
from .im import (
    IBUS_ENGINE_NAME,
    detect_framework,
    install_ibus_component,
    set_gnome_engine_listed,
    uninstall_ibus_component,
)
from .model import verify_model
from .outputs import SinkError
from .pipeline import transcribe_file
from .transcriber import TranscriberError

SAMPLE_CONFIG = """\
# Rechka configuration. Every key is optional; shown values are defaults
# unless noted. Restart the daemon after changes.

[model]
# Directory with the four GigaAM v3 files (downloaded separately — the
# model is NOT part of the rechka-cli package). ~326 MB total.
# dir = "~/.local/share/rechka/models/gigaam-v3"
threads = 4

[audio]
# Microphone device for `rechka start` / `toggle` ("default", or a device
# name/index understood by the capture backend).
device = "default"
sample_rate = 16000
# Capture backend: auto | sounddevice | pipewire | pulse | alsa.
# auto picks the first available of sounddevice (extra: `uv sync
# --extra mic`), pw-cat, parec, arecord.
backend = "auto"

[vad]
enabled = true
# silero_vad.onnx location; empty means "next to the GigaAM files".
# model = "/path/to/silero_vad.onnx"
min_silence_seconds = 0.5
file_enabled = true
chunk_seconds = 30.0

[text]
dictionary_enabled = false
# dictionary_path = "~/.config/rechka/dictionary.txt"
voice_suffix = false

[im]
# auto = start IBus or Fcitx5 engine matching the running session.
framework = "auto"

[daemon]
# socket_path = "$XDG_RUNTIME_DIR/rechka.sock"
start_im_engine = true

[download]
# GitHub release hosting the GigaAM files (`rechka download`).
repository = "dm1sh/rechka"
tag = "model-gigaam-v3"
include_vad = true
"""


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to rechka.toml (default: $RECHKA_CONFIG or ~/.config/rechka/rechka.toml)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rechka",
        description="Offline Russian speech recognition (GigaAM v3) for Linux.",
    )
    parser.add_argument("--version", action="version", version=f"rechka {__version__}")
    _add_config_arg(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("serve", help="Run the daemon in the foreground")
    for name, help_text in (
        ("toggle", "Start listening if stopped, stop if listening (via daemon)"),
        ("start", "Start listening (via daemon)"),
        ("stop", "Stop listening (via daemon)"),
        ("status", "Show daemon status"),
        ("shutdown", "Ask the daemon to exit"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        _add_config_arg(cmd)

    transcribe = sub.add_parser("transcribe", help="Recognize an audio/video file")
    _add_config_arg(transcribe)
    transcribe.add_argument("file", type=Path, help="Audio or video file to transcribe")
    transcribe.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="OUTPUT",
        help="Where the transcript goes: '-' = stdout; 'ibus'/'fcitx5' = "
        "input-method engine; an existing folder = <source name>.txt inside "
        "it; anything else = output file path. Default: stdout",
    )
    transcribe.add_argument(
        "--show-progress",
        action="store_true",
        help="Print progress to stderr (even when the transcript goes to stdout)",
    )

    benchmark = sub.add_parser(
        "benchmark", help="Measure model startup and file pipeline speed"
    )
    _add_config_arg(benchmark)
    benchmark.add_argument("file", type=Path, help="Audio or video file to benchmark")
    benchmark.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Timed repetitions per mode; the best is reported (default: 3)",
    )
    benchmark.add_argument(
        "--show-text", action="store_true", help="Also print the recognized text"
    )

    check = sub.add_parser("check", help="Verify configuration and model files")
    _add_config_arg(check)
    check.add_argument("--fast", action="store_true", help="Skip SHA-256 hashing")

    download = sub.add_parser(
        "download", help="Download model files from the GitHub release"
    )
    _add_config_arg(download)
    download.add_argument("--repo", default=None, help="GitHub owner/repo (overrides config)")
    download.add_argument("--tag", default=None, help="Release tag (overrides config)")
    download.add_argument(
        "--force", action="store_true", help="Re-download even verified files"
    )
    download.add_argument(
        "--no-vad", action="store_true", help="Skip silero_vad.onnx"
    )

    config_cmd = sub.add_parser("config", help="Config helpers")
    _add_config_arg(config_cmd)
    config_sub = config_cmd.add_subparsers(dest="config_command", required=True)
    init = config_sub.add_parser("init", help="Write a commented sample config")
    init.add_argument("--force", action="store_true", help="Overwrite existing config")
    config_sub.add_parser("path", help="Print the config file path in use")
    config_sub.add_parser("show", help="Print the effective configuration as JSON")

    im_cmd = sub.add_parser("im", help="Input-method framework helpers")
    _add_config_arg(im_cmd)
    im_sub = im_cmd.add_subparsers(dest="im_command", required=True)
    im_sub.add_parser("detect", help="Detect whether this session uses IBus or Fcitx5")
    im_install = im_sub.add_parser(
        "install",
        help="Install the IBus component XML (GNOME can then select Rechka)",
    )
    im_install.add_argument(
        "--system",
        action="store_true",
        help="Install into /usr/share/ibus/component (needs root/write access)",
    )
    im_install.add_argument(
        "--exec-start",
        default=None,
        help="Custom <exec> command for the component "
        "(default: generated distro-python launcher)",
    )
    im_sub.add_parser(
        "hide",
        help="Remove Rechka from the GNOME input-source switcher "
        "(it still activates while dictating)",
    )
    im_sub.add_parser(
        "unhide", help="Put Rechka back into the GNOME input-source switcher"
    )
    im_uninstall = im_sub.add_parser("uninstall", help="Remove the installed IBus component")
    im_uninstall.add_argument(
        "--system",
        action="store_true",
        help="Remove from /usr/share/ibus/component instead of the user locations",
    )

    engine = sub.add_parser("im-engine", help=argparse.SUPPRESS)
    _add_config_arg(engine)
    engine.add_argument("framework", choices=["ibus", "fcitx5"])
    engine.add_argument(
        "--ibus",
        action="store_true",
        help="IBus only: launched by ibus-daemon from component XML "
        "(claim the bus name; do not register a component at runtime)",
    )

    systemd_cmd = sub.add_parser("systemd", help="systemd --user autostart unit")
    systemd_sub = systemd_cmd.add_subparsers(dest="systemd_command", required=True)
    systemd_sub.add_parser("print", help="Print the unit file")
    install = systemd_sub.add_parser("install", help="Install the unit for this user")
    install.add_argument(
        "--exec-start",
        default=None,
        help="ExecStart command (default: '<rechka binary on PATH> serve')",
    )
    return parser


def _load_or_die(config_path: Path | None):
    try:
        return load_config(config_path)
    except ConfigError as exc:
        print(f"rechka: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _daemon_command(cfg, action: str) -> int:
    try:
        response = send_command(cfg.daemon.socket_path, {"cmd": action})
    except DaemonUnavailableError as exc:
        print(f"rechka: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0 if response.get("ok") else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command
    cfg = _load_or_die(getattr(args, "config", None))

    if command == "serve":
        daemon = Daemon(cfg)
        try:
            daemon.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0

    if command in ("toggle", "start", "stop", "status", "shutdown"):
        return _daemon_command(cfg, command)

    if command == "im-engine":
        # Engine processes are spawned by the daemon (im.start_engine) and
        # serve commits on the engine socket until told to shut down.
        if args.framework == "ibus":
            from .ibus_engine import run as run_ibus_engine

            return run_ibus_engine(cfg, exec_by_ibus=args.ibus)
        from .fcitx5_engine import run as run_fcitx5_engine

        return run_fcitx5_engine(cfg)

    if command == "transcribe":
        try:
            transcribe_file(
                args.file,
                cfg,
                output=args.output,
                progress=args.show_progress,
            )
        except (AudioDecodeError, TranscriberError, ValueError, SinkError) as exc:
            print(f"rechka: {exc}", file=sys.stderr)
            return 1
        return 0

    if command == "benchmark":
        from .benchmark import run_benchmark

        try:
            report = run_benchmark(args.file, cfg, runs=args.runs)
        except (AudioDecodeError, TranscriberError, ValueError) as exc:
            print(f"rechka: {exc}", file=sys.stderr)
            return 1
        print(report.render(show_text=args.show_text))
        return 0

    if command == "check":
        problems = verify_model(cfg.model.dir, full_hash=not args.fast)
        detection = detect_framework(cfg)
        print(f"config: {cfg.source}")
        print(f"model dir: {cfg.model.dir}")
        if problems:
            for problem in problems:
                print(f"  MISSING/BAD {problem.name}: {problem.detail}")
            print("model: NOT READY")
        else:
            print("model: OK" + (" (sizes only)" if args.fast else " (sha256 verified)"))
        print(f"IM framework: {detection.framework} ({detection.evidence})")
        from .im import engine_prerequisites
        from .mic import available_backends

        backends = available_backends()
        print(
            "mic backends: "
            + (", ".join(backends) if backends else "none available "
               "(install the sounddevice extra or pw-cat/parec/arecord)")
        )
        if detection.framework != "none":
            ok, detail = engine_prerequisites(detection.framework)
            print(f"IM engine: {'ready' if ok else 'unavailable'} ({detail})")
        return 1 if problems else 0

    if command == "download":
        from .download import download_all

        results = download_all(
            cfg,
            repository=args.repo,
            tag=args.tag,
            include_vad=False if args.no_vad else None,
            force=args.force,
        )
        failed = False
        for result in results:
            print(f"{result.status:>10}  {result.name}  {result.detail}".rstrip())
            failed = failed or result.status == "failed"
        return 1 if failed else 0

    if command == "config":
        if args.config_command == "init":
            target = args.config or paths.default_config_path()
            if target.exists() and not args.force:
                print(f"rechka: {target} already exists (use --force)", file=sys.stderr)
                return 1
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(SAMPLE_CONFIG, encoding="utf-8")
            print(f"Wrote sample config to {target}")
            return 0
        if args.config_command == "path":
            print(args.config or paths.default_config_path())
            return 0
        if args.config_command == "show":
            print(json.dumps(effective_summary(cfg), ensure_ascii=False, indent=2))
            return 0

    if command == "im":
        if args.im_command == "detect":
            detection = detect_framework(cfg)
            print(
                json.dumps(
                    {"framework": detection.framework, "evidence": detection.evidence},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.im_command == "install":
            try:
                result = install_ibus_component(
                    cfg, exec_start=args.exec_start, system=args.system
                )
            except OSError as exc:
                print(f"rechka: cannot install the IBus component: {exc}", file=sys.stderr)
                if args.system:
                    print(
                        "rechka: --system writes to /usr/share/ibus/component; "
                        "re-run with sudo or use a per-user install.",
                        file=sys.stderr,
                    )
                return 1
            print(f"Installed IBus component: {result.component_path}")
            if result.launcher_path is not None:
                print(f"Engine launcher: {result.launcher_path}")
            print(f"Component <exec>: {result.exec_command}")
            if result.dropin_path is not None:
                print(f"systemd drop-in: {result.dropin_path}")
                print("Activate it with:")
                print("  systemctl --user daemon-reload")
                print("  systemctl --user restart org.freedesktop.IBus.session.GNOME.service")
                print("  (or simply log out and back in)")
            else:
                print("Restart IBus (ibus restart, or log out and back in).")
            print("No need to add 'Rechka' in Settings -> Keyboard -> Input Sources:")
            print("the daemon switches to it automatically while dictating and back")
            print("afterwards. If you added it there earlier and do not want it in")
            print("the input-source switcher, run: rechka im hide")
            return 0
        if args.im_command in ("hide", "unhide"):
            result = set_gnome_engine_listed(
                IBUS_ENGINE_NAME, listed=args.im_command == "unhide"
            )
            print(result.detail, file=sys.stderr if not result.ok else sys.stdout)
            return 0 if result.ok else 1
        if args.im_command == "uninstall":
            try:
                removed = uninstall_ibus_component(system=args.system)
            except OSError as exc:
                print(f"rechka: cannot uninstall the IBus component: {exc}", file=sys.stderr)
                return 1
            if removed:
                for target in removed:
                    print(f"Removed {target}")
                print("Restart IBus (or log out and back in) to drop the engine.")
            else:
                print("No Rechka IBus component was installed there.")
            return 0

    if command == "systemd":
        if args.systemd_command == "print":
            print(systemd.unit_text(systemd.default_exec_start()), end="")
            return 0
        if args.systemd_command == "install":
            target = systemd.install_unit(args.exec_start)
            print(f"Installed {target}")
            print("Enable autostart at login with:")
            print("  systemctl --user daemon-reload")
            print("  systemctl --user enable --now rechka.service")
            print("The unit starts in the background and never blocks login;")
            print("if it fails, systemd restarts it without affecting the session.")
            return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
