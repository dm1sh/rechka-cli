from rechka.cli import build_parser


def parse(argv):
    return build_parser().parse_args(argv)


def test_transcribe_defaults():
    args = parse(["transcribe", "lecture.wav"])
    assert args.output is None
    assert args.show_progress is False


def test_show_progress_flag():
    args = parse(["transcribe", "lecture.wav", "--show-progress"])
    assert args.show_progress is True


def test_output_flag_forms():
    assert parse(["transcribe", "a.wav", "-o", "-"]).output == "-"
    assert parse(["transcribe", "a.wav", "--output", "IBUS"]).output == "IBUS"


def test_im_engine_command_parses():
    args = parse(["im-engine", "ibus"])
    assert args.framework == "ibus"


def test_im_install_and_uninstall_parse():
    args = parse(["im", "install"])
    assert args.im_command == "install" and args.system is False
    args = parse(["im", "uninstall", "--system"])
    assert args.im_command == "uninstall" and args.system is True


def test_im_engine_ibus_flag_parses():
    args = parse(["im-engine", "ibus", "--ibus"])
    assert args.framework == "ibus" and args.ibus is True
