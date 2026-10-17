from pathlib import Path

from rechka.dictionary import Dictionary


def write_dict(tmp_path: Path) -> Dictionary:
    path = tmp_path / "dictionary.txt"
    path.write_text(
        "# comment\nречка=Речка\nлишний\nпривет=здравствуй\n",
        encoding="utf-8",
    )
    return Dictionary.load(path)


def test_replacement_and_deletion(tmp_path):
    dictionary = write_dict(tmp_path)
    assert dictionary.apply("это речка и лишний привет") == "это Речка и здравствуй"


def test_word_boundaries(tmp_path):
    dictionary = write_dict(tmp_path)
    # 'речка' inside a longer word must not be replaced.
    assert dictionary.apply("речкой") == "речкой"


def test_missing_file_is_empty(tmp_path):
    assert Dictionary.load(tmp_path / "nope.txt").apply("текст") == "текст"
