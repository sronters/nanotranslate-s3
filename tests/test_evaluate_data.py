from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
from evaluate import read_parallel_pairs


def test_references_are_aligned(tmp_path):
    norm, raw = tmp_path / "norm.tsv", tmp_path / "raw.tsv"
    norm.write_text("привет\thello\nпока\tbye\n", encoding="utf-8")
    raw.write_text("привет\tHello!\nпока\tGoodbye!\n", encoding="utf-8")
    assert list(read_parallel_pairs(norm, raw)) == [
        ("привет", "hello", "Hello!"),
        ("пока", "bye", "Goodbye!"),
    ]


def test_misaligned_raw_reference_fails(tmp_path):
    norm, raw = tmp_path / "norm.tsv", tmp_path / "raw.tsv"
    norm.write_text("привет\thello\nпока\tbye\n", encoding="utf-8")
    raw.write_text("привет\tHello!\nдругая фраза\tGoodbye!\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch at row 2"):
        list(read_parallel_pairs(norm, raw))


def test_missing_raw_reference_fails(tmp_path):
    norm, raw = tmp_path / "norm.tsv", tmp_path / "raw.tsv"
    norm.write_text("привет\thello\nпока\tbye\n", encoding="utf-8")
    raw.write_text("привет\tHello!\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ends before row 2"):
        list(read_parallel_pairs(norm, raw))
