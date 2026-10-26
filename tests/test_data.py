from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
from data import TSVPairs
from tokenizer import FlashPiece, EOS


def test_overlong_pair_is_skipped_instead_of_truncated(tmp_path):
    data = tmp_path / "pairs.tsv"
    data.write_text("привет\thello\n" + ("я" * 100) + "\thello\n", encoding="utf-8")
    tok = FlashPiece.build(["привет", "hello"], vocab_size=260, min_count=1000)
    dataset = TSVPairs(data, tok, max_src=40, max_tgt=40)
    assert len(dataset) == 1
    assert dataset[0][0] == tok.encode("привет")
    assert dataset[0][0][-1] == EOS
