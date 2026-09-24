from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'model'))
from tokenizer import FlashPiece,normalize_english

def test_roundtrip():
    texts=['привет как дела','hello world','это тест!']
    t=FlashPiece.build(texts,vocab_size=300,min_count=1)
    for s in texts: assert t.decode(t.encode(s))==s

def test_english_ascii():
    assert normalize_english('“Café”—test…')=='"Cafe"-test...'
