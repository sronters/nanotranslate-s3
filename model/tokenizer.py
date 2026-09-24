#!/usr/bin/env python3
"""FlashPiece: tiny UTF-8 byte-piece tokenizer for MCU translation.

The vocabulary always contains 256 raw byte pieces so encoding is lossless.
Additional pieces are frequent UTF-8 byte n-grams selected from training text.
Encoding is deterministic greedy longest-match, which is cheap to reproduce in C++.
"""
from __future__ import annotations
import argparse, collections, json, unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

SPECIAL = [b"<pad>", b"<bos>", b"<eos>", b"<unk>"]
PAD, BOS, EOS, UNK = range(4)

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s.strip())
    table = str.maketrans({
        "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
        "’": "'", "‘": "'", "—": "-", "–": "-", "…": "...",
        "\u00a0": " ",
    })
    s = s.translate(table)
    return " ".join(s.split())

def normalize_english(s: str) -> str:
    s = normalize_text(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return " ".join(s.split())

@dataclass
class FlashPiece:
    pieces: List[bytes]

    @classmethod
    def build(cls, texts: Iterable[str], vocab_size: int = 1536,
              max_piece_bytes: int = 8, min_count: int = 8) -> "FlashPiece":
        if vocab_size < 260:
            raise ValueError("vocab_size must be >= 260 (4 specials + 256 bytes)")
        counts = collections.Counter()
        for text in texts:
            b = normalize_text(text).encode("utf-8")
            for n in range(2, max_piece_bytes + 1):
                for i in range(0, len(b) - n + 1):
                    p = b[i:i+n]
                    if p.count(b"\n") or p.count(b"\t"):
                        continue
                    if b"  " in p or p.count(b" ") > 1:
                        continue
                    counts[p] += 1
        ranked = [(c * (len(p) - 1), c, len(p), p)
                  for p, c in counts.items() if c >= min_count]
        ranked.sort(reverse=True)
        pieces = list(SPECIAL) + [bytes([i]) for i in range(256)]
        seen = set(pieces)
        for _, _, _, p in ranked:
            if p in seen:
                continue
            pieces.append(p); seen.add(p)
            if len(pieces) >= vocab_size:
                break
        return cls(pieces)

    def __post_init__(self):
        self._by_first = [[] for _ in range(256)]
        for idx, p in enumerate(self.pieces[4:], start=4):
            if p:
                self._by_first[p[0]].append((len(p), idx, p))
        for bucket in self._by_first:
            bucket.sort(key=lambda x: (-x[0], x[1]))

    @property
    def vocab_size(self) -> int:
        return len(self.pieces)

    def encode(self, text: str, bos: bool = True, eos: bool = True,
               max_tokens: int | None = None) -> List[int]:
        data = normalize_text(text).encode("utf-8")
        out = [BOS] if bos else []
        i = 0
        while i < len(data):
            match = None
            for _, idx, p in self._by_first[data[i]]:
                if data.startswith(p, i):
                    match = (idx, len(p)); break
            if match is None:
                out.append(UNK); i += 1
            else:
                out.append(match[0]); i += match[1]
            if max_tokens is not None and len(out) >= max_tokens - int(eos):
                break
        if eos:
            out.append(EOS)
        return out

    def decode(self, ids: Iterable[int]) -> str:
        buf = bytearray()
        for i in ids:
            if i in (PAD, BOS):
                continue
            if i == EOS:
                break
            if 0 <= i < len(self.pieces) and i >= 4:
                buf.extend(self.pieces[i])
        return bytes(buf).decode("utf-8", errors="replace")

    def save(self, path: str | Path):
        obj = {"type":"flashpiece-v1", "pieces_hex":[p.hex() for p in self.pieces]}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FlashPiece":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        if obj.get("type") != "flashpiece-v1":
            raise ValueError("unsupported tokenizer")
        return cls([bytes.fromhex(x) for x in obj["pieces_hex"]])

def iter_tsv_text(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                yield parts[0]; yield parts[1]

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--input", required=True)
    b.add_argument("--output", required=True)
    b.add_argument("--vocab-size", type=int, default=1536)
    b.add_argument("--max-piece-bytes", type=int, default=8)
    b.add_argument("--min-count", type=int, default=8)
    args = ap.parse_args()
    if args.cmd == "build":
        tok = FlashPiece.build(iter_tsv_text(args.input), args.vocab_size,
                               args.max_piece_bytes, args.min_count)
        tok.save(args.output)
        print(f"saved {tok.vocab_size} pieces -> {args.output}")

if __name__ == "__main__":
    main()
