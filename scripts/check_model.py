#!/usr/bin/env python3
"""Validate the .s3mt header, payload length and CRC32 (not inference quality)."""
from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path

MAGIC = b"S3MT0001"
HEADER_BYTES = 128
NAMES = (
    "version", "flags", "vocab", "d_model", "d_ff", "heads",
    "enc", "dec", "max_src", "max_tgt", "pad", "bos", "eos", "unk",
    "tokenizer_bytes", "payload_bytes", "crc32", "total_bytes",
)


def validate(data: bytes) -> tuple[bool, str]:
    if len(data) < HEADER_BYTES or data[:8] != MAGIC:
        return False, "missing or invalid S3MT header"
    values = struct.unpack_from("<18I", data, 8)
    fields = dict(zip(NAMES, values))
    if fields["version"] != 1:
        return False, "unsupported version"
    if fields["total_bytes"] != len(data):
        return False, "total length mismatch"
    if fields["payload_bytes"] != len(data) - HEADER_BYTES:
        return False, "payload length mismatch"
    if fields["tokenizer_bytes"] > fields["payload_bytes"]:
        return False, "tokenizer length exceeds payload"
    checksum = zlib.crc32(data[HEADER_BYTES:]) & 0xFFFFFFFF
    if checksum != fields["crc32"]:
        return False, "CRC32 mismatch"
    return True, "OK (header/length/CRC32 only; not a full model validation)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    ok, detail = validate(args.model.read_bytes())
    print(detail if ok else f"INVALID: {detail}", file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
