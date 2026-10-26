import struct
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_model.py"


def fake_header(payload: bytes) -> bytes:
    vals = [1, 0, 260, 32, 64, 4, 1, 1, 24, 32,
            0, 1, 2, 3, 0, len(payload), zlib.crc32(payload) & 0xFFFFFFFF,
            128 + len(payload)]
    return struct.pack("<8s18I48s", b"S3MT0001", *vals, b"\0" * 48)


def test_validator_exits_nonzero_on_corrupt_payload(tmp_path):
    model = tmp_path / "test.s3mt"
    payload = b"abc"
    model.write_bytes(fake_header(payload) + payload)
    ok = subprocess.run([sys.executable, str(SCRIPT), str(model)], capture_output=True, text=True)
    assert ok.returncode == 0
    model.write_bytes(fake_header(payload) + b"abd")
    bad = subprocess.run([sys.executable, str(SCRIPT), str(model)], capture_output=True, text=True)
    assert bad.returncode != 0
    assert "CRC32 mismatch" in bad.stderr


def test_validator_rejects_wrong_total_length(tmp_path):
    model = tmp_path / "wrong.s3mt"
    model.write_bytes(fake_header(b"") + b"extra")
    result = subprocess.run([sys.executable, str(SCRIPT), str(model)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "total length" in result.stderr
