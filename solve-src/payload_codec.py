import zlib
from typing import List

MAGIC = b"VR"
VERSION = 1


def bytes_to_bits(data: bytes) -> List[int]:
    bits = []
    for b in data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits


def bits_to_bytes(bits: List[int]) -> bytes:
    usable = len(bits) // 8 * 8
    bits = bits[:usable]
    out = bytearray()
    for i in range(0, len(bits), 8):
        val = 0
        for b in bits[i:i + 8]:
            val = (val << 1) | b
        out.append(val)
    return bytes(out)


def bits_to_int(bits: List[int]) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | b
    return v


def int_to_bits(v: int, width: int) -> List[int]:
    return [(v >> i) & 1 for i in range(width - 1, -1, -1)]


def text_to_canonical_values(text: str) -> List[int]:
    body = text.encode("utf-8")
    if len(body) > 255:
        raise ValueError("text too long")

    core = MAGIC + bytes([VERSION, len(body)]) + body
    crc = zlib.crc32(core) & 0xFFFFFFFF
    payload = core + crc.to_bytes(4, "big")

    bits = bytes_to_bits(payload)
    while len(bits) % 5 != 0:
        bits.append(0)

    values = [bits_to_int(bits[i:i + 5]) for i in range(0, len(bits), 5)]
    return values


def canonical_values_to_bytes(values: List[int]) -> bytes:
    bits = []
    for v in values:
        if not (0 <= v < 32):
            raise ValueError("bad canonical value")
        bits.extend(int_to_bits(v, 5))
    return bits_to_bytes(bits)


def canonical_values_to_text(values: List[int]) -> str:
    raw = canonical_values_to_bytes(values)

    if len(raw) < 2 + 1 + 1 + 4:
        raise ValueError("payload too short")

    if raw[:2] != MAGIC:
        raise ValueError("bad magic")

    version = raw[2]
    if version != VERSION:
        raise ValueError("bad version")

    n = raw[3]
    need = 2 + 1 + 1 + n + 4
    if len(raw) < need:
        raise ValueError("truncated payload")

    core = raw[:4 + n]
    body = raw[4:4 + n]
    crc_expected = int.from_bytes(raw[4 + n:4 + n + 4], "big")
    crc_actual = zlib.crc32(core) & 0xFFFFFFFF

    if crc_expected != crc_actual:
        raise ValueError("bad crc")

    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError("body is not valid UTF-8") from e

    return text
