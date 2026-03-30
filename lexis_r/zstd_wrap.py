"""zstd post-compression wrapper for .lexisr payloads.

Usage
-----
Plain (no dictionary):
    from lexis_r.zstd_wrap import compress_payload, decompress_payload
    compressed = compress_payload(msgpack_bytes, level=19)
    original   = decompress_payload(compressed)

With dictionary (train once you have 10+ .lexisr samples):
    from lexis_r.zstd_wrap import train_dictionary, compress_with_dict, decompress_with_dict
    zdict = train_dictionary(list_of_sample_bytes, size=4096)
    zdict.save('lexisr.dict')

    import zstandard as zstd
    zdict  = zstd.ZstdCompressionDict(open('lexisr.dict','rb').read())
    cdata  = compress_with_dict(msgpack_bytes, zdict, level=19)
    orig   = decompress_with_dict(cdata, zdict)

Auto-detection
--------------
The decompressor checks for the zstd magic bytes (0xFD2FB528 LE)
at offset 0. Old plain-msgpack .lexisr files load unchanged.
"""

from __future__ import annotations

_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"  # little-endian 0xFD2FB528

try:
    import zstandard as zstd  # type: ignore
    _ZSTD_AVAILABLE = True
except ImportError:
    _ZSTD_AVAILABLE = False


def _require_zstd() -> None:
    if not _ZSTD_AVAILABLE:
        raise ImportError(
            "zstandard is required for zstd support. "
            "Install with: pip install zstandard"
        )


def is_zstd(data: bytes) -> bool:
    """Return True if data starts with a zstd frame magic."""
    return data[:4] == _ZSTD_MAGIC


def compress_payload(data: bytes, level: int = 19) -> bytes:
    """Compress msgpack payload bytes with zstd (no dictionary)."""
    _require_zstd()
    cctx = zstd.ZstdCompressor(level=level)
    return cctx.compress(data)


def decompress_payload(data: bytes) -> bytes:
    """Decompress a plain zstd-wrapped payload."""
    _require_zstd()
    dctx = zstd.ZstdDecompressor()
    return dctx.decompress(data)


def train_dictionary(samples: list[bytes], size: int = 4096) -> "zstd.ZstdCompressionDict":
    """Train a zstd dictionary from a list of .lexisr payload bytes.

    Recommended: 10+ samples, dict size 2048-8192 bytes.
    The trained dict should be saved and bundled with the codec.
    """
    _require_zstd()
    return zstd.train_dictionary(size, samples)


def compress_with_dict(
    data: bytes,
    zdict: "zstd.ZstdCompressionDict",
    level: int = 19,
) -> bytes:
    """Compress with a pre-trained dictionary for maximum gain on small files."""
    _require_zstd()
    cctx = zstd.ZstdCompressor(level=level, dict_data=zdict)
    return cctx.compress(data)


def decompress_with_dict(
    data: bytes,
    zdict: "zstd.ZstdCompressionDict",
) -> bytes:
    """Decompress with a pre-trained dictionary."""
    _require_zstd()
    dctx = zstd.ZstdDecompressor(dict_data=zdict)
    return dctx.decompress(data)
