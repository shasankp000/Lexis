"""LZ77-style pointer encoding for the POS tag token stream.

Operates on the FLAT integer sequence of POS tag ids across all sentences
(token-level, not byte-level). A sliding window back-reference replaces
any repeated run of >= MIN_MATCH tokens with a (distance, length) pointer.

Wire format (flat bytes, self-delimiting):
  Each element is one of:
    Literal  : 0x00  vlq(tag_id)          (2B minimum)
    Backref  : 0xFF  vlq(dist) vlq(run)   (3B minimum, wins at run >= 3)

Sentence boundaries are stored separately as a uint8 list of per-sentence
token counts (already present in the payload as packed_pos_n_tags).
The decoder uses those counts to reconstruct the nested list.

Search strategy: O(n*w) greedy longest-match with window=WINDOW_SIZE.
For 5k chars (~1250 tokens) this is fast enough at encode time.

Constants
---------
  WINDOW_SIZE  : how far back to search for matches (tokens)
  MIN_MATCH    : minimum run length for a pointer to save bytes
                 pointer = 1+vlq(dist)+vlq(run) bytes
                 literals = MIN_MATCH * 2 bytes
                 break-even at run=3: 1+1+1=3B vs 3*2=6B  -> save 3B
"""

from __future__ import annotations

from typing import List, Tuple

WINDOW_SIZE = 1024   # tokens
MIN_MATCH   = 3      # tokens


# ---------------------------------------------------------------------------
# VLQ helpers (self-contained)
# ---------------------------------------------------------------------------

def _vlq_enc(v: int) -> bytes:
    assert v >= 0
    groups: List[int] = []
    while True:
        groups.append(v & 0x7F)
        v >>= 7
        if v == 0:
            break
    groups.reverse()
    return bytes([(g | 0x80) if i < len(groups) - 1 else g
                  for i, g in enumerate(groups)])


def _vlq_dec(data: bytes, off: int) -> Tuple[int, int]:
    v = 0
    while True:
        b = data[off]; off += 1
        v = (v << 7) | (b & 0x7F)
        if not (b & 0x80):
            break
    return v, off


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

def encode_lz77(tokens: List[int]) -> bytes:
    """
    LZ77-encode a flat list of POS tag integer ids.
    Returns raw bytes (flag + vlq fields, no length prefix).
    """
    out  = bytearray()
    i    = 0
    n    = len(tokens)

    while i < n:
        # Search backward in the sliding window for the longest match
        win_start  = max(0, i - WINDOW_SIZE)
        best_dist  = 0
        best_len   = 0

        # Only search if there are enough tokens ahead for MIN_MATCH
        if i + MIN_MATCH <= n:
            for j in range(win_start, i):
                # How long does this match run?
                run = 0
                while (
                    i + run < n
                    and tokens[j + run] == tokens[i + run]
                    and run < 255          # cap run to keep vlq 1B
                ):
                    run += 1
                    # Prevent j+run from going past i (overlapping ok in LZ77
                    # but only up to distance boundary)
                    if j + run >= i:
                        # overlapping match — still valid in LZ77
                        pass
                if run >= MIN_MATCH and run > best_len:
                    best_len  = run
                    best_dist = i - j

        if best_len >= MIN_MATCH:
            out += b"\xFF"
            out += _vlq_enc(best_dist)
            out += _vlq_enc(best_len)
            i   += best_len
        else:
            out += b"\x00"
            out += _vlq_enc(tokens[i])
            i   += 1

    return bytes(out)


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def decode_lz77(data: bytes, n_tokens: int) -> List[int]:
    """
    Decode LZ77-encoded POS tag stream back to flat integer list.
    n_tokens: expected total number of tokens (for safety check).
    """
    tokens: List[int] = []
    off = 0

    while off < len(data) and len(tokens) < n_tokens:
        flag = data[off]; off += 1
        if flag == 0x00:
            tag, off = _vlq_dec(data, off)
            tokens.append(tag)
        elif flag == 0xFF:
            dist, off = _vlq_dec(data, off)
            run,  off = _vlq_dec(data, off)
            start = len(tokens) - dist
            for k in range(run):
                tokens.append(tokens[start + k])
        else:
            # Corrupt stream — skip
            break

    return tokens


# ---------------------------------------------------------------------------
# Nested list helpers
# ---------------------------------------------------------------------------

def pack_pos_tags_lz77(
    sentences: List[List[str]],
    pos_to_idx: dict,
) -> bytes:
    """
    Encode nested POS tag strings to LZ77 bytes.
    Sentence structure is implicit via token counts stored in packed_pos_n_tags.
    """
    flat = [pos_to_idx.get(tag, 17) for sent in sentences for tag in sent]
    return encode_lz77(flat)


def unpack_pos_tags_lz77(
    data: bytes,
    n_tags_per_sent: List[int],
    pos_vocab: List[str],
) -> List[List[str]]:
    """
    Decode LZ77 bytes back to nested POS tag string lists.
    n_tags_per_sent: list of token counts per sentence (from packed_pos_n_tags).
    """
    n_total = sum(n_tags_per_sent)
    flat    = decode_lz77(data, n_total)
    result: List[List[str]] = []
    idx = 0
    for count in n_tags_per_sent:
        sent = [pos_vocab[flat[idx + k]] if (idx + k) < len(flat) else "X"
                for k in range(count)]
        result.append(sent)
        idx += count
    return result
