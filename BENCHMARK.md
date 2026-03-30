# Lexis-R Benchmark Results

## Baseline (2026-03-30)

**Test corpus:** `moby_dick.txt` first 5,000 characters  
**Variant:** Lexis-R

| Metric | Lexis-E | Lexis-R baseline |
|---|---|---|
| Full-payload bpb | 23.83 | **15.9984** |
| Char-stream bpb | — | 2.7388 |
| Payload size | — | 10,059 bytes |
| Improvement | — | **−37% vs Lexis-E** |

### Notes
- Roundtrip is semantically correct; cosmetic differences are:
  - BOM (`\ufeff`) stripped by stage1 normalizer
  - Case normalised to lowercase during pipeline
  - Newlines collapsed by `_join_words`
- Context model serialised into payload (no re-training at decompress time)
- `root_lengths` stored as VLQ (no 4-bit clamping)

## Planned improvements
- [ ] zlib-compress model bytes inside payload
- [ ] Improve context mixing weights
- [ ] Tighter pos-delta encoding
- [ ] Compact metadata (reduce structural overhead)
