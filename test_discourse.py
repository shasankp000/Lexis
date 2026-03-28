from compression.pipeline.stage4_discourse import DiscourseAnalyser
from compression.pipeline.stage5_discourse_symbols import decode_symbols, encode_symbols

analyser = DiscourseAnalyser(use_spacy=True)
text = open("moby_dick.txt").read()[:5000]
stage4 = analyser.analyse_document(text)

# Show raw Stage 4 chain data
print("=== Stage 4 chains ===")
for eid, mentions in stage4["coreference_chains"].items():
    print(f"Chain {eid}: {mentions}")

print()

# Show what text surrounds char 1750
print("=== Text around char 1750 ===")
print(repr(text[1720:1800]))

print()

# Encode and show what changed
compressed, table = encode_symbols(text, stage4)
print("=== Symbol table ===")
print(table)

print()

# Find first diff
decoded = decode_symbols(compressed, table)
for i, (a, b) in enumerate(zip(decoded, text)):
    if a != b:
        print(f"First mismatch at {i}:")
        print(f"  ORIG:    {repr(text[i - 30 : i + 30])}")
        print(f"  DECODED: {repr(decoded[i - 30 : i + 30])}")
        print(f"  COMPRESSED at same pos: {repr(compressed[i - 30 : i + 30])}")
        break
