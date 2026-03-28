from compression.pipeline.stage4_discourse import DiscourseAnalyser
from compression.pipeline.stage5_discourse_symbols import validate_round_trip

analyser = DiscourseAnalyser(use_spacy=True)
text = open("moby_dick.txt").read()[:5000]
stage4 = analyser.analyse_document(text)

result = validate_round_trip(text, stage4)

print(f"Round-trip OK:     {result['round_trip_ok']}")
print(f"Original tokens:   {result['original_tokens']}")
print(f"Compressed tokens: {result['compressed_tokens']}")
print(f"Reduction:         {result['reduction_pct']}%")
print(f"Symbols used:      {result['symbols_used']}")
print(f"Symbol table:      {result['symbol_table']}")
if not result["round_trip_ok"]:
    idx = result["first_mismatch_idx"]
    print(f"\nFirst mismatch at char {idx}:")
    print(f"  ORIG: ...{text[idx - 20 : idx + 20]}...")
