import re

from main import compress_to_file, decompress

original = open("moby_dick.txt").read().lower().lstrip("\ufeff")
compress_to_file(original, "moby_dick.bin")
decompressed = decompress("moby_dick.bin").lower()

orig = re.sub(r"\s+", " ", original).strip()
dec = re.sub(r"\s+", " ", decompressed).strip()

i = next(
    (j for j in range(min(len(orig), len(dec))) if orig[j] != dec[j]),
    min(len(orig), len(dec)),
)
print("First mismatch:", i)
print("ORIG:", orig[i - 40 : i + 80])
print("DEC :", dec[i - 40 : i + 80])
print("Lengths:", len(orig), len(dec))
print("Match:", orig == dec)
