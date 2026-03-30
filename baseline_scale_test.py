import json, time, os, gzip
import zstandard as zstd

sizes = [5000, 10000, 20000, 50000, 100000]
text  = open('moby_dick.txt').read().lstrip('\ufeff')
os.makedirs('scale_test_results', exist_ok=True)

cctx = zstd.ZstdCompressor(level=19)
dctx = zstd.ZstdDecompressor()

for size in sizes:
    chunk = text[:size].encode('utf-8')
    orig_bytes = len(chunk)

    # gzip level 9
    t0 = time.time()
    gz_data = gzip.compress(chunk, compresslevel=9)
    gz_compress = round(time.time() - t0, 4)
    t1 = time.time()
    gzip.decompress(gz_data)
    gz_decompress = round(time.time() - t1, 4)

    # zstd level 19
    t2 = time.time()
    zst_data = cctx.compress(chunk)
    zst_compress = round(time.time() - t2, 4)
    t3 = time.time()
    dctx.decompress(zst_data)
    zst_decompress = round(time.time() - t3, 4)

    result = {
        'input_chars':        size,
        'original_bytes':     orig_bytes,
        'gzip_bytes':         len(gz_data),
        'gzip_bpb':           round(len(gz_data) * 8 / orig_bytes, 4),
        'gzip_compress_sec':  gz_compress,
        'gzip_decompress_sec':gz_decompress,
        'zstd_bytes':         len(zst_data),
        'zstd_bpb':           round(len(zst_data) * 8 / orig_bytes, 4),
        'zstd_compress_sec':  zst_compress,
        'zstd_decompress_sec':zst_decompress,
    }
    fname = f'scale_test_results/baseline_{size}.json'
    with open(fname, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))

print('\nAll done.')
