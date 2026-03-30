import json, time, os, re
from lexis_r.compress import compress
from lexis_r.decompress import decompress

sizes = [5000, 10000, 20000, 50000, 100000]
text  = open('moby_dick.txt').read().lstrip('\ufeff')
os.makedirs('scale_test_results', exist_ok=True)

def tokens(t):
    return set(re.findall(r"[a-z']+", t.lower()))

for size in sizes:
    chunk = text[:size]
    out   = f'/tmp/moby_r_{size}.lexisr'
    print(f'\n=== SIZE {size} ===')

    t0 = time.time()
    stats = compress(chunk, out)
    t_compress = round(time.time() - t0, 2)

    t1 = time.time()
    restored = decompress(out)
    t_decompress = round(time.time() - t1, 2)

    orig_t  = tokens(chunk)
    rest_t  = tokens(restored)
    overlap = round(len(orig_t & rest_t) / len(orig_t) * 100, 2)

    result = {
        'input_chars':      size,
        'original_bytes':   stats['original_size'],
        'payload_bytes':    stats['payload_size'],
        'char_stream_bpb':  round(stats['bpb'], 4),
        'full_payload_bpb': round(stats['full_payload_bpb'], 4),
        'word_overlap_pct': overlap,
        'restored_len':     len(restored),
        'compress_sec':     t_compress,
        'decompress_sec':   t_decompress,
    }
    fname = f'scale_test_results/lexis_r_{size}.json'
    with open(fname, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))

print('\nAll done.')
