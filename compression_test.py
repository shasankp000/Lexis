"""
Compression Pipeline - Dummy Test Script
Tests three core ideas:
  1. Character transition graph
  2. Delta encoding
  3. Factoriadic encoding

Uses toy text data, CPU only, no external dependencies.
"""

from collections import defaultdict
import math

# ─────────────────────────────────────────────
# TOY DATA
# ─────────────────────────────────────────────

TEXT = """
the old man walked slowly home
the tired cat sat on the mat
the young dog ran quickly away
the small bird flew silently past
"""

# ─────────────────────────────────────────────
# TEST 1 — CHARACTER TRANSITION GRAPH
# ─────────────────────────────────────────────
# We build a graph where each node is a character,
# and edges represent "what character tends to follow this one".
# Then we try to reconstruct words procedurally from just their
# first character, following the most probable path.

def build_transition_graph(text):
    """
    For every character in the text, count how often
    each other character follows it.
    Returns a dict of { char -> { next_char -> count } }

    Special symbols used:
      '^' = start of word
      '$' = end of word
      '_' = whitespace (space between words)
      '.' = generic punctuation placeholder
    """
    graph = defaultdict(lambda: defaultdict(int))

    # Normalize punctuation to a single '.' symbol
    # so rare punctuation doesn't fragment the graph
    import re
    normalized = re.sub(r'[^\w\s]', '.', text)

    lines = normalized.strip().split('\n')
    for line in lines:
        words = line.strip().split(' ')
        words = [w for w in words if w]
        for idx, word in enumerate(words):
            # Add start and end markers per word
            word = "^" + word + "$"
            for i in range(len(word) - 1):
                current = word[i]
                next_char = word[i + 1]
                graph[current][next_char] += 1
            # After end of word, whitespace follows
            # (unless it's the last word on the line)
            if idx < len(words) - 1:
                graph['$']['_'] += 1   # word end → space
                graph['_']['^'] += 1   # space → next word start
    return graph


def normalize_graph(graph):
    """
    Convert raw counts to probabilities.
    { char -> { next_char -> probability } }
    """
    prob_graph = {}
    for char, transitions in graph.items():
        total = sum(transitions.values())
        prob_graph[char] = {
            next_char: count / total
            for next_char, count in transitions.items()
        }
    return prob_graph


def procedural_generate(prob_graph, start_char, max_len=15):
    """
    Given a starting character, follow the most probable
    transition at each step until we hit end marker '$'.
    This is procedural word generation — no explicit storage
    of each character, just follow the chain.
    """
    result = start_char
    current = start_char
    for _ in range(max_len):
        if current not in prob_graph:
            break
        transitions = prob_graph[current]
        # Follow the most probable next character
        next_char = max(transitions, key=transitions.get)
        if next_char == "$":
            break
        result += next_char
        current = next_char
    return result


def bits_for_char(prob_graph, char, prev_char):
    """
    How many bits does it cost to encode this character
    given the previous character?
    Cost = -log2(probability)
    Higher probability = fewer bits needed.

    Unknown characters fall back gracefully:
      - Unknown context → assume uniform over known alphabet
      - Unknown transition → assume rare (1 in 1000)
    """
    if prev_char not in prob_graph:
        alphabet_size = len(prob_graph) or 1
        return math.log2(alphabet_size)  # uniform distribution fallback
    transitions = prob_graph[prev_char]
    if char not in transitions:
        return math.log2(1000)  # rare but not impossible — ~10 bits
    prob = transitions[char]
    return -math.log2(prob)


print("=" * 55)
print("TEST 1 — CHARACTER TRANSITION GRAPH")
print("=" * 55)

graph = build_transition_graph(TEXT)
prob_graph = normalize_graph(graph)

# Show the transitions from '^' (word start)
print("\nMost probable first characters after word start (^):")
start_transitions = prob_graph.get("^", {})
sorted_starts = sorted(start_transitions.items(), key=lambda x: -x[1])
for char, prob in sorted_starts:
    bar = "█" * int(prob * 40)
    print(f"  '{char}' : {prob:.2f}  {bar}")

# Try procedural generation from each starting character
print("\nProcedural generation from each start character:")
for start_char, _ in sorted_starts:
    generated = procedural_generate(prob_graph, start_char)
    print(f"  Starting with '{start_char}' → '{generated}'")

# Show bit cost for encoding each word character by character
print("\nBit cost per word (lower = more compressible):")
words = [w for w in TEXT.split() if w]
for word in sorted(set(words)):
    total_bits = 0
    word_with_markers = "^" + word + "$"
    for i in range(1, len(word_with_markers)):
        cost = bits_for_char(
            prob_graph,
            word_with_markers[i],
            word_with_markers[i-1]
        )
        total_bits += cost
    bits_per_char = total_bits / len(word)
    print(f"  '{word}' : {total_bits:.2f} total bits  "
          f"({bits_per_char:.2f} bits/char)")


# ─────────────────────────────────────────────
# TEST 2 — DELTA ENCODING
# ─────────────────────────────────────────────
# Instead of storing symbol values absolutely,
# store the difference (delta) between consecutive values.
# If symbols appear in predictable sequences, deltas
# will be small and clustered around zero — much cheaper to encode.

def assign_symbol_ids(prob_graph):
    """
    Assign a numeric ID to each character based on
    frequency — most common characters get lowest IDs.
    This is your abc -> 123 mapping (Idea 1).
    """
    # Count total occurrences across all transitions
    freq = defaultdict(int)
    for transitions in prob_graph.values():
        for char, prob in transitions.items():
            freq[char] += prob  # use prob as proxy for frequency
    # Sort by frequency descending, assign IDs
    sorted_chars = sorted(freq.items(), key=lambda x: -x[1])
    symbol_map = {char: idx for idx, (char, _) in enumerate(sorted_chars)}
    reverse_map = {idx: char for char, idx in symbol_map.items()}
    return symbol_map, reverse_map


def delta_encode(sequence):
    """
    Convert absolute symbol IDs to deltas.
    First value stored as-is, rest stored as difference
    from previous value.
    """
    if not sequence:
        return []
    deltas = [sequence[0]]
    for i in range(1, len(sequence)):
        deltas.append(sequence[i] - sequence[i-1])
    return deltas


def delta_decode(deltas):
    """
    Reconstruct absolute values from deltas.
    Perfectly reversible.
    """
    if not deltas:
        return []
    sequence = [deltas[0]]
    for i in range(1, len(deltas)):
        sequence.append(sequence[-1] + deltas[i])
    return sequence


print("\n" + "=" * 55)
print("TEST 2 — DELTA ENCODING")
print("=" * 55)

symbol_map, reverse_map = assign_symbol_ids(prob_graph)

print("\nSymbol ID mapping (most frequent = lowest ID):")
for char, idx in sorted(symbol_map.items(), key=lambda x: x[1])[:10]:
    print(f"  '{char}' → {idx}")

# Encode a sample sentence as absolute IDs then as deltas
# We use our special symbols: '^' word start, '$' word end, '_' space
# Representing "the old man" as the graph would see it:
# ^the$ _ ^old$ _ ^man$
sample_raw = "the old man"
sample_words = sample_raw.split()
sample_symbols = []
for i, word in enumerate(sample_words):
    sample_symbols.append('^')
    sample_symbols.extend(list(word))
    sample_symbols.append('$')
    if i < len(sample_words) - 1:
        sample_symbols.append('_')  # whitespace symbol

print(f"\nSample text: '{sample_raw}'")
print(f"As symbol sequence: {sample_symbols}")

abs_ids = [symbol_map.get(c, len(symbol_map)) for c in sample_symbols]
deltas = delta_encode(abs_ids)
recovered_ids = delta_decode(deltas)
recovered_symbols = [reverse_map.get(i, '?') for i in recovered_ids]

# Reconstruct readable text from symbol sequence
recovered_text = ""
for sym in recovered_symbols:
    if sym == '^' or sym == '$':
        pass         # markers are invisible in output
    elif sym == '_':
        recovered_text += ' '
    else:
        recovered_text += sym

print(f"\nAbsolute IDs : {abs_ids}")
print(f"Delta encoded: {deltas}")
print(f"Recovered IDs: {recovered_ids}")
print(f"Recovered text: '{recovered_text}'")

# Compare the range of values — smaller range = cheaper to encode
abs_range = max(abs_ids) - min(abs_ids)
delta_range = max(deltas) - min(deltas)
print(f"\nAbsolute ID range : {abs_range}  (larger = more bits needed)")
print(f"Delta range       : {delta_range}  (smaller = fewer bits needed)")
print(f"Range reduction   : {abs_range - delta_range} "
      f"({((abs_range-delta_range)/abs_range*100):.1f}% smaller)")


# ─────────────────────────────────────────────
# TEST 3 — FACTORIADIC ENCODING
# ─────────────────────────────────────────────
# Represent numbers in the factorial number system.
# Small numbers (common patterns) get very compact representations.
# Large numbers (rare patterns) cost more but are proportionally rare.

def to_factoriadic(n):
    """
    Convert integer n to factoriadic representation.
    Returns list of digits [d_k, ..., d_2, d_1, d_0]
    where d_i < i+1 and n = sum(d_i * i!)
    """
    if n == 0:
        return [0]
    digits = []
    i = 1
    while n > 0:
        digits.append(n % i)
        n //= i
        i += 1
    return list(reversed(digits))


def from_factoriadic(digits):
    """
    Convert factoriadic digits back to integer.
    Perfectly reversible.
    """
    digits = list(reversed(digits))
    result = 0
    for i, d in enumerate(digits):
        result += d * math.factorial(i)
    return result


def factoriadic_bit_cost(n):
    """
    Estimate how many bits it takes to store
    the factoriadic representation of n.
    Approximation: log2(n!) grows slower than log2(2^n)
    so factorial encoding is efficient for small n.
    """
    if n == 0:
        return 1
    return math.log2(math.factorial(len(to_factoriadic(n))))


print("\n" + "=" * 55)
print("TEST 3 — FACTORIADIC ENCODING")
print("=" * 55)

print("\nNumber → Factoriadic → Recovered:")
test_numbers = [0, 1, 2, 3, 5, 7, 10, 15, 20, 50, 100]
for n in test_numbers:
    fact_digits = to_factoriadic(n)
    recovered = from_factoriadic(fact_digits)
    standard_bits = math.ceil(math.log2(n + 1)) if n > 0 else 1
    print(f"  {n:4d} → {str(fact_digits):25s} → {recovered:4d}  "
          f"(standard: {standard_bits} bits)")

# Now apply factoriadic to the delta values from Test 2
print(f"\nApplying factoriadic to delta sequence from Test 2:")
print(f"Deltas: {deltas}")
print()
for d in deltas:
    abs_d = abs(d)  # factoriadic works on positive integers
    sign = "-" if d < 0 else "+"
    fact = to_factoriadic(abs_d)
    recovered = from_factoriadic(fact)
    print(f"  delta {sign}{abs_d:2d} → factoriadic {str(fact):15s}"
          f" → recovered {sign}{recovered}")


# ─────────────────────────────────────────────
# COMBINED SUMMARY
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("COMBINED SUMMARY")
print("=" * 55)
print("""
Pipeline flow demonstrated:

  Raw text
    ↓
  [Graph] Character transitions built
          Common words predictable from start char alone
    ↓
  [Delta] Absolute symbol IDs → small delta values
          Range shrinks significantly
    ↓
  [Factoriadic] Small deltas → compact factorial representation
                Common patterns cost very few bits
    ↓
  Compressed output

Each stage makes the next stage's job easier.
The combination is more powerful than any single stage alone.
""")
