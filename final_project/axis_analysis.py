import numpy as np
from gensim.models import Word2Vec
import json
from axes_config import axes
import os

model_a = Word2Vec.load("models/model_a.w2v")
model_b = Word2Vec.load("models/model_b.w2v")

def neighbor_diagnostic(model, name):
    print(f"\n=== {name} Neighbors ===")

    test_words = ["heaven", "death", "king", "body"]

    for w in test_words:
        if w in model.wv:
            sims = model.wv.most_similar(w, topn=5)
            print(f"{w:>8} → {[x[0] for x in sims]}")

def cosine_diagnostic(model, name):
    print(f"\n=== {name} Cosine Distribution ===")

    words = list(model.wv.key_to_index)[:1000]
    sims = []

    for i in range(len(words)-1):
        sims.append(model.wv.similarity(words[i], words[i+1]))

    mean_sim = np.mean(sims)
    print(f"Mean cosine similarity: {mean_sim:.4f}")

def axis_quality_check(model, axes):
    print("\n=== AXIS QUALITY CHECK ===")

    for name, config in axes.items():
        valid_p1 = [w for w in config["pole1"] if w in model.wv]
        valid_p2 = [w for w in config["pole2"] if w in model.wv]

        if len(valid_p1) < 2 or len(valid_p2) < 2:
            print(f"{name}: ⚠️ weak (too many missing anchors)")
        else:
            print(f"{name}: OK ({len(valid_p1)} vs {len(valid_p2)} anchors)")

# --- diagnostics ---

neighbor_diagnostic(model_a, "Corpus A")
neighbor_diagnostic(model_b, "Corpus B")

cosine_diagnostic(model_a, "Corpus A")
cosine_diagnostic(model_b, "Corpus B")

axis_quality_check(model_a, axes)
axis_quality_check(model_b, axes)

def pole_score(model, word, anchors):
    """Average cosine similarity from word to all available anchor words."""
    scores = []
    missing = []
    anchors = [a for a in anchors if a in model.wv]
    for anchor in anchors:
        if anchor in model.wv and word in model.wv:
            scores.append(model.wv.similarity(word, anchor))
        else:
            missing.append(anchor)
    if missing:
        print(f"  [!] missing anchors: {missing}")
    return float(np.mean(scores)) if scores else None

def analyze_word(word, config, model_a, model_b):
    p1, p2 = config["pole1_name"], config["pole2_name"]
    print(f"\n{'='*55}")
    print(f"  WORD: '{word.upper()}'")
    print(f"  Axis: [{p1}] <-----> [{p2}]")
    print(f"{'='*55}")

    results = {}
    for label, model in [("Corpus A (translated)", model_a),
                          ("Corpus B (original)",  model_b)]:

        if word not in model.wv:
            print(f"\n  {label}: '{word}' not in vocabulary — skipping")
            results[label] = None
            continue

        s1 = pole_score(model, word, config["pole1"])
        s2 = pole_score(model, word, config["pole2"])

        if s1 is None or s2 is None:
            results[label] = None
            continue

        diff = s1 - s2
        lean = p1 if diff > 0 else p2
        strength = abs(diff)

        results[label] = {
            "pole1": float(s1),
            "pole2": float(s2),
            "diff": float(diff)
        }

        print(f"\n  {label}:")
        print(f"    {p1:<20} score: {s1:.4f}")
        print(f"    {p2:<20} score: {s2:.4f}")
        print(f"    difference:          {diff:+.4f}")
        print(f"    --> leans [{lean}]  (strength: {strength:.4f})")

        # Top 5 neighbors for qualitative texture
        neighbors = [w for w, _ in model.wv.most_similar(word, topn=8)
                     if w not in config["pole1"] + config["pole2"]][:5]
        print(f"    top neighbors: {neighbors}")

    # Cross-corpus comparison
    a = results.get("Corpus A (translated)")
    b = results.get("Corpus B (original)")
    if a and b:
        shift = a["diff"] - b["diff"]
        print(f"\n  AXIS SHIFT (A minus B): {shift:+.4f}")
        if abs(shift) < 0.02:
            print(f"  --> minimal difference between corpora")
        elif shift > 0:
            print(f"  --> Corpus A pulls '{word}' more toward [{p1}]")
        else:
            print(f"  --> Corpus B pulls '{word}' more toward [{p1}]")

    return results

all_results = {}
for word, config in axes.items():
    all_results[word] = analyze_word(word, config, model_a, model_b)

# Summary table
print(f"\n\n{'='*55}")
print("  SUMMARY — axis lean by corpus")
print(f"{'='*55}")
print(f"  {'word':<12} {'A leans':<20} {'B leans':<20} {'shift':>8}")
print(f"  {'-'*58}")
for word, config in axes.items():
    res = all_results[word]
    a = res.get("Corpus A (translated)")
    b = res.get("Corpus B (original)")
    p1, p2 = config["pole1_name"], config["pole2_name"]
    a_lean = p1 if (a and a["diff"] > 0) else p2 if a else "N/A"
    b_lean = p1 if (b and b["diff"] > 0) else p2 if b else "N/A"
    shift = (a["diff"] - b["diff"]) if (a and b) else float("nan")
    print(f"  {word:<12} {a_lean:<20} {b_lean:<20} {shift:>+8.4f}")

os.makedirs("results", exist_ok=True)
with open("results/axis_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("Saved results to results/axis_results.json")