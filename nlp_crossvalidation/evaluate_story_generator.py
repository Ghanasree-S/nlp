"""
Story Generator Evaluation — 5-Fold Cross-Validation

Evaluates the N-gram Markov Chain story generator using 5 evaluation metrics:
1. Keyword Coverage    — % of input keywords incorporated into generated story
2. Fluency Score       — Bigram transition probability (how natural the text flows)
3. Lexical Diversity   — Type-Token Ratio (vocabulary richness)
4. Narrative Structure — Opening/middle/closing arc completeness
5. Grammatical Quality — % of well-formed sentences (proper punctuation, length)

Each fold uses a different set of keyword prompts to evaluate generation quality.
"""

import os
import sys
import re
import json
import random
import numpy as np
from collections import Counter

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from story_gen.story_generator import StoryGenerator


# ─── Test Keyword Sets (5 folds × 5 prompts each) ───────────────────────────

FOLD_KEYWORDS = [
    # Fold 1: Classic fairy tale themes
    [
        ["princess", "dragon", "castle", "sword"],
        ["forest", "wolf", "grandmother", "basket"],
        ["magic", "mirror", "queen", "beauty"],
        ["giant", "beanstalk", "golden", "harp"],
        ["witch", "oven", "children", "candy"],
    ],
    # Fold 2: Adventure themes
    [
        ["pirate", "treasure", "island", "ship"],
        ["knight", "quest", "mountain", "fire"],
        ["wizard", "spell", "tower", "crystal"],
        ["sailor", "storm", "sea", "whale"],
        ["thief", "diamond", "palace", "guard"],
    ],
    # Fold 3: Nature themes
    [
        ["river", "fish", "bear", "mountain"],
        ["garden", "butterfly", "flower", "rain"],
        ["eagle", "cliff", "wind", "sky"],
        ["deer", "snow", "winter", "fire"],
        ["lion", "savanna", "hunt", "sunrise"],
    ],
    # Fold 4: Emotional/character themes
    [
        ["brave", "warrior", "battle", "victory"],
        ["kind", "orphan", "village", "hope"],
        ["wise", "elder", "journey", "truth"],
        ["lonely", "wanderer", "star", "home"],
        ["curious", "child", "secret", "door"],
    ],
    # Fold 5: Mixed/complex themes
    [
        ["enchanted", "lake", "swan", "moonlight"],
        ["ancient", "scroll", "prophecy", "destiny"],
        ["shadow", "lantern", "courage", "darkness"],
        ["golden", "ring", "dwarf", "forge"],
        ["sleeping", "thorn", "prince", "kiss"],
    ],
]


def evaluate_keyword_coverage(keywords, story_text):
    """Metric 1: What fraction of input keywords appear in the generated story."""
    story_lower = story_text.lower()
    found = sum(1 for kw in keywords if kw.lower() in story_lower)
    return found / len(keywords) if keywords else 0.0


def evaluate_fluency(generator, story_text):
    """Metric 2: Average bigram transition probability from the trained model."""
    words = story_text.lower().split()
    if len(words) < 2:
        return 0.0

    valid_transitions = 0
    total_transitions = 0

    for i in range(len(words) - 1):
        w = words[i].strip('.,!?;:"\'-()[]')
        if w in generator.bigrams:
            total_transitions += 1
            next_w = words[i + 1].strip('.,!?;:"\'-()[]')
            if next_w in [x.lower().strip('.,!?;:"\'-()[]') for x in generator.bigrams[w]]:
                valid_transitions += 1

    return valid_transitions / total_transitions if total_transitions > 0 else 0.0


def evaluate_lexical_diversity(story_text):
    """Metric 3: Type-Token Ratio — unique words / total words."""
    words = re.findall(r'\b[a-zA-Z]+\b', story_text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def evaluate_narrative_structure(story_text):
    """Metric 4: Check for opening, middle development, and closing structure."""
    sentences = re.split(r'(?<=[.!?])\s+', story_text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    if len(sentences) < 3:
        return 0.3

    score = 0.0

    # Check opening (first sentence should have story-like beginning)
    opening_patterns = ['once', 'long ago', 'there was', 'there lived',
                        'in a', 'a long', 'one day', 'upon a time']
    first_lower = sentences[0].lower()
    if any(p in first_lower for p in opening_patterns):
        score += 0.30
    elif sentences[0][0].isupper() and len(sentences[0].split()) >= 5:
        score += 0.20  # At least a proper sentence

    # Check middle development (multiple sentences with varying content)
    if len(sentences) >= 5:
        score += 0.25
    elif len(sentences) >= 3:
        score += 0.15

    # Check middle has some variety (not all same words)
    middle = sentences[1:-1] if len(sentences) > 2 else sentences
    middle_text = " ".join(middle).lower()
    middle_words = re.findall(r'\b[a-zA-Z]+\b', middle_text)
    if middle_words:
        middle_diversity = len(set(middle_words)) / len(middle_words)
        score += min(0.20, middle_diversity * 0.25)

    # Check closing (last sentence has concluding markers)
    closing_patterns = ['and so', 'in the end', 'finally', 'thus',
                        'from that day', 'at last', 'ever after',
                        'the end', 'happily']
    last_lower = sentences[-1].lower()
    if any(p in last_lower for p in closing_patterns):
        score += 0.25
    elif sentences[-1].endswith('.'):
        score += 0.10

    return min(1.0, score)


def evaluate_grammatical_quality(story_text):
    """Metric 5: % of sentences with proper structure (capitalization, punctuation, length)."""
    sentences = re.split(r'(?<=[.!?])\s+', story_text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 3]

    if not sentences:
        return 0.0

    good_count = 0
    for sent in sentences:
        is_good = True
        # Check: starts with capital letter
        if not sent[0].isupper():
            is_good = False
        # Check: ends with punctuation
        if not sent.rstrip()[-1] in '.!?':
            is_good = False
        # Check: reasonable length (4-40 words)
        word_count = len(sent.split())
        if word_count < 4 or word_count > 50:
            is_good = False
        if is_good:
            good_count += 1

    return good_count / len(sentences)


def run_evaluation():
    """Run 5-fold evaluation of story generator."""
    print("=" * 60)
    print(" Story Generator — 5-Fold Evaluation")
    print("=" * 60)

    # Initialize generator
    print("\nLoading story generator...")
    generator = StoryGenerator()

    if not generator._trained:
        print("ERROR: Story generator model not trained!")
        return None

    print(f"Model loaded: {len(generator.bigrams)} bigram keys, "
          f"{len(generator.trigrams)} trigram keys\n")

    all_fold_results = []

    for fold_idx, keyword_sets in enumerate(FOLD_KEYWORDS):
        fold_num = fold_idx + 1
        print(f"--- Fold {fold_num} ---")

        fold_metrics = {
            "keyword_coverage": [],
            "fluency_score": [],
            "lexical_diversity": [],
            "narrative_structure": [],
            "grammatical_quality": [],
        }

        for kw_set in keyword_sets:
            # Generate story
            result = generator.generate(kw_set, num_sentences=12)
            story = result.get("story", "")

            if not story:
                continue

            # Compute all 5 metrics
            kc = evaluate_keyword_coverage(kw_set, story)
            fl = evaluate_fluency(generator, story)
            ld = evaluate_lexical_diversity(story)
            ns = evaluate_narrative_structure(story)
            gq = evaluate_grammatical_quality(story)

            fold_metrics["keyword_coverage"].append(kc)
            fold_metrics["fluency_score"].append(fl)
            fold_metrics["lexical_diversity"].append(ld)
            fold_metrics["narrative_structure"].append(ns)
            fold_metrics["grammatical_quality"].append(gq)

        # Average per fold
        fold_avg = {}
        for metric, values in fold_metrics.items():
            fold_avg[metric] = np.mean(values) if values else 0.0

        all_fold_results.append({
            "fold": fold_num,
            **{k: round(v, 4) for k, v in fold_avg.items()}
        })

        print(f"  Keyword Coverage:    {fold_avg['keyword_coverage']:.4f}")
        print(f"  Fluency Score:       {fold_avg['fluency_score']:.4f}")
        print(f"  Lexical Diversity:   {fold_avg['lexical_diversity']:.4f}")
        print(f"  Narrative Structure: {fold_avg['narrative_structure']:.4f}")
        print(f"  Grammatical Quality: {fold_avg['grammatical_quality']:.4f}")
        print()

    # Overall averages
    avg_metrics = {}
    for metric in ["keyword_coverage", "fluency_score", "lexical_diversity",
                    "narrative_structure", "grammatical_quality"]:
        values = [f[metric] for f in all_fold_results]
        avg_metrics[metric] = round(np.mean(values), 4)

    print("=== OVERALL AVERAGES ===")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save results
    results = {
        "model": "Story Generator (N-gram Markov Chain)",
        "corpus": "Fairy Tales (cleaned_merged_fairy_tales_without_eos.txt)",
        "n_folds": 5,
        "prompts_per_fold": 5,
        "fold_results": all_fold_results,
        "avg_metrics": avg_metrics,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/story_generator_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/story_generator_results.json")

    return results


if __name__ == "__main__":
    run_evaluation()
