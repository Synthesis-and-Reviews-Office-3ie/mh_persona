"""
M3c: Context Erosion Conversation Quality Analysis
====================================================
Reviewer concern: Are the 100-turn synthetic conversations producing
meaningful, diverse dialogue, or are they repetitive/degenerate?

Analyses:
1. Response diversity: unique n-grams, self-BLEU (intra-conversation)
2. Topic progression: do conversations follow the prescribed phases?
3. Response repetition: exact/near-duplicate detection across turns
4. User-message quality: are GPT-4o-mini user messages diverse?
5. Example excerpts: representative conversation fragments

Output: Quality report + supplementary table + example excerpts
"""

import json
import os
import re
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
MODELS = ["llama3", "qwen2", "mistral"]

CONVERSATION_PHASES = {
    (1, 25): "general_chat",
    (26, 50): "mild_emotional",
    (51, 75): "deeper_issues",
    (76, 100): "crisis_signals",
}


def get_phase(turn_num):
    for (lo, hi), phase in CONVERSATION_PHASES.items():
        if lo <= turn_num <= hi:
            return phase
    return "unknown"


def tokenize(text):
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r'\b\w+\b', text.lower())


def get_ngrams(tokens, n):
    """Extract n-grams from token list."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def jaccard_similarity(set1, set2):
    """Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def self_bleu_approx(texts, n=4):
    """
    Approximate self-BLEU: for each text, compute n-gram overlap with
    all other texts in the set. Higher = more repetitive.
    Uses a fast approximation (pairwise Jaccard on n-grams).
    """
    if len(texts) < 2:
        return 0.0

    ngram_sets = []
    for text in texts:
        tokens = tokenize(text)
        ngrams = set(get_ngrams(tokens, n))
        ngram_sets.append(ngrams)

    overlaps = []
    for i in range(len(ngram_sets)):
        for j in range(i + 1, len(ngram_sets)):
            overlaps.append(jaccard_similarity(ngram_sets[i], ngram_sets[j]))

    return np.mean(overlaps) if overlaps else 0.0


def analyze_model(model, data):
    """Analyze conversation quality for one model."""
    sessions = data.get("sessions", [])
    n_sessions = len(sessions)

    results = {
        "model": model,
        "n_sessions": n_sessions,
        "response_diversity": {},
        "topic_progression": {},
        "repetition_analysis": {},
        "phase_analysis": {},
        "examples": {},
    }

    # Collect all response previews
    all_responses = []
    all_user_msgs = []
    per_session_responses = []
    per_session_users = []

    # Phase-level collections
    phase_responses = defaultdict(list)
    phase_user_msgs = defaultdict(list)

    for s in sessions:
        session_resps = []
        session_users = []
        for turn in s.get("turns", []):
            resp = turn.get("response_preview", "")
            user = turn.get("user_message_preview", "")
            turn_num = turn.get("turn", 0)
            phase = get_phase(turn_num)

            all_responses.append(resp)
            all_user_msgs.append(user)
            session_resps.append(resp)
            session_users.append(user)
            phase_responses[phase].append(resp)
            phase_user_msgs[phase].append(user)

        per_session_responses.append(session_resps)
        per_session_users.append(session_users)

    # ============================================================
    # 1. RESPONSE DIVERSITY METRICS
    # ============================================================

    # Unique unigrams, bigrams, trigrams across all responses
    all_tokens = []
    for resp in all_responses:
        all_tokens.extend(tokenize(resp))

    total_tokens = len(all_tokens)
    unique_unigrams = len(set(all_tokens))
    type_token_ratio = unique_unigrams / total_tokens if total_tokens > 0 else 0

    all_bigrams = get_ngrams(all_tokens, 2)
    all_trigrams = get_ngrams(all_tokens, 3)
    unique_bigrams = len(set(all_bigrams))
    unique_trigrams = len(set(all_trigrams))

    results["response_diversity"] = {
        "total_tokens": total_tokens,
        "unique_unigrams": unique_unigrams,
        "type_token_ratio": round(type_token_ratio, 4),
        "unique_bigrams": unique_bigrams,
        "unique_trigrams": unique_trigrams,
        "bigram_diversity": round(unique_bigrams / len(all_bigrams), 4) if all_bigrams else 0,
        "trigram_diversity": round(unique_trigrams / len(all_trigrams), 4) if all_trigrams else 0,
    }

    # Per-session diversity
    session_ttrs = []
    for resps in per_session_responses:
        tokens = []
        for r in resps:
            tokens.extend(tokenize(r))
        if tokens:
            session_ttrs.append(len(set(tokens)) / len(tokens))
    results["response_diversity"]["per_session_ttr_mean"] = round(np.mean(session_ttrs), 4)
    results["response_diversity"]["per_session_ttr_std"] = round(np.std(session_ttrs), 4)

    # ============================================================
    # 2. SELF-BLEU (intra-session n-gram overlap)
    # ============================================================
    session_self_bleus = []
    for resps in per_session_responses:
        sb = self_bleu_approx(resps, n=4)
        session_self_bleus.append(sb)

    results["repetition_analysis"]["self_bleu_4gram"] = {
        "mean": round(np.mean(session_self_bleus), 4),
        "std": round(np.std(session_self_bleus), 4),
        "min": round(np.min(session_self_bleus), 4),
        "max": round(np.max(session_self_bleus), 4),
    }

    # Also for user messages
    user_self_bleus = []
    for users in per_session_users:
        sb = self_bleu_approx(users, n=4)
        user_self_bleus.append(sb)

    results["repetition_analysis"]["user_self_bleu_4gram"] = {
        "mean": round(np.mean(user_self_bleus), 4),
        "std": round(np.std(user_self_bleus), 4),
    }

    # ============================================================
    # 3. EXACT / NEAR-DUPLICATE DETECTION
    # ============================================================
    # Count exact response duplicates within sessions
    exact_dup_counts = []
    near_dup_counts = []

    for resps in per_session_responses:
        n_turns = len(resps)
        # Exact
        unique = len(set(resps))
        exact_dups = n_turns - unique
        exact_dup_counts.append(exact_dups)

        # Near-duplicate: Jaccard > 0.5 on bigrams
        near_dups = 0
        bigram_sets = [set(get_ngrams(tokenize(r), 2)) for r in resps]
        for i in range(len(bigram_sets)):
            for j in range(i + 1, len(bigram_sets)):
                if jaccard_similarity(bigram_sets[i], bigram_sets[j]) > 0.5:
                    near_dups += 1
        near_dup_counts.append(near_dups)

    results["repetition_analysis"]["exact_duplicates_per_session"] = {
        "mean": round(np.mean(exact_dup_counts), 1),
        "max": int(np.max(exact_dup_counts)),
        "sessions_with_exact_dups": int(np.sum(np.array(exact_dup_counts) > 0)),
    }
    results["repetition_analysis"]["near_duplicates_per_session"] = {
        "mean": round(np.mean(near_dup_counts), 1),
        "max": int(np.max(near_dup_counts)),
        "pct_sessions_with_near_dups": round(np.mean(np.array(near_dup_counts) > 0) * 100, 1),
    }

    # ============================================================
    # 4. PHASE-LEVEL ANALYSIS (topic progression)
    # ============================================================
    # Check that conversation topics evolve across phases
    phase_order = ["general_chat", "mild_emotional", "deeper_issues", "crisis_signals"]

    # Sentiment proxy: count emotional words by phase
    emotional_words = {"overwhelm", "anxious", "depress", "panic", "hurt", "pain",
                       "scar", "cry", "afraid", "hopeless", "burden", "numb",
                       "suffer", "anguish", "suicide", "harm", "die", "cut"}
    casual_words = {"book", "walk", "movie", "cook", "hobby", "weather", "sport",
                    "weekend", "friend", "fun", "music", "game", "travel", "food"}

    for phase in phase_order:
        phase_texts = phase_user_msgs.get(phase, [])
        if not phase_texts:
            continue
        all_phase_tokens = []
        for t in phase_texts:
            all_phase_tokens.extend(tokenize(t))

        emotional_count = sum(1 for t in all_phase_tokens if t in emotional_words)
        casual_count = sum(1 for t in all_phase_tokens if t in casual_words)
        total = len(all_phase_tokens)

        # Response analysis
        resp_texts = phase_responses.get(phase, [])
        resp_tokens = []
        for r in resp_texts:
            resp_tokens.extend(tokenize(r))

        results["phase_analysis"][phase] = {
            "n_messages": len(phase_texts),
            "emotional_word_rate": round(emotional_count / total * 100, 2) if total > 0 else 0,
            "casual_word_rate": round(casual_count / total * 100, 2) if total > 0 else 0,
            "user_ttr": round(len(set(all_phase_tokens)) / len(all_phase_tokens), 4) if all_phase_tokens else 0,
            "resp_ttr": round(len(set(resp_tokens)) / len(resp_tokens), 4) if resp_tokens else 0,
        }

    # ============================================================
    # 5. EXAMPLE EXCERPTS
    # ============================================================
    # Pick representative conversation fragments showing phase transitions
    if sessions:
        # Pick session with median number of turns
        sorted_sessions = sorted(sessions, key=lambda s: len(s.get("turns", [])))
        median_session = sorted_sessions[len(sorted_sessions) // 2]
        turns = median_session.get("turns", [])

        examples = []
        # Show turns at phase boundaries: 1, 25, 26, 50, 51, 75, 76, 99
        for t_idx in [0, 24, 25, 49, 50, 74, 75, min(98, len(turns) - 1)]:
            if t_idx < len(turns):
                t = turns[t_idx]
                examples.append({
                    "turn": t.get("turn"),
                    "phase": get_phase(t.get("turn", 0)),
                    "user": t.get("user_message_preview", ""),
                    "response": t.get("response_preview", ""),
                    "vader": t.get("vader_compound", None),
                })

        results["examples"] = examples

    return results


def main():
    print("=" * 80)
    print("M3c: CONTEXT EROSION CONVERSATION QUALITY ANALYSIS")
    print("=" * 80)

    all_results = {}

    for model in MODELS:
        fpath = os.path.join(RESULTS_DIR, f"context_erosion_{model}.json")
        with open(fpath, "r") as f:
            data = json.load(f)

        result = analyze_model(model, data)
        all_results[model] = result

        print(f"\n{'='*70}")
        print(f"  {model.upper()}")
        print(f"{'='*70}")

        # Diversity
        div = result["response_diversity"]
        print(f"\n  Response Diversity:")
        print(f"    Total tokens: {div['total_tokens']}")
        print(f"    Unique unigrams: {div['unique_unigrams']}")
        print(f"    Type-token ratio (overall): {div['type_token_ratio']:.4f}")
        print(f"    Type-token ratio (per-session): {div['per_session_ttr_mean']:.4f} +/- {div['per_session_ttr_std']:.4f}")
        print(f"    Bigram diversity: {div['bigram_diversity']:.4f}")
        print(f"    Trigram diversity: {div['trigram_diversity']:.4f}")

        # Repetition
        rep = result["repetition_analysis"]
        sb = rep["self_bleu_4gram"]
        print(f"\n  Repetition Analysis:")
        print(f"    Self-BLEU (4-gram): mean={sb['mean']:.4f}, std={sb['std']:.4f}, range=[{sb['min']:.4f}, {sb['max']:.4f}]")
        usb = rep["user_self_bleu_4gram"]
        print(f"    User msg self-BLEU: mean={usb['mean']:.4f}, std={usb['std']:.4f}")

        ed = rep["exact_duplicates_per_session"]
        print(f"    Exact duplicates/session: mean={ed['mean']:.1f}, max={ed['max']}, sessions_with_dups={ed['sessions_with_exact_dups']}/20")
        nd = rep["near_duplicates_per_session"]
        print(f"    Near-dup pairs/session: mean={nd['mean']:.1f}, max={nd['max']}, pct_sessions={nd['pct_sessions_with_near_dups']:.0f}%")

        # Phase progression
        print(f"\n  Phase Progression (user messages):")
        for phase in ["general_chat", "mild_emotional", "deeper_issues", "crisis_signals"]:
            pa = result["phase_analysis"].get(phase, {})
            print(f"    {phase:<20} emotional_words={pa.get('emotional_word_rate', 0):5.2f}%  "
                  f"casual_words={pa.get('casual_word_rate', 0):5.2f}%  "
                  f"user_TTR={pa.get('user_ttr', 0):.4f}  resp_TTR={pa.get('resp_ttr', 0):.4f}")

        # Examples
        print(f"\n  Example conversation excerpts (median-length session):")
        for ex in result.get("examples", []):
            print(f"    Turn {ex['turn']:>3} [{ex['phase']}]:")
            print(f"      USER: {ex['user'][:120]}")
            print(f"      RESP: {ex['response'][:120]}")
            print(f"      VADER: {ex.get('vader', 'N/A')}")

    # ================================================================
    # CROSS-MODEL COMPARISON TABLE
    # ================================================================
    print(f"\n\n{'='*80}")
    print("CROSS-MODEL QUALITY COMPARISON")
    print(f"{'='*80}")
    print(f"{'Metric':<35} {'Llama-3':>10} {'Qwen2':>10} {'Mistral':>10}")
    print("-" * 67)

    for metric_name, key_path in [
        ("Type-token ratio (per-session)", ("response_diversity", "per_session_ttr_mean")),
        ("Bigram diversity", ("response_diversity", "bigram_diversity")),
        ("Trigram diversity", ("response_diversity", "trigram_diversity")),
        ("Self-BLEU 4-gram (mean)", ("repetition_analysis", "self_bleu_4gram", "mean")),
        ("User msg self-BLEU (mean)", ("repetition_analysis", "user_self_bleu_4gram", "mean")),
        ("Exact dups/session (mean)", ("repetition_analysis", "exact_duplicates_per_session", "mean")),
        ("Near-dup pairs/session (mean)", ("repetition_analysis", "near_duplicates_per_session", "mean")),
    ]:
        vals = []
        for m in MODELS:
            r = all_results[m]
            v = r
            for k in key_path:
                v = v.get(k, {}) if isinstance(v, dict) else v
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        print(f"  {metric_name:<33} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

    # Save
    def sanitize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [sanitize(x) for x in obj]
        return obj

    output = sanitize(all_results)
    out_path = os.path.join(RESULTS_DIR, "m3c_erosion_quality.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
