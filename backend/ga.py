#!/usr/bin/env python3
import os
import random
from typing import List, Tuple, Dict, Iterable, Any
import math
import subprocess
from pathlib import Path
import pandas as pd
import time

# GA Hyperparameters
POPULATION_SIZE = 100
GENERATIONS = 50
TOURNAMENT_SIZE = 4
ELITISM = True
ELITISM_COUNT = 2
CROSSOVER_RATE = 0.9
MUTATION_RATE_START = 0.03  # will anneal
MUTATION_RATE_END = 0.01
SEQUENCE_LENGTH = 20  # gRNA length
MINIMUM_HAMMING = 2  # diversity control

# Off-target aggregation and tradeoff
LAMBDA = 0.5  # trade-off weight
OFFTARGET_TOPK = 10  # top-k mean over off-target scores

# GC window and constraints
GC_MIN = 0.35
GC_MAX = 0.75
MAX_HOMOPOLYMER = 4
FORBIDDEN_MOTIFS = ("TTTT",)  # U6 termination, adjust as needed

# Off-target mini-batch
OFF_BATCH_SIZE = 120  # number of off-targets sampled per fitness call
ROTATE_OFFSETS = True  # rotate subset generation-by-generation

# FASTA format whole-genome reference for homo sapiens
PARENT_DIR = Path(__file__).resolve().parent.parent
REFERENCE_GENOME = Path(os.path.join(PARENT_DIR, 'data/cas_offinder/hg38.fa'))
GUIDE_FILE = Path(os.path.join(PARENT_DIR, 'data/cas_offinder/guide.txt'))
OFF_TARGETS_FILE = Path(
    os.path.join('./', PARENT_DIR, 'data/cas_offinder/off_targets.txt')
)
PAM = 'NGG'
MAX_MISMATCH = 3

# Path to Cas-OFFinder executable
CAS_OFFINDER_PATH = PARENT_DIR / 'src/cas-offinder'

# Surrogate model hooks
def predict_prob_on_target(guide: str, target: str) -> float:
    """Return model-predicted probability guide cuts intended target.
    TODO: Replace with your trained regression model inference.
    """
    matches = sum(1 for a, b in zip(guide[-8:], target[-8:]) if a == b)
    return min(1.0, 0.1 + 0.1 * matches)


def predict_prob_off_target(guide: str, target: str) -> float:
    """Return model-predicted probability guide cuts an off-target.
    TODO: Replace with your trained regression model inference.
    """

    # Use Hayden's model



    mismatches = sum(1 for a, b in zip(guide, target) if a != b)
    return max(0.0, 0.4 - 0.03 * (SEQUENCE_LENGTH - mismatches))



def predict_off_targets(guide: str):
    # Returns a dataframe of potential off targets

    ## Write the guide RNA to a text file
    with GUIDE_FILE.open('w') as f:
        f.write(f"{REFERENCE_GENOME}\n")
        f.write(f"{PAM}\n")
        f.write(f"{guide} {MAX_MISMATCH}\N")

    ## Call the Cas-OFFinder executable
    subprocess.run([CAS_OFFINDER_PATH, str(GUIDE_FILE), str(OFF_TARGETS_FILE)])

    ## Read the off-target outputs as a dataframe
    off_targets = pd.read_csv(
        OFF_TARGETS_FILE, sep = '\t', header = None,
        names = ["chrom", "position", "strand", "target_seq", "mismatches"]
    )
    off_targets['position'] = off_targets['position'].astype(int)
    off_targets['mismatches'] = off_targets['mismatches'].astype(int)

    return off_targets


# Example off targets
INTENDED_TARGET = "ACGTACGTACGTACGTACGT"[:SEQUENCE_LENGTH]

# Provide a synthetic set of off-targets (replace with your real pool)
OFF_TARGETS_FULL = [
    "".join(random.choice("ACGT") for _ in range(SEQUENCE_LENGTH))
    for _ in range(1000)
]

# Utilities
NUCS = "ACGT"


def gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    gc = sum(1 for ch in seq if ch in "GC")
    return gc / len(seq)


def longest_homopolymer(seq: str) -> int:
    best = run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))


# Penalty & Repair
def penalties(seq: str) -> float:
    p = 0.0
    gc = gc_content(seq)
    if gc < GC_MIN:
        p += 0.2 * (GC_MIN - gc)
    if gc > GC_MAX:
        p += 0.2 * (gc - GC_MAX)
    hp = longest_homopolymer(seq)
    if hp > MAX_HOMOPOLYMER:
        p += 0.1 * (hp - MAX_HOMOPOLYMER)
    for motif in FORBIDDEN_MOTIFS:
        if motif in seq:
            p += 0.3
    # PAM checks can be added here if you evolve PAM-proximal region
    return p


def repair(seq: str) -> str:
    """Softly nudge sequence back into valid ranges (GC, homopolymer)."""
    s = list(seq)
    # Fix GC if needed
    gcf = gc_content(seq)
    attempts = 0
    while (gcf < GC_MIN or gcf > GC_MAX) and attempts < 10:
        i = random.randrange(len(s))
        if gcf < GC_MIN and s[i] in "AT":
            s[i] = random.choice("GC")
        elif gcf > GC_MAX and s[i] in "GC":
            s[i] = random.choice("AT")
        gcf = gc_content("".join(s))
        attempts += 1
    # Break long homopolymers
    # Flip middle of longest run if needed
    run = longest_homopolymer("".join(s))
    if run > MAX_HOMOPOLYMER:
        # naive pass to break any 5-run
        for i in range(2, len(s) - 2):
            if s[i - 2] == s[i - 1] == s[i] == s[i + 1] == s[i + 2]:
                choices = [n for n in NUCS if n != s[i]]
                s[i] = random.choice(choices)
    # Forbidden motifs
    seq2 = "".join(s)
    for motif in FORBIDDEN_MOTIFS:
        if motif in seq2:
            j = seq2.index(motif) + len(motif) // 2
            choices = [n for n in NUCS if n != seq2[j]]
            s[j] = random.choice(choices)
            seq2 = "".join(s)
    return "".join(s)


# Memoization cache
pred_cache_on: Dict[Tuple[str, str], float] = {}
pred_cache_off: Dict[Tuple[str, str], float] = {}


def cached_predict_on(g: str, t: str) -> float:
    key = (g, t)
    if key not in pred_cache_on:
        pred_cache_on[key] = predict_prob_on_target(g, t)
    return pred_cache_on[key]


def cached_predict_off(g: str, t: str) -> float:
    key = (g, t)
    if key not in pred_cache_off:
        pred_cache_off[key] = predict_prob_off_target(g, t)
    return pred_cache_off[key]


# Fitness
def off_agg(scores: List[float], k: int = OFFTARGET_TOPK) -> float:
    if not scores:
        return 0.0
    arr = sorted(scores, reverse=True)
    k = min(k, len(arr))
    return sum(arr[:k]) / k


def fitness_components(
    seq: str, off_targets_subset: Iterable[str]
) -> Tuple[float, float, float, float]:
    """Return (fitness, on, off, penalty) for logging."""
    # repair (light) before scoring to avoid wasting evals
    seq = repair(seq)
    on = cached_predict_on(seq, INTENDED_TARGET)
    off_scores = [cached_predict_off(seq, t) for t in off_targets_subset]
    off = off_agg(off_scores, OFFTARGET_TOPK)
    pen = penalties(seq)
    fit = on - LAMBDA * off - pen
    return fit, on, off, pen


# GA ops
def random_seq(L: int) -> str:
    return "".join(random.choice(NUCS) for _ in range(L))


def mutate(seq: str, rate: float) -> str:
    s = list(seq)
    for i in range(len(s)):
        if random.random() < rate:
            s[i] = random.choice([n for n in NUCS if n != s[i]])
    return "".join(s)


def crossover(a: str, b: str) -> Tuple[str, str]:
    if len(a) != len(b):
        L = min(len(a), len(b))
        a, b = a[:L], b[:L]
    if random.random() > CROSSOVER_RATE:
        return a, b  # no crossover this time
    cut1 = random.randint(1, len(a) - 2)
    cut2 = random.randint(cut1 + 1, len(a) - 1)
    c1 = a[:cut1] + b[cut1:cut2] + a[cut2:]
    c2 = b[:cut1] + a[cut1:cut2] + b[cut2:]
    return c1, c2


def tournament_select(pop: List[str], scores: Dict[str, float]) -> str:
    contenders = random.sample(pop, TOURNAMENT_SIZE)
    contenders.sort(key=lambda g: scores[g], reverse=True)
    return contenders[0]


def mean_hamming(pop: List[str]) -> float:
    if len(pop) < 2:
        return 0.0
    total = 0
    count = 0
    for i in range(len(pop)):
        for j in range(i + 1, len(pop)):
            total += hamming(pop[i], pop[j])
            count += 1
    return total / count


# Main GA driver
def anneal_mutation_rate(gen: int) -> float:
    # cosine anneal from start to end
    if GENERATIONS <= 1:
        return MUTATION_RATE_END
    frac = gen / (GENERATIONS - 1)
    cos = 0.5 * (1 + math.cos(math.pi * frac))
    return MUTATION_RATE_END + (MUTATION_RATE_START - MUTATION_RATE_END) * cos


def genetic_algorithm() -> Tuple[str, Dict]:
    # Initialize rotating offsets for off-target subsampling
    off_targets = OFF_TARGETS_FULL
    offset = 0

    # Seed population
    population = []
    retry_budget = 2000
    while len(population) < POPULATION_SIZE and retry_budget > 0:
        cand = random_seq(SEQUENCE_LENGTH)
        # diversity check against what's in population so far
        if all(hamming(cand, ex) >= MINIMUM_HAMMING for ex in population):
            population.append(cand)
        else:
            retry_budget -= 1
            if retry_budget % 200 == 0 and retry_budget > 0:
                # progressively relax if stuck
                pass
    if len(population) < POPULATION_SIZE:
        # fill whatever is missing without diversity constraint (graceful fallback)
        population.extend(
            random_seq(SEQUENCE_LENGTH)
            for _ in range(POPULATION_SIZE - len(population))
        )

    best_global = None  # (fit, seq, on, off, pen)
    logs = {
        "best_fit": [],
        "mean_fit": [],
        "best_on": [],
        "best_off": [],
        "best_gc": [],
        "diversity": [],
    }

    for gen in range(GENERATIONS):
        # Build the off-target mini-batch subset
        if ROTATE_OFFSETS:
            start = (offset + gen * OFF_BATCH_SIZE) % max(1, len(off_targets))
            subset = [
                off_targets[(start + i) % len(off_targets)]
                for i in range(min(OFF_BATCH_SIZE, len(off_targets)))
            ]
        else:
            subset = random.sample(
                off_targets, min(OFF_BATCH_SIZE, len(off_targets))
            )

        # Evaluate population
        scores: Dict[str, float] = {}
        comps: Dict[str, Tuple[float, float, float, float]] = {}
        for g in population:
            fit, on, off, pen = fitness_components(g, subset)
            scores[g] = fit
            comps[g] = (fit, on, off, pen)

        # Logging
        sorted_pop = sorted(population, key=lambda z: scores[z], reverse=True)
        best = sorted_pop[0]
        best_fit, best_on, best_off, best_pen = comps[best]
        mean_fit = sum(scores[g] for g in population) / len(population)
        logs["best_fit"].append(best_fit)
        logs["mean_fit"].append(mean_fit)
        logs["best_on"].append(best_on)
        logs["best_off"].append(best_off)
        logs["best_gc"].append(gc_content(best))
        logs["diversity"].append(mean_hamming(population))

        if (best_global is None) or (best_fit > best_global[0]):
            best_global = (best_fit, best, best_on, best_off, best_pen)

        # Next generation
        new_pop: List[str] = []
        if ELITISM:
            new_pop.extend(sorted_pop[:ELITISM_COUNT])

        # produce children in pairs
        mut_rate = anneal_mutation_rate(gen)
        retry_budget_children = 2000

        while len(new_pop) < POPULATION_SIZE and retry_budget_children > 0:
            p1 = tournament_select(population, scores)
            p2 = tournament_select(population, scores)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(repair(c1), mut_rate)
            c2 = mutate(repair(c2), mut_rate)
            # enforce diversity vs what's already in new_pop
            appended = 0
            if all(hamming(c1, ex) >= MINIMUM_HAMMING for ex in new_pop):
                new_pop.append(c1)
                appended += 1
            if len(new_pop) < POPULATION_SIZE and all(
                hamming(c2, ex) >= MINIMUM_HAMMING for ex in new_pop
            ):
                new_pop.append(c2)
                appended += 1
            if appended == 0:
                retry_budget_children -= 1
                if (
                    retry_budget_children % 200 == 0
                    and retry_budget_children > 0
                ):
                    # slight relaxation if stuck
                    pass

        if len(new_pop) < POPULATION_SIZE:
            # Fill by relaxing diversity
            while len(new_pop) < POPULATION_SIZE:
                new_pop.append(random_seq(SEQUENCE_LENGTH))

        population = new_pop

    # Final thorough evaluation on full off-targets for the best
    fit_final, on_final, off_final, pen_final = fitness_components(
        best_global[1], OFF_TARGETS_FULL
    )
    best_global = (fit_final, best_global[1], on_final, off_final, pen_final)
    return best_global[1], {"summary": best_global, "logs": logs}


if __name__ == "__main__":
    best_seq, info = genetic_algorithm()
    fit, seq, on, off, pen = info["summary"]
    print(f"Best sequence: {best_seq}")
    print(
        f"Fitness: {fit:.4f} | On-target: {on:.4f} |"
        f" Off-target(top-{OFFTARGET_TOPK} mean): {off:.4f} | Penalties:"
        f" {pen:.4f}"
    )
    print("Logs keys:", list(info["logs"].keys()))


def _logo_counts(population, top_k=40):
    # frequency counts for sequence logo
    if not population:
        return {}
    top = population[:top_k]
    L = len(top[0])
    counts = [dict(A=0, C=0, G=0, T=0) for _ in range(L)]
    for seq in top:
        for i, ch in enumerate(seq):
            counts[i][ch] += 1
    # shape as { "0": {"A":n,...}, "1": {...}, ... }
    return {str(i): counts[i] for i in range(L)}


def run_ga_stream(params: Dict[str, Any] = None) -> Iterable[Dict[str, Any]]:
    """Yields a JSON-serializable dict per generation and a final 'done' frame."""
    global POPULATION_SIZE, GENERATIONS, LAMBDA, INTENDED_TARGET
    if params:
        POPULATION_SIZE = int(params.get("population_size", POPULATION_SIZE))
        GENERATIONS = int(params.get("generations", GENERATIONS))
        LAMBDA = float(params.get("lambda", LAMBDA))
        # Update intended target if provided
        if "target_sequence" in params:
            INTENDED_TARGET = str(params["target_sequence"])[:SEQUENCE_LENGTH]

    # Copy of your genetic_algorithm loop, but yielding each gen’s snapshot.
    off_targets = OFF_TARGETS_FULL
    offset = 0

    population = []
    retry_budget = 2000
    while len(population) < POPULATION_SIZE and retry_budget > 0:
        cand = random_seq(SEQUENCE_LENGTH)
        if all(hamming(cand, ex) >= MINIMUM_HAMMING for ex in population):
            population.append(cand)
        else:
            retry_budget -= 1
    if len(population) < POPULATION_SIZE:
        population.extend(
            random_seq(SEQUENCE_LENGTH)
            for _ in range(POPULATION_SIZE - len(population))
        )

    best_global = None  # (fit, seq, on, off, pen)
    logs = {
        "best_fit": [],
        "mean_fit": [],
        "best_on": [],
        "best_off": [],
        "best_gc": [],
        "diversity": [],
    }

    for gen in range(GENERATIONS):
        # rotating off-target subset
        start = (offset + gen * OFF_BATCH_SIZE) % max(1, len(off_targets))
        subset = [
            off_targets[(start + i) % len(off_targets)]
            for i in range(min(OFF_BATCH_SIZE, len(off_targets)))
        ]

        # evaluate
        scores = {}
        comps = {}
        for g in population:
            fit, on, off, pen = fitness_components(g, subset)
            scores[g] = fit
            comps[g] = (fit, on, off, pen)

        sorted_pop = sorted(population, key=lambda z: scores[z], reverse=True)
        best = sorted_pop[0]
        best_fit, best_on, best_off, best_pen = comps[best]
        mean_fit = sum(scores[g] for g in population) / len(population)

        logs["best_fit"].append(best_fit)
        logs["mean_fit"].append(mean_fit)
        logs["best_on"].append(best_on)
        logs["best_off"].append(best_off)
        logs["best_gc"].append(gc_content(best))
        logs["diversity"].append(mean_hamming(population))

        if (best_global is None) or (best_fit > best_global[0]):
            best_global = (best_fit, best, best_on, best_off, best_pen)

        # emit frame for this generation
        yield {
            "gen": gen,
            "summary": {
                "best_fit": best_fit,
                "mean_fit": mean_fit,
                "best_on": best_on,
                "best_off": best_off,
                "best_gc": gc_content(best),
                "diversity_hamming": mean_hamming(population),
                "mutation_rate": anneal_mutation_rate(gen),
                "crossover_rate": CROSSOVER_RATE,
            },
            "top": [
                {
                    "id": f"g_{gen}_{i:03d}",
                    "seq": s,
                    "fit": float(scores[s]),
                    "on": float(comps[s][1]),
                    "off": float(comps[s][2]),
                    "gc": gc_content(s),
                }
                for i, s in enumerate(sorted_pop[: min(20, len(sorted_pop))])
            ],
            "pareto": [
                {
                    "id": f"p_{gen}_{i:03d}",
                    "on": float(comps[s][1]),
                    "off": float(comps[s][2]),
                }
                for i, s in enumerate(sorted_pop[: min(100, len(sorted_pop))])
            ],
            "logo_counts": _logo_counts(sorted_pop, top_k=40),
            "off_subset_meta": {"k": OFFTARGET_TOPK, "agg": "top_k_mean"},
            "done": False,
        }

        # next generation (same as your code)
        new_pop = []
        if ELITISM:
            new_pop.extend(sorted_pop[:ELITISM_COUNT])
        mut_rate = anneal_mutation_rate(gen)
        retry_budget_children = 2000
        while len(new_pop) < POPULATION_SIZE and retry_budget_children > 0:
            p1 = tournament_select(population, scores)
            p2 = tournament_select(population, scores)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(repair(c1), mut_rate)
            c2 = mutate(repair(c2), mut_rate)
            appended = 0
            if all(hamming(c1, ex) >= MINIMUM_HAMMING for ex in new_pop):
                new_pop.append(c1)
                appended += 1
            if len(new_pop) < POPULATION_SIZE and all(
                hamming(c2, ex) >= MINIMUM_HAMMING for ex in new_pop
            ):
                new_pop.append(c2)
                appended += 1
            if appended == 0:
                retry_budget_children -= 1
        if len(new_pop) < POPULATION_SIZE:
            while len(new_pop) < POPULATION_SIZE:
                new_pop.append(random_seq(SEQUENCE_LENGTH))
        population = new_pop
        # (Optional) small sleep to throttle stream in dev
        # time.sleep(0.02)

    # final thorough eval on full off-targets for the best
    fit_final, on_final, off_final, pen_final = fitness_components(
        best_global[1], OFF_TARGETS_FULL
    )

    # Also get the subset-based fitness for consistency with training
    final_subset_start = (offset + (GENERATIONS - 1) * OFF_BATCH_SIZE) % max(
        1, len(off_targets)
    )
    final_subset = [
        off_targets[(final_subset_start + i) % len(off_targets)]
        for i in range(min(OFF_BATCH_SIZE, len(off_targets)))
    ]
    fit_subset, _, _, _ = fitness_components(best_global[1], final_subset)

    yield {
        "gen": GENERATIONS,
        "summary": {
            "best_fit": (
                fit_subset
            ),  # Use subset fitness for consistency with training
            "mean_fit": logs["mean_fit"][-1],
            "best_on": (
                on_final
            ),  # Use full eval for on-target (doesn't change)
            "best_off": (
                off_final
            ),  # Use full eval for off-target (final validation)
            "best_gc": gc_content(best_global[1]),
            "diversity_hamming": logs["diversity"][-1],
            "mutation_rate": anneal_mutation_rate(GENERATIONS - 1),
            "crossover_rate": CROSSOVER_RATE,
        },
        "best_sequence": best_global[1],
        "validation_fitness": (
            fit_final
        ),  # Add full dataset fitness as separate field
        "training_fitness": fit_subset,  # Make the distinction clear
        "done": True,
    }
