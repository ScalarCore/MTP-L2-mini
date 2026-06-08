"""
kibo_fair_test.py — fair, reproducible benchmark for the MTP-L2 layer.

Design principles (so the numbers survive a hostile technical reviewer)
----------------------------------------------------------------------
1. IDENTICAL INPUT to both paths. Both the raw baseline and the MTP pipeline
   receive the *same* extracted LLM output. We are not prompting one side nicely
   and the other side adversarially — the only variable is the post-processing
   layer, which is the actual product.

2. NEUTRAL, FUZZED INPUTS. Angle proposals are drawn from a fixed-seed random
   distribution that deliberately includes out-of-range and malformed values
   (this is what real LLMs occasionally emit). No hand-authored "the LLM got
   confused" anecdotes — the inputs are mechanical and biasable-free.

3. WE MEASURE GUARANTEES, NOT MARKETING. MTP-L2's claim is a *guarantee*:
   every output is schema-conform and within physical bounds, by construction.
   The raw baseline forwards whatever it extracts. The benchmark reports how
   often each property holds over the fuzz set, plus compile latency and the
   payload the executor must consume.

4. LATENCY IS HONEST. We measure only the deterministic local step
   (microseconds, real). Both paths still pay the same upstream LLM call, which
   dominates wall-clock and is therefore not a differentiator. No fabricated
   round-trip numbers.

Run:  python kibo_fair_test.py            # offline, reproducible
      python kibo_fair_test.py --n 5000   # bigger fuzz set
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import time

from mtp_l2_mini import MTPCompiler, RawBaseline, ANGLE_MIN, ANGLE_MAX, VALID_DIRECTIONS

SEED = 20260608


def build_fuzz_set(n: int):
    """Neutral PROSE inputs — what an unconstrained LLM actually emits — fed
    IDENTICALLY to both paths. Fixed rates of in-range / out-of-range /
    multi-number / malformed. Phrasing is neutral (no adversarial 'be confused'
    instructions); the only variation is the value and how many numbers appear."""
    rng = random.Random(SEED)
    cases = []
    for i in range(n):
        direction = rng.choice(VALID_DIRECTIONS)
        roll = rng.random()
        if roll < 0.45:                      # in-range, single number
            angle = round(rng.uniform(ANGLE_MIN, ANGLE_MAX), 1)
            llm = f"Rotate the arm {direction} by {angle} degrees."
        elif roll < 0.70:                    # out-of-range value (LLM overshoot)
            angle = round(rng.choice([1, -1]) * rng.uniform(ANGLE_MAX + 1, 360), 1)
            llm = f"Rotate the arm {direction} by {angle} degrees."
        elif roll < 0.85:                    # verbose, single number
            angle = round(rng.uniform(ANGLE_MIN, ANGLE_MAX), 1)
            llm = (f"Sure — I'll move the arm {direction}. The rotation should be "
                   f"{angle} degrees. Executing now.")
        elif roll < 0.95:                    # two numbers (an estimate then the value)
            angle = round(rng.uniform(ANGLE_MIN, ANGLE_MAX), 1)
            decoy = round(rng.uniform(ANGLE_MIN, ANGLE_MAX), 1)
            llm = f"It's roughly {decoy} degrees, more precisely {angle} degrees {direction}."
        else:                                # malformed (no parseable number)
            angle = 0.0
            llm = f"Turn the arm {direction} a little."
        cases.append({
            "id": i,
            "direction_intent": direction,
            "angle_intent": angle,
            "llm_output": llm,
        })
    return cases


def run(n: int, out_csv: str):
    mtp = MTPCompiler()
    raw = RawBaseline()
    cases = build_fuzz_set(n)

    rows = []
    agg = {
        "MTP_DSL": {"valid": 0, "safe": 0, "correct": 0, "payload": [], "lat": []},
        "RAW_LLM": {"valid": 0, "safe": 0, "correct": 0, "payload": [], "lat": []},
    }

    for c in cases:
        for pipe_name, engine, fn in (
            ("MTP_DSL", mtp, "compile"),
            ("RAW_LLM", raw, "process"),
        ):
            method = getattr(engine, fn)
            # direction is detected from the SAME prose for both paths
            ctx = c["llm_output"]
            # time the post-LLM step over repeats for a stable microsecond reading
            t0 = time.perf_counter()
            for _ in range(50):
                r = method(c["llm_output"], ctx, c["angle_intent"], c["direction_intent"])
            lat_us = (time.perf_counter() - t0) / 50 * 1e6

            safe = ANGLE_MIN <= r.final_angle <= ANGLE_MAX
            a = agg[pipe_name]
            a["valid"] += int(r.schema_valid)
            a["safe"] += int(safe)
            a["correct"] += int(r.correct)
            a["payload"].append(r.payload_bytes)
            a["lat"].append(lat_us)
            rows.append({
                "id": c["id"], "pipeline": pipe_name,
                "schema_valid": r.schema_valid, "bounds_safe": safe,
                "correct": r.correct, "payload_bytes": r.payload_bytes,
                "final_angle": r.final_angle, "latency_us": round(lat_us, 3),
            })

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def pct(x):
        return f"{x / n * 100:.1f}%"

    def p(vals, q):
        return statistics.quantiles(vals, n=100)[q - 1] if len(vals) > 1 else vals[0]

    print(f"\n  MTP-L2 fair benchmark  (seed={SEED}, N={n}, identical input to both)\n")
    print(f"  {'metric':<26}{'RAW_LLM':>14}{'MTP_DSL':>14}")
    print("  " + "-" * 54)
    print(f"  {'schema-conform output':<26}{pct(agg['RAW_LLM']['valid']):>14}{pct(agg['MTP_DSL']['valid']):>14}")
    print(f"  {'within physical bounds':<26}{pct(agg['RAW_LLM']['safe']):>14}{pct(agg['MTP_DSL']['safe']):>14}")
    print(f"  {'correct vs intent':<26}{pct(agg['RAW_LLM']['correct']):>14}{pct(agg['MTP_DSL']['correct']):>14}")
    print(f"  {'payload bytes (mean)':<26}{statistics.mean(agg['RAW_LLM']['payload']):>14.1f}{statistics.mean(agg['MTP_DSL']['payload']):>14.1f}")
    print(f"  {'compile latency p50 (us)':<26}{p(agg['RAW_LLM']['lat'],50):>14.2f}{p(agg['MTP_DSL']['lat'],50):>14.2f}")
    print(f"  {'compile latency p95 (us)':<26}{p(agg['RAW_LLM']['lat'],95):>14.2f}{p(agg['MTP_DSL']['lat'],95):>14.2f}")
    print("\n  Notes:")
    print("  - 'within physical bounds' = final angle in [-90, 90]. MTP clamps; raw forwards.")
    print("  - payload = bytes the executor consumes (compiled DSL vs full LLM text).")
    print("  - latency is the LOCAL deterministic step only; both paths pay the same")
    print("    upstream LLM call, which dominates and is not a differentiator.")
    print(f"  - per-case rows written to {out_csv}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="fuzz set size")
    ap.add_argument("--out", default="kibo_results.csv")
    args = ap.parse_args()
    run(args.n, args.out)
