"""Refill the RF+Morgan rows of results/baseline_metrics.csv with the current METRICS.

Same job as rescore.py does for the GNN, but the baseline has no checkpoint to reload --
it has the predict_proba arrays that adjust_rf.py cached under results/logits/RF_*.npz.
Only the (split, seed) pairs with a cached file are refilled; everything else is left
alone. Re-running src/baselines.py regenerates the whole file from scratch, but that
refits every forest, so this fills in the rows needed for the S1 comparison meanwhile.

    python src/rescore_rf.py --seeds 0 1 2
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from baselines import CLASSES_0BASED, METRICS, scores

ROOT = Path(__file__).resolve().parent.parent
BASELINE_CSV = ROOT / "results" / "baseline_metrics.csv"
CACHE = ROOT / "results" / "logits"
KEY_COLS = ["split", "seed", "model", "set"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="S1")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    with open(BASELINE_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    by_key = {(r["split"], int(r["seed"]), r["model"], r["set"]): r for r in rows}

    updated = 0
    for seed in args.seeds:
        f = CACHE / f"RF_{args.split}_seed{seed}.npz"
        if not f.exists():
            print(f"seed {seed}: no cached probabilities at {f.name}; skipped")
            continue
        z = np.load(f, allow_pickle=True)
        for set_name in ("val", "test"):
            m = scores(z[f"{set_name}_y"], z[f"{set_name}_logits"].argmax(1), CLASSES_0BASED)
            key = (args.split, seed, "RF+Morgan", set_name)
            if key not in by_key:
                print(f"{key}: no such row; skipped")
                continue
            row = by_key[key]
            before = row.get("macro_f1")
            # The unchanged metric must reproduce, or the cached probabilities don't
            # correspond to the forest that produced this row.
            if before and abs(float(before) - m["macro_f1"]) > 1e-6:
                raise SystemExit(
                    f"{key}: macro_f1 changed ({float(before):.6f} -> {m['macro_f1']:.6f}); "
                    "refusing to write"
                )
            row.update({k: m[k] for k in METRICS})
            updated += 1
        print(f"seed {seed}: val occurring={by_key[(args.split, seed, 'RF+Morgan', 'val')]['macro_f1_occurring']:.4f}"
              f"  test occurring={by_key[(args.split, seed, 'RF+Morgan', 'test')]['macro_f1_occurring']:.4f}")

    if args.dry_run:
        print(f"\ndry run: {updated} rows would be rewritten")
        return

    with open(BASELINE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=KEY_COLS + METRICS)
        writer.writeheader()
        writer.writerows(rows)
    blank = sum(1 for r in rows if not r.get("macro_f1_occurring"))
    print(f"\nrewrote {updated} rows in {BASELINE_CSV.relative_to(ROOT)}; "
          f"{blank} rows still lack the new metrics (re-run src/baselines.py to fill them)")


if __name__ == "__main__":
    main()
