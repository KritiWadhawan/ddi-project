"""Refill results/gnn_metrics.csv with the current METRICS, from saved checkpoints.

scores() gained macro_f1_occurring (the primary) and macro_f1_all86 after most runs had
already been recorded, so their rows carry only the old columns. This re-runs inference
for each checkpoint given and rewrites the matching rows in place, leaving every other
row untouched. Metrics come from the same scores() the training loop uses, so a rescored
row is identical to what that run would write today.

    python src/rescore.py --ckpt "checkpoints/*.pt"              # whatever is local
    python src/rescore.py --ckpt "/content/drive/MyDrive/ddi_project_runs/checkpoints/*.pt"

Only best-epoch checkpoints are read; *_last.pt files are skipped (they hold optimizer
state for resuming, and their weights are the last epoch, not the selected one).
"""
import argparse
import csv
import glob
from pathlib import Path

import torch

from adjust import logits_for
from baselines import CLASSES_0BASED, METRICS, scores

ROOT = Path(__file__).resolve().parent.parent
METRICS_CSV = ROOT / "results" / "gnn_metrics.csv"
KEY_COLS = ["split", "seed", "model", "best_epoch", "set"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True, help="checkpoint paths (globs allowed)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    args = p.parse_args()

    paths = sorted({Path(q) for pattern in args.ckpt for q in glob.glob(pattern)
                    if not q.endswith("_last.pt")})
    if not paths:
        raise SystemExit(f"no best-epoch checkpoints matched {args.ckpt}")

    with open(METRICS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    by_key = {(r["model"], r["set"]): r for r in rows}

    updated, missing = 0, []
    for q in paths:
        run = q.stem
        if (run, "val") not in by_key:
            missing.append(run)
            print(f"{run}: no row in {METRICS_CSV.name}; skipped "
                  f"(rescore only refills existing rows)")
            continue
        d = logits_for(q, args.device, args.batch_size)
        for set_name in ("val", "test"):
            m = scores(d[f"{set_name}_y"], d[f"{set_name}_logits"].argmax(1), CLASSES_0BASED)
            row = by_key[(run, set_name)]
            before = row.get("macro_f1")
            # The unchanged metric must reproduce exactly, or the reload is wrong.
            if before and abs(float(before) - m["macro_f1"]) > 1e-9:
                raise SystemExit(
                    f"{run}/{set_name}: macro_f1 changed on reload "
                    f"({float(before):.6f} -> {m['macro_f1']:.6f}); refusing to write"
                )
            row.update({k: m[k] for k in METRICS})
            updated += 1
        print(f"{run}: val macro_f1_occurring={by_key[(run, 'val')]['macro_f1_occurring']:.4f}"
              f"  test={by_key[(run, 'test')]['macro_f1_occurring']:.4f}")

    if args.dry_run:
        print(f"\ndry run: {updated} rows would be rewritten")
        return

    with open(METRICS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=KEY_COLS + METRICS)
        writer.writeheader()
        writer.writerows(rows)

    blank = sum(1 for r in rows if not r.get("macro_f1_occurring"))
    print(f"\nrewrote {updated} rows in {METRICS_CSV.relative_to(ROOT)}; "
          f"{blank} rows still lack the new metrics (checkpoint not available locally)")
    if missing:
        print("checkpoints with no matching row:", ", ".join(missing))


if __name__ == "__main__":
    main()
