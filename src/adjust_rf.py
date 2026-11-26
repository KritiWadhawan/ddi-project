"""Apply the same prior adjustment to the RandomForest baseline, for a fair comparison.

src/adjust.py tunes one scalar tau on validation to rescale the GNN's class priors, and
that lifted S1 test macro-F1. But macro-F1 here follows sklearn's default label set
(classes in y_true OR y_pred), so declining to predict rare classes removes zero-F1
entries from the average -- a model can gain without classifying anything better. Tuning
tau for the GNN and not for the baseline would hand the GNN that gain unilaterally.

So: refit the RF, take predict_proba, and run the identical procedure (one shared tau
chosen on mean validation macro-F1 across seeds, test read once at that tau). If the RF
gains as much, the effect is the metric, not the model.

    python src/adjust_rf.py --seeds 0 1 2

Writes results/adjusted_rf_metrics.csv.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from adjust import macro_f1_true_labels, predict
from baselines import CLASSES_0BASED, METRICS, _to_features, fit_rf, scores
from features import build_fp_table
from splits import load_data, make_split
from train import N_CLASSES

ROOT = Path(__file__).resolve().parent.parent
PROBA_DIR = ROOT / "results" / "logits"
OUT_CSV = ROOT / "results" / "adjusted_rf_metrics.csv"


def rf_logits(seed: int, fp_table) -> dict:
    """log predict_proba over all 86 classes, plus labels and the train prior."""
    cache = PROBA_DIR / f"RF_S1_seed{seed}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        print(f"RF seed {seed}: cached probabilities")
        return {k: z[k] for k in z.files}

    train, val, test = make_split("S1", load_data(), seed)
    print(f"RF seed {seed}: fitting on {len(train)} pairs...", flush=True)
    clf = fit_rf(train, fp_table)

    data = {}
    for name, split in [("val", val), ("test", test)]:
        proba = clf.predict_proba(_to_features(split, fp_table))
        # clf.classes_ holds the labels present in train (1..86); widen to all 86 at 0..85.
        full = np.zeros((len(split), N_CLASSES))
        full[:, np.asarray(clf.classes_) - 1] = proba
        # log-probabilities stand in for logits; clip so absent classes are -inf-ish, not nan.
        data[f"{name}_logits"] = np.log(np.clip(full, 1e-12, None))
        data[f"{name}_y"] = split.Y.values - 1
    counts = np.bincount(train.Y.values - 1, minlength=N_CLASSES).astype(float)
    data["prior"] = counts / counts.sum()
    data["meta"] = np.array(["S1", seed, 0], dtype=object)

    PROBA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **data)
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--tau-min", type=float, default=-0.8)
    p.add_argument("--tau-max", type=float, default=0.3)
    p.add_argument("--tau-step", type=float, default=0.05)
    args = p.parse_args()

    fp_table = build_fp_table()
    runs = [rf_logits(s, fp_table) for s in args.seeds]
    taus = np.round(np.arange(args.tau_min, args.tau_max + 1e-9, args.tau_step), 4)

    print("\ntau=0 (unadjusted, should match baseline_metrics.csv):")
    for s, d in zip(args.seeds, runs):
        v = scores(d["val_y"], d["val_logits"].argmax(1), CLASSES_0BASED)["macro_f1"]
        t = scores(d["test_y"], d["test_logits"].argmax(1), CLASSES_0BASED)["macro_f1"]
        print(f"  seed {s}: val {v:.4f}  test {t:.4f}")

    # Selected on macro_f1_occurring, whose label set is np.unique(y_true) and so does
    # not move with tau -- see the note in adjust.py's main() for why that matters.
    curve = np.array([
        np.mean([macro_f1_true_labels(d["val_y"], predict(d["val_logits"], d["prior"], tau))
                 for d in runs])
        for tau in taus
    ])
    tau_star = float(taus[int(curve.argmax())])
    print("\nvalidation macro-F1 vs tau (mean over seeds):")
    for tau, v in zip(taus, curve):
        print(f"  tau={tau:5.2f}  {v:.4f} {'#' * int(round(v * 200))}"
              f"{'   <-- selected' if float(tau) == tau_star else ''}")
    print(f"\nselected tau = {tau_star:.2f} (on validation only)\n")

    rows = []
    for s, d in zip(args.seeds, runs):
        row = {"model": "RF+Morgan", "split": "S1", "seed": s, "tau": tau_star}
        for name in ("val", "test"):
            base = scores(d[f"{name}_y"], d[f"{name}_logits"].argmax(1), CLASSES_0BASED)
            pred = predict(d[f"{name}_logits"], d["prior"], tau_star)
            adj = scores(d[f"{name}_y"], pred, CLASSES_0BASED)
            row |= {f"{name}_{k}": adj[k] for k in METRICS}
            row[f"{name}_macro_f1_base"] = base["macro_f1"]
            row[f"{name}_macro_f1_occurring"] = macro_f1_true_labels(d[f"{name}_y"], pred)
            row[f"{name}_macro_f1_occurring_base"] = macro_f1_true_labels(
                d[f"{name}_y"], d[f"{name}_logits"].argmax(1)
            )
        rows.append(row)
        print(f"  seed {s}: val {row['val_macro_f1_base']:.4f} -> {row['val_macro_f1']:.4f}"
              f"   test {row['test_macro_f1_base']:.4f} -> {row['test_macro_f1']:.4f}")

    for name in ("val", "test"):
        for tag, suffix in [("macro-F1 (default labels)", ""), ("macro-F1 (occurring only)", "_occurring")]:
            base = np.array([r[f"{name}_macro_f1{suffix}_base"] for r in rows])
            adj = np.array([r[f"{name}_macro_f1{suffix}"] for r in rows])
            print(f"\n{name} {tag}: {base.mean():.4f} -> {adj.mean():.4f}"
                  f"   (mean change {adj.mean() - base.mean():+.4f})")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
