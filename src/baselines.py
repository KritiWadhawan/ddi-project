"""Baselines, in order: majority class, then Morgan fingerprint + RandomForest.

Run for all three eval protocols (and every seed in splits.SEEDS) so results are read
against the same floor the doc calls out: always-predict-Y49 = 32% accuracy, macro-F1
is primary.

Three macro-F1 variants are recorded, because the averaging label set turns out to
matter more than expected:

  macro_f1_occurring  PRIMARY. Averaged over the classes present in y_true for that
                      eval set. Fixed by the data, so no choice the model makes can
                      change the divisor.
  macro_f1_all86      Averaged over all 86 classes. Classes absent from the eval set
                      score 0 for every model, which deflates the absolute number but
                      keeps one divisor across splits and seeds.
  macro_f1            sklearn's default: classes in y_true OR y_pred. Kept so earlier
                      results stay readable, but NOT comparable between models -- a
                      model that predicts fewer rare classes removes its own zero-F1
                      entries and scores higher without classifying anything better.
                      results/logit_adjustment.md documents where this was found.

Micro-F1 equals accuracy for single-label multiclass; it's reported because the results
table asks for it.
"""
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from features import build_fp_table
from splits import SEEDS, load_data, make_split

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS_CSV = RESULTS / "baseline_metrics.csv"
SUMMARY_CSV = RESULTS / "baseline_summary.csv"
N_CLASSES = 86
METRICS = ["acc", "macro_f1_occurring", "macro_f1_all86", "macro_f1", "micro_f1"]

# The label encoding differs by caller: this module scores the raw Y column (1..86),
# while train.py and adjust.py score the model's 0..85. macro_f1_all86 needs the right
# one, so callers pass it explicitly rather than letting it be guessed from the data.
CLASSES_1BASED = np.arange(1, N_CLASSES + 1)
CLASSES_0BASED = np.arange(N_CLASSES)


def scores(y_true, y_pred, all_labels) -> dict:
    """All three macro-F1 variants; see the module docstring for which to report.

    all_labels: the project's full 86-class label set in the caller's encoding, i.e.
    CLASSES_1BASED when scoring the raw Y column, CLASSES_0BASED for model outputs.
    """
    return {
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1_occurring": f1_score(
            y_true, y_pred, average="macro", labels=np.unique(y_true), zero_division=0
        ),
        "macro_f1_all86": f1_score(
            y_true, y_pred, average="macro", labels=all_labels, zero_division=0
        ),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
    }


def majority_baseline(train, val, test):
    majority = train.Y.mode().iloc[0]
    return {
        name: scores(split.Y, np.full(len(split), majority), CLASSES_1BASED)
        for name, split in [("val", val), ("test", test)]
    }


def _to_features(split, fp_table):
    fp1 = np.stack([fp_table[i] for i in split.ID1])
    fp2 = np.stack([fp_table[i] for i in split.ID2])
    return np.concatenate([fp1, fp2], axis=1)


def fit_rf(train, fp_table):
    clf = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=0, class_weight="balanced_subsample"
    )
    clf.fit(_to_features(train, fp_table), train.Y)
    return clf


def rf_baseline(clf, val, test, fp_table):
    return {
        name: scores(split.Y, clf.predict(_to_features(split, fp_table)), CLASSES_1BASED)
        for name, split in [("val", val), ("test", test)]
    }


if __name__ == "__main__":
    df = load_data()
    fp_table = build_fp_table()

    # S1 and S2 with the same seed train on the identical set (both drugs in the
    # train bucket), so runs are ordered by seed and each forest is reused for the
    # other protocol. Only the latest forest is kept; they're large.
    runs = sorted(
        ((name, seed) for name, seeds in SEEDS.items() for seed in seeds),
        key=lambda r: (r[0] != "transductive", r[1]),
    )
    fitted_key, clf = None, None

    rows = []
    print(f"{'split':12s} {'seed':>4s} {'model':10s} {'set':5s} {'acc':>7s} {'macro-F1':>9s}")
    for split_name, seed in runs:
        train, val, test = make_split(split_name, df, seed)

        t0 = time.time()
        key = (seed, hash(train.index.values.tobytes()))
        if key != fitted_key:
            clf = None
            clf, fitted_key = fit_rf(train, fp_table), key
        results = {
            "majority": majority_baseline(train, val, test),
            "RF+Morgan": rf_baseline(clf, val, test, fp_table),
        }
        for model, by_set in results.items():
            for set_name, m in by_set.items():
                print(
                    f"{split_name:12s} {seed:4d} {model:10s} {set_name:5s} "
                    f"{m['acc']:7.3f} {m['macro_f1']:9.3f}",
                    flush=True,
                )
                rows.append({"split": split_name, "seed": seed, "model": model, "set": set_name, **m})
        print(f"  ({time.time() - t0:.1f}s, train size={len(train)})\n", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "seed", "model", "set", *METRICS])
        writer.writeheader()
        writer.writerows(rows)

    summary = (
        pd.DataFrame(rows)
        .groupby(["split", "model", "set"], sort=False)[METRICS]
        .agg(["mean", "std"])
        .round(3)
    )
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary.insert(0, "n_seeds", pd.DataFrame(rows).groupby(["split", "model", "set"], sort=False).size())
    summary.to_csv(SUMMARY_CSV)
    print(summary.to_string())
    print(f"\nwrote {RESULTS_CSV}\nwrote {SUMMARY_CSV}")
