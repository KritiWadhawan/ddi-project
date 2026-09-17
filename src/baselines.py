"""Baselines, in order: majority class, then Morgan fingerprint + RandomForest.

Run for all three eval protocols so results are read against the same floor
the doc calls out: always-predict-Y49 = 32% accuracy, macro-F1 is primary.
"""
import csv
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from features import build_fp_table
from splits import SPLIT_FNS, load_data

RESULTS_CSV = Path(__file__).resolve().parent.parent / "results" / "baseline_metrics.csv"


def majority_baseline(train, val, test):
    majority = train.Y.mode().iloc[0]
    results = {}
    for name, split in [("val", val), ("test", test)]:
        pred = np.full(len(split), majority)
        results[name] = {
            "acc": accuracy_score(split.Y, pred),
            "macro_f1": f1_score(split.Y, pred, average="macro", zero_division=0),
        }
    return results


def _to_features(split, fp_table):
    fp1 = np.stack([fp_table[i] for i in split.ID1])
    fp2 = np.stack([fp_table[i] for i in split.ID2])
    return np.concatenate([fp1, fp2], axis=1)


def rf_baseline(train, val, test, fp_table):
    X_train = _to_features(train, fp_table)
    clf = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=0, class_weight="balanced_subsample"
    )
    clf.fit(X_train, train.Y)

    results = {}
    for name, split in [("val", val), ("test", test)]:
        X = _to_features(split, fp_table)
        pred = clf.predict(X)
        results[name] = {
            "acc": accuracy_score(split.Y, pred),
            "macro_f1": f1_score(split.Y, pred, average="macro", zero_division=0),
        }
    return results


if __name__ == "__main__":
    df = load_data()
    fp_table = build_fp_table(df)

    rows = []
    print(f"{'split':12s} {'model':10s} {'set':5s} {'acc':>7s} {'macro-F1':>9s}")
    for split_name, fn in SPLIT_FNS.items():
        train, val, test = fn(df, seed=0)

        maj = majority_baseline(train, val, test)
        for set_name, m in maj.items():
            print(f"{split_name:12s} {'majority':10s} {set_name:5s} {m['acc']:7.3f} {m['macro_f1']:9.3f}")
            rows.append({"split": split_name, "model": "majority", "set": set_name, **m})

        t0 = time.time()
        rf = rf_baseline(train, val, test, fp_table)
        dt = time.time() - t0
        for set_name, m in rf.items():
            print(f"{split_name:12s} {'RF+Morgan':10s} {set_name:5s} {m['acc']:7.3f} {m['macro_f1']:9.3f}")
            rows.append({"split": split_name, "model": "RF+Morgan", "set": set_name, **m})
        print(f"  (RF train+eval took {dt:.1f}s, train size={len(train)})\n")

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "model", "set", "acc", "macro_f1"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {RESULTS_CSV}")
