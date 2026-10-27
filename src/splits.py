"""The three evaluation splits for DDI type prediction.

- transductive_split: random 60/20/20 over pairs. Upper bound, not the real result.
- s2_split: one drug per test pair unseen during training (split by drug).
- s1_split: both drugs per test pair unseen during training (split by drug).

S1/S2 partition drugs into disjoint train/val/test buckets first, then keep only
pairs consistent with that bucket's definition. Pairs that straddle two non-train
buckets (e.g. one val drug + one test drug) are dropped rather than assigned
arbitrarily.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clean import pair_key

DATA = Path(__file__).resolve().parent.parent / "data" / "drugbank_ddi_clean.parquet"
SPLITS_DIR = Path(__file__).resolve().parent.parent / "results" / "splits"


def load_data() -> pd.DataFrame:
    return pd.read_parquet(DATA)


def _save_drug_lists(name: str, seed: int, **buckets) -> Path:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    out = {k: sorted(v) for k, v in buckets.items()}
    path = SPLITS_DIR / f"{name}_seed{seed}_drugs.json"
    path.write_text(json.dumps(out, indent=2))
    return path


def transductive_split(df: pd.DataFrame, seed: int = 0, ratios=(0.6, 0.2, 0.2)):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_train = int(ratios[0] * len(df))
    n_val = int(ratios[1] * len(df))
    train = df.iloc[idx[:n_train]]
    val = df.iloc[idx[n_train : n_train + n_val]]
    test = df.iloc[idx[n_train + n_val :]]
    return train, val, test


def _split_drugs(df: pd.DataFrame, seed: int, ratios):
    drugs = sorted(set(df.ID1) | set(df.ID2))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(drugs)
    n_train = int(ratios[0] * len(perm))
    n_val = int(ratios[1] * len(perm))
    train_d = set(perm[:n_train])
    val_d = set(perm[n_train : n_train + n_val])
    test_d = set(perm[n_train + n_val :])
    return train_d, val_d, test_d


def s1_split(df: pd.DataFrame, seed: int = 0, ratios=(0.6, 0.2, 0.2)):
    """Both drugs in each test/val pair are unseen during training."""
    train_d, val_d, test_d = _split_drugs(df, seed, ratios)
    _save_drug_lists("S1", seed, train=train_d, val=val_d, test=test_d)

    def both_in(bucket):
        return df.ID1.isin(bucket) & df.ID2.isin(bucket)

    return df[both_in(train_d)], df[both_in(val_d)], df[both_in(test_d)]


def s2_split(df: pd.DataFrame, seed: int = 0, ratios=(0.6, 0.2, 0.2)):
    """Exactly one drug in each test/val pair is unseen during training."""
    train_d, val_d, test_d = _split_drugs(df, seed, ratios)
    _save_drug_lists("S2", seed, train=train_d, val=val_d, test=test_d)

    train = df[df.ID1.isin(train_d) & df.ID2.isin(train_d)]
    # A train-bucket drug whose partners all fell in other buckets has no training
    # pairs, so it's unseen too; only drugs that actually occur in train count as seen.
    seen = set(train.ID1) | set(train.ID2)

    def one_train_one_other(other_bucket):
        return (df.ID1.isin(seen) & df.ID2.isin(other_bucket)) | (
            df.ID2.isin(seen) & df.ID1.isin(other_bucket)
        )

    val = df[one_train_one_other(val_d)]
    test = df[one_train_one_other(test_d)]
    return train, val, test


SPLIT_FNS = {"transductive": transductive_split, "S1": s1_split, "S2": s2_split}

# S1's test set is small (both drugs held out), so it gets several seeds. S1 and S2
# with the same seed share one drug partition. Transductive test sets are large
# enough that one seed is stable.
SEEDS = {"transductive": [0], "S1": [0, 1, 2, 3, 4], "S2": [0, 1, 2, 3, 4]}

STATS_CSV = Path(__file__).resolve().parent.parent / "results" / "split_stats.csv"


def _drugs(split: pd.DataFrame) -> set:
    return set(split.ID1) | set(split.ID2)


def verify_split(name: str, train, val, test) -> None:
    """Fail loudly on pair leakage (either direction) or drug leakage for S1/S2."""
    keys = {k: set(pair_key(s)) for k, s in [("train", train), ("val", val), ("test", test)]}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        shared = keys[a] & keys[b]
        assert not shared, f"{name}: {len(shared)} pairs shared between {a} and {b}"

    if name == "transductive":
        return
    train_drugs = _drugs(train)
    for set_name, split in [("val", val), ("test", test)]:
        n_seen = split.ID1.isin(train_drugs).astype(int) + split.ID2.isin(train_drugs).astype(int)
        expected = {"S1": 0, "S2": 1}[name]
        assert (n_seen == expected).all(), (
            f"{name} {set_name}: {(n_seen != expected).sum()} pairs don't have exactly "
            f"{expected} training drug(s)"
        )
    if name == "S1":
        assert not (_drugs(val) | _drugs(test)) & train_drugs


def split_stats(name: str, seed: int, train, val, test, n_pairs: int) -> dict:
    """Sizes and class coverage: rare classes drop out of small test sets, and classes
    absent from train can't be predicted at all."""
    train_classes = set(train.Y)
    return {
        "split": name,
        "seed": seed,
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "kept_pct": round(100 * (len(train) + len(val) + len(test)) / n_pairs, 1),
        "classes_train": len(train_classes),
        "classes_val": val.Y.nunique(),
        "classes_test": test.Y.nunique(),
        "test_classes_unseen_in_train": len(set(test.Y) - train_classes),
    }


def make_split(name: str, df: pd.DataFrame, seed: int):
    train, val, test = SPLIT_FNS[name](df, seed=seed)
    verify_split(name, train, val, test)
    return train, val, test


if __name__ == "__main__":
    df = load_data()
    print(f"loaded {len(df)} pairs, {df.Y.nunique()} classes\n")
    rows = []
    for name, seeds in SEEDS.items():
        for seed in seeds:
            train, val, test = make_split(name, df, seed)
            rows.append(split_stats(name, seed, train, val, test, len(df)))
    stats = pd.DataFrame(rows)
    stats.to_csv(STATS_CSV, index=False)
    print(stats.to_string(index=False))
    print(f"\nall splits verified; wrote {STATS_CSV}")
