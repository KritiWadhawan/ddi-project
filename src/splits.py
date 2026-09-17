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

DATA = Path(__file__).resolve().parent.parent / "data" / "drugbank_ddi.parquet"
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

    def one_train_one_other(other_bucket):
        return (df.ID1.isin(train_d) & df.ID2.isin(other_bucket)) | (
            df.ID2.isin(train_d) & df.ID1.isin(other_bucket)
        )

    train = df[df.ID1.isin(train_d) & df.ID2.isin(train_d)]
    val = df[one_train_one_other(val_d)]
    test = df[one_train_one_other(test_d)]
    return train, val, test


SPLIT_FNS = {"transductive": transductive_split, "S1": s1_split, "S2": s2_split}


if __name__ == "__main__":
    df = load_data()
    print(f"loaded {len(df)} pairs, {df.Y.nunique()} classes\n")
    for name, fn in SPLIT_FNS.items():
        train, val, test = fn(df, seed=0)
        kept = len(train) + len(val) + len(test)
        print(
            f"{name:12s} train={len(train):7d} val={len(val):6d} test={len(test):6d} "
            f"(kept {kept}/{len(df)} = {100*kept/len(df):.1f}%)"
        )
