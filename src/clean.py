"""Phase 1 cleaning: drop conflicting pairs and unparseable drugs, build the drug table.

- Conflicting pairs: the raw file has no exact duplicate rows, but some drug pairs
  appear more than once (as (A,B) twice, or as (A,B) and (B,A)) with different Y.
  A single-label classifier can't fit both targets, and in the transductive split
  the copies can straddle train/test, so every row of such a pair is dropped.
- Unparseable drugs: drugs whose SMILES RDKit can't parse are dropped, along with
  every pair they appear in.

Writes data/drugbank_ddi_clean.parquet, data/drugs.parquet (ID -> SMILES), and the
counts to results/cleaning_log.json (data/ is gitignored, the log is not).

Run once after convert_to_parquet.py: python src/clean.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "drugbank_ddi.parquet"
CLEAN = ROOT / "data" / "drugbank_ddi_clean.parquet"
DRUGS = ROOT / "data" / "drugs.parquet"
LOG = ROOT / "results" / "cleaning_log.json"


def pair_key(df: pd.DataFrame) -> pd.Series:
    """Order-independent pair ID, so (A,B) and (B,A) map to the same key."""
    a, b = df.ID1.to_numpy(), df.ID2.to_numpy()
    return pd.Series(np.where(a < b, a + "|" + b, b + "|" + a), index=df.index)


def build_drug_table(df: pd.DataFrame) -> pd.DataFrame:
    stacked = pd.concat(
        [
            df[["ID1", "X1"]].set_axis(["ID", "SMILES"], axis=1),
            df[["ID2", "X2"]].set_axis(["ID", "SMILES"], axis=1),
        ]
    )
    # One SMILES per ID: catch the dataset disagreeing with itself before we dedupe on ID.
    n_smiles = stacked.groupby("ID").SMILES.nunique()
    assert (n_smiles == 1).all(), n_smiles[n_smiles > 1]
    return stacked.drop_duplicates("ID").sort_values("ID").reset_index(drop=True)


if __name__ == "__main__":
    RDLogger.DisableLog("rdApp.*")
    df = pd.read_parquet(RAW)
    log = {"raw_rows": len(df), "raw_classes": int(df.Y.nunique())}

    # 1.1 duplicates
    log["exact_duplicate_rows"] = int(df.duplicated(["ID1", "ID2", "Y"]).sum())
    key = pair_key(df)
    labels_per_pair = df.groupby(key).Y.nunique()
    conflicting = set(labels_per_pair[labels_per_pair > 1].index)
    is_conflicting = key.isin(conflicting)
    log["conflicting_pairs"] = len(conflicting)
    log["conflicting_rows_dropped"] = int(is_conflicting.sum())
    df = df[~is_conflicting]
    # Any remaining repeats of a pair would carry the same label; none expected, but check.
    log["same_label_repeat_rows_dropped"] = int(pair_key(df).duplicated().sum())
    df = df[~pair_key(df).duplicated()]

    # 1.2 + 1.3 drug table, SMILES validation
    drugs = build_drug_table(df)
    log["drugs_before_validation"] = len(drugs)
    valid = drugs.SMILES.map(lambda s: Chem.MolFromSmiles(s) is not None)
    bad_ids = sorted(drugs.ID[~valid])
    log["invalid_smiles_drugs"] = {i: drugs.SMILES[drugs.ID == i].iloc[0] for i in bad_ids}
    touches_bad = df.ID1.isin(bad_ids) | df.ID2.isin(bad_ids)
    log["rows_dropped_for_invalid_smiles"] = int(touches_bad.sum())
    df = df[~touches_bad].reset_index(drop=True)
    drugs = build_drug_table(df)

    log["clean_rows"] = len(df)
    log["clean_drugs"] = len(drugs)
    log["clean_classes"] = int(df.Y.nunique())
    log["classes_lost"] = sorted(
        int(y) for y in set(pd.read_parquet(RAW).Y) - set(df.Y)
    )

    df.to_parquet(CLEAN, index=False)
    drugs.to_parquet(DRUGS, index=False)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))
    print(f"\nwrote {CLEAN}\nwrote {DRUGS}\nwrote {LOG}")
