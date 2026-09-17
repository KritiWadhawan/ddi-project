"""One-off conversion of the raw DrugBank DDI tab file to parquet.

Run once: python src/convert_to_parquet.py
"""
import pandas as pd
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "drugbank_ddi.tab"
OUT = Path(__file__).resolve().parent.parent / "data" / "drugbank_ddi.parquet"

if __name__ == "__main__":
    df = pd.read_csv(RAW, sep="\t")
    assert set(df.columns) == {"ID1", "ID2", "X1", "X2", "Y", "Map"}, df.columns.tolist()
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} rows, {df.Y.nunique()} classes)")
