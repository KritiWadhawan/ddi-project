"""Morgan fingerprints for the RandomForest baseline, cached per drug ID."""
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

CACHE = Path(__file__).resolve().parent.parent / "data" / "morgan_fp_cache.npz"

_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def _fp(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(1024, dtype=np.uint8)
    return np.array(_GEN.GetFingerprint(mol), dtype=np.uint8)


def build_fp_table(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Map drug ID -> 1024-bit Morgan fingerprint, computed once per unique drug."""
    if CACHE.exists():
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}

    id_to_smiles = dict(zip(df.ID1, df.X1)) | dict(zip(df.ID2, df.X2))
    table = {drug_id: _fp(smi) for drug_id, smi in id_to_smiles.items()}
    np.savez_compressed(CACHE, **table)
    return table


if __name__ == "__main__":
    from splits import load_data

    df = load_data()
    table = build_fp_table(df)
    print(f"cached {len(table)} drug fingerprints -> {CACHE}")
