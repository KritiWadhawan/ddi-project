"""Morgan fingerprints for the RandomForest baseline, cached per drug ID."""
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parent.parent
DRUGS = ROOT / "data" / "drugs.parquet"

RADIUS = 2
N_BITS = 2048
CACHE = ROOT / "data" / f"morgan_r{RADIUS}_{N_BITS}_cache.npz"

_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)


def _fp(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES (clean.py should have dropped it): {smiles}")
    return np.array(_GEN.GetFingerprint(mol), dtype=np.uint8)


def build_fp_table() -> dict[str, np.ndarray]:
    """Map drug ID -> Morgan fingerprint, computed once per drug in the cleaned drug table."""
    if CACHE.exists():
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}

    drugs = pd.read_parquet(DRUGS)
    table = {drug_id: _fp(smi) for drug_id, smi in zip(drugs.ID, drugs.SMILES)}
    np.savez_compressed(CACHE, **table)
    return table


if __name__ == "__main__":
    table = build_fp_table()
    print(f"cached {len(table)} drug fingerprints ({N_BITS} bits) -> {CACHE}")
