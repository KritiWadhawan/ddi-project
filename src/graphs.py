"""Molecular graphs for the GNN, built once per drug and cached to a single pickle.

Atom features (55-dim): copied from AutoDDI's data_preprocessing.atom_features
(github.com/Zhen-Peng-Wu/AutoDDI), which is itself SSI-DDI's featurizer, so node
inputs match the published DrugBank models:
    element one-hot (43 symbols + Unknown)              44
    degree/10, implicit valence, formal charge, radicals  4  (scalars, not one-hot)
    hybridization one-hot (SP, SP2, SP3, SP3D, SP3D2)     5
    aromatic                                              1
    total H count                                         1  (scalar)

Bond features (12-dim): AutoDDI uses topology only (no bond features). These follow
DGL-LifeSci's CanonicalBondFeaturizer: bond type one-hot (4), conjugated, in ring,
stereo one-hot (6). Edges are stored in both directions with matching edge_attr.

Atom order in each graph is RDKit's atom order for the stored SMILES, so attention
weights can be mapped back onto the molecule for the Phase 4 case studies.

Run after clean.py: python src/graphs.py
"""
import pickle
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parent.parent
DRUGS = ROOT / "data" / "drugs.parquet"
CACHE = ROOT / "data" / "drug_graphs.pkl"

ATOM_SYMBOLS = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe", "As", "Al",
    "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H",
    "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb", "Unknown",
]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREO = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]
ATOM_DIM = 55
BOND_DIM = 12


def _one_hot_unk(x, allowed):
    """One-hot over `allowed`; unseen values map to the last entry (as in AutoDDI)."""
    if x not in allowed:
        x = allowed[-1]
    return [float(x == a) for a in allowed]


def _one_hot(x, allowed):
    """One-hot over `allowed`; unseen values give all zeros."""
    return [float(x == a) for a in allowed]


def atom_features(atom: Chem.Atom) -> list[float]:
    return (
        _one_hot_unk(atom.GetSymbol(), ATOM_SYMBOLS)
        + [
            atom.GetDegree() / 10,
            atom.GetValence(Chem.ValenceType.IMPLICIT),  # AutoDDI's GetImplicitValence(), renamed in RDKit
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
        ]
        + _one_hot_unk(atom.GetHybridization(), HYBRIDIZATIONS)
        + [float(atom.GetIsAromatic())]
        + [atom.GetTotalNumHs()]
    )


def bond_features(bond: Chem.Bond) -> list[float]:
    return (
        _one_hot(bond.GetBondType(), BOND_TYPES)
        + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
        + _one_hot(bond.GetStereo(), BOND_STEREO)
    )


def mol_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES (clean.py should have dropped it): {smiles}")

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    src, dst, attr = [], [], []
    for b in mol.GetBonds():
        i, j, f = b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bond_features(b)
        src += [i, j]
        dst += [j, i]
        attr += [f, f]
    edge_index = torch.tensor([src, dst], dtype=torch.long).view(2, -1)
    edge_attr = torch.tensor(attr, dtype=torch.float).view(-1, BOND_DIM)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)


def load_graphs() -> dict[str, Data]:
    with open(CACHE, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    drugs = pd.read_parquet(DRUGS)
    graphs = {drug_id: mol_to_graph(smi) for drug_id, smi in zip(drugs.ID, drugs.SMILES)}

    assert len(graphs) == len(drugs)
    assert all(g.x.shape[1] == ATOM_DIM for g in graphs.values())
    assert all(g.edge_attr.shape[1] == BOND_DIM for g in graphs.values())

    with open(CACHE, "wb") as f:
        pickle.dump(graphs, f)

    n_atoms = torch.tensor([g.num_nodes for g in graphs.values()])
    n_edges = torch.tensor([g.num_edges for g in graphs.values()])
    unknown = sum(int(g.x[:, ATOM_SYMBOLS.index("Unknown")].sum()) for g in graphs.values())
    no_bonds = [d for d, g in graphs.items() if g.num_edges == 0]
    print(f"cached {len(graphs)} graphs -> {CACHE}")
    print(f"atoms/mol: min={n_atoms.min()} median={n_atoms.median()} max={n_atoms.max()}")
    print(f"directed edges/mol: median={n_edges.median()} max={n_edges.max()}")
    print(f"atoms mapped to 'Unknown' element: {unknown}")
    print(f"drugs with no bonds ({len(no_bonds)}): {no_bonds}")
