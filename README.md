# Predicting Drug–Drug Interaction Types for Unseen Drugs

Given two drugs, predict which of **86 DrugBank interaction types** occurs between them
(for example, one drug increasing the metabolism of the other), with a focus on
**drugs the model has never seen during training**, the realistic case for new drugs.

The model is a graph neural network over molecular graphs: AutoDDI's searched 8-layer
encoder with cross-drug graph attention, plus partner-conditioned co-attention pooling
(after MGDDI). It is compared against a Morgan fingerprint + Random Forest baseline under
three increasingly strict evaluation protocols.

**Status:** data preparation, evaluation splits and baselines are complete. The GNN is
implemented and trained on S1; model screening is in progress (see [Results](#results)).

## Dataset

[TDC DrugBank DDI](https://tdcommons.ai/multi_pred_tasks/ddi/) (Harvard Dataverse file 4139573):
191,808 directed drug pairs, 1,706 drugs (SMILES), 86 interaction types. The labels are
heavily imbalanced: the largest class (Y = 49) is 32% of all pairs.

Cleaning (`src/clean.py`, logged in [`results/cleaning_log.json`](results/cleaning_log.json)):

| Step | Removed |
|---|---|
| Pairs carrying more than one label | 406 pairs (812 rows) |
| Drug with unparseable SMILES (DB11630) | 1 drug (10 rows) |
| **Clean dataset** | **190,986 pairs, 1,705 drugs, 86 classes** |

Each drug becomes a molecular graph (RDKit): atoms are nodes with 55 features (element,
degree, implicit valence, formal charge, radicals, hybridization, aromaticity, hydrogens;
the SSI-DDI/AutoDDI scheme), bonds are edges.

## Evaluation protocols

Drugs are partitioned 60/20/20 into train/validation/test buckets; every split is checked
in code for pair and drug leakage ([`results/split_stats.csv`](results/split_stats.csv)).

| Protocol | Test pairs contain | Seeds |
|---|---|---|
| Transductive | Random pairs; drugs usually seen in training | 1 |
| S2 | Exactly one drug unseen in training | 5 |
| S1 | Two drugs both unseen in training (cold start) | 5 |

**Primary metric: Macro-F1** (every interaction type weighted equally); accuracy and
Micro-F1 are also reported. Models and epochs are chosen on validation Macro-F1 only; the
test set is evaluated once per run.

## Model

```
Drug1 graph ─┐                                              ┌─ Drug2 graph
             ▼        8 cells, weights shared by both drugs ▼
   intra: GCNConv over the drug's own bonds              (64-d)
   inter: GATConv from every atom of the partner drug    (2 heads x 32-d)
   concat → LayerNorm → ELU → dropout
             │  after every cell: co-attention pooling   │
             ▼  (atoms weighted by relevance to partner) ▼
        Drug1 vectors (8 x 128)          Drug2 vectors (8 x 128)
             └──────── concat [Drug1 ‖ Drug2] ───────────┘
                               ▼
            MLP 2048 → 512 → 86 classes (softmax, cross-entropy)
```

- Order is kept ([Drug1 ‖ Drug2]) because many interaction types are directional.
- Optional `--fp`: each drug's Morgan fingerprint is encoded and fused into the classifier.
- Training: Adam/AdamW, optional class weighting (`balanced` or `sqrt`), early stopping
  (patience 10) on validation Macro-F1, a checkpoint every epoch so runs can resume.

## Results

Baselines, test Macro-F1 (mean ± std over seeds; [`results/baseline_summary.csv`](results/baseline_summary.csv)):

| Model | Transductive | S2 | S1 |
|---|---|---|---|
| Majority class | 0.006 | 0.006 ± 0.000 | 0.009 ± 0.001 |
| Morgan fingerprint + Random Forest | 0.799 | 0.585 ± 0.019 | 0.260 ± 0.044 |

GNN screening on S1, seeds 0–2 ([`results/gnn_metrics.csv`](results/gnn_metrics.csv),
per-epoch logs in [`results/runs/`](results/runs/)):

| Model | Val Macro-F1 | Test Macro-F1 |
|---|---|---|
| GNN, plain | 0.210 ± 0.013 | 0.199 ± 0.051 |
| GNN, sqrt class weights + regularization | **0.212 ± 0.016** | **0.235 ± 0.036** |
| Random Forest (same seeds) | 0.230 | 0.278 |

On S1 the GNN currently trails the fingerprint baseline; validation peaks within a few
epochs while training loss keeps falling, which points to overfitting on ~1,000 training
drugs. Next: fingerprint fusion and a 4-layer encoder (round 1b), then the full S1/S2/
transductive runs, ablations (mean pooling, no cross-drug attention), per-class F1 and
attention case studies.

## Repository layout

| Path | Contents |
|---|---|
| `src/convert_to_parquet.py` | Raw TDC tab file → parquet |
| `src/clean.py` | Drops conflicting pairs and invalid SMILES; builds the drug table |
| `src/graphs.py` | Molecular graphs (55-d atom features), cached to `data/drug_graphs.pkl` |
| `src/splits.py` | Transductive / S2 / S1 splits with leakage checks |
| `src/features.py` | Morgan fingerprints (radius 2, 2048 bits) |
| `src/baselines.py` | Majority and Random Forest baselines |
| `src/model.py` | The GNN |
| `src/train.py` | Training, early stopping, evaluation |
| `notebooks/gpu_train.ipynb` | Rebuilds the data and trains on a free Colab/Kaggle GPU |
| `notebooks/explore.ipynb` | Initial data exploration |
| `results/` | Cleaning log, split statistics, baseline and GNN metrics, training logs |

`data/` and `checkpoints/` are not committed; they are rebuilt by the steps below.

## Reproducing

Python 3.11+.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

mkdir -p data
curl -L -o data/drugbank_ddi.tab https://dataverse.harvard.edu/api/access/datafile/4139573

python src/convert_to_parquet.py  # raw file → parquet
python src/clean.py               # clean data + drug table (matches results/cleaning_log.json)
python src/graphs.py              # molecular graph cache
python src/splits.py              # build and verify all splits → results/split_stats.csv
python src/baselines.py           # baselines → results/baseline_*.csv

# GNN (a GPU is strongly recommended: ~50 s/epoch on a T4 vs 20–40 min on a CPU)
python src/train.py --split S1 --seed 0 --class-weights sqrt \
    --gnn-dropout 0.3 --adamw --weight-decay 1e-2 --lr 5e-4
```

For a free GPU, open [`notebooks/gpu_train.ipynb`](notebooks/gpu_train.ipynb) in Colab
(Runtime → T4 GPU → Run all); it clones this repo, rebuilds the data, checks it against the
committed cleaning log and runs the configured experiments.

## References

- Therapeutics Data Commons, DrugBank DDI dataset — dataset and splits source.
- AutoDDI ([code](https://github.com/Zhen-Peng-Wu/AutoDDI)) — searched GNN encoder architecture and atom features.
- MGDDI — partner-conditioned (co-attention) pooling.
- SSI-DDI — original 55-dimensional atom featurization.
