# Predicting Drug–Drug Interaction Types for Unseen Drugs

Given two drugs, predict which of **86 DrugBank interaction types** occurs between them
(for example, one drug increasing the metabolism of the other), with a focus on
**drugs the model has never seen during training**, the realistic case for new drugs.

The model is a graph neural network over molecular graphs: AutoDDI's searched 8-layer
encoder with cross-drug graph attention, plus partner-conditioned co-attention pooling
(after MGDDI). It is compared against a Morgan fingerprint + Random Forest baseline under
three increasingly strict evaluation protocols.

**Status:** data preparation, evaluation splits and baselines are complete. The GNN is
implemented, and screening on the cold-start S1 split has selected a 4-layer encoder with
fingerprint fusion. It beats the Random Forest baseline as trained (+0.032 Macro-F1 on all
three seeds) but is level with that baseline once the baseline is given the same
validation-tuned prior adjustment, so S1 is parity. The full S1/S2/transductive runs and
ablations are next (see [Results](#results)).

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

**Primary metric: `macro_f1_occurring`** — Macro-F1 averaged over the interaction types
present in the evaluation set, so every type counts equally and the divisor is fixed by the
data. `macro_f1_all86` (all 86 types) and accuracy / Micro-F1 are reported alongside.

A third column, `macro_f1`, uses sklearn's default label set (types in `y_true` **or**
`y_pred`). It is kept so earlier runs stay readable but is **not comparable between
models**: a model that predicts fewer rare types deletes its own zero-F1 entries and scores
higher without classifying anything better. That is not hypothetical — it produced a
spurious +0.021 before being caught, which is why the primary metric changed
([`results/logit_adjustment.md`](results/logit_adjustment.md)).

Models and epochs are chosen on validation only; the test set is evaluated once per run.

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

Baselines, test Macro-F1 (mean ± std over seeds; [`results/baseline_summary.csv`](results/baseline_summary.csv)).
These are in the old `macro_f1` column — re-running `src/baselines.py` refits every forest
and refills them under the primary metric:

| Model | Transductive | S2 | S1 |
|---|---|---|---|
| Majority class | 0.006 | 0.006 ± 0.000 | 0.009 ± 0.001 |
| Morgan fingerprint + Random Forest | 0.799 | 0.585 ± 0.019 | 0.260 ± 0.044 |

GNN screening on S1, seeds 0–2 ([`results/gnn_metrics.csv`](results/gnn_metrics.csv),
per-epoch logs in [`results/runs/`](results/runs/)):

| Model | Params | Val Macro-F1 | Test Macro-F1 |
|---|---|---|---|
| GNN, 8 layers | 1.54 M | 0.212 ± 0.016 | 0.235 ± 0.036 |
| GNN, 4 layers | 0.79 M | 0.215 ± 0.005 | 0.242 ± 0.055 |
| GNN, 8 layers + fingerprint fusion | 2.33 M | **0.258 ± 0.020** | 0.270 ± 0.053 |
| GNN, 4 layers + fingerprint fusion | 1.57 M | 0.258 ± 0.008 | **0.273 ± 0.040** |
| Random Forest (same seeds) | — | 0.230 ± 0.014 | 0.278 ± 0.042 |

All GNN rows use sqrt class weights and the same regularization (dropout 0.3, AdamW,
weight decay 1e-2, lr 5e-4); the encoder depth and `--fp` are the only differences. These
screening numbers are in the old `macro_f1` column, which is consistent across the rows and
so still valid for *choosing between variants* — selection was by validation anyway. They
are not the numbers to quote; see the headline comparison below.

Fusing the Morgan fingerprint into the classifier is what closes the cold-start gap, and
depth is nearly irrelevant next to it (0.235 → 0.242 for 4 layers alone, versus 0.273 with
fusion). The two fusion variants are indistinguishable on validation (0.2583 vs 0.2579), so
the 4-layer model is carried forward on grounds independent of the test set: half the
parameters, ~40% less time per epoch, and a third of the variance across seeds.

### Headline S1 comparison

The selected configuration against the baseline, under the primary metric, S1 test, seeds
0–2. Only these rows have been rescored so far; `src/rescore.py` and `src/rescore_rf.py`
refill the rest from saved checkpoints.

| S1 test, seeds 0–2 | Macro-F1 (occurring) | Macro-F1 (all 86) |
|---|---|---|
| GNN, 4 layers + fingerprint fusion | **0.320 ± 0.040** | 0.221 ± 0.043 |
| Morgan fingerprint + Random Forest | 0.289 ± 0.043 | 0.199 ± 0.044 |
| Random Forest + validation-tuned prior adjustment | **0.322 ± 0.036** | — |

**As trained, the GNN beats the baseline on cold start by 0.032, on all three seeds
individually** (+0.029, +0.030, +0.036) — the first protocol where it does.

**But the fair comparison is parity.** Investigating prior adjustment for the GNN (where it
gave nothing) revealed that the same correction helps the *Random Forest* substantially,
because the GNN's sqrt class weights already correct for imbalance while the RF's
`predict_proba` argmax does not. Given the same validation-tuned treatment the RF reaches
0.322 against the GNN's 0.314, i.e. level. The adjusted RF is the honest baseline to beat,
and it is not beaten yet. Full reasoning in
[`results/logit_adjustment.md`](results/logit_adjustment.md).

Next: graph augmentation (DropEdge, atom masking) to attack the real bottleneck of ~1,023
training drugs; a milder class-weighting variant; then the full S1/S2/transductive runs,
ablations (mean pooling, no cross-drug attention), per-class F1 and attention case studies.

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
