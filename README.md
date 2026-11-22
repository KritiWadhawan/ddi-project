# Predicting Drug–Drug Interaction Types for Unseen Drugs

Given two drugs, predict which of **86 DrugBank interaction types** occurs between them
(for example, one drug increasing the metabolism of the other), with a focus on
**drugs the model has never seen during training**, the realistic case for new drugs.

The model is a graph neural network over molecular graphs: AutoDDI's searched encoder with
cross-drug graph attention, plus partner-conditioned co-attention pooling (after MGDDI);
the selected configuration uses 4 of those layers and fuses each drug's Morgan fingerprint
into the classifier. It is compared against a Morgan fingerprint + Random Forest baseline
under three increasingly strict evaluation protocols.

**Status:** data preparation, splits, baselines and the full three-protocol comparison are
complete. The headline finding is that **the GNN's advantage collapses as soon as a drug is
unseen**: it beats the Random Forest by 0.142 Macro-F1 when both drugs of a test pair were
seen in training, loses by 0.009 when one is unseen, and is level when both are. Unseeing a
single drug costs the GNN 38% of its transductive score and the fingerprint forest only
26%, so the advantage is memorised drug identity rather than transferable chemistry
([`results/protocol_collapse.md`](results/protocol_collapse.md)). Ablations at a matched
epoch budget find no measurable contribution from either distinctive architectural
component, which points the same way. Remaining: per-class F1 against class frequency, and
the write-up.

Picking this up for the analysis and write-up? Start with
[`HANDOVER.md`](HANDOVER.md) — what exists, what is still open, and what the results do
not support.

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
  (patience 10) on validation `macro_f1_occurring`, a checkpoint every epoch so runs can
  resume. Note that best-epoch selection is a maximum over however many epochs a run
  survives, which on a noisy validation plateau favours whichever run trained longest —
  worth ~0.01 here, so variants meant to be compared are trained to a fixed budget.

## Results

### Headline: the protocol decides the result

Test `macro_f1_occurring`, selected configuration (4-layer encoder + fingerprint fusion,
sqrt class weights) against the baselines. Full analysis, including the leakage check on
the transductive number, in [`results/protocol_collapse.md`](results/protocol_collapse.md).

| Protocol | Test pairs contain | GNN | RF + Morgan | Majority | GNN keeps |
|---|---|---|---|---|---|
| Transductive | both drugs usually seen | **0.950** | 0.808 | 0.006 | 100% |
| S2 | exactly one drug unseen | 0.585 ± 0.009 | **0.594 ± 0.013** | 0.006 ± 0.000 | 61.5% |
| S1 | both drugs unseen | **0.314 ± 0.043** | 0.270 ± 0.043 | 0.009 ± 0.001 | 33.0% |

The last column is the finding: the RF keeps 73.5% of its own transductive score on S2
against the GNN's 61.5%, so **the GNN degrades faster than the baseline** at the first step
away from the protocol it looks best under. A fingerprint forest cannot memorise which
pairs co-occur — it sees two bit vectors — so its decline measures how much harder the task
becomes; the GNN's steeper decline is the part of its lead that was never chemistry.

Two caveats that belong next to the table. On S1 the +0.044 is against the RF *as trained*;
against the RF with its own validation-tuned prior adjustment it is −0.001 over the same
five seeds, i.e. parity (below). And in both cold-start regimes the RF is the *more accurate*
model (S1 0.465 vs 0.444, S2 0.726 vs 0.695), the sqrt class weighting trading head-class
accuracy for tail recall.

Baselines in full, test Macro-F1 ([`results/baseline_summary.csv`](results/baseline_summary.csv)):

| Model | Transductive | S2 | S1 |
|---|---|---|---|
| Majority class | 0.006 | 0.006 ± 0.000 | 0.009 ± 0.001 |
| Morgan fingerprint + Random Forest | 0.808 | 0.594 ± 0.013 | 0.270 ± 0.043 |

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
are not the numbers to quote; the headline table above is.

Fusing the Morgan fingerprint into the classifier is what closes the cold-start gap, and
depth is nearly irrelevant next to it (0.235 → 0.242 for 4 layers alone, versus 0.273 with
fusion). The two fusion variants are indistinguishable on validation (0.2583 vs 0.2579), so
the 4-layer model is carried forward on grounds independent of the test set: half the
parameters, ~40% less time per epoch, and a third of the variance across seeds.

### Why S1 is parity rather than a win

| S1 test, 5 seeds | Macro-F1 (occurring) |
|---|---|
| GNN, 4 layers + fingerprint fusion | 0.3138 ± 0.0425 |
| Morgan fingerprint + Random Forest, as trained | 0.2696 ± 0.0433 |
| Random Forest + validation-tuned prior adjustment (τ = 0.15) | **0.3148 ± 0.0277** |

Investigating prior adjustment for the GNN (where it gave nothing) revealed that the same
correction helps the *Random Forest* substantially, because the GNN's sqrt class weights
already correct for imbalance while the RF's `predict_proba` argmax does not. Given the
same validation-tuned treatment the RF reaches 0.3148 against the GNN's 0.3138:
**−0.0011, with the GNN ahead on 3 seeds of 5** and a per-seed spread of ±0.018 that
dwarfs the difference between the means. S1 is parity in the strict sense — not a narrow
win for either model. The adjusted RF also has the lower seed-to-seed variance of the two.
Full reasoning in [`results/logit_adjustment.md`](results/logit_adjustment.md).

One shared τ is chosen on validation across all five seeds, not one per seed. Selecting it
over five seeds returns the same τ = 0.15 that three seeds did, so the earlier three-seed
figures are unchanged by the extension.

### Architecture ablations

Trained to a fixed 63 epochs each, transductive validation `macro_f1_occurring`:

| transductive, seed 0, N = 63 | best val | test |
|---|---|---|
| full (co-attention + bipartite cross-drug GATConv) | 0.9518 | 0.9505 |
| mean pooling instead of co-attention | 0.9488 (−0.0031) | 0.9501 (−0.0004) |
| no bipartite cross-drug GATConv | 0.9488 (−0.0031) | 0.9413 (−0.0092) |

**Neither distinctive component has a measurable effect.** The largest difference is 0.0031
against a validation plateau whose standard deviation is 0.0078, and removing the bipartite
layer leaves a model that beats the full one on 34 of 63 matched epochs. That cross-drug
message passing contributes nothing is worth stating plainly, since it is the defining idea
in this architecture family.

An earlier version of this comparison appeared to show a clear +0.013 for co-attention.
That gap was training length, not architecture: best-epoch validation is a maximum over
however many epochs a run survived, and early stopping had given the full model 29 more
epochs than mean pooling to draw it from. Fixing the budget removed three-quarters of it.
Details and all four estimators in
[`results/protocol_collapse.md`](results/protocol_collapse.md).

This supports the headline rather than denting it: if neither distinctive component is
identifiable and the model still reaches 0.950 transductive, that score is a property of
the protocol, not the architecture. It is also why attention case studies were dropped from
the planned analysis — presenting the weights of a component with no measurable effect as
learned chemistry would be contradicted by this project's own ablation table. Both caveats
stand: one seed, and the regime where the task is easiest.

### Graph augmentation: a negative result that locates the bottleneck

DropEdge (20% of bonds) and atom-feature masking (15% of atoms) were added to test whether
the cold-start ceiling comes from the encoder memorising its ~1,023 training drugs. Both
cut memorisation substantially — final training loss rose from 0.040 to ~0.12 on seed 0 —
and validation did not move; all four variants of the round, including one with class
weighting removed, lost on S1 test on all three seeds.

The combination is the result: the intervention worked on the training side and bought
nothing on the evaluation side, so what limits S1 is how few distinct drugs the split
leaves to train on, not overfitting to them. On S1 both drugs of a test pair are unseen,
and the fingerprint — not the learned graph representation — is what transfers. Details in
[`results/round1c_augmentation.md`](results/round1c_augmentation.md), which also covers a
correction it turned up: checkpoint selection had been early-stopping on the old `macro_f1`
rather than the reported metric, and now uses `macro_f1_occurring`. Re-running under the
new rule showed it is a wash on the numbers (−0.003 validation, +0.007 test) — it is kept
because selecting on the metric you report is a requirement, not because it scores better.

Next: per-class F1 against each class's training frequency, the matched-budget ablation
re-run, and the write-up.

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
