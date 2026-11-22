# Handover: experiments are done, analysis and write-up are next

Everything in Phases 1–3 (data, splits, baselines, model, training) is finished. This page
says what exists, what it means, what is still open, and — most importantly — **what the
results do not support**, so the write-up does not claim something this project's own
tables contradict.

Read [`README.md`](README.md) first for the numbers, then
[`results/protocol_collapse.md`](results/protocol_collapse.md) for the main result.

## Before you can start: you need the checkpoints

`checkpoints/`, `results/logits/` and `data/` are **gitignored**, so cloning this repo does
not give you the trained models. They live in Google Drive at
`MyDrive/ddi_project_runs/checkpoints`. Ask for access — nothing that needs a model can be
reproduced without them.

Each run writes two files, `<run>.pt` (the selected epoch) and `<run>_last.pt` (every
epoch, so interrupted sessions resume), at ~6.3 MB each. Across the 13 reported runs plus
the archived `*.round1b` ones that is roughly 200 MB. Only the `<run>.pt` files are needed
for analysis; `_last.pt` matters only if a run is to be continued. Check what is actually
there before planning around it:

```
!ls -la /content/drive/MyDrive/ddi_project_runs/checkpoints
```

What the repo *does* give you without any of that: every metric
([`results/gnn_metrics.csv`](results/gnn_metrics.csv), 94 rows), every per-epoch training
curve ([`results/runs/`](results/runs/), 52 logs), and the baselines, which refit from
scratch on CPU in about 20 minutes with `python src/baselines.py`.

`data/` rebuilds from the raw file in a few minutes — see **Reproducing** in the README.
It is checked against [`results/cleaning_log.json`](results/cleaning_log.json), so if the
rebuild silently differs you will be told.

## The result, in one table

Test `macro_f1_occurring`, selected model (4-layer encoder + Morgan fingerprint fusion,
sqrt class weights) vs the Morgan + Random Forest baseline:

| protocol | test pairs contain | seeds | GNN | RF | GNN keeps | RF keeps |
|---|---|---|---|---|---|---|
| transductive | both drugs usually seen | 1 | **0.950** | 0.808 | 100% | 100% |
| S2 | exactly one drug unseen | 5 | 0.585 ± 0.009 | **0.594 ± 0.013** | 61.5% | 73.5% |
| S1 | both drugs unseen | 5 | **0.314 ± 0.043** | 0.270 ± 0.043 | 33.0% | 33.4% |

The last two columns are the contribution. The GNN loses its advantage *faster than the
baseline does* at the first step away from the transductive protocol, so that advantage is
memorised drug identity rather than transferable chemistry. Transductive is the protocol
most DDI papers report.

## What the results do NOT support

Please check anything you write against this list.

1. **S1 is parity, not a win.** The +0.044 is against the RF *as trained*. The honest
   baseline is the RF with its own validation-tuned prior adjustment, which is a genuine
   gain for the RF and gave the GNN nothing
   ([`results/logit_adjustment.md`](results/logit_adjustment.md)). Against that bar the GNN
   is roughly level. Use the adjusted bar.
2. **The ablations are one seed, in the easiest regime.** Neither the co-attention pooling
   nor the bipartite cross-drug GATConv has a measurable effect on transductive (largest
   difference 0.0031 against a plateau sd of 0.0078). That is *not* "the architecture is
   useless" — it is "no effect we can resolve here". Settling it needs several seeds,
   most usefully on S2.
3. **Do not quote the `macro_f1` column.** It uses sklearn's default label set (classes in
   `y_true` **or** `y_pred`), so a model that declines to predict rare classes deletes its
   own zero-F1 entries and scores higher without classifying anything better. It is **not
   comparable between models**. It is kept only because the early screening tables are
   internally consistent in it. The primary metric is `macro_f1_occurring`.
4. **Attention case studies are dropped.** Co-attention has no measurable effect on
   performance, so presenting its weights as learned chemistry would be contradicted by our
   own ablation table. The replacement analysis is per-class F1 against training frequency
   (below).
5. **Transductive is one seed and is an upper bound**, documented as such in
   [`src/splits.py`](src/splits.py). Report it as a ceiling, not as a result.
6. **Selection is on validation, never test.** Every model and epoch in this project was
   chosen on validation; test was read once per run. Keep it that way.

## Still open

| item | what it is | cost |
|---|---|---|
| per-class F1 beyond S1 | [`results/per_class_f1.csv`](results/per_class_f1.csv) covers only S1 seeds 0–2 (258 rows). Extend with `src/rescore.py` against the Drive checkpoints. | GPU session, or slow on CPU |
| prior adjustment on S2 / transductive | Neither model gets it there. Symmetric, so fair — but the RF gained +0.033 from it on S1, and a reviewer may ask why the baseline only got its best shot in one regime. `adjust.py` and `adjust_rf.py` are both S1-hardcoded. | ~1 h code + CPU |
| literature comparison | No published method has been re-evaluated under S1/S2 here. If published work reports transductive DrugBank in the same 0.95 band and does not report cold-start, that is what turns this from a model paper into a protocol critique. Worth checking before writing the introduction. | reading, no compute |

## The analysis that replaces the case studies

Per-class F1 against each class's training frequency. The claim is that the GNN's
transductive advantage is memorisation, so the prediction is specific: its per-class edge
over the RF should track how often a class appears in training, and should flatten on S1
where neither model has seen the drugs. [`results/per_class_f1.csv`](results/per_class_f1.csv)
already has the right columns (`train_frac`, `test_support`, `f1`) — it just needs the
other regimes.

If that pattern holds it is direct evidence for the main claim. If it does not, the claim
needs weakening, and it is much better to find that now than after the paper is drafted.

## Things that will bite you

- **`--resume` skips any run whose name is already in `gnn_metrics.csv`**
  ([`src/train.py:140`](src/train.py)). To re-run something, move its rows to
  `gnn_metrics_superseded.csv` first. `--epochs` and `--patience` are *not* part of the run
  name, so changing them alone will not create a new run.
- **Best-epoch validation is a maximum over however many epochs a run survived.** On a
  noisy plateau that favours whichever run trained longest — worth ~0.01 here, and it
  silently produced a wrong ablation result once already. Variants meant to be compared
  must be trained to a fixed budget (`--epochs N --patience 999`).
- **The Colab notebook symlinks `results/runs` and `results/gnn_metrics.csv` into Drive.**
  Both are tracked by git, so the clone has a typechange on tracked paths. The repo-update
  cell handles this by hard-resetting rather than pulling; do not change it back to
  `git pull`.
- **Checkpoints are written every epoch to `<run>_last.pt`** so interrupted Colab sessions
  resume. Deleting it forces a retrain from scratch.

## Where things are

| path | contents |
|---|---|
| `results/gnn_metrics.csv` | every GNN run's best-epoch val/test metrics (94 rows) |
| `results/gnn_metrics_superseded.csv` | runs replaced by a later one, kept for inspection (10 rows) |
| `results/baseline_metrics.csv`, `baseline_summary.csv` | majority + RF, all three Macro-F1 variants |
| `results/adjusted_rf_metrics.csv` | the RF with validation-tuned prior adjustment (S1) |
| `results/per_class_f1.csv` | per-class F1 vs training frequency — **S1 seeds 0–2 only** |
| `results/runs/` | per-epoch training curves, 52 logs |
| `results/*.md` | the three write-ups: protocol collapse, prior adjustment, augmentation |
| `notebooks/gpu_train.ipynb` | Colab/Kaggle training; clones the repo, rebuilds data, runs the configured experiments |

Commit messages carry the reasoning behind each round, including two cases where a
reported number turned out to be an artifact of how it was measured. `git log` is worth
reading before trusting any single figure.
