# Round 1c: graph augmentation does not help cold start, and explains why

**Summary.** DropEdge and atom-feature masking were added to test whether the S1 cold-start
ceiling is caused by the encoder memorising its ~1,023 training drugs. Both reduced
memorisation substantially and neither improved generalisation; every variant lost on S1
test on all three seeds. The informative part is the combination: the intervention worked
on the training side and produced no validation gain, which points at drug coverage rather
than overfitting as the binding constraint on S1. Dropping class weighting entirely, tested
in the same round, was the worst variant of the four, so sqrt weighting stays.

All numbers below are `macro_f1_occurring`, the project's primary metric (see
[logit_adjustment.md](logit_adjustment.md) for why the old `macro_f1` column is not
comparable between models). Selection is by validation only.

## What was run

Four variants against the round-1b winner (`--layers 4 --fp`, sqrt class weights,
dropout 0.3, AdamW, weight decay 1e-2, lr 5e-4), S1 seeds 0–2, 12 runs:

| flag | what it does |
|---|---|
| `--drop-edge 0.2` | drops 20% of bond edges each forward pass (DropEdge, Rong et al., ICLR 2020) |
| `--mask-atom 0.15` | zeroes 15% of atom feature rows, which LayerNorm maps to a shared mask token |
| both | the two together |
| no `--class-weights` | same model, unweighted cross-entropy |

Augmentation is training-only and the Morgan fingerprint is deliberately left untouched —
it is the part of the input that already transfers to unseen drugs.

## Selection (S1 seeds 0–2, validation)

| variant | val Macro-F1 (occurring) |
|---|---|
| **baseline — 4 layers + fp, no augmentation** | **0.300 ± 0.011** |
| mask-atom 0.15 | 0.296 ± 0.029 |
| drop-edge 0.2 + mask-atom 0.15 | 0.294 ± 0.020 |
| drop-edge 0.2 | 0.287 ± 0.015 |
| no class weights | 0.271 ± 0.010 |

Seed variance dominates these spreads (seed 2 is a hard split for every variant), so the
paired per-seed differences are the ones to read:

| vs baseline | val Δ | val wins | test Δ | test wins |
|---|---|---|---|---|
| drop-edge 0.2 | −0.013 | 0/3 | −0.010 | 0/3 |
| mask-atom 0.15 | −0.004 | 2/3 | −0.010 | 0/3 |
| both | −0.007 | 0/3 | −0.019 | 0/3 |
| no class weights | −0.030 | 0/3 | −0.020 | 0/3 |

Mask-atom's two validation wins are the only positive signal anywhere in the round, and
they do not survive to test. The selected model is therefore unchanged from round 1b.

## Why this is informative rather than just a failure

The augmentation did exactly what it was built to do. Seed 0, final training loss:

| run | final train loss | best epoch | best val |
|---|---|---|---|
| baseline | 0.040 | 37 | 0.255 |
| drop-edge 0.2 | 0.116 | 6 | 0.246 |
| mask-atom 0.15 | 0.119 | 6 | 0.233 |
| both | 0.110 | 9 | 0.245 |

Memorisation fell by roughly 3×. Validation did not follow — it sat near 0.23 instead of
climbing to 0.255, and the curve is flat from about epoch 5 onward rather than rising and
then falling. The best epochs moved *earlier* (37 → 6, 6, 9), the opposite of what a model
whose overfitting has been relieved should do; with a flat curve the peak simply lands
wherever noise puts it.

So the original reading of the S1 logs — "training loss reaches 0.04 while validation
plateaus, therefore the encoder is memorising molecules" — was half right. It *is*
memorising, and stopping it changes nothing, because the capacity freed up has nothing
further to learn from ~1,023 drugs. On S1 both drugs in a test pair are unseen, and the
graph encoder has too little coverage to learn chemistry that transfers under that
condition; the fingerprint carries what transfers, which is also why the Morgan + Random
Forest baseline is so hard to beat in this regime. That is a statement about the data, and
stronger regularization, more depth or longer training will not move it.

A practical consequence: effort is better spent on the S2 and transductive regimes, where
at least one drug of each test pair is seen in training, than on further S1 tuning.

## Side result: checkpoint selection was using the wrong metric

Round 1c's per-epoch logs are the first to record `val_macro_f1_occurring` every epoch,
which made it possible to ask — without retraining — what the training loop's early
stopping would have done had it used the reported metric instead of the old `macro_f1`.

Across the 12 runs it would have kept a different epoch in **7**, worth +0.004 validation
on average (max +0.019). More importantly, 2 runs had been stopped while the primary metric
was still rising: `S1_seed1 … fp_de0.2` peaked at epoch 29 of 30, and `S1_seed2 … fp` with
no class weights peaked at epoch 20 of 20. Selecting on a metric other than the reported
one also carries a known bias — the old `macro_f1` rewards predicting fewer rare classes,
so it favours conservative checkpoints.

`src/train.py` now early-stops and checkpoints on `macro_f1_occurring`. The round-1 screening
rounds were internally consistent under the old rule, which is all a screening comparison
needs, so they were not re-run; S1 seeds 0–2 of the selected configuration *are* re-run in
round 2 so that every reported seed is selected the same way. Their superseded rows and
checkpoints are archived, not deleted (`results/gnn_metrics_superseded.csv`, `*.round1b`).

## Status

The bar on S1 test is unchanged: the selected GNN at 0.320 ± 0.040 against the
prior-adjusted Random Forest at 0.322, i.e. parity. Round 2 runs that configuration across
S1 (5 seeds), S2 (5 seeds) and transductive, plus the co-attention and bipartite-edge
ablations.
