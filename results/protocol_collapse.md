# The GNN's advantage collapses as soon as a drug is unseen

**Summary.** Across the three evaluation protocols the GNN beats a Morgan fingerprint +
Random Forest baseline by 0.142 Macro-F1 when both drugs of a test pair were seen in
training, loses to it by 0.009 when one drug is unseen, and is level with it when both
are. Unseeing a *single* drug costs the GNN 38% of its transductive score and the
fingerprint forest only 26%. The advantage is not transferable chemistry; it is
drug-specific interaction structure the model has memorised, and it is the first thing
lost. Transductive is the protocol most DDI work reports, which is what makes the size of
this gap worth measuring.

All figures are `macro_f1_occurring`, the project's primary metric (see
[logit_adjustment.md](logit_adjustment.md) for why the old `macro_f1` column is not
comparable between models). Model selection and early stopping use validation only; test
is read once per run.

## The three regimes

Test set, selected configuration (4-layer encoder, Morgan fingerprint fusion, sqrt class
weights, dropout 0.3, AdamW, wd 1e-2, lr 5e-4) against the baselines:

| protocol | test pairs contain | seeds | GNN | RF + Morgan | majority |
|---|---|---|---|---|---|
| transductive | random pairs, both drugs usually seen | 1 | **0.950** | 0.808 | 0.006 |
| S2 | exactly one drug unseen | 5 | 0.585 ± 0.009 | **0.594 ± 0.013** | 0.006 ± 0.000 |
| S1 | both drugs unseen (cold start) | 5 | **0.314 ± 0.043** | 0.270 ± 0.043 | 0.009 ± 0.001 |

Paired per-seed differences, GNN − RF, which are what to read given how much the seeds
move:

| protocol | mean Δ | GNN wins | per seed |
|---|---|---|---|
| transductive | +0.142 | 1/1 | +0.142 |
| S2 | −0.009 ± 0.011 | 1/5 | +0.007, −0.017, −0.017, −0.002, −0.018 |
| S1 | +0.044 ± 0.011 | 5/5 | +0.050, +0.030, +0.037, +0.058, +0.046 |

Transductive has one seed by design: [`src/splits.py`](../src/splits.py) documents it as an
upper bound rather than a result, and it is reported here to quantify that bound, not to
claim it.

## What the retention numbers say

Taking each model's own transductive score as its ceiling:

| protocol | GNN keeps | RF keeps |
|---|---|---|
| transductive | 100% | 100% |
| S2 | 61.5% | 73.5% |
| S1 | 33.0% | 33.4% |

This is the result. The GNN degrades *faster than the baseline* at the first step away
from the protocol it looks best under. A fingerprint forest has no mechanism for
memorising which pairs co-occur — it sees only two bit vectors — so its decline measures
how much harder the task itself becomes. The GNN's steeper decline is the part of its
transductive lead that was never chemistry.

By S1 both models have fallen to the same third of their transductive score, and the
GNN's apparent +0.044 does not survive a fair comparison (next section).

### The transductive number is not leakage

0.950 Macro-F1 over 86 classes invites the suspicion that the same pair appears in
training and test. It does not. [`src/clean.py`](../src/clean.py) keys pairs
order-independently (`pair_key`, so `(A,B)` and `(B,A)` collapse to one key) and
deduplicates before any split is drawn. Checked directly on the transductive split: the
train, validation and test sets share **zero** unordered pairs pairwise, and zero ordered
pairs. The 406 conflicting pairs the cleaning step removed are logged in
[`cleaning_log.json`](cleaning_log.json).

The number is high because transductive DDI is close to saturated, which is the point
being made about the protocol.

## S1 is parity, not a win

The +0.044 above is against the Random Forest as trained. The honest baseline is the RF
with its own validation-tuned prior adjustment, which is a real gain for the RF and gave
the GNN nothing — the asymmetry is explained in
[logit_adjustment.md](logit_adjustment.md). On the three seeds where that bar exists:

| S1 test, seeds 0–2 | Macro-F1 (occurring) |
|---|---|
| GNN, 4 layers + fingerprint fusion | 0.327 ± 0.046 |
| RF + Morgan, as trained | 0.289 ± 0.043 |
| RF + Morgan + validation-tuned prior adjustment (τ = 0.15) | **0.322 ± 0.035** |

+0.006, winning 2 of 3 seeds: parity, unchanged from earlier rounds. **Known gap:** the
GNN has 5 seeds on S1 and the adjusted baseline only 3, because `src/adjust_rf.py` has no
cached probabilities for seeds 3–4. Seeds 3 and 4 are where the unadjusted forest did
worst, so completing the bar could move the comparison either way, and the 5-seed S1
figure should not be quoted against the 3-seed bar until it is.

Prior adjustment has also not been extended to S2 or transductive. Neither model receives
it there, which is symmetric and therefore fair, but it does mean the S2 comparison is not
against the strongest available version of the baseline.

## The accuracy inversion

Worth reporting because it cuts against the model:

| test accuracy | GNN | RF + Morgan |
|---|---|---|
| transductive | **0.962** | 0.879 |
| S2 | 0.695 | **0.726** |
| S1 | 0.444 | **0.465** |

In both cold-start regimes the forest is *more accurate* while the GNN leads on S1
Macro-F1. That is the sqrt class weighting behaving as designed — trading head-class
accuracy for tail-class recall, with class 49 alone being 32% of the data. On S2 the RF
leads on both, so there is no metric on which the GNN takes S2.

## Architecture ablations: no effect that survives a matched budget

Two components distinguish this encoder: the bipartite GATConv that passes messages
between the two molecules, and the per-layer co-attention pooling taken from MGDDI. Both
were ablated on transductive — the regime where the GNN leads by 0.142, and so the one
where architecture has the best chance of showing up. (They were also run on S1 in the
previous round, where mean pooling *beat* co-attention on validation, 0.2811 vs 0.2803.
That is the expected null: [round1c_augmentation.md](round1c_augmentation.md) established
S1 is coverage-limited, so encoder changes cannot resolve there.)

As run, the result looks clear, and unlike on S1 validation and test agree in sign:

| transductive, seed 0 | val | test | best epoch | **epochs run** |
|---|---|---|---|---|
| full (co-attention + bipartite) | 0.9518 | 0.9505 | 53 | **63** |
| mean readout | 0.9389 | 0.9276 | 24 | **34** |
| no bipartite GATConv | 0.9479 | 0.9487 | 48 | **58** |

The last column is the problem. Best-epoch validation is a **maximum over however many
epochs a run survived**, and this plateau is noisy: over epochs 25–63 the full run has
sd 0.0078, a median epoch-to-epoch swing of 0.0063 and a total range of 0.0305. Sampling
that same plateau, the best of 30 consecutive epochs averages 0.9518 while the best of 10
averages 0.9468 — **about +0.005 bought by training longer, with nothing changed.** Early
stopping let the full model draw its maximum from 29 more epochs than mean pooling got.

### Redone at a matched budget

All three variants were then trained to a fixed 63 epochs with early stopping disabled,
so the maximum is drawn from the same number of draws in every case:

| transductive, seed 0, N = 63 | best val | best epoch | test | final train loss |
|---|---|---|---|---|
| full (co-attention + bipartite) | 0.9518 | 53 | 0.9505 | 0.0340 |
| mean readout | 0.9488 (−0.0031) | 47 | 0.9501 (−0.0004) | 0.0422 |
| no bipartite GATConv | 0.9488 (−0.0031) | 63 | 0.9413 (−0.0092) | 0.0352 |

**Co-attention's apparent +0.0129 advantage becomes +0.0031** — about three-quarters of it
was the budget. Truncating the original runs to a common 34 epochs had put the residual at
+0.0010 and the max-of-N estimate had attributed ~0.005 of the gap to length; the first was
the better predictor, and both pointed the right way.

Two further estimators on the same runs, which now agree with the first:

| | full | mean readout | no bipartite |
|---|---|---|---|
| mean of val over epochs 25–63 (n = 39) | 0.9370 ± 0.0078 | 0.9339 (−0.0031) | 0.9352 (−0.0018) |
| matched-epoch mean difference, all 63 epochs | — | −0.0026 | −0.0005 |
| epochs on which the ablation *beats* the full model | — | 25 / 63 | **34 / 63** |

The largest effect anywhere in the table is 0.0031, against a plateau standard deviation
of 0.0078 and a range of 0.0305 — roughly 0.4 sd, with one seed and no replicates. Removing
the bipartite GATConv leaves a model that beats the full one on *more than half* of all
matched epochs, which is as close to a null as a single run can show. Its best epoch is
63, the last one, so it was still improving when the budget ran out and is if anything
under-measured here.

Test agrees on mean pooling (−0.0004) and disagrees on the bipartite layer (−0.0092),
which is the signature of noise rather than effect. Mean pooling fits the training set
slightly less well (0.0422 vs 0.0340) and generalises identically, so the extra capacity
co-attention provides is spent on memorising rather than on anything that transfers.

**Conclusion: neither distinctive component has a measurable effect on transductive
performance.** That inter-molecular message passing contributes nothing is worth stating
plainly, because cross-drug attention is the defining idea in this architecture family.

**This supports the main result rather than denting it.** If neither distinctive component
is identifiable and the model still reaches 0.950 transductive, then 0.950 is a property
of the protocol, not of the architecture.

### Consequences

- **Attention case studies are dropped from the planned analysis.** Co-attention has no
  measurable effect on performance in either regime tested, so presenting its weights as
  learned chemistry would be unsupportable — and this project's own ablation table would
  be the evidence against it. A per-class F1 breakdown against each class's training
  frequency replaces it, which speaks to the memorisation claim directly.
- **Best-epoch selection on a noisy plateau favours whichever run trained longest**, worth
  roughly 0.01 here. Any single-seed comparison in this project decided on a margin under
  0.01 is suspect for the same reason, including the checkpoint-selection change discussed
  in [round1c_augmentation.md](round1c_augmentation.md).

## Status

The three-regime comparison and the ablations are both complete. The early-stopped
versions of the ablation runs are archived in
[`gnn_metrics_superseded.csv`](gnn_metrics_superseded.csv) rather than deleted, so the
confounded figures remain inspectable next to the matched-budget ones.

Two caveats attach to the ablations and should be carried into the write-up rather than
dropped: they are a single seed, and transductive is the regime where the task is easiest,
so "no measurable effect here" is not "no effect anywhere". Establishing either component
as genuinely useless would need several seeds, most usefully on S2, where the GNN's
deficit actually lives.

Outstanding elsewhere: the adjusted-RF bar on S1 seeds 3–4 (the headline S1 figure has 5
seeds against a 3-seed bar), and whether prior adjustment should be extended to S2 and
transductive for both models.
