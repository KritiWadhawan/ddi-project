# Phases 4 and 5: the analysis and the write-up

This is the working document for the two phases that remain. [`HANDOVER.md`](HANDOVER.md)
tells you what exists and what the results will not support; this one tells you what to do
next, in what order, with the exact commands and the places the tooling will mislead you.

Phases 1–3 (data, splits, baselines, model, training) are finished. No more training runs
are needed for the result the paper argues. Everything below is analysis, one small piece
of coding, reading, and writing.

**If you are driving this with Claude**, start by telling it to read `HANDOVER.md` and this
file, then see [Working with Claude on this repo](#working-with-claude-on-this-repo) at the
bottom. There are four traps in this repo that an agent will walk into unprompted, and two
of them silently destroy data.

---

## Where the project stands

Test `macro_f1_occurring`, selected model (4-layer encoder + Morgan fingerprint fusion,
sqrt class weights) against the Morgan + Random Forest baseline:

| protocol | test pairs contain | seeds | GNN | RF | GNN keeps | RF keeps |
|---|---|---|---|---|---|---|
| transductive | both drugs usually seen | 1 | **0.950** | 0.808 | 100% | 100% |
| S2 | exactly one drug unseen | 5 | 0.585 ± 0.009 | **0.594 ± 0.013** | 61.5% | 73.5% |
| S1 | both drugs unseen | 5 | **0.314 ± 0.043** | 0.270 ± 0.043 | 33.0% | 33.4% |

### The thesis, stated plainly

The GNN beats the fingerprint baseline by 0.142 Macro-F1 when both drugs of a test pair
were seen in training, and loses to it when one drug is unseen. Unseeing a *single* drug
costs the GNN 38% of its transductive score and the baseline only 26%. A fingerprint forest
has no mechanism for memorising which pairs co-occur — it sees two bit vectors — so its
decline measures how much harder the task itself becomes. **The GNN's steeper decline is
the part of its transductive lead that was never chemistry.** Transductive is the protocol
most published DDI work reports.

Everything in Phase 4 either sharpens that claim or tests it. Keep it in view: it decides
which analyses are worth doing and which numbers belong in the paper.

### Three supporting results already established

- **Architecture ablations found nothing measurable.** Neither the co-attention pooling nor
  the bipartite cross-drug GATConv changes transductive performance once variants are
  trained to an equal epoch budget (largest difference 0.0031 against a plateau sd of
  0.0078). This *supports* the thesis — if neither distinctive component is identifiable
  and the model still reaches 0.950, then 0.950 is a property of the protocol, not the
  architecture. [`results/protocol_collapse.md`](results/protocol_collapse.md)
- **Graph augmentation located the S1 bottleneck.** DropEdge and atom-feature masking cut
  memorisation substantially (training loss 0.040 → ~0.12) and bought nothing on
  validation. So what limits S1 is how few distinct drugs the split leaves to train on
  (~1,023), not overfitting to them.
  [`results/round1c_augmentation.md`](results/round1c_augmentation.md)
- **Prior adjustment is a negative result for the GNN and a real gain for the RF.** An
  apparent +0.021 was an artifact of how Macro-F1 was averaged. Finding that is why the
  project's primary metric changed.
  [`results/logit_adjustment.md`](results/logit_adjustment.md)

### Read this before you write anything

`HANDOVER.md` has a section called **"What the results do NOT support"** with six items.
In one line each: S1 is parity not a win; the ablations are one seed in the easiest regime;
never quote the `macro_f1` column; attention case studies are dropped; transductive is an
upper bound not a result; selection is on validation only, never test.

Two of those exist because a number looked better than it was and had to be corrected. The
list is the cheapest insurance in the project — check claims against it as you draft, not
after.

---

# Phase 4 — the analysis

Three tasks. Task 1 is the one the paper needs; Task 3 is the one that decides how the
paper is framed and costs no compute, so start it in parallel. Task 2 is for the reviewer
who asks why the baseline only got its best form in one regime of three.

## Task 1 — Per-class F1 against training frequency

**This is the analysis that replaces the dropped attention case studies, and it is the only
remaining piece of evidence that speaks to the memorisation claim directly.**

### Why this specific analysis

The thesis says the GNN's transductive advantage is memorised drug identity. That makes a
falsifiable prediction about *structure*, not just totals: the GNN's per-class F1 edge over
the RF should scale with how often a class appears in training, and should flatten on S1
where neither model has seen the drugs. A model that had learned transferable chemistry
would show no such pattern.

This matters because the retention table alone is consistent with a weaker story — "the GNN
is more sensitive to distribution shift" — which is true but much less interesting. The
per-class pattern is what separates "memorisation" from "fragility".

### What exists

[`results/per_class_f1.csv`](results/per_class_f1.csv), 258 rows = 86 classes × 3 seeds.
Columns: `model, split, seed, class, train_frac, test_support, f1`. **S1 only, seeds 0–2
only, GNN only.**

### What is missing, and what each piece costs

**(a) GNN per-class on S2 and transductive** — one command. `src/adjust.py` reads each
run's split out of its own checkpoint, so it is *not* S1-specific; pass it whatever
checkpoints you want. Requires the Drive checkpoints and a GPU session (or patience on
CPU).

```
python src/adjust.py \
  --ckpt "checkpoints/S1_seed?_coattn_L4_cwsqrt_do0.3_adamw_lr0.0005_wd0.01_fp.pt" \
         "checkpoints/S2_seed?_coattn_L4_cwsqrt_do0.3_adamw_lr0.0005_wd0.01_fp.pt" \
         "checkpoints/transductive_seed0_coattn_L4_cwsqrt_do0.3_adamw_lr0.0005_wd0.01_fp.pt" \
  --tau-min 0 --tau-max 0 --per-class
```

Three things about that command, each of which will cost you a run if you skip it:

- **`per_class_f1.csv` is opened with `"w"`.** Every invocation rewrites the file from
  scratch ([`src/adjust.py:229`](src/adjust.py)). Pass every checkpoint you want in **one**
  invocation, or back the file up first — running it on S2 alone deletes the S1 rows that
  are already there. `results/adjusted_metrics.csv` is written the same way.
- **`--tau-min 0 --tau-max 0` forces τ = 0.** Without it the default grid runs 0 → 1.5 and
  the per-class rows are written at whichever τ validation selects, which is *not* the model
  in the results tables. For this analysis you want the plain model as reported.
- **Be specific in the glob.** `S1_seed*_fp.pt` also matches the superseded round-1b runs
  (recognisable by a `_de0.2` or `_ma0.15` tag, or a missing `cwsqrt`). `seed?` with the
  full suffix, as above, matches only the reported runs. Check what it expanded to before
  trusting the output.

**(b) RF per-class — this does not exist and needs about 20 lines of code.**

`src/adjust_rf.py` writes only `results/adjusted_rf_metrics.csv`; it has no `--per-class`
option. Without the RF half there is no comparison at all, so this is not optional.

The S1 part is cheap: `results/logits/RF_S1_seed{0..4}.npz` already cache the
`predict_proba` arrays and the training prior for all five seeds, so nothing needs refitting.
Mirror the per-class block at [`src/adjust.py:217–232`](src/adjust.py) — it computes
`f1_score(..., average=None, labels=np.arange(N_CLASSES))`, takes `test_support` from
`np.bincount(y)`, and reads `train_frac` straight off `d["prior"]`. Write to a *separate*
file (`results/per_class_f1_rf.csv`) rather than appending to the GNN's, so neither run can
clobber the other.

For S2 and transductive the forests must be refit — `rf_logits()` in `adjust_rf.py` does
that and caches the result, roughly 20 minutes of CPU per split.

### How to read the outcome — including if it fails

Plot per-class F1 difference (GNN − RF) against `train_frac`, one panel per protocol.

- **If the gap tracks training frequency on transductive and flattens on S1**, that is
  direct evidence for the central claim, and it should lead the experiments section.
- **If it does not**, the memorisation claim needs weakening to the retention argument
  alone. That argument still stands by itself — the three-regime and retention tables do
  not depend on this analysis — so a negative result here costs the paper a figure, not its
  contribution.
- **Either way, report it.** This project has corrected itself twice in public (in the
  write-ups and the commit messages), and that consistency is doing real work for its
  credibility. A quietly dropped analysis is the one thing that would undo it.

A practical note: `train_frac` spans several orders of magnitude (class 49 alone is 32% of
DrugBank DDI), so use a log x-axis, and weight your reading of the plot by `test_support` —
classes with a handful of test examples have very noisy F1 and will dominate the visual
scatter without carrying information.

## Task 2 — Prior adjustment on S2 and transductive

### Why

The RF gained +0.033 on S1 from validation-tuned prior adjustment, which is what turned the
S1 comparison from "GNN ahead +0.044" into parity. Neither model receives it on S2 or
transductive. That is symmetric, and therefore fair — but a reviewer will ask why the
baseline was given its best form in only one regime of three, and on S2 the RF already
wins without it.

### What the code needs

- **GNN side: nothing.** `adjust.py` is split-agnostic (it calls
  `make_split(a["split"], ...)` using the split stored in the checkpoint). Point it at S2
  or transductive checkpoints and it works. *(`HANDOVER.md` said both scripts were
  S1-hardcoded. That was wrong about `adjust.py`; it is corrected there now.)*
- **RF side: `adjust_rf.py` genuinely is hardcoded to S1** — literal `"S1"` at lines 37,
  43, 58 and 97. Add a `--split` argument and thread it through `rf_logits()` (including
  the cache filename, or S2 runs will read S1's cached probabilities and silently produce
  nonsense).

### Rules that are not negotiable here

- **One shared τ per seed set, selected on validation**, never one τ per seed and never
  anything selected on test. The S1 result used a single shared τ across five seeds; do the
  same, or the numbers are not comparable with what is already published.
- **Do it for both models or neither.** Tuning τ for one and not the other hands that model
  whatever gain the *procedure* confers rather than the *model*, which is exactly the
  mistake the S1 fairness check caught.
- **Expect it to make the GNN look worse.** The RF already beats it on S2 on both Macro-F1
  and accuracy; giving the RF its best form will widen that. Report it. The paper's argument
  is that the protocol decides the result, so a baseline that beats the model in cold start
  is evidence *for* the thesis, not against it.

## Task 3 — The literature comparison

**No compute. Start this first — it decides how the whole paper is framed, and it is pure
reading.**

No published method has been re-evaluated under S1/S2 in this project, and that is the
honest limitation. What needs establishing is what published work actually reports.

For each relevant paper, record three things: **(i)** the evaluation protocol — random
split over pairs, or a cold-start split; **(ii)** the headline DrugBank DDI-type number and
the metric and label set it uses; **(iii)** whether any cold-start result is reported at all.

Worth checking: AutoDDI (the architecture search this encoder came from), SSI-DDI (whose
atom featurizer this project copies), MGDDI (the co-attention pooling), GMPNN-CS, DeepDDI,
and the TDC DrugBank DDI leaderboard itself.

The two outcomes diverge sharply:

- **If published work reports transductive DrugBank in the same 0.95 band and does not
  report cold start**, then this is a protocol critique with a measurement to back it, which
  is a stronger and more publishable contribution than a model paper. Frame the whole paper
  that way.
- **If published work does report cold start, and does better than 0.314 on S1**, the
  framing has to change and that must be stated plainly. Find this out before drafting the
  introduction, not after.

One caution while comparing: other papers' "Macro-F1" may use sklearn's default label set,
which is the manipulable one this project abandoned. If a paper's number is not clearly
computed over a fixed label set, say so rather than tabulating it as comparable.

---

# Phase 5 — the write-up

## What kind of paper this is

**A protocol paper, not a model paper.** The model is a reimplementation of an
AutoDDI-searched encoder with MGDDI's co-attention pooling and fingerprint fusion — and this
project's own ablations could not measure a contribution from either distinctive component.
Presenting it as a novel architecture would be contradicted by Table (ablations) in the same
paper.

The contribution is the measurement: *how much of a DDI model's reported performance is an
artifact of the evaluation protocol, quantified on the same model and data, with a baseline
that has no memorisation mechanism as the control.*

## Section by section

**Abstract / title.** Lead with the protocol claim and the retention numbers (38% vs 26%
loss from unseeing one drug). Do not lead with 0.950.

**Introduction.** The gap: DDI models are reported under random pair splits; the use case
is a new drug, where both or one of the pair is unseen. Then what Task 3 found about what
the literature reports.

**Related work.** Driven by Task 3. State explicitly that no published method was
re-evaluated here under S1/S2 — it is a limitation of this work, and naming it is better
than letting a reviewer find it.

**Data and protocols.** 191,808 → 190,986 cleaned pairs, 1,705 drugs, 86 classes, class 49
= 32%. The three splits with their sizes and seed counts. Two things must appear here:
- **The leakage check**, because 0.950 over 86 classes invites the suspicion. `pair_key` in
  [`src/clean.py`](src/clean.py) keys pairs order-independently and deduplicates before any
  split is drawn; train/val/test share zero unordered *and* zero ordered pairs.
- **Transductive is one seed and documented in [`src/splits.py`](src/splits.py) as an upper
  bound.** Report it as a ceiling.

**Method.** The encoder, honestly described as above. 1,574,148 parameters.

**Experiments.** In this order:
1. The three-regime table.
2. The retention table — this is the contribution; give it a figure.
3. The accuracy inversion (RF is *more accurate* in both cold-start regimes: S1 0.465 vs
   0.444, S2 0.726 vs 0.695). It cuts against the model, which is why it belongs.
4. S1 parity against the adjusted RF: 0.3148 ± 0.0277 vs 0.3138 ± 0.0425 over five seeds,
   GNN ahead on 3 of 5. Note the adjusted RF also has the *lower* variance.
5. The ablations, at matched budget, with the max-of-N caveat.
6. Task 1's per-class analysis, if it comes out.

**Metric justification.** This needs its own subsection, short. Macro-F1 over classes
occurring in `y_true` is primary because it is fixed by the data, so no choice a model makes
can alter the divisor. The alternative — sklearn's default `y_true` OR `y_pred` label set —
lets a model that declines to predict rare classes delete its own zero-F1 entries and score
higher without classifying anything better. This project found a +0.021 "improvement" that
was entirely that effect. Readers will not expect this subsection, and it is one of the more
interesting things the project turned up.

**Limitations.** Do not thin this out:
- transductive is a single seed and an upper bound;
- the ablations are a single seed in the regime where the task is easiest, so "no measurable
  effect here" is not "no effect anywhere" — settling it would need several seeds, most
  usefully on S2 where the GNN's deficit actually lives;
- prior adjustment is applied only on S1 (unless Task 2 lands);
- no published method was re-evaluated under these protocols;
- the per-class analysis covers whichever regimes Task 1 completed.

## Which number comes from where

Never retype a figure from prose, including this document's. Sources of truth:

| claim | file |
|---|---|
| any GNN run's val/test metrics | `results/gnn_metrics.csv` (94 rows) |
| majority + RF baselines, all three Macro-F1 variants | `results/baseline_metrics.csv`, `baseline_summary.csv` |
| the adjusted RF (S1) | `results/adjusted_rf_metrics.csv` |
| per-class F1 vs training frequency | `results/per_class_f1.csv` |
| split sizes and drug counts | `results/split_stats.csv` |
| what cleaning removed | `results/cleaning_log.json` |
| per-epoch training curves | `results/runs/` (52 logs) |
| the confounded early-stopped ablations, kept for inspection | `results/gnn_metrics_superseded.csv` |

**Use the `macro_f1_occurring` column.** The `macro_f1` column is the old sklearn-default
one, kept only so the early screening tables stay internally consistent, and it is **not
comparable between models**.

## Figures worth making

1. **Retention** — each model's score across the three protocols as a fraction of its own
   transductive score. This is the paper's main figure; the crossing lines are the argument.
2. **Per-class F1 difference vs training frequency**, three panels, log x-axis, point size by
   `test_support` (from Task 1).
3. **The τ sweep** for GNN and RF side by side, from
   `results/logit_adjustment.md` — optional, but it makes the fairness check legible in one
   glance and shows the two models' optima falling on opposite sides of zero.

---

## Working with Claude on this repo

Point it at `HANDOVER.md` and this file first. Then be aware of these, because an agent
will hit them unprompted:

- **`results/per_class_f1.csv` and `results/adjusted_metrics.csv` are opened with `"w"`.**
  Any `adjust.py` run rewrites them from scratch. An agent asked to "add S2 to the per-class
  analysis" will delete the S1 rows and report success. Tell it to pass all checkpoints in
  one invocation, and commit the CSV before any run so git can recover it.
- **`--resume` skips any run whose name is already in `gnn_metrics.csv`**
  ([`src/train.py:140`](src/train.py)). `--epochs` and `--patience` are *not* part of the
  run name, so changing them alone will not produce a new run — it will silently skip. To
  re-run something, move its rows to `gnn_metrics_superseded.csv` first.
- **Best-epoch validation is a maximum over however many epochs a run survived.** On a noisy
  plateau that favours whichever run trained longest — worth ~0.01 here, and it silently
  produced a wrong ablation result once already. Any variants being compared must be trained
  to a fixed budget: `--epochs N --patience 999`. Treat any single-seed comparison decided
  on a margin under 0.01 as unresolved.
- **The Colab notebook symlinks `results/runs` and `results/gnn_metrics.csv` into Drive.**
  Both are tracked by git, so the clone shows a typechange on tracked paths. The repo-update
  cell hard-resets rather than pulling; do not let anything "fix" it back to `git pull`.
- **Selection is on validation Macro-F1 only.** Not test, not accuracy, not the old
  `macro_f1`. If an agent proposes picking a variant because it scores better on test, that
  is the one hard rule in this project being broken.
- **Ask it to verify every number against the CSV** before it goes in the draft. Numbers in
  the prose of these documents are correct as of writing, but the CSVs are the source of
  truth, and several figures in this project were revised after first being reported.

The commit messages carry the reasoning behind each round, including the two cases where a
reported number turned out to be an artifact of how it was measured. `git log` is worth
reading before trusting any single figure.

## Getting set up

```bash
pip install -r requirements.txt
```

`data/` is gitignored and rebuilds from the raw file in a few minutes — see **Reproducing**
in [`README.md`](README.md). It is verified against `results/cleaning_log.json`, so a
rebuild that silently differs will tell you.

`checkpoints/` and `results/logits/` are also gitignored and live in Google Drive — see the
top of [`HANDOVER.md`](HANDOVER.md) for which 15 checkpoints matter and why you should not
copy the whole folder.

The baselines need no checkpoints and refit on CPU in about 20 minutes:

```bash
python src/baselines.py
```
