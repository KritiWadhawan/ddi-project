# Prior (logit) adjustment on S1: a negative result

**Summary: prior adjustment does not help the GNN — an apparent +0.021 test Macro-F1 was
an artefact of how Macro-F1 was being averaged, and finding that is why the project changed
its primary metric. Applying the same procedure to the Random Forest baseline, as a fairness
check, turned out to help it genuinely (+0.033 on seeds 0–2; +0.045 over all five), so
adjusted RF.**

## What was tried and why

The GNN reaches ~0.41 accuracy but only ~0.27 Macro-F1 on S1. That gap is the signature of
a classifier biased toward frequent classes, which costs a metric that weights all 86
classes equally (class 49 alone is 32% of DrugBank DDI). The standard correction
(Menon et al., *Long-tail learning via logit adjustment*, ICLR 2021) rescales the class
priors at inference:

```
predict  argmax_c  f_c(x) - tau * log pi_c
```

with `pi` the training class prior and `tau` a single scalar. `tau = 0` is the unmodified
model. One shared `tau` was selected on **validation** Macro-F1 across seeds 0–2 (one
hyperparameter, not one per seed), and test was read once at that value. Model:
`S1 L4 + fingerprint fusion`, the round-1b winner. Implementation: [`src/adjust.py`](../src/adjust.py).

## What happened

Positive `tau` — the direction the method prescribes — made things monotonically worse.
Negative `tau` appeared to help a lot:

| | Val | Test |
|---|---|---|
| `tau = 0` | 0.2579 | 0.2728 |
| `tau = -0.25` (selected on val) | 0.2673 | **0.2933** |

That would have put the GNN ahead of the Random Forest baseline (0.2775) on all three
seeds individually. It is not a real gain.

## Why it was not real

Macro-F1 was being computed with sklearn's default label set: the classes appearing in
`y_true` **or** `y_pred`. Under that definition a class the model predicts that never
occurs in the eval set scores 0 *and* enlarges the divisor. Negative `tau` suppresses rare
predictions, so it deletes those zero-F1 entries:

| | Spurious classes predicted | Macro-F1 divisor |
|---|---|---|
| `tau = 0` | 10.3 | 69.3 |
| `tau = -0.25` | 6.7 | 65.7 |

Re-selecting `tau` against label sets that a model's own predictions cannot change makes
the effect vanish:

| `tau` selected to optimise | `tau*` | Val gain | Test gain |
|---|---|---|---|
| Default label set (`y_true` or `y_pred`) | −0.25 | +0.0094 | **+0.0206** |
| Classes occurring in `y_true` | +0.05 | +0.0001 | **−0.0059** |
| All 86 classes | +0.05 | +0.0001 | **−0.0041** |

Under either fixed label set the optimum is `tau ≈ 0`, the validation gain is within
0.0001, and test is slightly *worse*. The model classified nothing better; it only stopped
guessing classes that were not there.

## The fairness check, which changed the conclusion

Tuning `tau` for the GNN and not for the baseline would hand the GNN any gain the
*procedure* confers rather than the *model*. So the identical procedure was run on the RF's
`predict_proba` ([`src/adjust_rf.py`](../src/adjust_rf.py)): same grid, one shared `tau`
chosen on mean validation Macro-F1, test read once.

The RF behaves oppositely to the GNN. Its optimum is **positive** (`tau = +0.15`), and its
gain **survives the fixed label set**, so it is real and not a divisor effect:

> **Later extension.** This section's figures are seeds 0–2, the seeds available when the
> experiment was run. The adjusted bar was afterwards extended to all five S1 seeds. The
> five-seed bar is **0.3150 ± 0.0272** against the GNN's **0.3138 ± 0.0425** — parity, with
> the RF nominally ahead by 0.0012 on the mean and the GNN ahead on 3 seeds of 5
> ([protocol_collapse.md](protocol_collapse.md)).
>
> **Selection-metric correction (2026-11-27).** Both adjust scripts originally selected
> `tau` on sklearn's *default* label set — the very metric this page exists to discredit.
> That is a real defect, and a specific one: the default label set is
> `union(y_true, y_pred)`, and `tau` is the knob controlling how many rare classes enter
> `y_pred`, so **the label set moved with the parameter being tuned**. Raising `tau` added
> newly-predicted absent classes to the average at F1 ≈ 0, penalising large `tau` for a
> reason unrelated to prediction quality, which biases the selected `tau` low. Both scripts
> now select on `macro_f1_occurring`, whose label set is `np.unique(y_true)` and so is fixed
> across the sweep. Re-run over five seeds: `tau*` moves **0.15 → 0.20** in the predicted
> direction, and the five-seed bar moves **0.3148 → 0.3150, i.e. +0.0001**. The conclusion
> does not change and is now demonstrably insensitive to the defect — but the selected
> `tau` is defensible if a reviewer asks how it was chosen, which it previously was not.
>
> The GNN figure below (0.3202) is the round-1b checkpoint, which is what this experiment's
> cached logits came from. That run was later superseded by a re-run under the corrected
> checkpoint-selection rule, scoring 0.3274 on the same three seeds. The conclusion here —
> that prior adjustment does nothing for the GNN and genuinely helps the RF — is unaffected,
> but do not quote 0.3202 as the model's S1 result.

| S1 test, seeds 0–2 | `tau = 0` | at its own `tau*` | Change |
|---|---|---|---|
| GNN L4+fp (`tau* = -0.25` default / `+0.05` occurring) | 0.3202 | 0.3143 | **−0.006** |
| RF + Morgan (`tau* = +0.15`) | 0.2885 | **0.3215** | **+0.033** |

(Macro-F1 over occurring classes, the new primary.)

The asymmetry has a clear cause: the GNN trains with sqrt class weights, which already
correct for imbalance, so there is nothing left for `tau` to fix and a positive `tau`
over-corrects. The RF's `class_weight="balanced_subsample"` reweights the bootstrap, but its
`predict_proba` argmax remains frequency-biased, so it was genuinely under-corrected.

**This makes the S1 comparison depend on what each model is allowed.** Both framings are
honest and both belong in the write-up:

| S1 test (occurring) | GNN L4+fp | RF + Morgan | Verdict |
|---|---|---|---|
| As trained, no inference-time tuning | **0.3202** | 0.2885 | GNN ahead +0.032, on all 3 seeds |
| Each with its own val-tuned `tau` | 0.3143 | **0.3215** | Parity (RF +0.007) |

The second row is the stronger baseline and the one a reviewer would ask for, so it should
be reported; the first should not be presented alone.

## Consequences

1. **Prior adjustment is not used for the GNN.** It is not part of the pipeline and should
   not appear in GNN results tables except as this negative result. For the RF it is a real
   improvement, so the adjusted RF is the baseline to beat.
2. **The primary metric changed.** `scores()` now records three variants, and
   `macro_f1_occurring` (averaged over classes present in `y_true`) is primary: it is
   fixed by the data, so no choice a model makes can alter the divisor. `macro_f1_all86`
   is reported alongside. The old `macro_f1` is retained so earlier rows stay readable but
   **is not comparable between models**.
3. Rows recorded before the change carry only the old column;
   [`src/rescore.py`](../src/rescore.py) refills them from saved checkpoints.

## A real finding that survives

Every seed's optimum sat at *negative* `tau` under the gameable metric, and positive `tau`
was monotonically harmful under all three. The sqrt class weighting used in training
already over-corrects for imbalance — there is no residual under-correction left to fix.
A milder or absent class weighting is worth one cell of the round-2 grid; that would be a
model that needs no inference-time correction at all.

## Reproducing

```bash
python src/adjust.py --ckpt "checkpoints/S1_seed*_L4_*_fp.pt" --device cpu \
    --tau-min -0.8 --tau-max 1.5 --tau-step 0.05 --per-class
```

Checkpoints live in Google Drive under `ddi_project_runs/checkpoints/`. Logits are cached
in `results/logits/` (git-ignored) so re-sweeps are instant.
