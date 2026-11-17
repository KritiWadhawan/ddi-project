"""Logit adjustment at inference: boost rare classes to raise macro-F1, no retraining.

A cross-entropy-trained argmax is biased toward frequent classes (class 49 alone is 32%
of DrugBank DDI), which costs macro-F1 since it weights all 86 classes equally. The fix
(Menon et al., "Long-tail learning via logit adjustment", ICLR 2021) is to predict

    argmax_c  f_c(x) - tau * log pi_c

where pi is the training class prior and tau a single scalar. tau=0 is the unmodified
model; tau=1 is the full correction. Training already uses sqrt class weights, so tau
picks up whatever correction remains.

tau is chosen on VALIDATION macro-F1 only -- one scalar shared across seeds, so the whole
procedure costs one hyperparameter -- and the test set is read once at that tau.

    python src/adjust.py --ckpt checkpoints/S1_seed0_..._fp.pt checkpoints/S1_seed1_..._fp.pt
    python src/adjust.py --ckpt "checkpoints/S1_seed*_fp.pt" --per-class

Logits are cached under results/logits/ so re-sweeps are instant. Writes the selected-tau
metrics to results/adjusted_metrics.csv and, with --per-class, per-class F1 to
results/per_class_f1.csv.
"""
import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from baselines import CLASSES_0BASED, METRICS, scores
from features import N_BITS, build_fp_table
from graphs import load_graphs
from model import DDIModel
from splits import load_data, make_split
from train import PairLoader, N_CLASSES

ROOT = Path(__file__).resolve().parent.parent
LOGITS_DIR = ROOT / "results" / "logits"
ADJUSTED_CSV = ROOT / "results" / "adjusted_metrics.csv"
PER_CLASS_CSV = ROOT / "results" / "per_class_f1.csv"


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    out, labels = [], []
    for a, b, y in loader:
        out.append(model(a.to(device), b.to(device)).cpu())
        labels.append(y)
    return torch.cat(out).numpy(), torch.cat(labels).numpy()


def logits_for(ckpt_path: Path, device: str, batch_size: int = 512) -> dict:
    """Raw val/test logits, labels and the train class prior for one checkpoint."""
    run = ckpt_path.stem
    cache = LOGITS_DIR / f"{run}.npz"
    # Keyed on the run name, so stamp the source file's size+mtime and recompute if the
    # checkpoint was replaced (a retrained run reuses its name).
    stamp = np.array([ckpt_path.stat().st_size, int(ckpt_path.stat().st_mtime)])
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if "stamp" in z.files and np.array_equal(z["stamp"], stamp):
            print(f"{run}: cached logits")
            return {k: z[k] for k in z.files}
        print(f"{run}: checkpoint changed since cache was written; recomputing")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = state["args"]
    print(f"{run}: split={a['split']} seed={a['seed']} best_epoch={state['epoch']}")

    graphs = load_graphs()
    if a.get("fp"):
        for drug_id, fp in build_fp_table().items():
            graphs[drug_id].fp = torch.tensor(fp, dtype=torch.float).unsqueeze(0)
    train, val, test = make_split(a["split"], load_data(), a["seed"])

    model = DDIModel(
        n_classes=N_CLASSES,
        n_layers=a["layers"],
        hidden=a["hidden"],
        inter=not a["no_inter"],
        readout=a["readout"],
        gnn_dropout=a["gnn_dropout"],
        fp_dim=N_BITS if a.get("fp") else 0,
    ).to(device)
    model.load_state_dict(state["model"])

    data = {}
    for name, split in [("val", val), ("test", test)]:
        loader = PairLoader(split, graphs, batch_size, shuffle=False)
        data[f"{name}_logits"], data[f"{name}_y"] = collect_logits(model, loader, device)
    counts = np.bincount(train.Y.values - 1, minlength=N_CLASSES).astype(float)
    data["prior"] = counts / counts.sum()
    data["meta"] = np.array([a["split"], a["seed"], state["epoch"]], dtype=object)
    data["stamp"] = stamp

    LOGITS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **data)
    return data


def macro_f1_true_labels(y, pred) -> float:
    """Macro-F1 averaged only over the classes present in y_true.

    scores() follows sklearn's default label set (classes in y_true OR y_pred), which is
    what every other table in this project uses, so that stays the headline. But raising
    tau makes the model predict more rare classes, and any it predicts that never occur
    score 0 and enter the average -- so this variant separates "predicts rare classes
    better" from "predicts rare classes that aren't there".
    """
    return f1_score(y, pred, average="macro", labels=np.unique(y), zero_division=0)


def predict(logits: np.ndarray, prior: np.ndarray, tau: float) -> np.ndarray:
    """argmax of logit - tau*log(prior), with classes absent from training excluded.

    At tau=0 and without the mask this is the plain argmax that produced gnn_metrics.csv.
    """
    if tau == 0:
        return logits.argmax(1)
    present = prior > 0
    adj = logits - tau * np.log(np.where(present, prior, 1.0))
    return np.where(present, adj, -np.inf).argmax(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True, help="checkpoint paths (globs allowed)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--tau-min", type=float, default=0.0, help="negative tau suppresses rare classes")
    p.add_argument("--tau-max", type=float, default=1.5)
    p.add_argument("--tau-step", type=float, default=0.05)
    p.add_argument("--per-class", action="store_true", help="also write per-class F1 at the chosen tau")
    args = p.parse_args()

    paths = sorted({Path(q) for pattern in args.ckpt for q in glob.glob(pattern)})
    if not paths:
        raise SystemExit(f"no checkpoints matched {args.ckpt}")

    runs = [logits_for(q, args.device, args.batch_size) for q in paths]
    taus = np.round(np.arange(args.tau_min, args.tau_max + 1e-9, args.tau_step), 4)

    # Sanity check: tau=0 must reproduce the committed gnn_metrics.csv rows.
    print("\ntau=0 (unadjusted, should match gnn_metrics.csv):")
    for q, d in zip(paths, runs):
        m = scores(d["test_y"], d["test_logits"].argmax(1), CLASSES_0BASED)
        print(f"  {q.stem[:46]:46s} val {scores(d['val_y'], d['val_logits'].argmax(1), CLASSES_0BASED)['macro_f1']:.4f}"
              f"  test {m['macro_f1']:.4f}")

    # One shared tau, selected on mean validation macro-F1 across seeds.
    curve = []
    for tau in taus:
        vals = [f1_score(d["val_y"], predict(d["val_logits"], d["prior"], tau),
                         average="macro", zero_division=0) for d in runs]
        curve.append(np.mean(vals))
    curve = np.array(curve)
    best = int(curve.argmax())
    tau_star = float(taus[best])

    print(f"\nvalidation macro-F1 vs tau (mean over {len(runs)} seeds):")
    for tau, v in zip(taus, curve):
        bar = "#" * int(round(v * 200))
        print(f"  tau={tau:4.2f}  {v:.4f} {bar}{'   <-- selected' if float(tau) == tau_star else ''}")

    print(f"\nselected tau = {tau_star:.2f} (on validation only)")
    # What a per-seed tau would have given, to show what sharing one costs. Diagnostic
    # only: three taus fitted on validation would be three hyperparameters, not one.
    print("per-seed best tau on validation (diagnostic, not used):")
    for d in runs:
        per = [f1_score(d["val_y"], predict(d["val_logits"], d["prior"], tau),
                        average="macro", zero_division=0) for tau in taus]
        j = int(np.argmax(per))
        shared = per[int(np.argmin(np.abs(taus - tau_star)))]
        print(f"  seed {d['meta'][1]}: best tau={taus[j]:4.2f} -> val {per[j]:.4f}"
              f"   (at the shared tau={tau_star:.2f}: {shared:.4f})")
    print()
    rows = []
    for q, d in zip(paths, runs):
        split, seed, epoch = d["meta"]
        row = {"model": q.stem, "split": split, "seed": int(seed), "best_epoch": int(epoch), "tau": tau_star}
        for name in ("val", "test"):
            base = scores(d[f"{name}_y"], d[f"{name}_logits"].argmax(1), CLASSES_0BASED)
            adj = scores(d[f"{name}_y"], predict(d[f"{name}_logits"], d["prior"], tau_star), CLASSES_0BASED)
            row |= {f"{name}_{k}": adj[k] for k in METRICS}
            row[f"{name}_macro_f1_base"] = base["macro_f1"]
            row[f"{name}_macro_f1_occurring"] = macro_f1_true_labels(
                d[f"{name}_y"], predict(d[f"{name}_logits"], d["prior"], tau_star)
            )
            row[f"{name}_macro_f1_occurring_base"] = macro_f1_true_labels(
                d[f"{name}_y"], d[f"{name}_logits"].argmax(1)
            )
        rows.append(row)
        print(f"  seed {seed}: val {row['val_macro_f1_base']:.4f} -> {row['val_macro_f1']:.4f}"
              f"   test {row['test_macro_f1_base']:.4f} -> {row['test_macro_f1']:.4f}")

    for name in ("val", "test"):
        base = np.array([r[f"{name}_macro_f1_base"] for r in rows])
        adj = np.array([r[f"{name}_macro_f1"] for r in rows])
        print(f"\n{name}: macro-F1 {base.mean():.4f} +/- {base.std(ddof=1):.4f}"
              f"  ->  {adj.mean():.4f} +/- {adj.std(ddof=1):.4f}   (mean change {adj.mean() - base.mean():+.4f})")
        tb = np.array([r[f"{name}_macro_f1_occurring_base"] for r in rows])
        ta = np.array([r[f"{name}_macro_f1_occurring"] for r in rows])
        print(f"{name}: macro-F1 over occurring classes only {tb.mean():.4f} -> {ta.mean():.4f}"
              f"   (mean change {ta.mean() - tb.mean():+.4f})")
        acc = np.array([r[f"{name}_acc"] for r in rows])
        print(f"{name}: accuracy at the selected tau {acc.mean():.4f} +/- {acc.std(ddof=1):.4f}")

    with open(ADJUSTED_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {ADJUSTED_CSV.relative_to(ROOT)}")

    if args.per_class:
        per_class = []
        for q, d in zip(paths, runs):
            split, seed, _ = d["meta"]
            y, pred = d["test_y"], predict(d["test_logits"], d["prior"], tau_star)
            f1s = f1_score(y, pred, average=None, labels=np.arange(N_CLASSES), zero_division=0)
            support = np.bincount(y, minlength=N_CLASSES)
            train_counts = d["prior"] * 1.0
            for c in range(N_CLASSES):
                per_class.append({"model": q.stem, "split": split, "seed": int(seed),
                                  "class": c + 1, "train_frac": train_counts[c],
                                  "test_support": int(support[c]), "f1": f1s[c]})
        with open(PER_CLASS_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_class[0]))
            writer.writeheader()
            writer.writerows(per_class)
        print(f"wrote {PER_CLASS_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
