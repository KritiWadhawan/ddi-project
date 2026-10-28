"""Train the DDI GNN on one split/seed, early-stopping on validation macro-F1.

    python src/train.py --split S1 --seed 0
    python src/train.py --split S1 --seed 0 --readout mean      # pooling ablation
    python src/train.py --split S1 --seed 0 --max-batches 5 --epochs 1   # smoke test

Writes the best checkpoint to checkpoints/<run>.pt, the per-epoch log to
results/runs/<run>.csv, and appends the best model's val/test metrics to
results/gnn_metrics.csv (same metric definitions as baselines.py).
"""
import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from baselines import METRICS, scores
from graphs import load_graphs
from model import DDIModel
from splits import load_data, make_split

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = ROOT / "checkpoints"
RUNS = ROOT / "results" / "runs"
METRICS_CSV = ROOT / "results" / "gnn_metrics.csv"
N_CLASSES = 86


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PairLoader:
    """Batches (Drug1 graphs, Drug2 graphs, labels) from the cached drug graphs.

    Labels in the file are 1..86; the model uses 0..85.
    """

    def __init__(self, split, graphs, batch_size, shuffle, seed=0, workers=0):
        self.pairs = list(zip(split.ID1, split.ID2, split.Y - 1))
        self.graphs = graphs
        self.gen = torch.Generator().manual_seed(seed)
        self.loader = DataLoader(
            self.pairs,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=self.gen,
            collate_fn=self.collate,
            num_workers=workers,
            persistent_workers=workers > 0,
        )

    def collate(self, batch):
        id1, id2, y = zip(*batch)
        a = Batch.from_data_list([self.graphs[i] for i in id1], exclude_keys=["smiles"])
        b = Batch.from_data_list([self.graphs[i] for i in id2], exclude_keys=["smiles"])
        return a, b, torch.tensor(y)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    preds, labels = [], []
    for a, b, y in loader:
        preds.append(model(a.to(device), b.to(device)).argmax(1).cpu())
        labels.append(y)
    return scores(torch.cat(labels).numpy(), torch.cat(preds).numpy())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["transductive", "S1", "S2"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--readout", choices=["coattn", "mean"], default="coattn")
    p.add_argument("--no-inter", action="store_true", help="drop the bipartite GATConv")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--max-batches", type=int, default=None, help="cap train batches per epoch")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")
    p.add_argument("--resume", action="store_true", help="continue from checkpoints/<run>_last.pt")
    p.add_argument(
        "--class-weights", choices=["balanced", "sqrt"], default=None,
        help="balanced: sklearn's inverse frequency, as the RF baseline uses (ranges ~0.04-970 on S1); "
        "sqrt: inverse square-root frequency, a gentler version",
    )
    p.add_argument("--gnn-dropout", type=float, default=0.0, help="dropout on node features after each layer")
    p.add_argument("--adamw", action="store_true", help="decoupled weight decay (AdamW) instead of Adam's L2")
    args = p.parse_args()

    device = torch.device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)
    run = f"{args.split}_seed{args.seed}_{args.readout}{'_nointer' if args.no_inter else ''}_L{args.layers}"
    # Non-default training options go in the name so variants never collide (or get
    # skipped by --resume as already finished).
    tags = [
        {"balanced": "cw", "sqrt": "cwsqrt", None: ""}[args.class_weights],
        f"do{args.gnn_dropout:g}" if args.gnn_dropout else "",
        "adamw" if args.adamw else "",
        f"lr{args.lr:g}" if args.lr != 1e-3 else "",
        f"wd{args.weight_decay:g}" if args.weight_decay != 5e-4 else "",
    ]
    run += "".join(f"_{t}" for t in tags if t)
    if args.resume and METRICS_CSV.exists():
        with open(METRICS_CSV, newline="") as f:
            if any(row["model"] == run for row in csv.DictReader(f)):
                print(f"{run} already finished (in {METRICS_CSV.name}); skipping")
                return

    graphs = load_graphs()
    train, val, test = make_split(args.split, load_data(), args.seed)
    train_loader = PairLoader(train, graphs, args.batch_size, shuffle=True, seed=args.seed, workers=args.workers)
    val_loader = PairLoader(val, graphs, args.batch_size * 2, shuffle=False, workers=args.workers)
    test_loader = PairLoader(test, graphs, args.batch_size * 2, shuffle=False, workers=args.workers)

    model = DDIModel(
        n_classes=N_CLASSES,
        n_layers=args.layers,
        hidden=args.hidden,
        inter=not args.no_inter,
        readout=args.readout,
        gnn_dropout=args.gnn_dropout,
    ).to(device)
    optim = torch.optim.AdamW if args.adamw else torch.optim.Adam
    opt = optim(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # balanced is sklearn's n_samples / (n_present_classes * count); sqrt uses 1/sqrt(count)
    # scaled so the per-sample mean weight is 1, like balanced. Classes missing from train
    # never appear as targets, so their weight is irrelevant; it's left at 0.
    class_weight = None
    if args.class_weights:
        counts = torch.bincount(torch.tensor(train.Y.values - 1), minlength=N_CLASSES).float()
        present = counts > 0
        if args.class_weights == "balanced":
            w = len(train) / (present.sum() * counts.clamp(min=1))
        else:
            w = counts.clamp(min=1).rsqrt()
            w = w * len(train) / (w * counts).sum()
        class_weight = torch.where(present, w, torch.zeros(())).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{run}: train={len(train)} val={len(val)} test={len(test)} params={n_params:,} device={device}")

    CHECKPOINTS.mkdir(exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINTS / f"{run}.pt"
    # Written every epoch so a run cut off by a Colab/Kaggle disconnect can --resume.
    last_ckpt = CHECKPOINTS / f"{run}_last.pt"
    log_rows, best_f1, best_epoch, start_epoch = [], -1.0, 0, 1
    if args.resume and last_ckpt.exists():
        # Load to CPU: RNG states must stay CPU ByteTensors; load_state_dict moves the
        # model and optimizer tensors to the model's device itself.
        state = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        train_loader.gen.set_state(state["loader_rng"])
        torch.set_rng_state(state["torch_rng"])
        log_rows, best_f1, best_epoch = state["log_rows"], state["best_f1"], state["best_epoch"]
        start_epoch = state["epoch"] + 1
        print(f"resumed after epoch {state['epoch']} (best val macro-F1 {best_f1:.3f} at epoch {best_epoch})")

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch - best_epoch > args.patience:
            break  # resumed a run that had already early-stopped
        model.train()
        t0, total_loss, n_batches = time.time(), 0.0, 0
        for a, b, y in train_loader:
            loss = F.cross_entropy(model(a.to(device), b.to(device)), y.to(device), weight=class_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
            if args.max_batches and n_batches >= args.max_batches:
                break
        train_time = time.time() - t0

        val_m = evaluate(model, val_loader, device)
        log_rows.append(
            {"epoch": epoch, "loss": total_loss / n_batches, **{f"val_{k}": v for k, v in val_m.items()},
             "train_s": round(train_time, 1), "total_s": round(time.time() - t0, 1)}
        )
        improved = val_m["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch = val_m["macro_f1"], epoch
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, ckpt)
        print(
            f"epoch {epoch:3d} loss={total_loss / n_batches:.4f} val_acc={val_m['acc']:.3f} "
            f"val_macro_f1={val_m['macro_f1']:.3f}{' *' if improved else ''} "
            f"({train_time:.0f}s train, {time.time() - t0:.0f}s total)",
            flush=True,
        )
        with open(RUNS / f"{run}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0]))
            writer.writeheader()
            writer.writerows(log_rows)
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
             "loader_rng": train_loader.gen.get_state(), "torch_rng": torch.get_rng_state(),
             "log_rows": log_rows, "best_f1": best_f1, "best_epoch": best_epoch},
            last_ckpt,
        )
        if epoch - best_epoch >= args.patience:
            print(f"early stop: no val macro-F1 gain for {args.patience} epochs")
            break

    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    final = {"val": evaluate(model, val_loader, device), "test": evaluate(model, test_loader, device)}
    write_header = not METRICS_CSV.exists()
    with open(METRICS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["split", "seed", "model", "best_epoch", "set", *METRICS]
        )
        if write_header:
            writer.writeheader()
        for set_name, m in final.items():
            writer.writerow(
                {"split": args.split, "seed": args.seed, "model": run, "best_epoch": best_epoch,
                 "set": set_name, **m}
            )
    print(f"best epoch {best_epoch}: val {final['val']}\n  test {final['test']}")


if __name__ == "__main__":
    main()
