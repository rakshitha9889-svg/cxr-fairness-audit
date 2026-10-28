"""Step 3: train a DenseNet-121 baseline (5 findings, multi-label) and save predictions.

On Kaggle (GPU on):
    python src/train.py --data-dir /kaggle/input --epochs 8

Outputs in --out-dir (default: outputs/):
    best_model.pt         weights with the best validation mean AUROC (do NOT commit)
    history.csv           loss and AUROC per epoch
    metrics.json          per-finding AUROC and AUPRC on the test split
    val_preds.csv         per-image probabilities + sex/age/view, used in Step 4 (fairness)
    test_preds.csv        same for the test split
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import models

from dataset import FINDINGS, ChestXrayDataset, build_image_index

try:
    import mlflow
except ImportError:
    mlflow = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(pretrained=True):
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, len(FINDINGS))
    return model


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def per_finding_metrics(probs, labels):
    out = {}
    for j, f in enumerate(FINDINGS):
        y = labels[:, j]
        if 0 < y.sum() < len(y):
            out[f] = {"auroc": float(roc_auc_score(y, probs[:, j])),
                      "auprc": float(average_precision_score(y, probs[:, j])),
                      "n_pos": int(y.sum())}
    out["mean_auroc"] = float(np.mean([v["auroc"] for k, v in out.items() if k in FINDINGS]))
    return out


def save_preds(ds, probs, path):
    df = ds.df[["image_index", "patient_id", "sex", "age", "age_bin", "view"] + FINDINGS].copy()
    for j, f in enumerate(FINDINGS):
        df[f"prob_{f}"] = probs[:, j]
    df.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/kaggle/input")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true", help="for quick offline tests only")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Indexing images (takes ~1 min on Kaggle)...")
    index = build_image_index(Path(args.data_dir))
    sp = Path(args.splits)
    train_ds = ChestXrayDataset(sp / "train.csv", index, args.img_size, train=True)
    val_ds = ChestXrayDataset(sp / "val.csv", index, args.img_size)
    test_ds = ChestXrayDataset(sp / "test.csv", index, args.img_size)
    print(f"train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)}")

    kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda")
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, shuffle=False, **kw)
    test_dl = DataLoader(test_ds, shuffle=False, **kw)

    model = build_model(pretrained=not args.no_pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=train_ds.pos_weight().to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    if mlflow:
        mlflow.set_experiment("cxr-fairness-audit")
        mlflow.start_run(run_name="densenet121-baseline")
        mlflow.log_params(vars(args))

    best, bad_epochs, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for step, (x, y) in enumerate(train_dl, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_dl)} loss {running / step:.4f}")

        val_m = per_finding_metrics(*predict(model, val_dl, device))
        scheduler.step(val_m["mean_auroc"])
        row = {"epoch": epoch, "train_loss": running / len(train_dl), "val_mean_auroc": val_m["mean_auroc"],
               **{f"val_auroc_{f}": val_m[f]["auroc"] for f in FINDINGS if f in val_m},
               "minutes": (time.time() - t0) / 60}
        history.append(row)
        print(f"Epoch {epoch}: train loss {row['train_loss']:.4f}, val mean AUROC {row['val_mean_auroc']:.4f}, "
              f"{row['minutes']:.1f} min")
        if mlflow:
            mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)

        if val_m["mean_auroc"] > best:
            best, bad_epochs = val_m["mean_auroc"], 0
            torch.save(model.state_dict(), out / "best_model.pt")
            print("  saved new best model")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs")
                break

    pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    model.load_state_dict(torch.load(out / "best_model.pt", map_location=device))
    val_probs, _ = predict(model, val_dl, device)
    test_probs, test_labels = predict(model, test_dl, device)
    save_preds(val_ds, val_probs, out / "val_preds.csv")
    save_preds(test_ds, test_probs, out / "test_preds.csv")

    test_m = per_finding_metrics(test_probs, test_labels)
    test_m["best_val_mean_auroc"] = best
    (out / "metrics.json").write_text(json.dumps(test_m, indent=2))
    print("\nTest results (best model):")
    for f in FINDINGS:
        if f in test_m:
            m = test_m[f]
            print(f"  {f:13s} AUROC {m['auroc']:.3f}  AUPRC {m['auprc']:.3f}  positives {m['n_pos']}")
    print(f"  mean AUROC {test_m['mean_auroc']:.3f}")
    if mlflow:
        mlflow.log_metrics({"test_mean_auroc": test_m["mean_auroc"]})
        mlflow.end_run()


if __name__ == "__main__":
    main()
