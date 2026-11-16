"""Train the paper-configured ASL CNN and report validation metrics only.

The test split is never loaded by this script. The best checkpoint is selected
by validation loss, and all reported accuracy/recall values come from the
validation set. No augmentation is enabled by default; if added later, pass it
only to the training dataset after the split.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from torch import nn
from torch.utils.data import DataLoader

from dataset import ASLAlphabetDataset
from model import ASLAlphabetCNN


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable training order/init."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_validation(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                        device: torch.device, class_names: list[str]) -> dict:
    """Compute loss, accuracy, and per-class recall on validation data only."""
    model.eval()
    total_loss = 0.0
    labels_all: list[int] = []
    predictions_all: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())
    label_ids = list(range(len(class_names)))
    recalls = recall_score(labels_all, predictions_all, labels=label_ids,
                           average=None, zero_division=0)
    matrix = confusion_matrix(labels_all, predictions_all, labels=label_ids)
    supports = matrix.sum(axis=1)
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(labels_all, predictions_all),
        "macro_recall": recall_score(labels_all, predictions_all, labels=label_ids,
                                      average="macro", zero_division=0),
        "weighted_recall": recall_score(labels_all, predictions_all, labels=label_ids,
                                         average="weighted", zero_division=0),
        "per_class": {
            name: {"recall": float(recalls[i]), "support": int(supports[i]),
                   "accuracy": float(matrix[i, i] / supports[i]) if supports[i] else 0.0}
            for i, name in enumerate(class_names)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader worker count; 0 is the safest default on macOS.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Intentionally load only train and validation manifests; test rows are never accessed.
    train_data = ASLAlphabetDataset(args.manifest, split="train")
    validation_data = ASLAlphabetDataset(args.manifest, split="validation")
    class_names = list("ABCDEFGHI") + list("KLMNOPQRSTUVWXY")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.workers, pin_memory=device.type == "cuda")

    model = ASLAlphabetCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history: list[dict] = []
    print(f"Device: {device}; train={len(train_data)}, validation={len(validation_data)}; test=not loaded")
    print(f"Training for {args.epochs} epochs, batch={args.batch_size}, Adam lr={args.learning_rate}", flush=True)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        validation = evaluate_validation(model, validation_loader, criterion, device, class_names)
        entry = {"epoch": epoch, "train_loss": running_loss / len(train_data),
                 "validation": validation, "seconds": time.time() - start}
        history.append(entry)
        print(f"Epoch {epoch:02d}/{args.epochs}: train_loss={entry['train_loss']:.5f} "
              f"val_loss={validation['loss']:.5f} val_accuracy={validation['accuracy']:.4f} "
              f"val_macro_recall={validation['macro_recall']:.4f} "
              f"elapsed={entry['seconds']:.1f}s", flush=True)
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "validation": validation,
                        "class_names": class_names, "seed": args.seed},
                       args.output / "best_model.pt")
            (args.output / "best_validation_metrics.json").write_text(
                json.dumps({"epoch": epoch, "metrics": validation,
                            "test_split_used": False}, indent=2) + "\n")
        (args.output / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")

    print(f"Best validation loss: {best_loss:.5f}; test split was not loaded or evaluated.")
    print(f"Best checkpoint and validation metrics saved under {args.output}")


if __name__ == "__main__":
    main()
