"""Train the paper-configured ASL CNN and report validation metrics only.

The test split is never loaded by this script. The best checkpoint is selected
by validation loss, and all reported accuracy/recall values come from the
validation set. No augmentation is enabled by default; if added later, pass it
only to the training dataset after the split.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
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


def atomic_torch_save(payload: dict, destination: Path) -> None:
    """Write a checkpoint beside its destination, then atomically replace it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp",
                                          dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        # Flush checkpoint bytes before the atomic rename so a completed save is durable.
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_save(payload: dict | list, destination: Path) -> None:
    """Write JSON through a temporary sibling to avoid partial reports."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp",
                                          dir=destination.parent, text=True)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_payload(epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer,
                       validation: dict, class_names: list[str], seed: int, best_loss: float,
                       history: list[dict], training_config: dict) -> dict:
    """Capture everything needed to restart at the next epoch reproducibly."""
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation": validation,
        "class_names": class_names,
        "seed": seed,
        "best_validation_loss": best_loss,
        "history": history,
        "training_config": training_config,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


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
    parser.add_argument("--output", type=Path,
                        default=Path(os.environ.get("ASL_ARTIFACT_DIR", "outputs")),
                        help="Checkpoint directory (defaults to ASL_ARTIFACT_DIR or ./outputs).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", nargs="?", const="auto", default=None, metavar="CHECKPOINT",
                        help="Resume from CHECKPOINT; with no path, find the latest checkpoint in OUTPUT or project outputs/.")
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader worker count; 0 is the safest default on macOS.")
    args = parser.parse_args()
    output_was_explicit = any(arg == "--output" or arg.startswith("--output=")
                              for arg in sys.argv[1:])
    args.output = Path(os.path.expandvars(str(args.output))).expanduser()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")

    resume_path: Path | None = None
    if args.resume is not None:
        if args.resume == "auto":
            resume_path = args.output / "latest_checkpoint.pt"
            project_checkpoint = Path(__file__).resolve().parent / "outputs" / "latest_checkpoint.pt"
            if not resume_path.is_file() and not output_was_explicit and project_checkpoint.is_file():
                # A stale or unavailable ASL_ARTIFACT_DIR should not hide an
                # existing project-local checkpoint when the user asks to resume.
                resume_path = project_checkpoint
                args.output = project_checkpoint.parent
                print(f"Using project checkpoint at {resume_path}", flush=True)
        else:
            resume_path = Path(os.path.expandvars(args.resume)).expanduser()
        if not resume_path.is_file():
            parser.error(f"resume checkpoint does not exist: {resume_path}. "
                         "Pass its full path with --resume CHECKPOINT, or place it at OUTPUT/latest_checkpoint.pt.")

    try:
        args.output.mkdir(parents=True, exist_ok=True)
        # Fail before loading data/training if the chosen checkpoint destination
        # cannot accept atomic checkpoint writes.
        with tempfile.NamedTemporaryFile(prefix=".asl_write_check_", dir=args.output):
            pass
    except OSError as exc:
        parser.error(f"output directory is not writable: {args.output} ({exc}). "
                     "Choose a writable local or mounted Drive folder with --output.")

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
    best_loss = float("inf")
    history: list[dict] = []
    start_epoch = 1
    training_config = {"batch_size": args.batch_size, "learning_rate": args.learning_rate,
                       "seed": args.seed, "num_classes": len(class_names), "input_size": 400}
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint["class_names"] != class_names:
            parser.error("checkpoint class ordering does not match the current dataset")
        saved_config = checkpoint.get("training_config", {})
        for key in ("batch_size", "learning_rate", "seed", "num_classes", "input_size"):
            if key in saved_config and saved_config[key] != training_config[key]:
                parser.error(f"resume configuration mismatch for {key}: checkpoint has "
                             f"{saved_config[key]!r}, requested {training_config[key]!r}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = checkpoint.get("history", [])
        best_loss = checkpoint.get("best_validation_loss", float("inf"))
        start_epoch = int(checkpoint["epoch"]) + 1
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
            # map_location=device may move these saved byte tensors to CUDA,
            # while set_rng_state_all requires CPU ByteTensors as input.
            cuda_rng_states = [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
            torch.cuda.set_rng_state_all(cuda_rng_states)
        print(f"Resumed from {resume_path} at epoch {start_epoch}/{args.epochs}", flush=True)
        if start_epoch > args.epochs:
            print("Checkpoint already reached the requested epoch count; nothing to train.")
            return
    print(f"Device: {device}; train={len(train_data)}, validation={len(validation_data)}; test=not loaded")
    print(f"Training for {args.epochs} epochs, batch={args.batch_size}, Adam lr={args.learning_rate}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
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
            # Best is for inference/review, so omit optimizer and RNG state; the
            # resumable latest checkpoint below contains the complete run state.
            best_payload = {"epoch": epoch, "model_state_dict": model.state_dict(),
                            "validation": validation, "class_names": class_names,
                            "seed": args.seed, "training_config": training_config}
            atomic_torch_save(best_payload, args.output / "best_model.pt")
            atomic_json_save({"epoch": epoch, "metrics": validation,
                              "test_split_used": False}, args.output / "best_validation_metrics.json")

        # Latest is committed only after a full train+validation epoch. A crash
        # mid-epoch leaves the previous checkpoint intact, so resume repeats it.
        latest_payload = checkpoint_payload(epoch, model, optimizer, validation, class_names,
                                            args.seed, best_loss, history, training_config)
        atomic_torch_save(latest_payload, args.output / "latest_checkpoint.pt")
        atomic_json_save(history, args.output / "training_history.json")

    print(f"Best validation loss: {best_loss:.5f}; test split was not loaded or evaluated.")
    print(f"Best checkpoint and validation metrics saved under {args.output}")


if __name__ == "__main__":
    main()
