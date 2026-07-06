"""
Generic train / evaluation loop for EEGNet (or any nn.Module classifier).

Functions
---------
    train_one_epoch(model, loader, optimizer, criterion, device)
    evaluate(model, loader, criterion, device)
    train(model, train_loader, val_loader, ...)   -> history dict
    test(model, test_loader, device)              -> accuracy, per-class report
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from config import (
    LR, WEIGHT_DECAY, MAX_EPOCHS, EARLY_STOP_PATIENCE,
    CHECKPOINT_DIR, CLASS_NAMES,
)


# -----------------------------------------------------------------------
# Single-epoch helpers
# -----------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += len(y)

    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Returns (loss, accuracy, roc_auc).

    For binary classification ROC-AUC is computed from the predicted
    probability of the positive class (class index 1).
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_pos_probs, all_labels = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss   = criterion(logits, y)
        probs  = torch.softmax(logits, dim=1)   # (batch, 2)

        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += len(y)

        all_pos_probs.append(probs[:, 1].cpu().numpy())   # P(class=1)
        all_labels.append(y.cpu().numpy())

    pos_probs  = np.concatenate(all_pos_probs)   # (n_trials,)
    all_labels = np.concatenate(all_labels)

    try:
        auc = roc_auc_score(all_labels, pos_probs)
    except ValueError:
        auc = float("nan")   # guard: only one class present in batch

    return total_loss / n, correct / n, auc


# -----------------------------------------------------------------------
# Full training loop
# -----------------------------------------------------------------------

def train(
    model,
    train_loader,
    val_loader,
    checkpoint_name: str = "best_model.pt",
    lr:              float = LR,
    weight_decay:    float = WEIGHT_DECAY,
    max_epochs:      int   = MAX_EPOCHS,
    patience:        int   = EARLY_STOP_PATIENCE,
    device:          str   = None,
) -> dict:
    """
    Train `model` with Adam + early stopping on validation loss.

    Parameters
    ----------
    model           : nn.Module  (EEGNet or any compatible classifier)
    train_loader    : DataLoader
    val_loader      : DataLoader
    checkpoint_name : filename (saved under CHECKPOINT_DIR)
    lr, weight_decay: Adam hyperparameters
    max_epochs      : hard cap on training epochs
    patience        : stop if val loss does not improve for this many epochs
    device          : 'cuda', 'mps', or 'cpu'  (auto-detected if None)

    Returns
    -------
    history : dict with keys 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_auc'
              val_auc is the binary ROC-AUC on the validation set each epoch.
    """
    if device is None:
        device = _auto_device()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2)

    checkpoint_path = f"{CHECKPOINT_DIR}/{checkpoint_name}"
    best_val_loss   = float("inf")
    best_weights    = None
    epochs_no_impr  = 0

    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   [], "val_auc": []}

    print(f"\nTraining on {device} | max_epochs={max_epochs} | patience={patience}")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>9}  {'Val Acc':>8}  {'Val AUC':>8}  {'Time':>6}")
    print("-" * 70)

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc          = train_one_epoch(model, train_loader, optimizer,
                                                   criterion, device)
        vl_loss, vl_acc, vl_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["val_auc"].append(vl_auc)

        elapsed = time.time() - t0
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_acc:>9.4f}  "
              f"{vl_loss:>9.4f}  {vl_acc:>8.4f}  {vl_auc:>8.4f}  {elapsed:>5.1f}s")

        # ---- Early stopping -------------------------------------------------
        if vl_loss < best_val_loss - 1e-5:
            best_val_loss  = vl_loss
            best_weights   = copy.deepcopy(model.state_dict())
            epochs_no_impr = 0
            torch.save(best_weights, checkpoint_path)
        else:
            epochs_no_impr += 1
            if epochs_no_impr >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs).")
                break

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"Restored best weights from epoch with val_loss={best_val_loss:.4f}")

    return history


# -----------------------------------------------------------------------
# Test evaluation
# -----------------------------------------------------------------------

@torch.no_grad()
def test(model, test_loader, device: str = None) -> tuple:
    """
    Run the model on the test set and print a full classification report
    including ROC-AUC for binary classification.

    Returns
    -------
    (accuracy, auc, report_str, confusion_matrix_array)
    """
    if device is None:
        device = _auto_device()
    model.eval().to(device)

    all_preds, all_pos_probs, all_labels = [], [], []
    for X, y in test_loader:
        X      = X.to(device)
        logits = model(X)
        probs  = torch.softmax(logits, dim=1)   # (batch, 2)

        all_preds.append(logits.argmax(1).cpu().numpy())
        all_pos_probs.append(probs[:, 1].cpu().numpy())  # P(class=1)
        all_labels.append(y.numpy())

    y_pred     = np.concatenate(all_preds)
    y_pos_prob = np.concatenate(all_pos_probs)
    y_true     = np.concatenate(all_labels)

    acc    = (y_pred == y_true).mean()
    auc    = roc_auc_score(y_true, y_pos_prob)
    report = classification_report(y_true, y_pred,
                                   target_names=CLASS_NAMES, digits=4)
    cm     = confusion_matrix(y_true, y_pred)

    print(f"\n── Test Results ──────────────────────────────────────")
    print(f"Accuracy : {acc:.4f}  ({int(acc * len(y_true))}/{len(y_true)})")
    print(f"ROC-AUC  : {auc:.4f}")
    print(f"\n{report}")
    print("Confusion matrix:")
    print(cm)

    return acc, auc, report, cm


# -----------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------

def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"