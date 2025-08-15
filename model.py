
"""
model.py — Tabular Binary Classifier (Bank Term Deposit Subscription)
---------------------------------------------------------------------
Goal:
  Predict whether a client will subscribe to a term deposit (binary y).

Expected input:
  Batches from a PyTorch DataLoader yielding (X, y)
    - X: float32 tensor of shape (B, D) — already preprocessed (numeric + encoded categorical)
    - y: float/long tensor of shape (B,) with values in {0,1}

Artifacts saved by `fit()`:
  - best_model.pt
  - model_config.json (includes tuned threshold, pos_weight, etc.)

Note:
  This file does NOT do feature engineering. Handle preprocessing in your dataloader.
"""
from __future__ import annotations

import os, json, math, random
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from tqdm.auto import tqdm

# ------------------------------
# Utilities
# ------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _infer_input_dim(sample_batch) -> int:
    X, _ = sample_batch
    if isinstance(X, dict):
        X = X.get("X")
    if not torch.is_tensor(X):
        raise ValueError("Expected features X to be a tensor or a dict with key 'X'.")
    return int(X.shape[-1])


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_pos_weight(train_loader: DataLoader) -> torch.Tensor:
    """pos_weight = N_neg / N_pos for BCEWithLogitsLoss."""
    pos = 0
    total = 0
    for _, y in train_loader:
        y = y.detach().view(-1).float()
        pos += (y == 1).sum().item()
        total += y.numel()
    neg = max(total - pos, 1)
    pos = max(pos, 1)
    return torch.tensor([neg / pos], dtype=torch.float32)


# ------------------------------
# Model
# ------------------------------
class TabularMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (256, 128),
        dropout: float = 0.20,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

        # He/Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in = m.weight.size(1)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        logits = self.head(z).squeeze(-1)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(x))


# ------------------------------
# Training / Eval
# ------------------------------
@dataclass
class TrainConfig:
    input_dim: Optional[int] = None   # if None, infer from first batch
    hidden_dims: Tuple[int, ...] = (256, 128)
    dropout: float = 0.20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    seed: int = 42
    early_stop_patience: int = 8
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    grad_clip: float = 5.0
    show_train_pbar: bool = True
    show_val_pbar: bool = False


@torch.no_grad()
def _gather_preds(loader: DataLoader, model: nn.Module, device: torch.device):
    model.eval()
    all_p, all_y = [], []
    for X, y in loader:
        X = X.to(device).float()
        y = y.to(device).float().view(-1)
        probs = model.predict_proba(X)
        all_p.append(probs.detach().cpu())
        all_y.append(y.detach().cpu())
    return torch.cat(all_p), torch.cat(all_y)


def evaluate(loader: DataLoader, model: nn.Module, device: Optional[torch.device] = None) -> Dict[str, float]:
    device = device or _get_device()
    probs, ys = _gather_preds(loader, model, device)
    y_true = ys.numpy()
    y_prob = probs.numpy()
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y_true, y_prob)
    except Exception:
        auprc = float("nan")
    y_pred = (y_prob >= 0.5).astype("int32")
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"auroc": float(auroc), "auprc": float(auprc), "f1@0.5": float(f1)}


def _select_best_threshold(y_true: np.ndarray, y_prob: np.ndarray):
    ps, rs, ts = precision_recall_curve(y_true, y_prob)
    ts = list(ts)
    if not ts:
        return 0.5, 0.0
    best_t, best_f1 = 0.5, -1.0
    for t in ts:
        pred = (y_prob >= t).astype("int32")
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def train_one_epoch(
    loader: DataLoader,
    model: nn.Module,
    optimizer,
    criterion,
    device: torch.device,
    grad_clip: float,
    epoch: int = 1,
    total_epochs: int = 1,
    show_pbar: bool = False,
) -> float:
    model.train()
    total_loss, n = 0.0, 0
    iterator = tqdm(loader, desc=f"Epoch {epoch:03d}/{total_epochs}", unit="batch", leave=False) if show_pbar else loader
    for X, y in iterator:
        X = X.to(device).float()
        y = y.to(device).float().view(-1)
        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
        if show_pbar:
            iterator.set_postfix(loss=float(loss.item()))
    return total_loss / max(n, 1)


def fit(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[TrainConfig] = None,
    input_dim: Optional[int] = None,
    save_dir: str = ".",
) -> Dict[str, Any]:
    config = config or TrainConfig()
    if input_dim is None:
        for batch in train_loader:
            input_dim = _infer_input_dim(batch)
            break
        if input_dim is None:
            raise RuntimeError("Failed to infer input_dim — provide it explicitly.")

    set_seed(config.seed)
    device = _get_device()

    model = TabularMLP(
        input_dim=input_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)

    pos_weight = _compute_pos_weight(train_loader).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.scheduler_factor, patience=config.scheduler_patience
    )

    best = {"val_loss": float("inf"), "epoch": 0}
    wait = 0

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            train_loader, model, optimizer, criterion, device, config.grad_clip,
            epoch=epoch, total_epochs=config.epochs, show_pbar=config.show_train_pbar
        )

        # validation loss
        model.eval()
        with torch.no_grad():
            val_loss, n = 0.0, 0
            v_iter = tqdm(val_loader, desc="Valid", unit="batch", leave=False) if config.show_val_pbar else val_loader
            for X, y in v_iter:
                X = X.to(device).float()
                y = y.to(device).float().view(-1)
                logits = model(X)
                loss = criterion(logits, y)
                val_loss += loss.item() * y.size(0)
                n += y.size(0)
            val_loss /= max(n, 1)

        scheduler.step(val_loss)
        metrics = evaluate(val_loader, model, device)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | "
              f"val_AUPRC={metrics['auprc']:.4f} | val_AUROC={metrics['auroc']:.4f} | F1@0.5={metrics['f1@0.5']:.4f}")

        if val_loss < best["val_loss"]:
            probs, ys = _gather_preds(val_loader, model, device)
            y_true, y_prob = ys.numpy(), probs.numpy()
            t, f1_best = _select_best_threshold(y_true, y_prob)

            best.update({
                "val_loss": float(val_loss),
                "val_auprc": float(metrics["auprc"]),
                "val_auroc": float(metrics["auroc"]),
                "f1_at_0.5": float(metrics["f1@0.5"]),
                "f1_at_best_t": float(f1_best),
                "threshold": float(t),
                "epoch": int(epoch),
            })

            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            with open(os.path.join(save_dir, "model_config.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "input_dim": int(input_dim),
                    "hidden_dims": list(config.hidden_dims),
                    "dropout": float(config.dropout),
                    "lr": float(config.lr),
                    "weight_decay": float(config.weight_decay),
                    "pos_weight": float(pos_weight.item()),
                    "threshold": float(t),
                    "val_auprc": float(metrics["auprc"]),
                    "val_auroc": float(metrics["auroc"]),
                    "seed": int(config.seed),
                }, f, indent=2)
            wait = 0
        else:
            wait += 1
            if wait >= config.early_stop_patience:
                print("Early stopping.")
                break

    return best


def load_trained_model(save_dir: str = ".") -> Tuple[TabularMLP, Dict[str, Any]]:
    with open(os.path.join(save_dir, "model_config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model = TabularMLP(input_dim=cfg["input_dim"], hidden_dims=tuple(cfg["hidden_dims"]), dropout=cfg["dropout"])
    state = torch.load(os.path.join(save_dir, "best_model.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, cfg


if __name__ == "__main__":
    print("This module defines the TabularMLP model and training helpers.\n"
          "Integrate it with your existing dataloader to train the model.")
