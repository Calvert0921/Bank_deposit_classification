import os
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch


# ---------- 1) Utilities ----------
DEFAULT_TARGET_CANDIDATES = ["target", "y", "label", "is_subscribed"]

def _infer_target_col(df: pd.DataFrame, user_target: Optional[str] = None) -> str:
    """
    Try to detect the target column name automatically.
    Priority: user-specified -> default candidates.
    """
    if user_target is not None:
        if user_target not in df.columns:
            raise ValueError(f"target_col='{user_target}' not found in columns: {list(df.columns)}")
        return user_target
    for c in DEFAULT_TARGET_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"Target column not found. Please specify target_col "
                     f"(candidates: {DEFAULT_TARGET_CANDIDATES})")

def _split_num_cat(df: pd.DataFrame, exclude_cols: List[str]) -> Tuple[List[str], List[str]]:
    """
    Split columns into numeric and categorical lists.
    """
    num_cols, cat_cols = [], []
    for c in df.columns:
        if c in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols


# ---------- 2) Dataset ----------
class TabularDataset(Dataset):
    """
    A simple PyTorch Dataset for tabular data.
    """
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None, ids: Optional[np.ndarray] = None):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = None if y is None else torch.from_numpy(y.astype(np.float32)).view(-1, 1)
        self.ids = None if ids is None else torch.from_numpy(ids)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            if self.ids is None:
                return self.X[idx]
            return self.X[idx], self.ids[idx]
        return self.X[idx], self.y[idx]


# ---------- 3) Main function to create DataLoaders ----------
def get_dataloaders(
    train_csv: str,
    test_csv: Optional[str] = None,
    target_col: Optional[str] = None,
    id_col: Optional[str] = None,
    batch_size: int = 1024,
    num_workers: int = 2,
    val_size: float = 0.1,
    seed: int = 42,
    use_weighted_sampler: bool = True,
) -> Dict[str, Any]:
    """
    Build train/val/test DataLoaders for tabular binary classification.

    Returns:
      {
        "train_loader": DataLoader,
        "val_loader": DataLoader,
        "test_loader": Optional[DataLoader],
        "preprocess": {scaler, ohe, label_enc, num_cols, cat_cols, feature_names, target_name, classes_}
      }
    """
    assert os.path.exists(train_csv), f"{train_csv} not found"
    train_df = pd.read_csv(train_csv)

    # 1) Detect target column
    tgt = _infer_target_col(train_df, target_col)

    # Encode target (support yes/no, strings, numeric)
    label_enc = LabelEncoder()
    y_raw = train_df[tgt].values
    if pd.api.types.is_numeric_dtype(train_df[tgt]):
        y = y_raw.astype(int)
        label_enc.classes_ = np.array(sorted(np.unique(y)))
    else:
        y = label_enc.fit_transform(y_raw)

    # 2) Handle ID column (optional)
    ids_train = None
    if id_col and id_col in train_df.columns:
        ids_train = train_df[id_col].values

    # 3) Split numeric/categorical columns
    exclude_cols = [tgt] + ([id_col] if id_col and id_col in train_df.columns else [])
    num_cols, cat_cols = _split_num_cat(train_df, exclude_cols)

    # 4) Train/val split (Stratified)
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        train_df.drop(columns=[tgt]),
        y,
        test_size=val_size,
        random_state=seed,
        stratify=y
    )

    # Keep IDs if available
    ids_train_split = X_train_df[id_col].values if id_col and id_col in X_train_df.columns else None
    ids_val_split   = X_val_df[id_col].values if id_col and id_col in X_val_df.columns else None

    # Drop ID column from features
    if id_col and id_col in X_train_df.columns:
        X_train_df = X_train_df.drop(columns=[id_col])
        X_val_df = X_val_df.drop(columns=[id_col])

    # 5) Fit preprocessors (only on training set)
    scaler = StandardScaler() if num_cols else None
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False) if cat_cols else None

    # Numeric features
    if scaler:
        X_train_num = scaler.fit_transform(X_train_df[num_cols])
        X_val_num = scaler.transform(X_val_df[num_cols])
    else:
        X_train_num = np.empty((len(X_train_df), 0))
        X_val_num = np.empty((len(X_val_df), 0))

    # Categorical features
    if ohe:
        X_train_cat = ohe.fit_transform(X_train_df[cat_cols].astype(str))
        X_val_cat = ohe.transform(X_val_df[cat_cols].astype(str))
        cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
    else:
        X_train_cat = np.empty((len(X_train_df), 0))
        X_val_cat = np.empty((len(X_val_df), 0))
        cat_feature_names = []

    # Combine numeric + categorical
    X_train = np.hstack([X_train_num, X_train_cat])
    X_val = np.hstack([X_val_num, X_val_cat])

    feature_names = (num_cols or []) + cat_feature_names

    # 6) Build Dataset objects
    ds_train = TabularDataset(X_train, y_train, ids=ids_train_split)
    ds_val = TabularDataset(X_val, y_val, ids=ids_val_split)

    # 7) Build DataLoaders
    if use_weighted_sampler:
        # Handle class imbalance
        class_sample_count = np.bincount(y_train)
        class_sample_count = np.maximum(class_sample_count, 1)
        weights = 1.0 / class_sample_count
        sample_weights = weights[y_train]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            ds_train, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )

    val_loader = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    # 8) Test loader (optional)
    test_loader = None
    if test_csv is not None and os.path.exists(test_csv):
        test_df = pd.read_csv(test_csv)
        ids_test = test_df[id_col].values if (id_col and id_col in test_df.columns) else None
        if id_col and id_col in test_df.columns:
            test_df = test_df.drop(columns=[id_col])

        test_num = scaler.transform(test_df[num_cols]) if scaler else np.empty((len(test_df), 0))
        test_cat = ohe.transform(test_df[cat_cols].astype(str)) if ohe else np.empty((len(test_df), 0))
        X_test = np.hstack([test_num, test_cat])

        ds_test = TabularDataset(X_test, y=None, ids=ids_test)
        test_loader = DataLoader(
            ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "preprocess": {
            "scaler": scaler,
            "ohe": ohe,
            "label_enc": label_enc,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "feature_names": feature_names,
            "target_name": tgt,
            "classes_": list(getattr(label_enc, "classes_", [0, 1])),
        },
    }


# ---------- 4) Example usage ----------
if __name__ == "__main__":
    loaders = get_dataloaders(
        train_csv="data/train.csv",
        test_csv="data/test.csv",   # None if you don't have test.csv
        target_col=None,       # auto-detect target column
        id_col="id",           # None if you don't have an ID column
        batch_size=1024,
        num_workers=2,
        val_size=0.1,
        seed=42,
        use_weighted_sampler=True,
    )

    print("Train batches:", len(loaders["train_loader"]))
    print("Val batches:", len(loaders["val_loader"]))
    if loaders["test_loader"] is not None:
        print("Test batches:", len(loaders["test_loader"]))
    print("Num features:", len(loaders["preprocess"]["feature_names"]))
    print("Target:", loaders["preprocess"]["target_name"], "Classes:", loaders["preprocess"]["classes_"])
