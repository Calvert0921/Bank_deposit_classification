
import argparse, os
import numpy as np
import pandas as pd
import torch

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from model import load_trained_model

def build_preprocessor(df_train_X):
    num_cols = df_train_X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df_train_X.columns if c not in num_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    return pre

def main():
    ap = argparse.ArgumentParser(description="Generate predictions for test.csv")
    ap.add_argument("--train_csv", default="data/train.csv")
    ap.add_argument("--test_csv", default="data/test.csv")
    ap.add_argument("--sep", default=",")
    ap.add_argument("--target", default="y")
    ap.add_argument("--save_dir", default="outputs", help="where best_model.pt + model_config.json are saved")
    ap.add_argument("--out_csv", default="submission.csv")
    ap.add_argument("--id_col", default="id", help="name of id column if present; else we make a row index")
    args = ap.parse_args()

    # Load data
    train_df = pd.read_csv(args.train_csv, sep=args.sep)
    test_df = pd.read_csv(args.test_csv, sep=args.sep)

    # Drop leakage / non-informative
    drop_cols = []
    if "duration" in train_df.columns:
        drop_cols.append("duration")
    if args.id_col in train_df.columns:
        drop_cols.append(args.id_col)
    if drop_cols:
        train_df = train_df.drop(columns=drop_cols)
        test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    if args.target not in train_df.columns:
        raise ValueError(f"Target '{args.target}' not in train CSV.")

    y = train_df[args.target]
    X_train = train_df.drop(columns=[args.target])
    X_test = test_df.copy()

    # Build & fit preprocessor on full train
    pre = build_preprocessor(X_train)
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)

    # Load model
    model, cfg = load_trained_model(args.save_dir)
    model.eval()

    # Predict
    X_test_t = torch.tensor(np.asarray(X_test_t), dtype=torch.float32)
    with torch.no_grad():
        probs = model.predict_proba(X_test_t).cpu().numpy()

    thr = cfg.get("threshold", 0.5)
    preds = (probs >= thr).astype(int)

    # Build submission
    if args.id_col in test_df.columns:
        out = pd.DataFrame({args.id_col: test_df[args.id_col], "y": preds})
    else:
        out = pd.DataFrame({"y": preds})

    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} with {len(out)} rows. (threshold={thr})")

if __name__ == "__main__":
    main()
