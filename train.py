
import argparse, os, inspect, importlib

from model import fit, TrainConfig

def import_loader_fn():
    # Try user dataloader first
    try:
        dl = importlib.import_module("dataloader")
        for name in ("make_dataloaders", "get_dataloaders", "build_dataloaders"):
            if hasattr(dl, name):
                return getattr(dl, name)
        raise ImportError("No make_dataloaders/get_dataloaders/build_dataloaders in dataloader.py")
    except Exception as e:
        # Fallback to dataloader_min if present
        try:
            from dataloader_min import make_dataloaders as fn
            return fn
        except Exception:
            raise e

def call_loader_with_signature(fn, args):
    sig = inspect.signature(fn)
    params = sig.parameters

    # decide how to pass the training CSV
    pos_args = []
    kw = {}

    if "csv_path" in params:
        kw["csv_path"] = args.train_csv
    elif "train_csv" in params:
        kw["train_csv"] = args.train_csv
    elif "path" in params:
        kw["path"] = args.train_csv
    else:
        # assume first positional is the path
        pos_args.append(args.train_csv)

    # map optional args only if the function accepts them
    opt_map = {
        "target": "target",
        "sep": "sep",
        "val_size": "val_size",
        "random_state": "seed",
        "batch_size": "batch_size",
        "use_weighted_sampler": "use_weighted_sampler",
    }
    for fn_kw, arg_name in opt_map.items():
        if fn_kw in params:
            kw[fn_kw] = getattr(args, arg_name)

    # common default
    if "num_workers" in params and "num_workers" not in kw:
        kw["num_workers"] = 0

    loaders = fn(*pos_args, **kw)

    # support (train, val) or (train, val, test)
    if isinstance(loaders, (list, tuple)) and len(loaders) >= 2:
        return loaders[0], loaders[1]
    raise RuntimeError("Your dataloader function must return (train_loader, val_loader) (optionally plus test).")

def main():
    p = argparse.ArgumentParser(description="Train TabularMLP on Bank Marketing data")
    p.add_argument("--train_csv", type=str, default="data/train.csv")
    p.add_argument("--sep", type=str, default=",")
    p.add_argument("--target", type=str, default="y")
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="outputs")
    p.add_argument("--no_weighted_sampler", action="store_true", help="Disable WeightedRandomSampler")
    args = p.parse_args()
    args.use_weighted_sampler = not args.no_weighted_sampler

    os.makedirs(args.save_dir, exist_ok=True)

    make_or_get = import_loader_fn()
    train_loader, val_loader = call_loader_with_signature(make_or_get, args)

    cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        seed=args.seed,
    )

    summary = fit(train_loader, val_loader, config=cfg, save_dir=args.save_dir)
    print("\nTraining finished. Best validation metrics:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nSaved: {os.path.join(args.save_dir, 'best_model.pt')}")
    print(f"Saved: {os.path.join(args.save_dir, 'model_config.json')}")

if __name__ == "__main__":
    main()
