#!/usr/bin/env python3
"""Train YOLO26n or tune a small, chess-specific hyperparameter space."""

import argparse
from pathlib import Path

import yaml


RECIPE = Path(__file__).with_name("yolo26_chess.yaml")
TUNE_SPACE = {
    "lr0": (0.0005, 0.01),
    "lrf": (0.01, 0.2),
    "weight_decay": (0.0001, 0.001),
    "box": (4.0, 9.0),
    "cls": (0.4, 1.2),
    "dfl": (3.0, 12.0),
    "scale": (0.0, 0.25),
    "mosaic": (0.0, 0.2),
    "hsv_s": (0.0, 0.3),
    "hsv_v": (0.0, 0.3),
}


def load_settings(hyp=None, **overrides):
    with RECIPE.open(encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    if hyp:
        with Path(hyp).open(encoding="utf-8") as f:
            tuned = yaml.safe_load(f)
        if not isinstance(tuned, dict) or set(tuned) - TUNE_SPACE.keys():
            raise ValueError("--hyp must contain only parameters from the chess tuning space")
        settings.update(tuned)
    settings.update({k: v for k, v in overrides.items() if v is not None})
    return settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Dataset YAML (default: datasets/data.yaml)")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int, help="Batch size; -1 enables single-GPU AutoBatch")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", help="e.g. 0, 0,1, or cpu")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--optimizer", choices=("MuSGD", "AdamW", "SGD"))
    parser.add_argument("--lr0", type=float)
    parser.add_argument("--name")
    parser.add_argument("--hyp", help="best_hyperparameters.yaml from a prior tuning run")
    parser.add_argument("--tune", type=int, default=0, metavar="TRIALS", help="Search on validation data instead of training once")
    parser.add_argument("--time", type=float, help="Training time limit in hours")
    parser.add_argument("--dry-run", action="store_true", help="Print settings without loading weights or training")
    args = vars(parser.parse_args(argv))
    tune, dry_run, hyp = (args.pop(k) for k in ("tune", "dry_run", "hyp"))
    try:
        settings = load_settings(hyp, **args)
    except (OSError, ValueError, yaml.YAMLError) as e:
        parser.error(str(e))
    if tune < 0 or settings["epochs"] < 1 or settings["imgsz"] < 32:
        parser.error("--tune must be >= 0; --epochs and --imgsz must be positive")
    if settings["batch"] != -1 and settings["batch"] < 1:
        parser.error("--batch must be >= 1 or -1 for AutoBatch")
    if settings["workers"] < 0 or settings["patience"] < 0:
        parser.error("--workers and --patience must be >= 0")
    if not 0 < settings["lr0"] < 1 or (settings.get("time") is not None and not 0 < settings["time"] < float("inf")):
        parser.error("--lr0 must be between 0 and 1; --time must be finite and positive")
    if args["optimizer"] in ("AdamW", "SGD") and args["lr0"] is None:
        parser.error("Set --lr0 explicitly when changing the optimizer")
    if tune and args["name"] is None:
        settings["name"] = "yolo26n_chess_tune"
    settings["data"] = str(Path(settings["data"]).resolve())
    print(yaml.safe_dump(settings, sort_keys=False))
    if dry_run:
        if tune:
            print(yaml.safe_dump({"iterations": tune, "space": TUNE_SPACE}))
        return
    if not Path(settings["data"]).is_file():
        parser.error(f"Dataset YAML not found: {settings['data']}. Generate the dataset first.")

    from ultralytics import YOLO

    model = YOLO(settings.pop("model"))
    if tune:
        model.tune(**settings, iterations=tune, space=TUNE_SPACE, use_ray=False)
        print("Retrain from yolo26n.pt with --hyp <tuning-run>/best_hyperparameters.yaml.")
    else:
        model.train(**settings)
        print(f"Best checkpoint: {model.trainer.best}")
    return model


if __name__ == "__main__":
    main()
