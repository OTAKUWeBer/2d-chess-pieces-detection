# 2D chess detection with YOLO26n

Generate labeled chess screenshots and train a pretrained `yolo26n.pt` to detect 12 piece classes plus the board. The pipeline targets desktop, web, mobile, and broadcast screenshots, including boards embedded in photographic backgrounds.

There is no universal best set of parameters. These defaults are a starting point; accuracy improvements must be measured on held-out real screenshots. Synthetic mAP alone does not establish real-world FEN accuracy.

| File | Purpose |
| --- | --- |
| `generate_datasets.py` | Render images, labels, dataset YAML, and asset manifest. |
| `random_fen_gen.py` | Sample chess and recognition positions. |
| `train_yolo26.py` / `yolo26_chess.yaml` | Shared training defaults and bounded parameter search. |
| `verify_dataset.py` / `visualize_labels.py` | Validate data and inspect annotations. |
| `eval.py` | Run detection and compare predicted board placements with reference FENs. |
| `2d_chess_detection.ipynb` | Kaggle generation, training, evaluation, and ONNX export. |
| `.todo` / `notes` | Remaining real-image checks and unresolved classification errors. |

## Install

Use Python 3.10+ and install the appropriate [PyTorch build for your GPU](https://pytorch.org/get-started/locally/) first. For AMD Radeon, use AMD's [ROCm PyTorch installation guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html); NVIDIA CUDA wheels do not enable an AMD GPU. Then:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` covers generation, evaluation, the notebook, ONNX export/inference, asset downloaders, and SVG conversion. CairoSVG also needs the native Cairo library when rebuilding PNG sprites from SVGs. Existing PNG assets do not require conversion.

## Generate and verify

Run from the repository root:

```bash
python generate_datasets.py --train-count 20000 --val-count 1500 --test-count 1500 --workers 8
python verify_dataset.py --data-dir datasets
```

For a quick check, use `--train-count 100 --val-count 25 --test-count 25 --out-dir tmp/smoke`. Generation refuses to reuse a nonempty output directory; use a fresh path or explicitly pass `--clean` to delete the previous output.

The command above requests 23,000 images. Bare generator defaults are 1,000 training, 200 validation, and 200 test images; `--workers` defaults to available CPU cores. Reduce worker count if memory is limited.

The generator supplies:

- **800 x 800 images and boards**, retaining more small-piece detail than 640-pixel rendering. Match training and inference resolution when comparing runs.
- **32% full-board, 44% app UI, 17% embedded, and 7% multi-board scenes** before retries. All visible secondary-board pieces are labeled.
- **Position diversity:** 20% openings, 30% middlegames, 20% endgames, 20% recognition positions, and 10% constrained random positions. Recognition positions place one of every class at varied squares to teach appearance instead of starting-square shortcuts; they need not be legal games.
- **Every piece style in training**, cycling through styles before repeating. By default, validation/test reuse rendering styles with independent scenes and disjoint background files. This evaluates new screenshots of supported themes.
- **Optional unseen-style evaluation:** `--holdout-styles` partitions piece styles, board themes, and backgrounds. This is a different, harder benchmark. Training then excludes the held-out styles.
- **Reproducibility:** sample-specific Python and NumPy seeds produce identical images and labels across worker counts on the same software/platform.
- **A strict 18-pixel piece-size floor:** retry small samples; in mixed mode the final attempt uses a full board and still enforces the floor. A requested scene type stays fixed through every retry. Impossible settings fail instead of silently accepting tiny pieces.
- **Moderate artifacts:** highlights, move hints, arrows, coordinates, controlled blur/downsampling/JPEG compression, and occasional translucent distortions. Heavy corruption can erase the features separating similar pieces.

The output includes `images/{split}`, `labels/{split}`, a ready-to-train `data.yaml` with an absolute dataset path, and `manifest.json` recording settings and asset pools. Regenerate/update `data.yaml` after moving a dataset to another machine. Train-only generation (`--val-count 0 --test-count 0`) is supported, but training still needs a separate validation set.

Other useful arguments: `--composition` (`mixed`, `full`, `ui`, `embedded`, or `multi`), `--imgsz`, `--board-size` (divisible by 8), `--min-piece-px`, `--max-sample-attempts`, `--seed`, and `--no-board-labels` (writes a 12-class YAML). Missing or transparent sprites fail loudly.

QA checks image/label pairs, finite coordinates, box boundaries, class representation, object sizes, and identical decoded images across all three split pairs, even if their labels differ. It writes previews under `results/previews`. File partitions and exact hashes cannot prove that differently encoded or near-duplicate source assets are independent.

```bash
python visualize_labels.py --random --split val --count 5
```

### Train specifically on small boards in backgrounds

The previous small-board task is supported by the embedded compositor. To make an entire dataset of these scenes:

```bash
python generate_datasets.py --composition embedded --train-count 10000 --val-count 1000 --test-count 1000 --max-sample-attempts 32 --out-dir datasets/embedded
python verify_dataset.py --data-dir datasets/embedded
python visualize_labels.py --data-dir datasets/embedded --random --split train --count 8
python train_yolo26.py --data datasets/embedded/data.yaml --device 0 --name yolo26n_chess_embedded
```

The framed board spans 38–72% of image width, sampled around 54% (roughly 304–576 pixels at 800px). Piece and board labels scale and translate with the rendering. Dedicated embedded mode never substitutes a full-board scene when retrying. The size floor may reject the smallest or narrowest pieces; increase resolution before lowering the floor. Use mixed mode for a model expected to handle both full and embedded boards.

### Class IDs

| Piece | Black | White |
| --- | --- | --- |
| Pawn | 0 | 6 |
| Rook | 1 | 7 |
| Knight | 2 | 8 |
| Bishop | 3 | 9 |
| Queen | 4 | 10 |
| King | 5 | 11 |

Class 12 is `chess_board`, enclosing the playable 8 x 8 grid rather than coordinate borders. Black's perspective reverses square placement; piece images stay upright. Reference FENs are placement-only strings ordered by the top image row, without game-state or perspective metadata.

## Train

```bash
python train_yolo26.py --device 0
# Inspect the exact settings without importing PyTorch or loading weights:
python train_yolo26.py --dry-run
```

The CLI and Kaggle notebook share `yolo26_chess.yaml`. Runs get separate output directories; `best.pt`, `last.pt`, curves, and the actual arguments are saved by Ultralytics.

| Setting | Baseline | Reason |
| --- | --- | --- |
| Weights | `yolo26n.pt` | Fine-tune pretrained weights instead of training from scratch. |
| Resolution | 800 | Preserve fine details in embedded boards; compare 640 and 960 separately if memory/latency permits. |
| Epochs / patience | 100 / 30 | Retain the schedule reported to work best on real screenshots; see the historical results below. |
| Batch | 16 | Conservative starting point; lower if memory is insufficient. `--batch -1` enables single-GPU AutoBatch. |
| Optimizer | MuSGD | Explicit optimizer so the requested learning rate is used. |
| `lr0` / `lrf` | 0.0054 / 0.0495 | Published YOLO26n COCO recipe values, used as a starting point. |
| Momentum / decay | 0.947 / 0.00064 | Published nano optimizer values. |
| `box` / `cls` / `dfl` | 5.63 / 0.56 / 9.04 | Published nano loss gains; YOLO26 uses `dfl` for L1 distance regression. |
| Schedule / warmup | cosine / 3 epochs | Smooth decay with warmup for the new classification head. |
| Mosaic / MixUp / copy-paste | 0 / 0 / 0 | Preserve small piece details; generator already varies composition and scale. Tune low mosaic if helpful. |
| Image flips / rotation | disabled | Pieces remain upright; Black's perspective changes square placement without rotating sprites. |
| Scale / translation | 0.15 / 0.05 | Modest additional geometric variation. |
| HSV hue / saturation / value | 0.01 / 0.15 / 0.15 | Avoid stacking strong color corruption on generated artifacts. |

These chess-specific changes are hypotheses, not published chess benchmark results. Ultralytics explicitly recommends adjusting fine-tuning settings based on your own validation results. The public nano recipe uses much stronger natural-image augmentation and was trained at batch 128; its reported COCO metrics are not expected here.

To compare a longer schedule, run `python train_yolo26.py --epochs 150 --patience 30 --name yolo26n_chess_longer`. Other overrides include `--data`, `--imgsz`, `--batch`, `--workers`, `--seed`, `--name`, and `--time` (hours). Changing `--optimizer` requires an explicit `--lr0`.

## Find better parameters for your data

First save a baseline, then use the built-in Ultralytics tuner:

```bash
python train_yolo26.py --device 0 --tune 20 --epochs 60
python train_yolo26.py --device 0 --hyp runs/detect/yolo26n_chess_tune/best_hyperparameters.yaml --name yolo26n_chess_tuned
```

Use the actual tuning directory printed by Ultralytics if the name has a numeric suffix. Search varies learning rate, final LR fraction, weight decay, loss gains, mild HSV/scale, and mosaic between 0 and 0.2. Flips and blending stay disabled. Trials use validation data; the script never evaluates `test` during training/search.

Twenty trials of 60 epochs can cost roughly 1,200 training epochs. Short runs can favor fast convergence: retrain promising candidates at the full schedule, keep the baseline for comparison, and use multiple seeds for close results. The tuner selects detection fitness, not FEN exact-match accuracy.

For deployment accuracy, label representative real screenshots and add them to training/validation with splits grouped by game/session/source. Include examples of the confusing classes, low-contrast themes, board sizes, and overlays you actually see. Keep a separate untouched real test set; do not repeatedly select parameters against it.

## Final evaluation and export

Evaluate the selected checkpoint on the synthetic test set once:

```bash
yolo detect val model=runs/detect/yolo26n_chess/weights/best.pt data=datasets/data.yaml split=test imgsz=800
```

Then use the existing real screenshot/FEN evaluator:

```bash
python eval.py --model runs/detect/yolo26n_chess/weights/best.pt --input test_images --match-fen --imgsz 800 --output predictions/final
```

This checkout includes `test_images/fens.json` but ignores the actual test PNG/JPEG files; restore your screenshots before running that comparison. Track exact FEN accuracy, square error, class confusion, and latency alongside per-class AP. `--refined-search` also tests board-crop inference; it has its own accuracy/latency tradeoff.

To compare crop inference on the same final checkpoint, save its report separately:

```bash
python eval.py --model runs/detect/yolo26n_chess/weights/best.pt --input test_images --match-fen --imgsz 800 --refined-search --output predictions/final_refined
```

### Historical results and known failures

The earlier project notes reported the best-performing generator at commit [`8c16c3d`](https://github.com/Zai-Kun/2d-chess-pieces-detection/commit/8c16c3decf88d93a5f5497d98ec87b1cef65eeed), trained for 100 epochs with patience 30. Its position sampler randomized piece locations, which is one reason to retain appearance-focused recognition positions in the current generator. This is a plausible explanation, not a measured ablation.

The saved notes recorded these results on 35 screenshots:

| Inference | Exact FEN matches | Exact FEN accuracy | Mean square error | Square accuracy |
| --- | --- | --- | --- | --- |
| Full image | 27 / 35 | 77.14% | 2.23% | 97.77% |
| `--refined-search` | 26 / 35 | 74.29% | 2.10% | 97.90% |

Both reports recorded zero missing references and zero missing boards; the full-image run also recorded zero failed images. The logs called square accuracy the final error-weighted evaluation percentage; it is not an additional independent metric. Checkpoint identity and the complete training configuration were not saved in these notes, so these are historical user-reported results, not a verified benchmark of the current pipeline. Crop inference slightly improved square accuracy while reducing complete-board accuracy; choose the metric that matches your application.

Known confusions were **king/queen**, **bishop/pawn**, and occasionally **knight/rook**. Current generation puts all 12 classes into 20% of positions at varied squares, cycles through piece styles, and preserves more visual detail through an enforced size floor and milder corruption. These changes address likely training-data causes; confirming that the errors are fixed still requires retraining and evaluating real screenshots.

When testing a new recipe, keep the data split, seed, resolution, confidence threshold, and inference mode fixed, change one factor at a time, and save each run's `args.yaml` plus `fen_match_results.json`. Reproduce the 100/30 baseline before attributing gains to a longer schedule or new optimizer values. Use a separate validation collection for these comparisons so the final test set remains untouched.

```python
from ultralytics import YOLO
model = YOLO("runs/detect/yolo26n_chess/weights/best.pt")
model.export(format="onnx", imgsz=800, dynamic=False, nms=False)
```

YOLO26 supports an optional NMS-free inference head. Explicitly export it with `nms=False` and revalidate the exported artifact; compare it against the default inference path before deployment.

## Checks and sources

GitHub Actions runs regression tests and generates/validates small mixed and embedded datasets on Linux and Windows for pushes and pull requests. CI installs only the generation/test dependencies and does not download model weights or run GPU training. Model-quality checks remain in `.todo` and `notes`.

```bash
python -m unittest discover -s tests -v
```

Research sources checked November 12, 2026:

- [Official YOLO26 training recipe](https://docs.ultralytics.com/guides/yolo26-training-recipe): nano-specific optimizer/loss values and fine-tuning caveats.
- [Training arguments](https://docs.ultralytics.com/modes/train/): image size, batch, patience, reproducibility, and training time limits.
- [Hyperparameter tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning/): custom search spaces, saved hyperparameters, and limits of short searches.
- [YOLO26 model guide](https://docs.ultralytics.com/models/yolo26/): architecture and inference modes.
