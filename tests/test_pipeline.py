import ast
import contextlib
import io
import json
from pathlib import Path
import random
import re
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import yaml

import generate_datasets as generator
from random_fen_gen import flip_fen, generate_fen
import train_yolo26 as training
from verify_dataset import validate_label_file, verify_dataset


class GenerationTests(unittest.TestCase):
    def test_recognition_positions_cover_every_class_and_both_perspectives(self):
        kings = set()
        for seed in range(100):
            random.seed(seed)
            fen = generate_fen("recognition", allow_flip=False)
            squares = []
            for rank in fen.split("/"):
                row = "".join("." * int(c) if c.isdigit() else c for c in rank)
                self.assertEqual(len(row), 8)
                squares.extend(row)
            self.assertEqual(sorted(c for c in squares if c != "."), sorted(generator.FEN_TO_PIECE))
            self.assertNotIn("P", squares[:8] + squares[-8:])
            self.assertNotIn("p", squares[:8] + squares[-8:])
            self.assertEqual(flip_fen(flip_fen(fen)), fen)
            kings.add(squares.index("K"))
        self.assertGreater(len(kings), 40)

    def test_shared_styles_keep_backgrounds_disjoint(self):
        for shared in (False, True):
            parts = generator.split_assets(shared_styles=shared)
            for key in ("pieces", "boards", "bgs"):
                pools = [set(parts[s][key]) for s in ("train", "val", "test")]
                self.assertTrue(all(pools))
                if shared and key != "bgs":
                    self.assertEqual(pools[0], pools[1])
                    self.assertEqual(pools[0], pools[2])
                else:
                    self.assertFalse(pools[0] & pools[1] or pools[0] & pools[2] or pools[1] & pools[2])

    def test_insufficient_backgrounds_fail_instead_of_leaking(self):
        original = generator.os.listdir

        def listing(path):
            return ["one.png", "two.png"] if path == generator.BACKGROUND_NOISE_DIR else original(path)

        with patch.object(generator.os, "listdir", side_effect=listing):
            with self.assertRaisesRegex(ValueError, "three backgrounds"):
                generator.split_assets()

    def test_worker_reseeds_numpy_and_never_relaxes_size_floor(self):
        task = (1, "train", "style", "board", "bg", 800, 800, True, "images", "labels", ["style"], ["board"], "42:train:1", 18, 2, None)
        draws = []

        def sample(*args, **kwargs):
            draws.append((random.random(), np.random.random()))
            return True

        with patch.object(generator, "_generate_single_sample_once", side_effect=sample):
            generator.generate_single_sample(task)
            np.random.seed(999)
            generator.generate_single_sample(task)
        self.assertEqual(draws[0], draws[1])
        with patch.object(generator, "_generate_single_sample_once", return_value=False) as render:
            with self.assertRaisesRegex(RuntimeError, "pieces >= 18px"):
                generator.generate_single_sample(task)
            self.assertEqual(render.call_count, 2)
            for call in render.call_args_list:
                self.assertEqual(call.kwargs["min_piece_px"], 18)
            self.assertEqual(render.call_args.kwargs["composition"], "full")
        with patch.object(generator, "_generate_single_sample_once", return_value=False) as render:
            with self.assertRaises(RuntimeError):
                generator.generate_single_sample((*task[:-1], "embedded"))
            self.assertTrue(all(call.kwargs["composition"] == "embedded" for call in render.call_args_list))

    def test_embedded_scene_retains_background_and_scales_piece_labels(self):
        # Isolate compositing so pixel evidence can be compared with exact labels.
        with tempfile.TemporaryDirectory() as directory:
            background = Path(directory) / "background.png"
            Image.new("RGB", (800, 800), "blue").save(background)
            board = Image.new("RGB", (800, 800), "white")
            board.paste(Image.new("RGB", (80, 80), "red"), (100, 200))
            with patch.object(generator, "get_embedded_board_size", return_value=400), patch.object(generator.random, "randint", side_effect=[100, 150]), patch.object(generator, "draw_random_ui_clutter", side_effect=lambda canvas, *a, **kw: canvas):
                canvas, pieces, board_box = generator.compose_embedded_board(
                    board, [(4, 100, 200, 180, 280)], base_board_size=800,
                    bg_file=str(background.resolve()), canvas_size=800,
                )
            self.assertEqual(canvas.getpixel((0, 0)), (0, 0, 255))
            self.assertEqual(canvas.getpixel((170, 270)), (255, 0, 0))
            self.assertEqual(pieces, [(4, 150, 250, 190, 290)])
            self.assertEqual(board_box, (12, 100, 150, 500, 550))

    def test_transparent_or_missing_sprites_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bad").mkdir()
            Image.new("RGBA", (16, 16)).save(root / "bad" / "bP.png")
            with patch.object(generator, "PIECES_DIR", directory), patch.object(generator, "_PIECE_CACHE", {}):
                with self.assertRaisesRegex(ValueError, "no visible pixels"):
                    generator.load_piece_sprites("bad", 80)
                with self.assertRaisesRegex(ValueError, "Failed loading piece"):
                    generator.load_piece_sprites("missing", 80)

    def test_generation_writes_config_and_rejects_stale_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dataset"
            args = ["generate_datasets.py", "--out-dir", str(output), "--train-count", "1", "--val-count", "0", "--test-count", "0", "--no-board-labels"]
            # Exercise coordinator/config without paying for another rendering pass.
            def generate(**kwargs):
                Path(kwargs["images_dir"]).mkdir(parents=True)
                Path(kwargs["labels_dir"]).mkdir(parents=True)

            with patch("sys.argv", args), patch.object(generator, "generate_split_dataset", side_effect=generate), contextlib.redirect_stdout(io.StringIO()):
                generator.main()
                data = yaml.safe_load((output / "data.yaml").read_text())
                self.assertEqual(data["path"], str(output.resolve()))
                self.assertEqual(len(data["names"]), 12)
                self.assertNotIn("val", data)
                self.assertTrue(json.loads((output / "manifest.json").read_text())["assets"]["train"]["pieces"])
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    generator.main()


class QualityTests(unittest.TestCase):
    def test_nonfinite_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "sample.txt"
            label.write_text("0 0.5 0.5 nan 0.2\n0 0.5 0.5 0.2 inf\n")
            boxes, errors, _ = validate_label_file(label)
            self.assertFalse(boxes)
            self.assertEqual(len(errors), 2)

    def test_val_test_duplicate_is_leakage_even_with_different_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, label in (("val", "0"), ("test", "1")):
                (root / "images" / split).mkdir(parents=True)
                (root / "labels" / split).mkdir(parents=True)
                Image.new("RGB", (80, 80), "red").save(root / "images" / split / "1.png")
                (root / "labels" / split / "1.txt").write_text(f"{label} 0.5 0.5 0.2 0.2\n")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(verify_dataset(directory, splits=("val", "test"), preview=False), 1)


class TrainingTests(unittest.TestCase):
    def test_all_direct_imports_have_declared_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        imports = set()
        sources = [p.read_text(encoding="utf-8") for p in root.glob("*.py")]
        sources += [p.read_text(encoding="utf-8") for p in (root / "assets").glob("*.py")]
        notebook = json.loads((root / "2d_chess_detection.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                # IPython shell/magic lines are not Python syntax.
                sources.append("\n".join(re.sub(r"^(\s*)[!%].*", r"\1pass", line) for line in "".join(cell["source"]).splitlines()))
        for source in sources:
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Import):
                    imports.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split(".")[0])
        local = {p.stem for p in root.glob("*.py")}
        third_party = imports - sys.stdlib_module_names - local
        packages = {"PIL": "Pillow", "cv2": "opencv-python", "yaml": "PyYAML"}
        declared = {re.split(r"[<>=!~\s]", line)[0].lower() for line in (root / "requirements.txt").read_text().splitlines() if line and not line.startswith("#")}
        self.assertFalse({packages.get(module, module).lower() for module in third_party} - declared)

    def test_tuned_values_override_baseline_and_cli_overrides_tuning(self):
        with tempfile.TemporaryDirectory() as directory:
            hyp = Path(directory) / "best_hyperparameters.yaml"
            hyp.write_text("lr0: 0.002\ncls: 0.8\n")
            settings = training.load_settings(hyp, lr0=0.003, epochs=120, batch=None)
            self.assertEqual(settings["lr0"], 0.003)
            self.assertEqual(settings["cls"], 0.8)
            self.assertEqual(settings["epochs"], 120)
            self.assertEqual(settings["fliplr"], 0)
            hyp.write_text("data: secret_test_set.yaml\n")
            with self.assertRaises(ValueError):
                training.load_settings(hyp)

    def test_training_and_tuning_dispatch_with_guarded_augmentations(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data.yaml"
            data.write_text("train: images/train\nval: images/val\n")
            fake = MagicMock()
            with patch.dict("sys.modules", {"ultralytics": fake}), contextlib.redirect_stdout(io.StringIO()):
                training.main(["--data", str(data), "--epochs", "1"])
                settings = fake.YOLO.return_value.train.call_args.kwargs
                self.assertEqual(training.load_settings()["epochs"], 100)
                self.assertEqual(training.load_settings()["patience"], 30)
                self.assertEqual(settings["data"], str(data.resolve()))
                self.assertNotIn("split", settings)
                training.main(["--data", str(data), "--tune", "2"])
                settings = fake.YOLO.return_value.tune.call_args.kwargs
                self.assertEqual(settings["iterations"], 2)
                for key in ("fliplr", "flipud", "mixup", "degrees"):
                    self.assertEqual(settings[key], 0)
                    self.assertNotIn(key, settings["space"])


if __name__ == "__main__":
    unittest.main()
