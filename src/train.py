"""Tasks 2.2 - 2.5 and 3.3 - 3.5: train one model and log the whole run to MLflow.

One command per experiment. Change exactly one flag between runs and you get the
six comparable runs that sub-task 3.5 asks for.

Examples:
    python -m src.train --model baseline --run-name run1-baseline
    python -m src.train --model transfer --backbone MobileNetV2 --run-name run2-mnv2-frozen
    python -m src.train --model transfer --finetune --unfreeze 20 --run-name run3-mnv2-ft20
    python -m src.train --model transfer --lr 1e-4 --run-name run4-mnv2-lr1e-4
    python -m src.train --model transfer --no-augment --run-name run5-no-aug
    python -m src.train --model transfer --img-size 224 --run-name run6-img224
    python -m src.train --model transfer --backbone EfficientNetB0 --run-name run7-effb0
    python -m src.train --model transfer --autolog --run-name run8-autolog

--autolog swaps our hand-written logging for mlflow.tensorflow.autolog() so the
report can compare what each approach captures.
"""

import argparse
import json
import os
import platform
import subprocess
import sys

import mlflow
import numpy as np
import tensorflow as tf

from src import config, evaluate, models
from src.data_pipeline import make_dataset, save_augmented_batch_figure


def git_commit():
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return output.decode("utf-8").strip()
    except Exception:
        return "unknown"


def dvc_data_version():
    """The md5 recorded in data/raw.dvc identifies the exact dataset snapshot."""
    for candidate in ["data/raw.dvc", "data.dvc"]:
        if os.path.exists(candidate):
            handle = open(candidate, "r", encoding="utf-8")
            text = handle.read()
            handle.close()
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("-"):
                    stripped = stripped[1:].strip()
                if stripped.startswith("md5:"):
                    return stripped.split("md5:")[1].strip()
    return "unknown"


class MlflowEpochLogger(tf.keras.callbacks.Callback):
    """Manual per-epoch logging (the 'do it once manually' half of 3.3)."""

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        for key in logs:
            try:
                mlflow.log_metric(key, float(logs[key]), step=epoch)
            except Exception:
                pass


def build_callbacks(checkpoint_path, use_manual_logging):
    callbacks = []

    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        )
    )
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        )
    )
    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        )
    )
    if use_manual_logging:
        callbacks.append(MlflowEpochLogger())

    return callbacks


def parse_args():
    parser = argparse.ArgumentParser(description="Train a VisionGuard model.")
    parser.add_argument("--model", choices=["baseline", "transfer"], default="transfer")
    parser.add_argument(
        "--backbone",
        choices=["MobileNetV2", "EfficientNetB0", "ResNet50"],
        default="MobileNetV2",
    )
    parser.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--finetune-epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--unfreeze", type=int, default=20)
    parser.add_argument("--finetune", action="store_true", help="run stage 2 as well")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--autolog", action="store_true", help="use MLflow autologging")
    parser.add_argument("--run-name", default="run")
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    config.ensure_dirs()
    config.set_seeds(args.seed)

    if config.MLFLOW_TRACKING_URI != "":
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    use_augmentation = not args.no_augment

    # ---- build the model -------------------------------------------------
    preprocess = None
    backbone = None
    if args.model == "baseline":
        model = models.build_baseline_cnn(
            img_size=args.img_size, dropout=args.dropout, learning_rate=args.lr
        )
    else:
        model, preprocess, backbone = models.build_transfer_model(
            backbone_name=args.backbone,
            img_size=args.img_size,
            dropout=args.dropout,
            learning_rate=args.lr,
        )

    # ---- datasets --------------------------------------------------------
    train_dataset = make_dataset(
        "train",
        img_size=args.img_size,
        batch_size=args.batch_size,
        augment=use_augmentation,
        shuffle=True,
        preprocess_fn=preprocess,
    )
    validation_dataset = make_dataset(
        "val",
        img_size=args.img_size,
        batch_size=args.batch_size,
        preprocess_fn=preprocess,
    )
    test_dataset = make_dataset(
        "test",
        img_size=args.img_size,
        batch_size=args.batch_size,
        preprocess_fn=preprocess,
    )

    checkpoint_path = os.path.join(config.MODEL_DIR, args.run_name + ".keras")

    with mlflow.start_run(run_name=args.run_name) as active_run:
        run_id = active_run.info.run_id
        print("MLflow run id: " + run_id)

        if args.autolog:
            mlflow.tensorflow.autolog(log_models=True)

        # ---- parameters and tags ----------------------------------------
        parameters = {
            "model": args.model,
            "backbone": args.backbone if args.model == "transfer" else "none",
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "finetune": args.finetune,
            "finetune_epochs": args.finetune_epochs if args.finetune else 0,
            "learning_rate": args.lr,
            "finetune_lr": args.finetune_lr if args.finetune else 0,
            "dropout": args.dropout,
            "unfrozen_layers": args.unfreeze if args.finetune else 0,
            "augmentation": use_augmentation,
            "seed": args.seed,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "logging_mode": "autolog" if args.autolog else "manual",
        }
        mlflow.log_params(parameters)

        mlflow.set_tag("git_commit", git_commit())
        mlflow.set_tag("dvc_data_version", dvc_data_version())
        mlflow.set_tag("seed", str(args.seed))
        mlflow.set_tag("python", platform.python_version())
        mlflow.set_tag("tensorflow", tf.__version__)
        mlflow.set_tag("platform", platform.platform())

        # ---- model summary as an artifact -------------------------------
        summary_lines = []
        model.summary(print_fn=lambda line: summary_lines.append(line))
        summary_path = os.path.join(config.ARTIFACT_DIR, args.run_name + "_summary.txt")
        handle = open(summary_path, "w", encoding="utf-8")
        handle.write("\n".join(summary_lines))
        handle.close()
        mlflow.log_artifact(summary_path, artifact_path="model_info")
        print("\n".join(summary_lines))

        # ---- stage 1 -----------------------------------------------------
        callbacks = build_callbacks(checkpoint_path, use_manual_logging=not args.autolog)
        history_one = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=2,
        )
        combined_history = {}
        for key in history_one.history:
            combined_history[key] = list(history_one.history[key])

        # ---- stage 2, fine-tuning ---------------------------------------
        if args.finetune and args.model == "transfer":
            models.unfreeze_last_layers(
                model, backbone, how_many=args.unfreeze, learning_rate=args.finetune_lr
            )
            trainable = models.trainable_layer_names(model)
            trainable_path = os.path.join(
                config.ARTIFACT_DIR, args.run_name + "_trainable_layers.json"
            )
            handle = open(trainable_path, "w", encoding="utf-8")
            json.dump(trainable, handle, indent=2)
            handle.close()
            mlflow.log_artifact(trainable_path, artifact_path="model_info")
            mlflow.log_param("trainable_layer_count", len(trainable))

            start_epoch = len(combined_history["loss"])
            callbacks = build_callbacks(
                checkpoint_path, use_manual_logging=not args.autolog
            )
            history_two = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=start_epoch + args.finetune_epochs,
                initial_epoch=start_epoch,
                callbacks=callbacks,
                verbose=2,
            )
            for key in history_two.history:
                if key in combined_history:
                    combined_history[key] = combined_history[key] + list(
                        history_two.history[key]
                    )
                else:
                    combined_history[key] = list(history_two.history[key])

        # ---- best epoch --------------------------------------------------
        validation_losses = combined_history["val_loss"]
        best_epoch = int(np.argmin(validation_losses)) + 1
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("best_val_loss", float(np.min(validation_losses)))
        print("best epoch (lowest val_loss): " + str(best_epoch))

        curves_path = evaluate.plot_history(
            combined_history, best_epoch, args.run_name
        )
        mlflow.log_artifact(curves_path, artifact_path="figures")

        history_json = os.path.join(config.ARTIFACT_DIR, args.run_name + "_history.json")
        handle = open(history_json, "w", encoding="utf-8")
        json.dump(combined_history, handle, indent=2)
        handle.close()
        mlflow.log_artifact(history_json, artifact_path="history")

        # ---- threshold from validation, metrics on test ------------------
        validation_probabilities, validation_labels = evaluate.collect_predictions(
            model, validation_dataset
        )
        threshold, threshold_table = evaluate.choose_threshold(
            validation_probabilities, validation_labels
        )
        mlflow.log_param("decision_threshold", threshold)
        threshold_path = os.path.join(
            config.ARTIFACT_DIR, args.run_name + "_threshold_sweep.csv"
        )
        threshold_table.to_csv(threshold_path, index=False)
        mlflow.log_artifact(threshold_path, artifact_path="evaluation")
        print("chosen decision threshold (from validation): " + str(threshold))

        test_probabilities, test_labels = evaluate.collect_predictions(model, test_dataset)
        metrics, figure_paths = evaluate.full_evaluation(
            test_probabilities, test_labels, threshold, args.run_name
        )
        for key in metrics:
            mlflow.log_metric("test_" + key, float(metrics[key]))
        for path in figure_paths:
            mlflow.log_artifact(path, artifact_path="figures")

        print("")
        print("=== test metrics ===")
        for key in metrics:
            print("  " + key.ljust(12) + str(round(float(metrics[key]), 4)))

        # ---- error analysis (2.6) ---------------------------------------
        error_figure = evaluate.error_grid(
            test_probabilities, test_labels, threshold, args.run_name, args.img_size
        )
        if error_figure is not None:
            mlflow.log_artifact(error_figure, artifact_path="figures")

        # ---- artifacts for 3.4 ------------------------------------------
        for split_name in ["train", "val", "test"]:
            split_path = os.path.join(config.SPLIT_DIR, split_name + ".csv")
            if os.path.exists(split_path):
                mlflow.log_artifact(split_path, artifact_path="splits")

        freeze_path = os.path.join(config.ARTIFACT_DIR, "pip_freeze.txt")
        handle = open(freeze_path, "w", encoding="utf-8")
        try:
            frozen = subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL
            )
            handle.write(frozen.decode("utf-8"))
        except Exception:
            handle.write("pip freeze failed on this machine\n")
        handle.close()
        mlflow.log_artifact(freeze_path, artifact_path="environment")

        model.save(checkpoint_path)
        mlflow.log_artifact(checkpoint_path, artifact_path="keras_model")

        if not args.autolog:
            try:
                mlflow.tensorflow.log_model(model, artifact_path="model")
            except Exception as error:
                print("log_model skipped: " + str(error))

        print("")
        print("saved model: " + checkpoint_path)
        print("run finished. open it on DagsHub under experiment " + config.EXPERIMENT_NAME)


if __name__ == "__main__":
    main()
