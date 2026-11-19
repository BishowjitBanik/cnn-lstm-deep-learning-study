"""COIT29225 Assessment 3, Task 2.1: UTKFace age regression and gender classification.

Run in VS Code with Python 3.13. Menu options compare a baseline CNN with an
augmented CNN. Each option evaluates multiple parameter configurations.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "UTKFace"
RESULTS_DIR = ROOT / "results" / "cnn"
MODELS_DIR = ROOT / "models"
IMAGE_SIZE = (128, 128)
SEED = 42

# Three configurations tune architectural and algorithmic parameters.
# Five epochs per configuration; consider more epochs if training has not converged.
CONFIGS = [
    {"name": "A", "filters": (16, 32, 64), "dense": 64, "dropout": 0.25,
     "learning_rate": 0.001, "batch_size": 32, "epochs": 5},

     {"name": "B", "filters": (32, 64, 128), "dense": 128, "dropout": 0.35,
      "learning_rate": 0.0005, "batch_size": 32, "epochs": 5},

    {"name": "C", "filters": (32, 64, 128), "dense": 128, "dropout": 0.30,
     "learning_rate": 0.001, "batch_size": 32, "epochs": 5},
]

def read_metadata():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Dataset folder missing: {DATA_DIR}")
    paths, ages, genders = [], [], []
    for file in sorted(DATA_DIR.glob("*.jpg")):
        try:
            age, gender = (int(x) for x in file.name.split("_")[:2])
            if not 0 <= age <= 116 or gender not in (0, 1):
                continue
            paths.append(str(file))
            ages.append(age)
            genders.append(gender)
        except (ValueError, IndexError):
            continue
    if not paths:
        raise ValueError("No valid UTKFace images found")
    return np.array(paths), np.array(ages, dtype=np.float32), np.array(genders, dtype=np.float32)


def get_indices(n, genders):
    saved = ROOT / "results" / "utkface_split.npz"
    if saved.exists():
        with np.load(saved) as split:
            tr, va, te = (split[k].astype(int) for k in ("train_idx", "val_idx", "test_idx"))
        combined = np.concatenate([tr, va, te])
        if len(combined) == n and np.array_equal(np.sort(combined), np.arange(n)):
            return tr, va, te
        raise ValueError("Saved split does not match the current dataset")
    tr, remaining = train_test_split(np.arange(n), test_size=0.30, random_state=SEED,
                                     stratify=genders.astype(int))
    va, te = train_test_split(remaining, test_size=0.50, random_state=SEED,
                              stratify=genders[remaining].astype(int))
    saved.parent.mkdir(parents=True, exist_ok=True)
    np.savez(saved, train_idx=tr, val_idx=va, test_idx=te)
    return tr, va, te


def decode(path, age, gender):
    raw = tf.io.read_file(path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, {"age": tf.reshape(age, (1,)), "gender": tf.reshape(gender, (1,))}


def dataset(paths, ages, genders, indices, batch_size, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths[indices], ages[indices], genders[indices]))
    if training:
        ds = ds.shuffle(min(len(indices), 5000), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(config, augmented):
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="face_image")
    x = inputs
    if augmented:
        x = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.04),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomContrast(0.10),
        ], name="data_augmentation")(x)
    for i, filters in enumerate(config["filters"], start=1):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu",
                                    name=f"conv_{i}")(x)
        x = tf.keras.layers.MaxPooling2D(2, name=f"pool_{i}")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    age_branch = tf.keras.layers.Dense(config["dense"], activation="relu", name="age_hidden")(x)
    age_branch = tf.keras.layers.Dropout(config["dropout"], name="age_dropout")(age_branch)
    age_output = tf.keras.layers.Dense(1, name="age")(age_branch)
    gender_branch = tf.keras.layers.Dense(config["dense"], activation="relu", name="gender_hidden")(x)
    gender_branch = tf.keras.layers.Dropout(config["dropout"], name="gender_dropout")(gender_branch)
    gender_output = tf.keras.layers.Dense(1, activation="sigmoid", name="gender")(gender_branch)
    model = tf.keras.Model(inputs, {"age": age_output, "gender": gender_output}, name="UTKFace_CNN")
    model.compile(optimizer=tf.keras.optimizers.Adam(config["learning_rate"]),
                  loss={"age": "mae", "gender": "binary_crossentropy"},
                  # MAE in years is much larger than BCE, so balance gradients.
                  loss_weights={"age": 0.05, "gender": 1.0},
                  metrics={"age": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
                           "gender": [tf.keras.metrics.BinaryAccuracy(name="accuracy")]})
    return model


def save_architecture(model, path):
    try:
        tf.keras.utils.plot_model(model, to_file=str(path), show_shapes=True,
                                  show_layer_names=True, dpi=85, expand_nested=True)
        print("Architecture diagram:", path)
    except Exception as exc:
        print("plot_model unavailable (Graphviz may be missing):", exc)
    try:
        import visualkeras
        visualkeras.layered_view(model, legend=True).save(str(path.with_name(path.stem + "_visualkeras.png")))
    except Exception as exc:
        print("VisualKeras view unavailable for this multi-output model:", exc)
    # A readable architecture diagram even without Graphviz / VisualKeras.
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axis("off")
    blocks = ["Input: 128 x 128 x 3"]
    if any(layer.name == "data_augmentation" for layer in model.layers):
        blocks.append("Training-only data augmentation")
    blocks += ["Conv2D + MaxPool (x3)", "Flatten", "Separate heads"]
    ypos = [0.93, 0.79, 0.65, 0.51, 0.37] if len(blocks) == 5 else [0.93, 0.76, 0.59, 0.42]
    for title, y in zip(blocks, ypos):
        ax.text(0.5, y, title, ha="center", va="center", fontsize=13,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black"))
    ax.text(0.24, 0.16, "Age head: Dense -> Dropout -> Linear\nRegression (years)",
            ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="black"))
    ax.text(0.76, 0.16, "Gender head: Dense -> Dropout -> Sigmoid\nBinary classification",
            ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="black"))
    ax.set_title("CNN architecture: shared feature extractor and two prediction heads")
    fallback = path.with_name(path.stem + "_schematic.png")
    fig.savefig(fallback, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Architecture schematic:", fallback)
    # Always save a model summary as a fallback reference.
    with open(path.with_suffix(".txt"), "w", encoding="utf-8") as out:
        model.summary(print_fn=lambda line: out.write(line + "\n"))


def plot_history(history, destination):
    h = history.history
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for ax, key, title in [
        (axs[0, 0], "age_mae", "Age MAE (years)"),
        (axs[0, 1], "gender_accuracy", "Gender accuracy"),
        (axs[1, 0], "age_loss", "Age loss (MAE)"),
        (axs[1, 1], "gender_loss", "Gender loss (binary cross-entropy)"),
    ]:
        if key in h:
            epochs = np.arange(1, len(h[key]) + 1)
            ax.plot(epochs, h[key], marker="o", label="Training")
            if "val_" + key in h:
                ax.plot(epochs, h["val_" + key], marker="o", label="Validation")
        ax.set(title=title, xlabel="Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def sample_predictions(model, paths, ages, genders, indices, destination):
    rng = np.random.default_rng(SEED)
    selected = rng.choice(indices, size=min(10, len(indices)), replace=False)
    images = []
    for idx in selected:
        with Image.open(paths[idx]) as image:
            images.append(np.asarray(image.convert("RGB").resize(IMAGE_SIZE), dtype=np.float32) / 255.0)
    preds = model.predict(np.stack(images), verbose=0)
    age_pred = preds["age"].reshape(-1)
    gender_pred = preds["gender"].reshape(-1)
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for j, ax in enumerate(axes.flat):
        if j >= len(selected):
            ax.axis("off")
            continue
        idx = selected[j]
        ax.imshow(images[j])
        true_gender = "Male" if genders[idx] == 0 else "Female"
        predicted_gender = "Female" if gender_pred[j] >= 0.5 else "Male"
        ax.set_title(f"True: {int(ages[idx])}y, {true_gender}\n"
                     f"Pred: {age_pred[j]:.1f}y, {predicted_gender}", fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def evaluate(model, ds):
    result = model.evaluate(ds, verbose=1, return_dict=True)
    return {key: float(value) for key, value in result.items()}


def run_experiments(augmented):
    scenario = "augmented" if augmented else "original"
    print(f"\nRUNNING {scenario.upper()} CNN EXPERIMENTS")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    paths, ages, genders = read_metadata()
    train_idx, val_idx, test_idx = get_indices(len(paths), genders)
    print(f"Images: {len(paths)} | Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    rows = []
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for config in CONFIGS:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        tag = f"assessment3_{scenario}_{config['name']}_{run_id}"
        print(f"\n{'=' * 60}\nExperiment {tag}: {config}\n{'=' * 60}")
        train_ds = dataset(paths, ages, genders, train_idx, config["batch_size"], training=True)
        train_eval_ds = dataset(paths, ages, genders, train_idx, config["batch_size"])
        val_ds = dataset(paths, ages, genders, val_idx, config["batch_size"])
        test_ds = dataset(paths, ages, genders, test_idx, config["batch_size"])
        model = build_model(config, augmented)
        model.summary()
        save_architecture(model, RESULTS_DIR / f"architecture_{tag}.png")
        history = model.fit(train_ds, validation_data=val_ds, epochs=config["epochs"], verbose=1)
        train_metrics = evaluate(model, train_eval_ds)
        val_metrics = evaluate(model, val_ds)
        test_metrics = evaluate(model, test_ds)
        model.save(MODELS_DIR / f"cnn_{tag}.keras")
        plot_history(history, RESULTS_DIR / f"curves_{tag}.png")
        # Persist every real epoch metric for reproducible report plots.
        keys = list(history.history)
        with open(RESULTS_DIR / f"history_{tag}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch"] + keys)
            for i in range(len(history.history["loss"])):
                writer.writerow([i + 1] + [float(history.history[k][i]) for k in keys])
        sample_predictions(model, paths, ages, genders, test_idx,
                           RESULTS_DIR / f"predictions_{tag}.png")
        with open(RESULTS_DIR / f"metrics_{tag}.json", "w", encoding="utf-8") as f:
            json.dump({"config": config, "train": train_metrics,
                       "validation": val_metrics, "test": test_metrics}, f, indent=2)
        row = {"scenario": scenario, **config}
        for split, metrics in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
            for key, value in metrics.items():
                row[f"{split}_{key}"] = value
        rows.append(row)
        print(f"DONE {tag}: validation age MAE={val_metrics.get('age_mae', float('nan')):.2f} years; "
              f"gender accuracy={val_metrics.get('gender_accuracy', float('nan')):.3f}")
    csv_path = RESULTS_DIR / f"comparison_{scenario}_{run_id}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nAll {scenario} experiments completed. Comparison: {csv_path}")
    print("Open results/cnn to view graphs, predictions and metrics.")
    print("Choose the best configuration using validation metrics; reserve test results for final reporting.")


def main():
    while True:
        print("\n========== CNN MAIN MENU ==========")
        print("1. Run CNN")
        print("2. Run CNN with data augmentation")
        print("3. Quit")
        choice = input("Enter your choice (1-3): ").strip()
        if choice == "3":
            print("Goodbye!")
            break
        if choice in ("1", "2"):
            try:
                run_experiments(augmented=(choice == "2"))
            except Exception as exc:
                print(f"ERROR: {type(exc).__name__}: {exc}")
                print("Fix the error and try the option again.")
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
