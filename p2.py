"""COIT29225 Assessment 3 - Task 2.2: LSTM stock-price prediction.

Run from the project root with: python p2.py
Input CSV files: data/a3-p2-data/{CBA,NAB,QBE}.csv
"""

import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "a3-p2-data"
RESULTS_DIR = BASE_DIR / "results" / "lstm"
MODELS_DIR = BASE_DIR / "models" / "lstm"
STOCKS = ("CBA", "NAB", "QBE")
TRAIN_DAYS = 204
TEST_DAYS = 50
BATCH_SIZE = 16

CONFIGS = (
    {"name": "A", "time_step": 5, "neurons": 32, "epochs": 20},
    {"name": "B", "time_step": 5, "neurons": 50, "epochs": 30},
    {"name": "C", "time_step": 5, "neurons": 64, "epochs": 40},
    {"name": "D", "time_step": 10, "neurons": 32, "epochs": 20},
    {"name": "E", "time_step": 10, "neurons": 50, "epochs": 30},
    {"name": "F", "time_step": 10, "neurons": 64, "epochs": 40},
)


def load_stock(stock):
    path = DATA_DIR / f"{stock}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    for col in ("Date", "Close $"):
        if col not in df.columns:
            raise ValueError(f"{path.name} missing column: {col}")
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="raise")
    df["Close $"] = pd.to_numeric(
        df["Close $"].astype(str).str.replace(",", "", regex=False),
        errors="raise",
    )
    df = df.sort_values("Date").reset_index(drop=True)
    if len(df) != TRAIN_DAYS + TEST_DAYS:
        raise ValueError(f"{stock}: expected 254 rows, found {len(df)}")
    if df["Close $"].isna().any():
        raise ValueError(f"{stock}: missing closing price")
    return df


def make_sequences(values, time_step):
    xs, ys = [], []
    for i in range(time_step, len(values)):
        xs.append(values[i - time_step:i, 0])
        ys.append(values[i, 0])
    return np.asarray(xs, dtype=np.float32).reshape(-1, time_step, 1), np.asarray(ys, dtype=np.float32)


def build_model(time_step, neurons):
    model = Sequential([
        Input(shape=(time_step, 1)),
        LSTM(neurons),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def save_plots(prefix, history, dates, actual, predicted, rmse):
    plt.figure(figsize=(9, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{prefix}: Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (scaled prices)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{prefix}_loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.plot(dates, actual, marker="o", markersize=3, linewidth=1.5, label="Actual Close")
    plt.plot(dates, predicted, marker="x", markersize=3, linewidth=1.5, label="Predicted Close")
    plt.title(f"{prefix}: Actual vs Predicted (RMSE = {rmse:.4f} AUD)")
    plt.xlabel("Date")
    plt.ylabel("Closing Price (AUD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{prefix}_prediction.png", dpi=150)
    plt.close()


def run_experiment(stock, config, run_id):
    tf.keras.backend.clear_session()
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    df = load_stock(stock)
    time_step = config["time_step"]
    prefix = f"{stock}_{config['name']}_{run_id}"

    # Fit the scaler on the training period ONLY (avoid test-data leakage).
    train_prices = df.iloc[:TRAIN_DAYS][["Close $"]].to_numpy(dtype=float)
    all_prices = df[["Close $"]].to_numpy(dtype=float)
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_prices)
    x_train, y_train = make_sequences(scaled_train, time_step)

    model = build_model(time_step, config["neurons"])
    print(f"\n>>> {stock} | Config {config['name']} | steps={time_step} "
          f"| neurons={config['neurons']} | epochs={config['epochs']}", flush=True)
    history = model.fit(
        x_train, y_train,
        epochs=config["epochs"],
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        shuffle=False,
        verbose=2,
    )

    # One-step-ahead evaluation: each test-day input uses preceding ACTUAL prices.
    # This is not a 50-step recursive forecast.
    scaled_all = scaler.transform(all_prices)
    x_test = np.asarray([
        scaled_all[i - time_step:i, 0]
        for i in range(TRAIN_DAYS, TRAIN_DAYS + TEST_DAYS)
    ], dtype=np.float32).reshape(TEST_DAYS, time_step, 1)
    pred_scaled = model.predict(x_test, verbose=0)
    predicted = scaler.inverse_transform(pred_scaled).ravel()
    actual = all_prices[TRAIN_DAYS:TRAIN_DAYS + TEST_DAYS, 0]
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    print(f"{stock} Config {config['name']}: RMSE = {rmse:.4f} AUD", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / f"{prefix}.keras")
    with (RESULTS_DIR / f"{prefix}_architecture.txt").open("w", encoding="utf-8") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))

    dates = df.iloc[TRAIN_DAYS:TRAIN_DAYS + TEST_DAYS]["Date"]
    save_plots(prefix, history, dates, actual, predicted, rmse)
    pd.DataFrame({
        "epoch": np.arange(1, len(history.history["loss"]) + 1),
        "train_loss": history.history["loss"],
        "validation_loss": history.history["val_loss"],
    }).to_csv(RESULTS_DIR / f"{prefix}_history.csv", index=False)
    pd.DataFrame({
        "date": dates.dt.strftime("%Y-%m-%d").to_numpy(),
        "actual_close_aud": actual,
        "predicted_close_aud": predicted,
    }).to_csv(RESULTS_DIR / f"{prefix}_predictions.csv", index=False)

    result = {
        "stock": stock,
        "config": config["name"],
        "time_step": time_step,
        "neurons": config["neurons"],
        "epochs": config["epochs"],
        "batch_size": BATCH_SIZE,
        "train_days": TRAIN_DAYS,
        "test_days": TEST_DAYS,
        "evaluation": "one-step-ahead, previous actual prices supplied",
        "train_loss": float(history.history["loss"][-1]),
        "validation_loss": float(history.history["val_loss"][-1]),
        "test_rmse_aud": rmse,
    }
    (RESULTS_DIR / f"{prefix}_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def run_batch(stocks, configs):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / f"comparison_{run_id}.csv"
    fields = ["stock", "config", "time_step", "neurons", "epochs", "batch_size",
              "train_days", "test_days", "evaluation", "train_loss",
              "validation_loss", "test_rmse_aud"]
    results = []
    # Append after EACH experiment so an interrupted batch retains completed rows.
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for stock in stocks:
            for config in configs:
                result = run_experiment(stock, config, run_id)
                writer.writerow(result)
                f.flush()
                os.fsync(f.fileno())
                results.append(result)
    print(f"\nCompleted {len(results)} experiments. Comparison: {summary_path}")
    print("\nBest configuration for each company (lowest test RMSE):")
    for stock in stocks:
        stock_results = [r for r in results if r["stock"] == stock]
        best = min(stock_results, key=lambda r: r["test_rmse_aud"])
        print(f"{stock}: Config {best['config']} | RMSE {best['test_rmse_aud']:.4f} AUD")


def main():
    while True:
        print("\n===== Task 2.2: LSTM Stock Price Prediction =====")
        print("1. Time step = 5")
        print("2. Time step = 10")
        print("3. Quit")
        choice = input("Choose 1, 2 or 3: ").strip()
        if choice in ("1", "2"):
            time_step = 5 if choice == "1" else 10
            selected_configs = [c for c in CONFIGS if c["time_step"] == time_step]
            print(f"\nTime step {time_step}: 3 configurations x 3 stocks = 9 experiments")
            for config in selected_configs:
                print(f"  Config {config['name']}: neurons={config['neurons']}, epochs={config['epochs']}")
            confirm = input("Type YES to train these 9 models (or anything else to cancel): ").strip().upper()
            if confirm == "YES":
                run_batch(STOCKS, selected_configs)
            else:
                print("Cancelled; returning to menu.")
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Please choose 1, 2 or 3.")


if __name__ == "__main__":
    main()
