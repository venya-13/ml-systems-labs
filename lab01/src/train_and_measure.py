"""Lab 1: train two baselines on Breast Cancer Wisconsin and measure system cost."""
import os
import random
import statistics
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from memory_profiler import memory_usage
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

try:
    import torch
except ImportError:
    torch = None

# ---------- reproducibility ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if torch is not None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

TRAIN_REPEATS = 5
INFER_REPEATS = 100

# Upper limits per target (memory MB, latency ms, size KB)
BUDGETS = {
    "Cloud":  {"mem_mb": float("inf"), "lat_ms": 100, "size_kb": 500 * 1024},
    "Edge":   {"mem_mb": 1024,         "lat_ms": 50,  "size_kb": 50 * 1024},
    "Mobile": {"mem_mb": 256,          "lat_ms": 20,  "size_kb": 10 * 1024},
    "TinyML": {"mem_mb": 256 / 1024,   "lat_ms": 10,  "size_kb": 100},
}


def make_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=SEED),
    }


def peak_mb(func):
    """Peak process RSS (MiB) while func runs."""
    res = memory_usage((func, (), {}), interval=0.001, max_usage=True)
    return float(res[0] if isinstance(res, (list, tuple)) else res)


def main():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )
    sample = X_test[:1]
    proc = psutil.Process(os.getpid())
    baseline_mb = proc.memory_info().rss / 2**20

    acc_rows, sys_rows, fit_rows = [], [], []

    for name in make_models():
        # --- training time: 1 warm-up + 5 timed runs ---
        make_models()[name].fit(X_train, y_train)  # warm-up
        train_times = []
        for _ in range(TRAIN_REPEATS):
            m = make_models()[name]
            t0 = time.perf_counter()
            m.fit(X_train, y_train)
            train_times.append(time.perf_counter() - t0)
        model = m

        # --- accuracy ---
        acc = accuracy_score(y_test, model.predict(X_test))
        acc_rows.append({"model": name, "test_accuracy": round(acc, 4)})

        # --- single-sample inference: 1 warm-up + 100 timed runs ---
        model.predict(sample)
        lat = []
        for _ in range(INFER_REPEATS):
            t0 = time.perf_counter()
            model.predict(sample)
            lat.append(time.perf_counter() - t0)
        lat_ms = statistics.median(lat) * 1000

        # --- model size ---
        path = RESULTS / f"model_{name}.joblib"
        joblib.dump(model, path)
        size_b = path.stat().st_size

        # --- peak memory ---
        mem_train = peak_mb(lambda: make_models()[name].fit(X_train, y_train))
        mem_infer = peak_mb(lambda: model.predict(sample))

        sys_rows.append({
            "model": name,
            "train_time_median_s": round(statistics.median(train_times), 4),
            "infer_latency_median_ms": round(lat_ms, 4),
            "model_size_bytes": size_b,
            "model_size_kb": round(size_b / 1024, 2),
            "peak_mem_train_mb": round(mem_train, 1),
            "peak_mem_infer_mb": round(mem_infer, 1),
            "process_baseline_mb": round(baseline_mb, 1),
        })

        for target, b in BUDGETS.items():
            ok_mem = mem_infer <= b["mem_mb"]
            ok_lat = lat_ms <= b["lat_ms"]
            ok_size = size_b / 1024 <= b["size_kb"]
            fit_rows.append({
                "model": name, "target": target,
                "memory_ok": ok_mem, "latency_ok": ok_lat, "size_ok": ok_size,
                "fits": ok_mem and ok_lat and ok_size,
            })

    # File name required by the manual
    joblib.dump(make_models()["LogisticRegression"].fit(X_train, y_train),
                RESULTS / "model.joblib")

    acc_df = pd.DataFrame(acc_rows)
    sys_df = pd.DataFrame(sys_rows)
    fit_df = pd.DataFrame(fit_rows)
    acc_df.to_csv(RESULTS / "baseline_accuracy.csv", index=False)
    sys_df.to_csv(RESULTS / "system_metrics.csv", index=False)
    fit_df.to_csv(RESULTS / "deployment_fit.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Accuracy ===\n", acc_df.to_string(index=False))
    print("\n=== System metrics ===\n", sys_df.to_string(index=False))
    print("\n=== Deployment fit ===\n", fit_df.to_string(index=False))
    print(f"\nSaved CSVs and models to {RESULTS}")


if __name__ == "__main__":
    main()