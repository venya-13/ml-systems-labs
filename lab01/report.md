# Lab 1: Environment and first system measurements

## Goal

Set up a reproducible Python environment, train two baseline classifiers on the Breast Cancer Wisconsin dataset and measure their system cost (training time, inference latency, model size, peak memory). Use the measurements to decide which deployment targets (Cloud, Edge, Mobile, TinyML) each model fits.

## Method

- **Environment:** Python 3.11.8 in an isolated `.venv`, packages pinned in `requirements.txt`. Versions are printed by `src/print_versions.py` and saved to `results/versions.txt`.
  - Note: `pyarrow` is pinned to **15.0.2** instead of 16.1.0, because `mlflow==2.14.1` requires `pyarrow<16` and the original pins cannot be installed together.
- **Hardware:** ASUS ROG Strix 17 GL703GE, Intel Core i7-8750H @ 2.20 GHz (6 cores / 12 threads), 16 GB RAM, Windows 10. All measurements on CPU (GPU NVIDIA GTX 1050 Ti not used).
- **Data:** `load_breast_cancer(return_X_y=True)`, 569 samples, 30 features. Stratified 70/30 split, `random_state=42`. No feature scaling.
- **Models:** `LogisticRegression(max_iter=1000, random_state=42)` and `RandomForestClassifier(n_estimators=100, random_state=42)`.
- **Seeds:** `random`, `numpy` and `torch` were all seeded with 42.
- **Measurements:**
  - **Training time:** 1 warm-up, then 5 timed runs, median reported.
  - **Single-sample inference:** 1 warm-up, then 100 timed runs, median reported.
  - **Model size:** file size after `joblib.dump`.
  - **Peak memory:** peak process RSS via `memory_profiler` during training and during inference. The process baseline (RSS before any training) is reported separately.
- **Deployment fit:** each model was checked against the upper limits of the given budgets (memory, latency, model size).

## Results

### Accuracy (`results/baseline_accuracy.csv`)

| Model | Test accuracy |
|---|---|
| LogisticRegression | 0.9415 |
| RandomForest | 0.9357 |

### System cost (`results/system_metrics.csv`)

| Model | Train time (median, s) | Inference latency (median, ms) | Model size | Peak memory train (MB) | Peak memory inference (MB) | Process baseline (MB) |
|---|---|---|---|---|---|---|
| LogisticRegression | 0.7735 | 0.0950 | 1,055 B (1.03 KB) | 265.0 | 265.0 | 263.3 |
| RandomForest | 0.4266 | 9.2313 | 290,889 B (284.07 KB) | 266.3 | 266.3 | 263.3 |

### Deployment fit (`results/deployment_fit.csv`)

| Target (memory / latency / size) | LogisticRegression | RandomForest |
|---|---|---|
| Cloud (≥1 GB / ≤100 ms / ≤500 MB) | ✅ fits | ✅ fits |
| Edge (256–1024 MB / ≤50 ms / ≤50 MB) | ✅ fits | ✅ fits |
| Mobile (64–256 MB / ≤20 ms / ≤10 MB) | ❌ memory (265 MB > 256 MB) | ❌ memory (266 MB > 256 MB) |
| TinyML (≤256 KB / ≤10 ms / ≤100 KB) | ❌ memory | ❌ memory and size (284 KB > 100 KB) |

### Deployment-fit argument

- **Cloud and Edge:** both models fit with a large margin on every budget.
- **Mobile:** both models fail only on memory.
  - Peak RSS is about 265 MB, and almost all of it is the Python runtime plus the imported libraries (baseline 263.3 MB before training). The models themselves add only about 2–3 MB.
  - Latency and size are far below the Mobile limits: 0.095 ms and 1 KB for LogisticRegression, 9.2 ms and 284 KB for RandomForest.
  - With a lighter runtime (exported to ONNX, or without torch/pandas loaded), both models would very likely fit Mobile.
- **TinyML:** a Python process can never fit in 256 KB of RAM.
  - LogisticRegression is only 30 weights plus a bias (1 KB), so it could be ported to a microcontroller in plain C.
  - RandomForest cannot fit TinyML even after porting: 284 KB is above the 100 KB size limit, and its 9.2 ms latency is very close to the 10 ms limit.

## Conclusions

1. The accuracy of the two models is almost the same (about 94%), but the system cost is very different. The logistic regression file is only 1 KB and answers in less than 0.1 ms, while the random forest is 284 KB and needs about 9 ms per prediction. Since the accuracy is equal, I would choose logistic regression for deployment.
2. I expected the model to be the main memory user, but it was not. Python with all imported libraries already took around 263 MB before any training, and the models added only 2–3 MB on top. Because of this, both models failed the Mobile budget, even though they are tiny. To run them on a phone or microcontroller, the runtime has to change (for example ONNX or C code), not the model.
3. Surprisingly, the simpler model trained slower: 0.77 s for logistic regression vs 0.43 s for random forest. The reason is that I did not scale the features, so the solver needed all 1000 iterations and still showed a ConvergenceWarning. Adding StandardScaler should fix this. This shows that preprocessing affects training time, not only the model type.