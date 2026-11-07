# -*- coding: utf-8 -*-
"""
Shallow Models Training and Evaluation Pipeline
"""

import os
import time
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data_loader import fetch_input_output, fetch_scaler, normalize_data
from helper_functions import MODEL_NAMES, DATA_TYPES, get_model_filename_start
import model_creation

# List of shallow models to benchmark
SHALLOW_MODELS = [
    MODEL_NAMES.RANDOM_FOREST,
    MODEL_NAMES.EXTRA_TREES,
    MODEL_NAMES.XGBOOST,
    MODEL_NAMES.LIGHTGBM
]

MULTICLASS_SETTINGS = [False, True]  # False = Binary, True = Multi-class


def evaluate_and_train_shallow_model(model_enum: MODEL_NAMES, is_multiclass: bool):
    model_name = model_enum.value
    task_str = "multiclass" if is_multiclass else "binary"
    print('-'*20 + task_str.upper() + ' Training ' + model_name + '-'*20)

    # 1. Fetch raw datasets
    X_train, y_train = fetch_input_output(DATA_TYPES.TRAINING.value, is_multiclass)
    X_test, y_test = fetch_input_output(DATA_TYPES.TESTING.value, is_multiclass)

    # Convert targets to 1D arrays
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # 2. Scale 1D feature inputs (-1 to 1)
    scaler = fetch_scaler((-1, 1), is_multiclass)
    X_train = normalize_data(X_train, scaler)
    X_test = normalize_data(X_test, scaler)

    # 3. Instantiate model from model_creation module
    model_func = getattr(model_creation, model_name)
    model = model_func(is_multiclass=is_multiclass)

    # 4. Fit Model & measure training time
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    # 5. Measure CPU Inference Latency on Test Set
    start_infer = time.time()
    y_pred = model.predict(X_test)
    total_infer_time = time.time() - start_infer
    latency_ms_per_1k = (total_infer_time / len(X_test)) * 1000 * 1000  # ms per 1,000 samples

    # 6. Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )

    # 7. Save Trained Model Weights
    save_dir = get_model_filename_start(f"raw_16features/{task_str}_{model_name}")
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, 'model.joblib')
    joblib.dump(model, model_save_path)

    results = {
        "Model": model_name,
        "Task": task_str,
        "Accuracy": acc,
        "Macro Precision": macro_prec,
        "Macro Recall": macro_rec,
        "Macro F1": macro_f1,
        "Weighted Precision": weighted_prec,
        "Weighted Recall": weighted_rec,
        "Weighted F1": weighted_f1,
        "Train Time (s)": round(train_time, 3),
        "Latency (ms / 1k samples)": round(latency_ms_per_1k, 4)
    }

    print(f"Finished {model_name} [{task_str}]: Accuracy={acc:.5f}, Macro F1={macro_f1:.5f}, Latency={latency_ms_per_1k:.4f} ms/1k")
    return results


if __name__ == "__main__":
    all_results = []

    for is_multiclass in MULTICLASS_SETTINGS:
        for model_enum in SHALLOW_MODELS:
            res = evaluate_and_train_shallow_model(model_enum, is_multiclass)
            all_results.append(res)

    # Save summary results DataFrame to CSV
    results_df = pd.DataFrame(all_results)
    output_path = "../models/extended_october2023_start/shallow_models_benchmark_results.csv"
    results_df.to_csv(output_path, index=False)

    print('-'*20 + " SHALLOW MODELS TRAINING COMPLETE " + '-'*20)
    print("Summary Results:")
    print("-"*50)
    print(results_df.to_string(index=False))