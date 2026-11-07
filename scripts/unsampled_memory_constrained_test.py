# -*- coding: utf-8 -*-
"""
Low-Memory Streaming Evaluation Pipeline for AWID3 Dataset
Peak RAM Usage: < 2 GB
"""

import glob
import os
import time
import gc
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from helper_functions import (
    FEATURES_LIST,
    MATRIX_TYPES,
    MODEL_NAMES,
    get_model_filename_start
)
from data_loader import fetch_scaler, normalize_data, perform_matrix_transform
from update_csv_files import get_antsignal_value

AWID3_DATASET_ROOT = '../datasets/AWID3/*_Same'
SHALLOW_MODEL_NAMES = [
    MODEL_NAMES.RANDOM_FOREST.value,
    MODEL_NAMES.EXTRA_TREES.value,
    MODEL_NAMES.XGBOOST.value,
    MODEL_NAMES.LIGHTGBM.value
]


def compute_metrics_from_cm(cm):
    """
    Computes exact global classification metrics from an aggregated confusion matrix.
    Mathematically identical to sklearn.metrics.precision_recall_fscore_support.
    """
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp, dtype=float), where=(precision + recall) != 0)

    total_samples = np.sum(cm)
    accuracy = np.sum(tp) / total_samples if total_samples > 0 else 0
    macro_p = np.mean(precision)
    macro_r = np.mean(recall)
    macro_f1 = np.mean(f1)

    weights = np.sum(cm, axis=1) / total_samples if total_samples > 0 else 0
    weighted_f1 = np.sum(weights * f1)

    return accuracy, macro_p, macro_r, macro_f1, weighted_f1


def preprocess_chunk(df_chunk, is_multiclass: bool, scaler):
    """
    Preprocesses a single chunk/file of data in-memory.
    """
    if is_multiclass:
        label_map = {
            'Normal': 0, 'Deauth': 1, 'Disas': 2, '(Re)Assoc': 3,
            'RogueAP': 4, 'Krack': 5, 'Kr00k': 6, 'Kr00K': 6, 'Evil_Twin': 7
        }
    else:
        label_map = {
            'Normal': 0, 'Kr00k': 1, 'Kr00K': 1, 'Evil_Twin': 1, 'Disas': 1,
            'Krack': 1, 'Deauth': 1, '(Re)Assoc': 1, 'RogueAP': 1
        }

    df_chunk['Label'] = df_chunk['Label'].map(label_map)
    df_chunk.rename(columns={'Label': 'attack_map'}, inplace=True)

    # Filter features and drop NaNs
    df_filtered = df_chunk[FEATURES_LIST].dropna().copy()
    if df_filtered.empty:
        return None, None

    # Parse feature formats
    df_filtered['wlan.fc.ds'] = df_filtered['wlan.fc.ds'].apply(lambda x: int(str(x), 16) if pd.notnull(x) else 0)
    df_filtered['radiotap.present.tsft'] = df_filtered['radiotap.present.tsft'].map({'1-0-0': 1, '0-0-0': 0}).fillna(0)
    df_filtered['radiotap.dbm_antsignal'] = df_filtered['radiotap.dbm_antsignal'].apply(get_antsignal_value)

    X = df_filtered.drop('attack_map', axis=1).copy()
    y = df_filtered['attack_map'].to_numpy(dtype=int)

    # Scale raw 1D feature vector
    X_scaled = normalize_data(X, scaler)

    return X_scaled, y


def evaluate_shallow_models_streaming(all_csv_files, is_multiclass: bool, num_classes: int, scaler):
    """
    Evaluates saved shallow tree models (.joblib) by streaming CSV files one by one.
    """
    task_str = "multiclass" if is_multiclass else "binary"
    results = []

    print(f"\n==================== [STREAMING SHALLOW MODELS - {task_str.upper()}] ====================")

    for model_name in SHALLOW_MODEL_NAMES:
        model_path = os.path.join(
            get_model_filename_start(f"raw_16features/{task_str}_{model_name}"),
            'model.joblib'
        )

        if not os.path.exists(model_path):
            print(f"Skipping {model_name} ({task_str}): Model checkpoint not found at {model_path}")
            continue

        print(f"Loading {model_name} from {model_path}...")
        model = joblib.load(model_path)

        running_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        start_time = time.time()

        for csv_file in all_csv_files:
            # Stream file in chunks of 100,000 rows
            for chunk in pd.read_csv(csv_file, chunksize=100000):
                X_chunk, y_chunk = preprocess_chunk(chunk, is_multiclass, scaler)
                if X_chunk is None or len(X_chunk) == 0:
                    continue

                y_pred = model.predict(X_chunk)
                
                # Accumulate confusion matrix
                cm_chunk = confusion_matrix(y_chunk, y_pred, labels=list(range(num_classes)))
                running_cm += cm_chunk

                del X_chunk, y_chunk, y_pred
                gc.collect()

        total_time = time.time() - start_time
        acc, macro_p, macro_r, macro_f1, weighted_f1 = compute_metrics_from_cm(running_cm)

        results.append({
            "Model": model_name,
            "Matrix / Input": "Raw 16-Vector",
            "Task": task_str,
            "Accuracy": acc,
            "Macro Precision": macro_p,
            "Macro Recall": macro_r,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1,
            "Total Evaluated Samples": np.sum(running_cm),
            "Evaluation Time (s)": round(total_time, 2)
        })

        print(f"--> {model_name} [{task_str}]: Acc={acc*100:.2f}%, Macro F1={macro_f1*100:.2f}% (Processed {np.sum(running_cm):,} samples)")

    return results


def evaluate_dl_models_streaming(all_csv_files, is_multiclass: bool, num_classes: int, scaler):
    """
    Evaluates saved DL/CNN models (.keras) by streaming and transforming data chunk-by-chunk.
    """
    task_str = "multiclass" if is_multiclass else "binary"
    results = []

    print(f"\n==================== [STREAMING DL MODELS - {task_str.upper()}] ====================")

    for matrix_type in MATRIX_TYPES:
        for model_enum in MODEL_NAMES:
            model_name = model_enum.value
            if model_name in SHALLOW_MODEL_NAMES:
                continue

            full_model_identifier = f"{matrix_type.value}/{task_str}_{model_name}"
            model_dir = get_model_filename_start(full_model_identifier)
            model_path = os.path.join(model_dir, 'best_model.keras')

            if not os.path.exists(model_path):
                continue

            print(f"\nEvaluating DL Model: {full_model_identifier}...")
            model = tf.keras.models.load_model(model_path)

            running_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
            start_time = time.time()

            for csv_file in all_csv_files:
                for chunk in pd.read_csv(csv_file, chunksize=100000):
                    X_chunk, y_chunk = preprocess_chunk(chunk, is_multiclass, scaler)
                    if X_chunk is None or len(X_chunk) == 0:
                        continue

                    # Apply 2D image matrix transformation on current chunk
                    X_trans = perform_matrix_transform(X_chunk, matrix_type.value)

                    preds = model.predict(X_trans, batch_size=2048, verbose=0)
                    if is_multiclass:
                        y_pred = np.argmax(preds, axis=1)
                    else:
                        y_pred = (preds > 0.5).astype(int).flatten()

                    cm_chunk = confusion_matrix(y_chunk, y_pred, labels=list(range(num_classes)))
                    running_cm += cm_chunk

                    del X_chunk, y_chunk, X_trans, preds, y_pred
                    gc.collect()

            total_time = time.time() - start_time
            acc, macro_p, macro_r, macro_f1, weighted_f1 = compute_metrics_from_cm(running_cm)

            results.append({
                "Model": model_name,
                "Matrix / Input": matrix_type.value,
                "Task": task_str,
                "Accuracy": acc,
                "Macro Precision": macro_p,
                "Macro Recall": macro_r,
                "Macro F1": macro_f1,
                "Weighted F1": weighted_f1,
                "Total Evaluated Samples": np.sum(running_cm),
                "Evaluation Time (s)": round(total_time, 2)
            })

            print(f"--> {model_name} ({matrix_type.value}): Acc={acc*100:.2f}%, Macro F1={macro_f1*100:.2f}%")

            tf.keras.backend.clear_session()
            gc.collect()

    return results


if __name__ == "__main__":
    # Get all CSV paths from datasets/AWID3/*_Same/
    all_folders = glob.glob(AWID3_DATASET_ROOT)
    all_csv_files = []
    for folder in all_folders:
        all_csv_files.extend(glob.glob(f'{folder}/*.csv'))

    if not all_csv_files:
        raise FileNotFoundError(f"No CSV files found under {AWID3_DATASET_ROOT}")

    print(f"Found {len(all_csv_files)} CSV dataset files to evaluate.")

    all_results = []

    for is_multiclass in [False, True]:
        task_label = "multiclass" if is_multiclass else "binary"
        num_classes = 8 if is_multiclass else 2
        
        # Load scaler fitted on training set
        scaler = fetch_scaler((-1, 1), is_multiclass)

        # 1. Evaluate Shallow Models
        shallow_res = evaluate_shallow_models_streaming(all_csv_files, is_multiclass, num_classes, scaler)
        all_results.extend(shallow_res)

        # 2. Evaluate DL / CNN Models
        dl_res = evaluate_dl_models_streaming(all_csv_files, is_multiclass, num_classes, scaler)
        all_results.extend(dl_res)

    # Export aggregated summary table
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_csv = "../models/extended_october2023_start/unsampled_awid3_low_memory_results.csv"
        results_df.to_csv(output_csv, index=False)
        print("\n" + "="*80)
        print(f"STREAMING EVALUATION COMPLETE. Results saved to {output_csv}")
        print("="*80)
        print(results_df.to_string(index=False))