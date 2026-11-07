# -*- coding: utf-8 -*-
"""
Full Unsampled AWID3 Evaluation Pipeline for Deep Learning & Shallow Models
"""

import glob
import os
import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from helper_functions import (
    FEATURES_LIST,
    MATRIX_TYPES,
    MODEL_NAMES,
    get_model_filename_start
)
from data_loader import fetch_scaler, normalize_data, perform_matrix_transform
from update_csv_files import get_antsignal_value

AWID3_DATASET_ROOT = '../datasets/AWID3/*_Same'
CLASS_NAMES_MULTICLASS = ['Normal', 'Deauth', 'Disas', '(Re)Assoc', 'RogueAP', 'Krack', 'Kr00k', 'Evil_Twin']
CLASS_NAMES_BINARY = ['Normal', 'Attack']

SHALLOW_MODEL_NAMES = [
    MODEL_NAMES.RANDOM_FOREST.value,
    MODEL_NAMES.EXTRA_TREES.value,
    MODEL_NAMES.XGBOOST.value,
    MODEL_NAMES.LIGHTGBM.value
]


def load_and_preprocess_unsampled_awid3(is_multiclass: bool):
    """
    Loads raw CSV files from datasets/AWID3/*_Same without downsampling normal traffic.
    Applies feature filtering, hex parsing, NaN handling, and Min-Max normalization.
    """
    print(f"\n[Data Prep] Loading unsampled AWID3 dataset from {AWID3_DATASET_ROOT}...")
    all_folders = glob.glob(AWID3_DATASET_ROOT)
    
    if not all_folders:
        raise FileNotFoundError(f"No AWID3 directories found at path: {AWID3_DATASET_ROOT}")

    dfs = []
    for folder in all_folders:
        csv_files = glob.glob(f'{folder}/*.csv')
        for f in csv_files:
            df_temp = pd.read_csv(f)
            dfs.append(df_temp)

    df_full = pd.concat(dfs, ignore_index=True)
    print(f"[Data Prep] Total raw records loaded: {len(df_full):,}")

    # Map labels
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

    df_full['Label'] = df_full['Label'].map(label_map)
    df_full.rename(columns={'Label': 'attack_map'}, inplace=True)

    # Filter features and drop NaNs
    df_full = df_full[FEATURES_LIST].dropna()

    # Apply feature conversions matching update_csv_files.py
    df_full['wlan.fc.ds'] = df_full['wlan.fc.ds'].apply(lambda x: int(str(x), 16) if pd.notnull(x) else 0)
    df_full['radiotap.present.tsft'] = df_full['radiotap.present.tsft'].map({'1-0-0': 1, '0-0-0': 0}).fillna(0)
    df_full['radiotap.dbm_antsignal'] = df_full['radiotap.dbm_antsignal'].apply(get_antsignal_value)

    X = df_full.drop('attack_map', axis=1).copy()
    y = df_full['attack_map'].to_numpy()

    # Scale features (-1, 1) using training scaler
    scaler = fetch_scaler((-1, 1), is_multiclass)
    X_scaled = normalize_data(X, scaler)

    print(f"[Data Prep] Final processed dataset shape: {X_scaled.shape}")
    return X_scaled, y


def evaluate_shallow_models(X_raw, y, is_multiclass: bool):
    """
    Evaluates saved shallow tree-based models (.joblib) on raw unsampled 16-feature vector.
    """
    task_str = "multiclass" if is_multiclass else "binary"
    results = []

    print(f"\n==================== [EVALUATING SHALLOW MODELS - {task_str.upper()}] ====================")

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

        start_time = time.time()
        y_pred = model.predict(X_raw)
        infer_time = time.time() - start_time

        acc = accuracy_score(y, y_pred)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y, y_pred, average='macro', zero_division=0)
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted', zero_division=0)

        results.append({
            "Model": model_name,
            "Matrix / Input": "Raw 16-Vector",
            "Task": task_str,
            "Accuracy": acc,
            "Macro Precision": macro_p,
            "Macro Recall": macro_r,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1,
            "Inference Time (s)": round(infer_time, 2)
        })

        print(f"--> {model_name} [{task_str}]: Accuracy={acc*100:.2f}%, Macro F1={macro_f1*100:.2f}%")

    return results


def evaluate_deep_learning_models(X_raw, y, is_multiclass: bool):
    """
    Evaluates saved Keras CNN and ANN models across all matrix transformation techniques.
    """
    task_str = "multiclass" if is_multiclass else "binary"
    results = []

    print(f"\n==================== [EVALUATING DL MODELS - {task_str.upper()}] ====================")

    for matrix_type in MATRIX_TYPES:
        print(f"\n---> Transforming data for Matrix Type: {matrix_type.value}...")
        X_trans = perform_matrix_transform(X_raw, matrix_type.value)

        for model_enum in MODEL_NAMES:
            model_name = model_enum.value
            if model_name in SHALLOW_MODEL_NAMES:
                continue

            full_model_identifier = f"{matrix_type.value}/{task_str}_{model_name}"
            model_dir = get_model_filename_start(full_model_identifier)
            model_path = os.path.join(model_dir, 'best_model.keras')

            if not os.path.exists(model_path):
                continue

            print(f"Evaluating DL Model: {full_model_identifier}...")
            model = tf.keras.models.load_model(model_path)

            start_time = time.time()
            preds = model.predict(X_trans, batch_size=2048, verbose=0)
            infer_time = time.time() - start_time

            if is_multiclass:
                y_pred = np.argmax(preds, axis=1)
            else:
                y_pred = (preds > 0.5).astype(int).flatten()

            acc = accuracy_score(y, y_pred)
            macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y, y_pred, average='macro', zero_division=0)
            weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted', zero_division=0)

            results.append({
                "Model": model_name,
                "Matrix / Input": matrix_type.value,
                "Task": task_str,
                "Accuracy": acc,
                "Macro Precision": macro_p,
                "Macro Recall": macro_r,
                "Macro F1": macro_f1,
                "Weighted F1": weighted_f1,
                "Inference Time (s)": round(infer_time, 2)
            })

            print(f"--> {model_name} ({matrix_type.value}): Accuracy={acc*100:.2f}%, Macro F1={macro_f1*100:.2f}%")

            tf.keras.backend.clear_session()

    return results


if __name__ == "__main__":
    all_evaluation_results = []

    for is_multiclass in [False, True]:
        task_label = "multiclass" if is_multiclass else "binary"
        
        # 1. Load full unsampled AWID3 dataset
        X_unsampled, y_unsampled = load_and_preprocess_unsampled_awid3(is_multiclass)

        # 2. Evaluate Shallow Models
        shallow_res = evaluate_shallow_models(X_unsampled, y_unsampled, is_multiclass)
        all_evaluation_results.extend(shallow_res)

        # 3. Evaluate Neural Networks / CNNs
        dl_res = evaluate_deep_learning_models(X_unsampled, y_unsampled, is_multiclass)
        all_evaluation_results.extend(dl_res)

    # 4. Export aggregated summary to CSV
    if all_evaluation_results:
        results_df = pd.DataFrame(all_evaluation_results)
        output_csv = "../models/extended_october2023_start/unsampled_awid3_evaluation_results.csv"
        results_df.to_csv(output_csv, index=False)
        print("\n" + "="*80)
        print(f"EVALUATION COMPLETE. Summary results saved to {output_csv}")
        print("="*80)
        print(results_df.to_string(index=False))
    else:
        print("\nNo trained model checkpoints were found to evaluate. Please check model saved paths.")