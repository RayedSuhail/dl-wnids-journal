# -*- coding: utf-8 -*-
"""
Empirical Memory Footprint Profiler for Tree Ensembles vs. CNN Models
"""

import os
import sys

# OPTIONAL: Uncomment the line below if benchmarking CPU-only edge memory
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import gc
import joblib
import psutil
import pandas as pd
import tensorflow as tf

from helper_functions import MODEL_NAMES, MATRIX_TYPES, get_model_filename_start

SHALLOW_MODELS = [
    MODEL_NAMES.RANDOM_FOREST.value,
    MODEL_NAMES.EXTRA_TREES.value,
    MODEL_NAMES.XGBOOST.value,
    MODEL_NAMES.LIGHTGBM.value
]


def warmup_tensorflow():
    """
    Warms up the TensorFlow C++ runtime and CUDA driver context before profiling,
    preventing one-time framework initialization from inflating the first model's RAM count.
    """
    print("Warming up TensorFlow C++ / GPU runtime context...")
    tf.keras.backend.clear_session()
    # Build a tiny dummy model to trigger C++ and CUDA initialization
    dummy_input = tf.zeros((1, 16, 16, 1))
    dummy_model = tf.keras.Sequential([
        tf.keras.layers.Input((16, 16, 1)),
        tf.keras.layers.Conv2D(2, 3, padding='same')
    ])
    _ = dummy_model(dummy_input)
    
    del dummy_input, dummy_model
    tf.keras.backend.clear_session()
    gc.collect()


def count_tree_nodes(model):
    """
    Counts the total number of structural decision nodes across all trees.
    """
    total_nodes = 0
    if hasattr(model, 'estimators_'):  # RandomForest / ExtraTrees
        total_nodes = sum(tree.tree_.node_count for tree in model.estimators_)
    elif hasattr(model, 'get_booster'):  # XGBoost
        booster = model.get_booster()
        dump_list = booster.get_dump()
        total_nodes = sum(tree.count('leaf') + tree.count('->') for tree in dump_list)
    elif hasattr(model, 'booster_'):  # LightGBM
        df_dump = model.booster_.dump_model()
        total_nodes = sum(tree['num_leaves'] * 2 - 1 for tree in df_dump['tree_info'])
    return total_nodes


def profile_shallow_model(model_name: str, task: str = "multiclass"):
    model_dir = get_model_filename_start(f"raw_16features/{task}_{model_name}")
    model_path = os.path.join(model_dir, 'model.joblib')

    if not os.path.exists(model_path):
        return None

    # Measure disk size
    disk_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    # Measure RAM before loading
    process = psutil.Process(os.getpid())
    gc.collect()
    ram_before = process.memory_info().rss / (1024 * 1024)

    model = joblib.load(model_path)

    # Measure RAM after loading
    ram_after = process.memory_info().rss / (1024 * 1024)
    active_ram_mb = max(0.0, ram_after - ram_before)

    total_nodes = count_tree_nodes(model)

    del model
    gc.collect()

    return {
        "Model": model_name,
        "Type": "Tree Ensemble (Shallow)",
        "Input Representation": "Raw 16 Vector",
        "Disk File Size (MB)": round(disk_size_mb, 3),
        "Active RAM Overhead (MB)": round(active_ram_mb, 2),
        "Structural Units (Nodes / Params)": f"{total_nodes:,} nodes"
    }


def profile_cnn_model(model_name: str, matrix_type: str = "16_gaf_transform", task: str = "multiclass"):
    full_identifier = f"{matrix_type}/{task}_{model_name}"
    model_dir = get_model_filename_start(full_identifier)
    model_path = os.path.join(model_dir, 'best_model.keras')

    if not os.path.exists(model_path):
        return None

    # Measure disk size
    disk_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    # Measure RAM before loading
    process = psutil.Process(os.getpid())
    tf.keras.backend.clear_session()
    gc.collect()
    ram_before = process.memory_info().rss / (1024 * 1024)

    model = tf.keras.models.load_model(model_path)

    # Measure RAM after loading
    ram_after = process.memory_info().rss / (1024 * 1024)
    active_ram_mb = max(0.0, ram_after - ram_before)

    param_count = model.count_params()

    tf.keras.backend.clear_session()
    gc.collect()

    return {
        "Model": model_name,
        "Type": "Deep Convolutional Neural Net",
        "Input Representation": matrix_type,
        "Disk File Size (MB)": round(disk_size_mb, 3),
        "Active RAM Overhead (MB)": round(active_ram_mb, 2),
        "Structural Units (Nodes / Params)": f"{param_count:,} parameters"
    }


if __name__ == "__main__":
    results = []

    print("Profiling Shallow Tree Models...")
    for model_name in SHALLOW_MODELS:
        res = profile_shallow_model(model_name, task="multiclass")
        if res:
            results.append(res)

    # Pre-warm TensorFlow backend before profiling deep models
    warmup_tensorflow()

    print("Profiling Deep GAF Models...")
    dl_models = [
        MODEL_NAMES.CNN2_2D.value,
        MODEL_NAMES.CNN1_2D.value,
        MODEL_NAMES.CNN2_1D.value,
        MODEL_NAMES.CNN1_1D.value
    ]
    for model_name in dl_models:
        res = profile_cnn_model(model_name, matrix_type="16_gaf_transform", task="multiclass")
        if res:
            results.append(res)

    df_results = pd.DataFrame(results)
    print("\n" + "="*90)
    print("EMPIRICAL MODEL MEMORY FOOTPRINT COMPARISON")
    print("="*90)
    print(df_results.to_string(index=False))