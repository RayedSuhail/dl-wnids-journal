# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 17:43:49 2023

@author: ahmad104
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from pyts.image import GramianAngularField

from helper_functions import get_dataset_filename_start, DATA_TYPES, FEATURES_LIST, MATRIX_TYPES


def fetch_input_output(data_type: str, is_multiclass: bool):
    if data_type not in DATA_TYPES:
        raise Exception
    
    base_filename = get_dataset_filename_start(is_multiclass)
    
    X = pd.read_csv(f'{base_filename}_combined_awid3_{data_type}_input.csv')
    y = pd.read_csv(f'{base_filename}_combined_awid3_{data_type}_output.csv')
    
    return (X, y.to_numpy())

def fetch_scaler(feature_range=(-1, 1), is_multiclass=False):
    training_data, _ = fetch_input_output(DATA_TYPES.TRAINING.value, is_multiclass)
    scaler = MinMaxScaler(feature_range)
    return scaler.fit(training_data)

def normalize_data(X, scaler):
    return scaler.transform(X)

def perform_matrix_transform(X, method: str, gaf_method='s'):
    if method == MATRIX_TYPES.CORRELATION_TRANSFORM_16x16.value:
        X = np.apply_along_axis(lambda row: create_corr_matrix(row, 16), axis=1, arr=X)
        X = X.reshape(*X.shape, 1)
    elif method == MATRIX_TYPES.CIRCULANT_GRAYSCALE_TRANSFORM.value:
        X = np.apply_along_axis(lambda row: create_grayscale_matrix(row, True, 16), axis=1, arr=X)
        X = X.reshape(*X.shape, 1)
    elif method == MATRIX_TYPES.CYCLIC_GRAYSCALE_TRANSFORM_16x16.value:
        X = np.apply_along_axis(lambda row: create_grayscale_matrix(row, False, 16), axis=1, arr=X)
        X = X.reshape(*X.shape, 1)
    elif method == MATRIX_TYPES.GAF_TRANSFORM.value:
        X = perform_gramian_transform(X, gaf_method)
    elif method == MATRIX_TYPES.CIRCULANT_TRANSFORM.value:
        X = np.apply_along_axis(lambda row: create_circulant_matrix(row, 16), axis=1, arr=X)
        X = X.reshape(*X.shape, 1)
    
    return X

def perform_gramian_transform(X, method='s'):
    gaf = GramianAngularField(image_size=X.shape[1], method=method)
    image_shape = (X.shape[1], X.shape[1], 1)
    return gaf.transform(X).reshape(X.shape[0], *image_shape)

def create_grayscale_matrix(row, use_circulant: bool, size=16):
    mat = np.empty([size, size])
    circulant_idx = create_circulant_matrix(list(range(size)), size)
    for idx_i in range(size):
        for idx_j in range(size):
            if size == (len(FEATURES_LIST) - 1) and use_circulant:
                mat[idx_i, idx_j] = row[int(circulant_idx[idx_i, idx_j])]
            else:
                if (size != (len(FEATURES_LIST) - 1)):
                    idx = (idx_i * size) + idx_j
                else:
                    idx = (idx_i + idx_j) % size
                mat[idx_i, idx_j] = row[idx]
    return mat

def create_corr_matrix(row, size):
    mat = np.empty([size, size])
    for idx_i in range(size):
        for idx_j in range(size):
            if (size != (len(FEATURES_LIST) - 1)):
                idx = (idx_i * size) + idx_j
            else:
                idx = (idx_i + idx_j) % size
            mat[idx_i, idx_j] = row[idx]
    # Correlation same in cyclic or circulant
    corr_mat = pd.DataFrame(mat).corr()
    return corr_mat

def create_circulant_matrix(row, size):
    mat = np.empty((size, size))
    for i in range(size):
        mat[i, i:] = row[:size-i]
        mat[i, :i] = row[size-i:]
    return mat.T

def test_images(normalize_data = False):
    X = np.array([[0, 1, 2, 3, 4 ,5 ,6, 7 ,8 ,9 ,10, 11, 12, 13, 14, 15]])
    globals()['X'] = X
    if normalize_data:
        X = normalize_data(X, fetch_scaler((-1, 1), False))
        globals()['X_scaled'] = X
    for mat_type in MATRIX_TYPES:
        globals()[mat_type.value] = perform_matrix_transform(X, mat_type.value)[0, :, :, 0]