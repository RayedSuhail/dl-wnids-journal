# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 18:36:08 2023

@author: ahmad104
"""

import tensorflow as tf
import multiprocessing
import pickle

from model_creation import get_full_ML_model
from helper_functions import MODEL_NAMES, DATA_TYPES, model_checkpoint, get_model_filename_start, MATRIX_TYPES
from data_loader import fetch_input_output, fetch_scaler, normalize_data, perform_matrix_transform

BATCH_SIZE = 32
EPOCHS = 1
MULTICLASSES = [True, False]


def train_model(model_name: str, multiclass: bool, matrix_type: str):
    if multiclass:
        MONITORING_METRIC = 'sparse_categorical_accuracy'
    else :
        MONITORING_METRIC = 'binary_accuracy'
    
    X_train, y_train = fetch_input_output(DATA_TYPES.TRAINING.value, multiclass)
    X_val, y_val = fetch_input_output(DATA_TYPES.VALIDATION.value, multiclass)
    
    scaler = fetch_scaler((-1, 1), multiclass)
    
    X_train = normalize_data(X_train, scaler)
    X_val = normalize_data(X_val, scaler)
    
    X_train = perform_matrix_transform(X_train, matrix_type)
    X_val = perform_matrix_transform(X_val, matrix_type)
    
    tf.keras.backend.clear_session()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    model = get_full_ML_model(model_name, X_train[0].shape, multiclass, matrix_type)
    
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
                        callbacks=model_checkpoint(get_model_filename_start(model.name) + 'best_model.keras', MONITORING_METRIC))
    
    with open(get_model_filename_start(model.name) + 'history.pickle', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

if __name__ == "__main__":
    for matrix_type in MATRIX_TYPES:
        for model_name in MODEL_NAMES:
            for multiclass in MULTICLASSES:
                # if not matrix_type.value == MATRIX_TYPES.CIRCULANT_GRAYSCALE_TRANSFORM.value:
                #     continue
                print('-'*20 + matrix_type.value + ': ' + model_name.value + '-'*20)
                p = multiprocessing.Process(target=train_model, args=(model_name.value, multiclass, matrix_type.value))
                p.start()
                p.join()


# =============================================================================
# TEST CODE
# =============================================================================
# train_model('ANN_512Neurons_20Dropout', True, '16_corr_transform')

# model_name =  get_model_name('ANN_512Neurons_20Dropout', True, '16_corr_transform')
# with open(get_model_filename_start(model_name) + 'history.pickle', "rb") as file_pi:
#     history = pickle.load(file_pi)

