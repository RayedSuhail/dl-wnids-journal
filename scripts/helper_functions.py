# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 18:14:08 2023

@author: ahmad104
"""

import numpy as np
import logging
import os

from enum import Enum, EnumMeta

from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

class MetaEnum(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True

class BaseEnum(Enum, metaclass=MetaEnum):
    pass

class DATA_TYPES(BaseEnum):
    TRAINING = 'train'
    TESTING = 'test'
    VALIDATION = 'val'

class MATRIX_TYPES(BaseEnum):
    GAF_TRANSFORM = '16_gaf_transform'
    CORRELATION_TRANSFORM_16x16 = '16_corr_transform'
    CYCLIC_GRAYSCALE_TRANSFORM_16x16 = '16_cyclic_grayscale_transform'
    CIRCULANT_GRAYSCALE_TRANSFORM = '16_circulant_grayscale_transform'
    CIRCULANT_TRANSFORM = '16_circulant_transform'

class MODEL_NAMES(BaseEnum):
    CNN2_2D = 'CNN2D_2xConvPool'
    CNN1_2D = 'CNN2D_1xConvPool'
    CNN2_1D = 'CNN1D_2xConvPool'
    CNN1_1D = 'CNN1D_1xConvPool'

class CustomModelCheckpoint(ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1

        if self.save_freq == "epoch":
            self._save_model(epoch=epoch, batch=None, logs=logs)
            
        current = logs.get(self.monitor)
        log_filename_start = get_model_filename_start(self.model.name)
        if os.path.isfile(log_filename_start + 'best_model.keras') \
        or (self.save_best_only and current is not None and self.monitor_op(current, self.best)):
            logging.basicConfig(filename=log_filename_start + "best_model.custom.log", 
					format = '%(asctime)s %(message)s', 
					filemode = 'w') 

            logger = logging.getLogger() 
            
            #Now we are going to Set the threshold of logger to DEBUG 
            logger.setLevel(logging.DEBUG) 
            logger.info(f'For model {self.model.name}, '
                        f'best model saved at epoch {epoch + 1}') 

FEATURES_LIST = open('../features_awid3.txt').read().split('\n')[:-1]
CLASSES = list(range(8))

def model_checkpoint(save_dir: str, metric: str):
    early_stopping = EarlyStopping(monitor=metric, patience=3,
                              verbose=1, mode="max",
                              restore_best_weights=True)
    checkpoint = CustomModelCheckpoint(filepath=save_dir, 
                monitor=metric, verbose=1, mode='max',
                save_best_only=True)
    csvwriter = CSVLogger(filename=save_dir + '_model.csv')
    return [early_stopping, checkpoint, csvwriter]

def get_model_name(model_name: str, is_multiclass: bool, input_type: str):
    if model_name not in MODEL_NAMES or input_type not in MATRIX_TYPES:
        raise Exception
    if is_multiclass:
        ext = 'multiclass'
    else:
        ext = 'binary'
    return input_type + '/' + ext + '_' + model_name

def get_dataset_filename_start(is_multiclass: bool):
    if is_multiclass:
        return '../datasets/multiclass'
    else:
        return '../datasets/binary'

def get_model_filename_start(model_name: str):
    return f'../models/extended_october2023_start/{model_name}/'