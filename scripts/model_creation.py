# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 18:35:57 2023

@author: ahmad104
"""

import keras
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import Dropout, Flatten, Reshape
from keras.layers import Conv2D, AveragePooling2D
from keras.layers import Conv1D, AveragePooling1D
from tensorflow.keras.optimizers import Adam

from helper_functions import get_model_name, MODEL_NAMES, CLASSES

def get_full_ML_model(model_name: MODEL_NAMES, input_shape, is_multiclass, matrix_type):
    base_model = globals()[model_name](input_shape, is_multiclass, matrix_type)
    
    if is_multiclass:
        loss_str = 'sparse_categorical_crossentropy'
        metrics_list = ['sparse_categorical_accuracy']
    else:
        loss_str = 'binary_crossentropy'
        metrics_list = ['binary_accuracy']
    
    base_model.compile(
        optimizer = Adam(learning_rate=0.001, amsgrad=True),
        loss = loss_str,
        metrics = metrics_list, # F1Score(average='weighted', threshold=0.5, name='f1_score')
    )
    return base_model

def CNN2D_2xConvPool(input_shape, is_multiclass: bool, matrix_type) -> keras.engine.functional.Functional:
    """
    Model: "{data_input_type}/multiclass_CNN2D_2xConvPool"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 16 / 4, 16 / 4, 1)]       0         
    _________________________________________________________________
    conv2d (Conv2D)              (None, 16, 16, 16)        160       
    _________________________________________________________________
    average_pooling2d (AveragePo (None, 8, 8, 16)          0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 8, 8, 16)          2320      
    _________________________________________________________________
    average_pooling2d_1 (Average (None, 4, 4, 16)          0         
    _________________________________________________________________
    dropout (Dropout)            (None, 4, 4, 16)          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 256)               0         
    _________________________________________________________________
    dense (Dense)                (None, 32)                8224      
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 32)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 8)                 264       
    =================================================================
    Total params: 10,968
    Trainable params: 10,968
    Non-trainable params: 0
    _________________________________________________________________
    
    Model: "{data_input_type}/binary_CNN2D_2xConvPool"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 16 / 4, 16 / 4, 1)]       0         
    _________________________________________________________________
    conv2d (Conv2D)              (None, 16, 16, 16)        160       
    _________________________________________________________________
    average_pooling2d (AveragePo (None, 8, 8, 16)          0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 8, 8, 16)          2320      
    _________________________________________________________________
    average_pooling2d_1 (Average (None, 4, 4, 16)          0         
    _________________________________________________________________
    dropout (Dropout)            (None, 4, 4, 16)          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 256)               0         
    _________________________________________________________________
    dense (Dense)                (None, 32)                8224      
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 32)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 33        
    =================================================================
    Total params: 10,737
    Trainable params: 10,737
    Non-trainable params: 0
    _________________________________________________________________
    """
    input_x = Input(input_shape)

    X = Conv2D(filters=16, kernel_size=3, activation='relu', padding='same')(input_x)
    X = AveragePooling2D(pool_size=2)(X)
    X = Conv2D(filters=16, kernel_size=3, activation='relu', padding='same')(X)
    X = AveragePooling2D(pool_size=2)(X)
    X = Dropout(0.2)(X)
    X = Flatten()(X)
    
    X = Dense(32, activation='relu')(X)
    X = Dropout(0.5)(X)
    
    if is_multiclass:
        X = Dense(len(CLASSES), activation='softmax')(X)
    else:        
        X = Dense(1, activation='sigmoid')(X)
    
    model = Model(inputs=input_x, outputs=X,
                  name=get_model_name(MODEL_NAMES.CNN2_2D.value, is_multiclass, matrix_type))
    return model

def CNN2D_1xConvPool(input_shape, is_multiclass: bool, matrix_type) -> keras.engine.functional.Functional:
    """
    Model: "{data_input_type}/multiclass_CNN2D_1xConvPool"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 16 / 4, 16 / 4, 1)]       0         
    _________________________________________________________________
    conv2d (Conv2D)              (None, 16, 16, 16)        160       
    _________________________________________________________________
    average_pooling2d (AveragePo (None, 8, 8, 16)          0         
    _________________________________________________________________
    dropout (Dropout)            (None, 8, 8, 16)          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 1024)              0         
    _________________________________________________________________
    dense (Dense)                (None, 32)                32800     
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 32)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 8)                 264       
    =================================================================
    Total params: 33,224
    Trainable params: 33,224
    Non-trainable params: 0
    _________________________________________________________________
    
    Model: "{data_input_type}/binary_CNN2D_1xConvPool"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 16 / 4, 16 / 4, 1)]       0         
    _________________________________________________________________
    conv2d (Conv2D)              (None, 16, 16, 16)        160       
    _________________________________________________________________
    average_pooling2d (AveragePo (None, 8, 8, 16)          0         
    _________________________________________________________________
    dropout (Dropout)            (None, 8, 8, 16)          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 1024)              0         
    _________________________________________________________________
    dense (Dense)                (None, 32)                32800     
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 32)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 33        
    =================================================================
    Total params: 32,993
    Trainable params: 32,993
    Non-trainable params: 0
    _________________________________________________________________
    """
    input_x = Input(input_shape)

    X = Conv2D(filters = 16, kernel_size = 3, activation = 'relu', padding = 'same')(input_x)
    X = AveragePooling2D(pool_size = 2)(X)
    X = Dropout(0.2)(X)
    X = Flatten()(X)
    
    X = Dense(32, activation = 'relu')(X)
    X = Dropout(0.5)(X)
    
    if is_multiclass:
        X = Dense(len(CLASSES), activation = 'softmax')(X)
    else:        
        X = Dense(1, activation = 'sigmoid')(X)
    
    model = Model(inputs = input_x, outputs = X,
                  name = get_model_name(MODEL_NAMES.CNN1_2D.value, is_multiclass, matrix_type))
    return model

def CNN1D_2xConvPool(input_shape, is_multiclass: bool, matrix_type) -> keras.engine.functional.Functional:
    """
    Model: "{data_input_type}/multiclass_CNN1D_2xConvPool"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 16 / 4, 16 / 4, 1)]       0         
    _________________________________________________________________
    reshape (Reshape)            (None, 256, 1)            0         
    _________________________________________________________________
    conv1d (Conv1D)              (None, 256, 16)           64        
    _________________________________________________________________
    average_pooling1d (AveragePo (None, 128, 16)           0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 128, 16)           784       
    _________________________________________________________________
    average_pooling1d_1 (Average (None, 64, 16)            0         
    _________________________________________________________________
    dropout (Dropout)            (None, 64, 16)            0         
    _________________________________________________________________
    flatten (Flatten)            (None, 1024)              0         
    _________________________________________________________________
    dense (Dense)                (None, 32)                32800     
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 32)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 8)                 264       
    =================================================================
    Total params: 33,912
    Trainable params: 33,912
    Non-trainable params: 0
    _________________________________________________________________
    
    Model: "{data_input_type}/binary_CNN1D_2xConvPool"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 16 / 4, 16 / 4, 1)]       0         
    _________________________________________________________________
    reshape (Reshape)            (None, 256, 1)            0         
    _________________________________________________________________
    conv1d (Conv1D)              (None, 256, 16)           64        
    _________________________________________________________________
    average_pooling1d (AveragePo (None, 128, 16)           0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 128, 16)           784       
    _________________________________________________________________
    average_pooling1d_1 (Average (None, 64, 16)            0         
    _________________________________________________________________
    dropout (Dropout)            (None, 64, 16)            0         
    _________________________________________________________________
    flatten (Flatten)            (None, 1024)              0         
    _________________________________________________________________
    dense (Dense)                (None, 32)                32800     
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 32)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 33        
    =================================================================
    Total params: 33,681
    Trainable params: 33,681
    Non-trainable params: 0
    _________________________________________________________________
    """
    input_x = Input(input_shape)

    X = Reshape((input_shape[0] * input_shape[1], 1))(input_x)
    X = Conv1D(filters = 16, kernel_size = 3, activation = 'relu', padding = 'same')(X)
    X = AveragePooling1D(pool_size = 2)(X)
    X = Conv1D(filters = 16, kernel_size = 3, activation = 'relu', padding = 'same')(X)
    X = AveragePooling1D(pool_size = 2)(X)
    X = Dropout(0.2)(X)
    X = Flatten()(X)
    
    X = Dense(32, activation = 'relu')(X)
    X = Dropout(0.5)(X)
    
    if is_multiclass:
        X = Dense(len(CLASSES), activation = 'softmax')(X)
    else:        
        X = Dense(1, activation = 'sigmoid')(X)
    
    model = Model(inputs = input_x, outputs = X,
                  name = get_model_name(MODEL_NAMES.CNN2_1D.value, is_multiclass, matrix_type))
    return model

def CNN1D_1xConvPool(input_shape, is_multiclass: bool, matrix_type) -> keras.engine.functional.Functional:
    """
    Model: "{data_input_type}/multiclass_CNN1D_1xConvPool"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 16 / 4, 16 / 4, 1)]       0         
    _________________________________________________________________
    reshape (Reshape)            (None, 256, 1)            0         
    _________________________________________________________________
    conv1d (Conv1D)              (None, 256, 16)           64        
    _________________________________________________________________
    average_pooling1d (AveragePo (None, 128, 16)           0         
    _________________________________________________________________
    dropout (Dropout)            (None, 128, 16)           0         
    _________________________________________________________________
    flatten (Flatten)            (None, 2048)              0         
    _________________________________________________________________
    dense (Dense)                (None, 32)                65568     
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 32)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 8)                 264       
    =================================================================
    Total params: 65,896
    Trainable params: 65,896
    Non-trainable params: 0
    _________________________________________________________________
    
    Model: "{data_input_type}/binary_CNN1D_1xConvPool"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 16 / 4, 16 / 4, 1)]       0         
    _________________________________________________________________
    reshape (Reshape)            (None, 256, 1)            0         
    _________________________________________________________________
    conv1d (Conv1D)              (None, 256, 16)           64        
    _________________________________________________________________
    average_pooling1d (AveragePo (None, 128, 16)           0         
    _________________________________________________________________
    dropout (Dropout)            (None, 128, 16)           0         
    _________________________________________________________________
    flatten (Flatten)            (None, 2048)              0         
    _________________________________________________________________
    dense (Dense)                (None, 32)                65568     
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 32)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 33        
    =================================================================
    Total params: 65,665
    Trainable params: 65,665
    Non-trainable params: 0
    _________________________________________________________________
    """
    input_x = Input(input_shape)

    X = Reshape((input_shape[0] * input_shape[1], 1))(input_x)
    X = Conv1D(filters = 16, kernel_size = 3, activation = 'relu', padding = 'same')(X)
    X = AveragePooling1D(pool_size = 2)(X)
    X = Dropout(0.2)(X)
    X = Flatten()(X)
    
    X = Dense(32, activation = 'relu')(X)
    X = Dropout(0.5)(X)
    
    if is_multiclass:
        X = Dense(len(CLASSES), activation = 'softmax')(X)
    else:        
        X = Dense(1, activation = 'sigmoid')(X)
    
    model = Model(inputs = input_x, outputs = X,
                  name = get_model_name(MODEL_NAMES.CNN1_1D.value, is_multiclass, matrix_type))
    return model

