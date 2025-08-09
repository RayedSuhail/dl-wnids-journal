# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 18:35:41 2023

@author: ahmad104
"""

import matplotlib.pyplot as plt
import pandas as pd

from data_loader import fetch_input_output
from helper_functions import DATA_TYPES

def plot_history_metric(history, metric, title, has_valid=True):
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.show()

def compare_dataset_shapes(is_multiclass: bool):
    _, train_outputs = fetch_input_output(DATA_TYPES.TRAINING.value, is_multiclass)
    _, test_outputs = fetch_input_output(DATA_TYPES.TESTING.value, is_multiclass)
    _, val_outputs = fetch_input_output(DATA_TYPES.VALIDATION.value, is_multiclass)
    
    unchanged_dataset = pd.read_csv('../datasets/combined_awid3_unchanged.csv')
    
    unchanged_samples = unchanged_dataset.shape[0]
    train_samples = train_outputs.shape[0]
    test_samples = test_outputs.shape[0]
    val_samples = val_outputs.shape[0]

    train_test_val_total = train_samples + test_samples + val_samples
    train_val_total = train_samples + val_samples
    
    
    print('\nTotal unchanged samples: ', unchanged_samples)
    print('\nTotal training samples: ', train_samples)
    print('\nTotal testing samples: ', test_samples)
    print('\nTotal validation samples: ', val_samples)
    
    print('\nSum of training, testing, and validation: ', train_test_val_total)
    print('\nSum of training, and validation: ', train_val_total)
    
    print('\nPercentages of final training and validation: \
          \n\t\t %.2f training and %.2f validation' % \
              (train_samples / train_val_total * 100, val_samples / train_val_total * 100))
    print('\nPercentages of initial training and testing: \
          \n\t\t %.2f initial training and %.2f testing' % \
              ((train_samples + val_samples) / train_test_val_total * 100, test_samples / train_test_val_total * 100))
    print('\nPercentages of final training, testing, and validation: \
          \n\t\t %.2f final training and %.2f testing and % .2f validation' % \
              (train_samples / train_test_val_total * 100, test_samples / train_test_val_total * 100, val_samples / train_test_val_total * 100))
    
    print('\nValue counts of outputs in training dataset is: ')
    print(pd.DataFrame(train_outputs).value_counts())
    
    print('\nValue counts of outputs in testing dataset is: ')
    print(pd.DataFrame(test_outputs).value_counts())
    
    print('\nValue counts of outputs in validation dataset is: ')
    print(pd.DataFrame(val_outputs).value_counts())

    print('\nRatio of outputs in training dataset is: ')
    print(pd.DataFrame(train_outputs).value_counts() / len(train_outputs) * 100)
    
    print('\nRatio of outputs in testing dataset is: ')
    print(pd.DataFrame(test_outputs).value_counts() / len(test_outputs) * 100)
    
    print('\nRatio of outputs in validation dataset is: ')
    print(pd.DataFrame(val_outputs).value_counts() / len(val_outputs) * 100)
    
    return

# =============================================================================
# print('--------------------BINARY--------------------')
# compare_dataset_shapes(False)
# =============================================================================
# Total unchanged samples:  3489017

# Total training samples:  1709596

# Total testing samples:  1046692

# Total validation samples:  732685

# Sum of training, testing, and validation:  3488973

# Sum of training, and validation:  2442281

# Percentages of final training and validation:           
# 		 70.00 training and 30.00 validation

# Percentages of initial training and testing:           
# 		 70.00 initial training and 30.00 testing

# Percentages of final training, testing, and validation:           
# 		 49.00 final training and 30.00 testing and  21.00 validation

# Value counts of outputs in training dataset is: 
# 0.0    1480519
# 1.0     229077
# dtype: int64

# Value counts of outputs in testing dataset is: 
# 0.0    906440
# 1.0    140252
# dtype: int64

# Value counts of outputs in validation dataset is: 
# 0.0    634509
# 1.0     98176
# dtype: int64

# Ratio of outputs in training dataset is: 
# 0.0    86.600518
# 1.0    13.399482
# dtype: float64

# Ratio of outputs in testing dataset is: 
# 0.0    86.600452
# 1.0    13.399548
# dtype: float64

# Ratio of outputs in validation dataset is: 
# 0.0    86.600517
# 1.0    13.399483
# dtype: float64

# =============================================================================
# print('--------------------MULTICLASS--------------------')
# compare_dataset_shapes(True)
# =============================================================================
# Total unchanged samples:  3489017

# Total training samples:  1709596

# Total testing samples:  1046692

# Total validation samples:  732685

# Sum of training, testing, and validation:  3488973

# Sum of training, and validation:  2442281

# Percentages of final training and validation:           
# 		 70.00 training and 30.00 validation

# Percentages of initial training and testing:           
# 		 70.00 initial training and 30.00 testing

# Percentages of final training, testing, and validation:           
# 		 49.00 final training and 30.00 testing and  21.00 validation

# Value counts of outputs in training dataset is: 
# 0.0    1480519
# 6.0      93983
# 7.0      51365
# 2.0      36815
# 5.0      24495
# 1.0      19081
# 3.0       2696
# 4.0        642
# dtype: int64

# Value counts of outputs in testing dataset is: 
# 0.0    906440
# 6.0     57541
# 7.0     31448
# 2.0     22539
# 5.0     14997
# 1.0     11683
# 3.0      1651
# 4.0       393
# dtype: int64

# Value counts of outputs in validation dataset is: 
# 0.0    634509
# 6.0     40279
# 7.0     22014
# 2.0     15777
# 5.0     10498
# 1.0      8178
# 3.0      1155
# 4.0       275
# dtype: int64

# Ratio of outputs in training dataset is: 
# 0.0    86.600518
# 6.0     5.497381
# 7.0     3.004511
# 2.0     2.153433
# 5.0     1.432795
# 1.0     1.116112
# 3.0     0.157698
# 4.0     0.037553
# dtype: float64

# Ratio of outputs in testing dataset is: 
# 0.0    86.600452
# 6.0     5.497415
# 7.0     3.004513
# 2.0     2.153356
# 5.0     1.432800
# 1.0     1.116183
# 3.0     0.157735
# 4.0     0.037547
# dtype: float64

# Ratio of outputs in validation dataset is: 
# 0.0    86.600517
# 6.0     5.497451
# 7.0     3.004565
# 2.0     2.153313
# 5.0     1.432812
# 1.0     1.116169
# 3.0     0.157639
# 4.0     0.037533
# dtype: float64