# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 12:58:11 2023

@author: ahmad104
"""

import os
import glob
import pandas as pd
import traceback

from sklearn.model_selection import train_test_split

from helper_functions import get_dataset_filename_start, FEATURES_LIST

AWID3_DATASET_ROOT = '../datasets/AWID3/*_Same'
MULTICLASS_FLAG = False  # Set to True if you want to use multiclass labels, False for binary classification

def update_temp_dataset():
    if not os.path.exists('../datasets/AWID3_temp/'):
        os.makedirs('../datasets/AWID3_temp/')
    all_folders = glob.glob(AWID3_DATASET_ROOT)
    
    successful_file_writes = []
    
    for folder in all_folders:
        try:
            all_files_in_dir = glob.glob(f'{folder}/*.csv')
            df = pd.concat(map(pd.read_csv, all_files_in_dir), ignore_index=True)
            normal_reduced = df.groupby(df.columns[-1]).sample(frac=0.2, random_state=42)[df['Label'] == 'Normal']
            attacks_only = df[df['Label'] != 'Normal']
            reduced_df = pd.concat([normal_reduced, attacks_only], ignore_index=True)
            csvFileName = folder.split('\\')[-1].split('.')[-1]
            csvFileName = f'reduced_{csvFileName}'
            reduced_df.to_csv(f'../datasets/AWID3_temp/{csvFileName}.csv', index=False, na_rep='NaN')
            successful_file_writes.append(csvFileName)
            del df, normal_reduced, attacks_only, reduced_df
        except:
            print(f'Failed for {folder}', traceback.format_exc())
    return successful_file_writes

def combine_temp_dataset():
    all_csv_files = glob.glob('../datasets/AWID3_temp/*.csv')
    df = pd.concat(map(pd.read_csv, all_csv_files), ignore_index=True)
    df.to_csv('../datasets/combined_awid3_unchanged.csv', index=False, na_rep='NaN')
    
    return df

def get_antsignal_value(x):
    # Get maximum value in the radiotap.dbm_antsignal feature
    x = str(x)
    if len(x.split('-')) > 2:
        values = [i for i in x.split('-') if i]
        value = min(values)
        return int(f'-{value}')
    else:
        return int(float(x))

def update_train_test_val(is_multiclass=False):
    df = pd.read_csv('../datasets/combined_awid3_unchanged.csv')
    
    base_filename = get_dataset_filename_start(is_multiclass)
    
    if is_multiclass:
        df['Label'] = df['Label'].map(
            {'Normal': 0, 'Deauth': 1, 'Disas': 2, '(Re)Assoc': 3, 'RogueAP': 4,
             'Krack': 5, 'Kr00k': 6, 'Kr00K': 6, 'Evil_Twin': 7})
    else:
        df['Label'] = df['Label'].map(
                {'Normal': 0, 'Kr00k': 1, 'Kr00K': 1, 'Evil_Twin': 1, 'Disas': 1, 'Krack': 1,
                  'Deauth': 1, '(Re)Assoc': 1, 'RogueAP': 1})
    
    df.rename(columns={'Label': 'attack_map'}, inplace=True)
    
    df = df[FEATURES_LIST]
    df.dropna(inplace=True)
    df['wlan.fc.ds'] = df['wlan.fc.ds'].apply(int, base=16)
    df['radiotap.present.tsft'] = df['radiotap.present.tsft'].map({'1-0-0': 1, '0-0-0': 0})
    df['radiotap.dbm_antsignal'] = df['radiotap.dbm_antsignal'].apply(get_antsignal_value)
    
    X = df.drop('attack_map', axis=1).copy()
    y = df['attack_map']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, stratify=y_train, random_state=42)
    
    X_train.to_csv(f'{base_filename}_combined_awid3_train_input.csv', index=False)
    X_test.to_csv(f'{base_filename}_combined_awid3_test_input.csv', index=False)
    X_val.to_csv(f'{base_filename}_combined_awid3_val_input.csv', index=False)
    y_train.to_csv(f'{base_filename}_combined_awid3_train_output.csv', index=False)
    y_test.to_csv(f'{base_filename}_combined_awid3_test_output.csv', index=False)
    y_val.to_csv(f'{base_filename}_combined_awid3_val_output.csv', index=False)
    
    return (X_train, X_test, X_val, y_train, y_test, y_val)

if __name__ == '__main__':
    # Comment out update_temp_dataset() and combine_temp_dataset() once you have run it once
    # This is to avoid reprocessing the same data multiple times
    update_temp_dataset()
    combine_temp_dataset()
    update_train_test_val(is_multiclass=MULTICLASS_FLAG)