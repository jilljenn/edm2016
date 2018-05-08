import pandas as pd
import numpy as np
import pickle


df = pd.read_csv('skill_builder_data_processed.csv', index_col=0).rename(columns={
    'user_id': 'user_idx',
    'problem_id': 'item_idx',
    'skill_id': 'concept_idx',
    'order_id': 'time_idx',
    # 'correct': correct
})[['user_idx', 'item_idx', 'correct', 'time_idx', 'concept_idx']]
print(df.head())

train_folds = pd.read_csv('Assist_09_train_fold.csv')
test_folds = pd.read_csv('Assist_09_test_fold.csv')

data_folds = []
for i_fold in range(1, 6):
    this_train_fold = train_folds.query('fold_id == @i_fold')['student_id']
    this_test_fold = test_folds.query('fold_id == @i_fold')['student_id']
    train_data = df.query('user_idx in @this_train_fold')
    test_data = df.query('user_idx in @this_test_fold')
    data_folds.append((train_data, test_data))

with open('assist09.pickle', 'wb') as f:
    pickle.dump(data_folds, f)
