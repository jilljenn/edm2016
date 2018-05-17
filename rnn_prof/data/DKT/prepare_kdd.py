import pandas as pd
import numpy as np
import pickle
from collections import Counter


df = pd.read_csv('algebra_2005_2006_processed.csv', index_col=0).rename(columns={
    'AnonStudentId': 'user_idx',
    'Problem': 'item_idx',
    'KCs': 'concept_idx',
    'StepStartTime': 'time_idx',
    'CorrectFirstAttempt': 'correct'
})[['user_idx', 'item_idx', 'correct', 'time_idx', 'concept_idx']].sort('time_idx')

nb = Counter()
wins = []
fails = []
for user_id, skill_id, is_correct in np.array(df[['user_idx', 'concept_idx', 'correct']]):
    wins.append( nb[user_id, skill_id, 1])
    fails.append(nb[user_id, skill_id, 0])
    nb[user_id, skill_id, is_correct] += 1
df['wins'] = wins
df['fails'] = fails
print(df.head())

train_folds = pd.read_csv('KDD_train_fold.csv')
test_folds = pd.read_csv('KDD_test_fold.csv')

data_folds = []
for i_fold in range(1, 6):
    this_train_fold = train_folds.query('fold_id == @i_fold')['student_id']
    this_test_fold = test_folds.query('fold_id == @i_fold')['student_id']
    train_data = df.query('user_idx in @this_train_fold')
    test_data = df.query('user_idx in @this_test_fold')
    data_folds.append((train_data, test_data))

with open('kdd.pickle', 'wb') as f:
    pickle.dump(data_folds, f)
