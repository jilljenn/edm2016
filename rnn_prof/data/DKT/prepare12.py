import pandas as pd
import numpy as np
import pickle
from collections import Counter


df = pd.read_csv('2012-2013_processed.csv', index_col=0).rename(columns={
    'user_id': 'user_idx',
    'problem_id': 'item_idx',
    'skill': 'concept_idx',
    # 'correct': correct
})
df['time_idx'] = df.index
df = df[['user_idx', 'item_idx', 'correct', 'time_idx', 'concept_idx']]

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

train_folds = pd.read_csv('Assist_12_train_fold.csv')
test_folds = pd.read_csv('Assist_12_test_fold.csv')

data_folds = []
for i_fold in range(1, 6):
    this_train_fold = train_folds.query('fold_id == @i_fold')['student_id']
    this_test_fold = test_folds.query('fold_id == @i_fold')['student_id']
    train_data = df.query('user_idx in @this_train_fold')
    test_data = df.query('user_idx in @this_test_fold')
    data_folds.append((train_data, test_data))

with open('assist12.pickle', 'wb') as f:
    pickle.dump(data_folds, f)
