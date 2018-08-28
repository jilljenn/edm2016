import pandas as pd
import numpy as np
import pickle
import yaml
from collections import Counter


renaming = {
    'user': 'user',
    'problem_id': 'item',
    'outcome': 'correct',
    'time': 'time_idx',
}
df = pd.read_csv('movie100k_train.csv').rename(columns=renaming)
df['skill'] = df['item']
df_train = df[['user', 'item', 'correct', 'time_idx', 'skill']].sort('time_idx')

df = pd.read_csv('movie100k_test.csv').rename(columns=renaming)
df['skill'] = df['item']
df_test = df[['user', 'item', 'correct', 'time_idx', 'skill']].sort('time_idx')

# nb = Counter()
# wins = []
# fails = []
# for user_id, skill_id, is_correct in np.array(df[['user', 'skill', 'correct']]):
#     wins.append( nb[user_id, skill_id, 1])
#     fails.append(nb[user_id, skill_id, 0])
#     nb[user_id, skill_id, is_correct] += 1
# df['wins'] = wins
# df['fails'] = fails

# with open('/Users/jilljenn/code/ktm/data/assistments09/config.yml', 'w') as f:
#     yaml.safe_dump({
#         'nb_users': int(1 + df['user'].max()),
#         'nb_items': int(1 + df['item'].max()),
#         'nb_skills': int(1 + df['skill'].max()),
#     }, f, default_flow_style=False)

# print(df[['user', 'item', 'skill', 'correct', 'wins', 'fails']].to_csv('assist09.csv', index=False))
# .to_csv('assist09.csv', index=False)

# train_folds = pd.read_csv('Assist_09_train_fold.csv')
# test_folds = pd.read_csv('Assist_09_test_fold.csv')

data_folds = []
for i_fold in range(1, 2):
    data_folds.append((df_train, df_test))

with open('movie100k-1fold.pickle', 'wb') as f:
    pickle.dump(data_folds, f)
