# coding=utf8
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sandbox import onehotize, sigmoid, df_to_sparse
from scipy.sparse import coo_matrix, hstack
from scipy.optimize import brentq
import pandas as pd
import numpy as np
import argparse
import logging
import os.path
import random
import pickle
import yaml
import sys


parser = argparse.ArgumentParser(description='Run PFA and variants')
parser.add_argument('--dataset', type=str, nargs='?', default='assist09')
options = parser.parse_args()

# Config
active_features = ['users', 'skills', 'wins', 'fails']
# active_features = ['skills', 'wins', 'fails']  # PFA
suffix = ''.join(category[0] for category in active_features)

# Files
DATA = options.dataset
logging.warning('Dataset: %s', DATA)
config_file = os.path.join(DATA, 'config.yml')

with open('{:s}.pickle'.format(DATA), 'rb') as f:
    data_folds = pickle.load(f)

# Get important config values
train_data, test_data = data_folds[0]

if os.path.isfile(config_file):
    with open(config_file) as f:
        config = yaml.load(f)
else:
    train_max = train_data.max()[['user_idx', 'item_idx', 'concept_idx']]
    test_max = test_data.max()[['user_idx', 'item_idx', 'concept_idx']]
    nb_users, nb_items, nb_skills = 1 + np.column_stack((train_max.values, test_max.values)).max(axis=1)
    config = {'nb_users': int(nb_users), 'nb_items': int(nb_items), 'nb_skills': int(nb_skills)}
    with open(config_file, 'w') as f:
        yaml.safe_dump(config, f)

LAMBDA = 0.5  # 1e-3

def f(th, logits, y_valid):
    # print('test', th)
    # print('y', y_valid)
    # print('log', logits)
    # print((y_valid - 1 / (1 + np.exp(th - logits))))
    return (y_valid - sigmoid(th + logits)).sum() - 2 * LAMBDA * th

auc_folds = []
rmse_folds = []
for fold_num, (train_data, test_data) in enumerate(data_folds, start=1):
    test_data = test_data.sort(['user_idx', 'time_idx']).reset_index()

    X_train, y_train = df_to_sparse(train_data, config, active_features)
    X_test, y_test = df_to_sparse(test_data, config, active_features)

    # Train weights
    # trained_file = os.path.join(DATA, 'weights{:d}-{:s}.npy'.format(fold_num, suffix))
    model_file = os.path.join(DATA, 'logreg{:d}-{:s}.pickle'.format(fold_num, suffix))
    if os.path.isfile(model_file):
        logging.warning('Model already trained, loading it')
        with open(model_file, 'rb') as f:
            logreg = pickle.load(f)
        print(logreg.get_params())
        # weights = np.load(trained_file).reshape(-1)
    else:
        logging.warning('Model not found. Training…')
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        # print(logreg.get_params())
        # weights = logreg.coef_.reshape(-1)
        # np.save(trained_file, weights)
        with open(model_file, 'wb') as f:
            pickle.dump(logreg, f)
        print('done')
    weights = logreg.coef_.reshape(-1)
    bias = logreg.intercept_

    # Test
    logits_test = X_test.dot(weights) + bias
    # print(sigmoid(logits_test[:5]))
    # print(logreg.predict_proba(X_test[:5]))

    # Test on new users
    print(test_data.head(5))

    if 'u' not in suffix:
        train_pred = logreg.predict_proba(X_train)[:, 1]
        print('Train ACC', logreg.score(X_train, y_train))
        print('Train AUC', roc_auc_score(y_train, train_pred))

        test_pred = logreg.predict_proba(X_test)[:, 1]
        print('Test ACC', logreg.score(X_test, y_test))
        test_auc = roc_auc_score(y_test, test_pred)
        print('Test AUC', test_auc)
        test_rmse = mean_squared_error(y_test, test_pred) ** 0.5
        print('Test RMSE', test_rmse)
        auc_folds.append(test_auc)
        rmse_folds.append(test_rmse)
        continue

    test_pred = []

    user_considered = None
    for index in range(len(test_data)):
        current_user = test_data.iloc[index]['user_idx']
        if current_user != user_considered:
            if random.random() < 1./500:
                logging.warning('%d: %s %d %%', index, 'change', index * 100. / len(test_data))
            user_considered = current_user
            start = index
            t = 0
        else:
            t += 1
        # start_id visible_id
        X_valid = X_test[start:start + t]
        y_valid = y_test[start:start + t]
        logits_valid = logits_test[start:start + t]
        # print('was', logits)
        # print('p', 1 / (1 + np.exp(-logits)))
        # print('but', y_valid)

        if True:#t == 0:
            th_opt = 0
        else:
            # print('points', f(-300, logits, y_valid), f(-6, logits, y_valid), f(0, logits, y_valid), f(300, logits, y_valid))
            th_opt = brentq(lambda th: f(th, logits_valid, y_valid), -30, 30)
        # print('found', th_opt)
        # print('is now', th_opt - logits)
        # print('p', 1 / (1 + np.exp(-(th_opt + logits))))
        # print('so', y_valid)
        pred = sigmoid(th_opt + logits_test[index])
        # print(th_opt, pred, y_test[index])
        test_pred.append(pred)
    # print(test_data['correct'].values[:MAX])
    # print(test_pred)
    test_auc = roc_auc_score(y_test, test_pred)
    test_rmse = mean_squared_error(y_test, test_pred)
    auc_folds.append(test_auc)
    rmse_folds.append(test_rmse)
    print('AUC', test_auc)

logging.warning('Dataset: %s', DATA)
logging.warning('Final AUC: %f', np.mean(auc_folds))
logging.warning('Final RMSE: %f', np.mean(rmse_folds))
