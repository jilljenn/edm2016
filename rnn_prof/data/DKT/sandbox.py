from scipy.sparse import lil_matrix, coo_matrix, save_npz, load_npz, hstack, diags
from scipy.optimize import newton, brentq
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def onehotize(col, depth):
    nb_events = len(col)
    rows = list(range(nb_events))
    return coo_matrix(([1] * nb_events, (rows, col)), shape=(nb_events, depth))

def df_to_sparse(df, config, active_features):
    # Prepare sparse features
    X = {}
    X['users'] = onehotize(df['user_idx'], config['nb_users'])
    # X['users'] = coo_matrix((len(test_data), nb_users))  # This is equivalent for the test set (all zeroes)
    X['items'] = onehotize(df['item_idx'], config['nb_items'])
    X['skills'] = onehotize(df['concept_idx'], config['nb_skills'])
    X['wins'] = X['skills'].copy()
    X['wins'].data = df['wins']
    X['fails'] = X['skills'].copy()
    X['fails'].data = df['fails']
    X_train = hstack([X[agent] for agent in active_features]).tocsr()
    y_train = df['correct'].values
    return X_train, y_train

X = onehotize([1, 2, 4, 3, 2], 5)
Y = X.copy()
X.data = np.array([2, 3, 5, 9, 11])
# print(X.toarray())
# print(Y.toarray())

LAMBDA = 1e-3

def p(th, d):
    print('p', th, d)
    return 1 / (1 + np.exp(-(th - d)))

def dll(th, seq):
    s = -2 * LAMBDA * th
    for d, y in seq:
        s += y - p(th, d)
    return s

def f(th):
    return dll(th, SEQ)

def df(th):
    s = -2 * LAMBDA
    for d, y in SEQ:
        pj = p(th, d)
        s -= pj * (1 - pj)
    return s

# SEQ = [(3, 1), (4, 0)]
SEQ = [(3, 1)]
# print(newton(f, 1, fprime=df))
# print(brentq(f, -30, 30))
