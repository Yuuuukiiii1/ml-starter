import os

import numpy as np
from hyperopt import hp
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from cv import parse_args
from cv.no_img import ObjectiveCV, CmdFactory
from stg1_data import load_train_data, load_test_data

np.random.seed(777)


class _CV(ObjectiveCV):
    def make_predictive(self, space, X_tr, y_tr, X_val, y_val):
        clf = LogisticRegression(C=space['C'],
                                 penalty=space['penalty'],
                                 multi_class='multinomial',
                                 solver='saga',
                                 tol=0.1,
                                 )
        clf.fit(X_tr, y_tr)

        # return lambda d: clf.predict_proba(d)[:, 1]
        # return lambda d: clf.predict_log_proba(d)[:, 1]
        return lambda d: clf.predict_proba(d)

    def pre_fit(self, space):
        if space['scaler'] == 'standard':
            self._pipe = Pipeline([
                ('scaler', StandardScaler()),
            ])
        else:
            self._pipe = Pipeline([
                ('scaler', MinMaxScaler()),
            ])


def main():
    base_name = 'stg0/02_lr'
    seed = 102_000

    args = parse_args()
    cmd = CmdFactory(_CV)

    if not os.path.exists('artifacts/stg0'):
        os.makedirs('artifacts/stg0')

    X, y = load_train_data()
    X_test, _ = load_test_data()

    X = X.reshape((X.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    if args.type == 'hpt':
        space = {
            'C': hp.loguniform('C', np.log(1e-3), np.log(1e-1)),
            'penalty': hp.choice('penalty', ['l2', 'l1']),
            'scaler': hp.choice('scaler', ['minmax', 'standard']),
        }
        cmd.hpt(train_data=(X, y),
                space=space,
                seed=seed,
                trials_file='artifacts/{}_trials.pickle'.format(base_name),
                max_iter=args.max_iter,
                steps=args.trial_steps,
                n_class=10,
                )
    elif args.type == 'pred':
        # 0.28481185045504903 / 0.35116803537013547
        params = {
            'C': 0.09020209050071439,
            'penalty': 'l2',
            'scaler': 'minmax',
        }
        cmd.pred(train_data=(X, y),
                 test_data=X_test,
                 params=params,
                 seed=seed,
                 out_tr='artifacts/{}_train.npy'.format(base_name),
                 out_test='artifacts/{}_test.npy'.format(base_name),
                 n_bags=args.n_bags,
                 n_class=10,
                 )
    else:
        raise ValueError('type must be hpt or pred.')


if __name__ == '__main__':
    main()
