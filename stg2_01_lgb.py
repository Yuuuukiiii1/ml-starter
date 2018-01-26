import os

import lightgbm as lgb
import numpy as np
from hyperopt import hp
from hyperopt.pyll.base import scope

from cv import parse_args
from cv.no_dnn import ObjectiveCV, CmdFactory
from stg2_00_data import load_train_data, load_test_data

np.random.seed(777)


class _CV(ObjectiveCV):
    def make_predictive(self, space, X_tr, y_tr, X_val, y_val):
        params = {
            'learning_rate': space['learning_rate'],
            'max_depth': space['max_depth'],
            'num_leaves': space['num_leaves'],
            'feature_fraction': space['feature_fraction'],
            'bagging_fraction': space['bagging_fraction'],
            'max_bin': space['max_bin'],
            # 'objective': 'binary',
            # 'metric': 'binary_logloss',
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'num_class': 10,
            'verbose': -1,
        }
        lgb_tr = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_tr, free_raw_data=False)

        model = lgb.train(params,
                          lgb_tr,
                          num_boost_round=2000,
                          valid_sets=lgb_val,
                          early_stopping_rounds=100,
                          verbose_eval=False)

        return lambda d: model.predict(d, num_iteration=model.best_iteration)


def main():
    base_name = 'stg2/01_lgb'
    seed = 201_000

    args = parse_args()
    cmd = CmdFactory(_CV)

    if not os.path.exists('artifacts/stg1'):
        os.makedirs('artifacts/stg1')

    X, y = load_train_data()
    X_test, _ = load_test_data()

    if args.type == 'hpt':
        space = {
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
            'max_depth': scope.int(hp.quniform("max_depth", 1, 7, 1)),
            'num_leaves': scope.int(hp.quniform('num_leaves', 2, 20, 1)),
            'feature_fraction': hp.uniform('feature_fraction', 0.4, 0.8),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.4, 1.0),
            'max_bin': scope.int(hp.quniform("max_bin", 400, 700, 1)),
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
        params = {
            'learning_rate': 0.029191104512834937,
            'max_depth': 4,
            'num_leaves': 9,
            'feature_fraction': 0.7115094613220588,
            'bagging_fraction': 0.4897073688073762,
            'max_bin': 639,
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
