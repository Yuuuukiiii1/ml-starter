from abc import abstractmethod, ABC
from typing import Type

import keras
import numpy as np
from hyperopt import STATUS_OK
from sklearn.model_selection import StratifiedKFold

from cv.ensemble import bagging
from cv.hpt import resume_trials


class ObjectiveCV(ABC):
    _X_pseudo = None
    _X_pseudo_aux = None
    _y_pseudo = None

    def __init__(self, X, X_aux, y, seed=1, pipe=None, n_splits=5, n_class=None):
        self._X = X
        self._X_aux = X_aux
        self._y = y
        self._seed = seed
        self._pipe = pipe
        self._n_splits = n_splits
        self._n_class = n_class

    @abstractmethod
    def fit(self, sp, X_tr, X_tr_aux, y_tr, X_val, X_val_aux, y_val):
        pass

    def pre_fit(self, space):
        pass

    def _pipe_fit(self, X, y):
        if self._pipe:
            self._pipe.fit(X, y)

    def _pipe_transform(self, X):
        if self._pipe:
            return self._pipe.transform(X)
        return X

    def set_pseudo(self, X, X_aux, y):
        self._X_pseudo = X
        self._X_pseudo_aux = X_aux
        self._y_pseudo = y

    # noinspection PyCallingNonCallable
    # noinspection PyTupleAssignmentBalance
    def cv(self, space, X_test=None, X_test_aux=None):
        X, X_aux, y = self._X, self._X_aux, self._y

        scores_tr = []
        scores_val = []
        test_preds = []

        if self._n_class:
            blended = np.zeros((len(y), self._n_class)).astype(float)
        else:
            blended = np.zeros((len(y), 1)).astype(float)

        skf = StratifiedKFold(n_splits=self._n_splits, shuffle=True, random_state=self._seed)

        for n_th, (train_idx, val_idx) in enumerate(skf.split(np.zeros_like(y), y)):
            if self._n_class:
                y_cat = keras.utils.to_categorical(y, self._n_class)
                X_tr, y_tr = X[train_idx], y_cat[train_idx]
                X_val, y_val = X[val_idx], y_cat[val_idx]
            else:
                X_tr, y_tr = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

            X_tr_aux = X_aux[train_idx]
            X_val_aux = X_aux[val_idx]

            # Add pseudo labeled data into training data
            if self._X_pseudo is not None:
                X_tr = np.concatenate((X_tr, self._X_pseudo))
                y_tr = np.concatenate((y_tr, self._y_pseudo))
                X_tr_aux = np.concatenate((X_tr_aux, self._X_pseudo_aux))

            # Preprocess
            self.pre_fit(space)
            self._pipe_fit(X_tr_aux, y_tr)
            X_tr_aux = self._pipe_transform(X_tr_aux)
            X_val_aux = self._pipe_transform(X_val_aux)

            # Train
            model = self.fit(space, X_tr, X_tr_aux, y_tr, X_val, X_val_aux, y_val)
            # model.save('model_%s.h5' % n_th)

            # Evaluate
            score_tr, _ = model.evaluate([X_tr, X_tr_aux], y_tr, verbose=0)
            score_val, _ = model.evaluate([X_val, X_val_aux], y_val, verbose=0)

            if X_test is not None:
                preds = model.predict([X_test, self._pipe_transform(X_test_aux)])
                test_preds.append(preds)

            scores_tr.append(score_tr)
            scores_val.append(score_val)

            blended[val_idx] = model.predict([X_val, X_val_aux])

        test_preds = np.array(test_preds)

        return np.mean(scores_tr), np.mean(scores_val), blended, test_preds

    def objective(self, sp):
        score_tr, score_val, _, _ = self.cv(sp)
        print("---", score_tr, score_val, sp)

        return {
            'loss': score_val,
            'status': STATUS_OK,
            'score_tr': score_tr,
            'score_val': score_val,
        }


class CmdFactory:
    def __init__(self, cls: Type[ObjectiveCV]):
        self._cls = cls

    def hpt(self, train_data, space, trials_file, max_iter, seed,
            pipe=None, n_class=None):
        X, X_aux, y = train_data

        if X_aux is None:
            # set empty array to avoid error
            X_aux = np.zeros((X.shape[0], 0))

        cv = self._cls(X, X_aux, y, seed=seed, pipe=pipe, n_class=n_class)

        resume_trials(cv, space, trials_file, max_iter)

    def pred(self, train_data, params, out_tr, out_test, n_bags, seed,
             test_data=(None, None), pipe=None, n_class=None):
        X, X_aux, y = train_data
        X_test, X_test_aux = test_data

        if X_aux is None:
            # set empty array to avoid error
            X_aux = np.zeros((X.shape[0], 0))
            X_test_aux = np.zeros((X_test.shape[0], 0))

        def pred_nth_bagging(n):
            cv = self._cls(X, X_aux, y, seed=seed + n, pipe=pipe, n_class=n_class)
            score_tr, score_val, train_preds, test_preds = cv.cv(params,
                                                                 X_test=X_test,
                                                                 X_test_aux=X_test_aux)
            print('Score: ', score_tr, score_val)
            return train_preds, test_preds

        train_bag_preds, test_bag_preds = bagging(n_bags=n_bags,
                                                  pred_fn=pred_nth_bagging)

        np.save(out_tr, train_bag_preds)
        np.save(out_test, test_bag_preds)
