import numpy as np

import stg1_00_data as stg1


# --------------------------------------
# Example for multi class classification
# --------------------------------------

def load_train_data():
    """
    Customize it for your own purpose.

    :return: X : train inputs
    :return: y : train labels
    """
    preds = [
        np.load('artifacts/stg1/01_cnn4l_train.npy'),
        np.load('artifacts/stg1/02_lr_train.npy'),
    ]
    # Average across bags
    preds = [
        np.mean(p, axis=0)
        for p in preds
    ]
    X = np.concatenate(preds, axis=1)

    _, y = stg1.load_train_data()

    return X, y


def load_test_data():
    """
    Customize it for your own purpose.
    There must be no labels in competition.

    :return: X : test inputs
    :return: y : test labels
    """
    preds = [
        np.load('artifacts/stg1/01_cnn4l_test.npy'),
        np.load('artifacts/stg1/02_lr_test.npy'),
    ]
    # Average across bags
    preds = [np.mean(p, axis=0) for p in preds]
    # Average across CVs
    preds = [np.mean(p, axis=0) for p in preds]

    X = np.concatenate(preds, axis=1)
    _, y = stg1.load_test_data()

    return X, y


# Debug
if __name__ == '__main__':
    load_train_data()
    load_test_data()
