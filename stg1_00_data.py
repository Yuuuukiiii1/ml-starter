import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

seed = 123


# --------------------------------------
# Example for multi class classification
# --------------------------------------

def load_train_data():
    """
    Customize it for your own purpose.

    :return: X : train inputs
    :return: y : train labels
    """
    mnist = fetch_mldata('MNIST original')
    X = mnist.data.astype('float64')
    y = mnist.target

    X = X.reshape(-1, 28, 28, 1) / 255

    if False:
        plt.imshow(X[0])
        plt.show()

    X_tr, X_test, y_tr, y_test = train_test_split(X, y,
                                                  stratify=y,
                                                  shuffle=True,
                                                  random_state=seed,
                                                  train_size=4000,
                                                  test_size=10000,
                                                  # test_size=0.2,
                                                  )
    return X_tr, y_tr


def load_test_data():
    """
    Customize it for your own purpose.
    There must be no labels in competition.

    :return: X : test inputs
    :return: y : test labels
    """
    mnist = fetch_mldata('MNIST original')
    X = mnist.data.astype('float64')
    y = mnist.target

    X = X.reshape(-1, 28, 28, 1) / 255

    if False:
        plt.imshow(X[0])
        plt.show()

    X_tr, X_test, y_tr, y_test = train_test_split(X, y,
                                                  stratify=y,
                                                  shuffle=True,
                                                  random_state=seed,
                                                  train_size=4000,
                                                  test_size=10000,
                                                  )
    return X_test, y_test


# ---------------------------------
# Example for binary classification
# ---------------------------------

def load_train_data2():
    """
    Customize it for your own purpose.

    :return: X : train inputs
    :return: y : train labels
    """
    mnist = fetch_mldata('MNIST original')
    X = mnist.data.astype('float64')
    y = mnist.target

    neg_mask = y == 0
    pos_mask = y == 1

    X = np.concatenate((
        X[neg_mask],
        X[pos_mask],
    )).reshape(-1, 28, 28, 1) / 255

    y = np.concatenate((
        y[neg_mask],
        y[pos_mask],
    ))

    X_tr, X_test, y_tr, y_test = train_test_split(X, y,
                                                  stratify=y,
                                                  shuffle=True,
                                                  random_state=seed,
                                                  train_size=4000,
                                                  test_size=10000,
                                                  )
    return X_tr, y_tr


def load_test_data2():
    """
    Customize it for your own purpose.

    :return: X : test inputs
    :return: y : test labels
    """
    mnist = fetch_mldata('MNIST original')
    X = mnist.data.astype('float64')
    y = mnist.target

    neg_mask = y == 0
    pos_mask = y == 1

    X = np.concatenate((
        X[neg_mask],
        X[pos_mask],
    )).reshape(-1, 28, 28, 1) / 255

    y = np.concatenate((
        y[neg_mask],
        y[pos_mask],
    ))

    X_tr, X_test, y_tr, y_test = train_test_split(X, y,
                                                  stratify=y,
                                                  shuffle=True,
                                                  random_state=seed,
                                                  train_size=4000,
                                                  test_size=10000,
                                                  )
    return X_test, y_test


# Debug
if __name__ == '__main__':
    load_train_data()
    load_test_data()
