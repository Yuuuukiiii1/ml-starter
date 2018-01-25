import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

seed = 123


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
                                                  train_size=5000,
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
                                                  train_size=5000,
                                                  test_size=10000,
                                                  )
    return X_test, y_test


if __name__ == '__main__':
    load_train_data()
    load_test_data()
