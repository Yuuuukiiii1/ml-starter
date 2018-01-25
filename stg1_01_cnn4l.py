import os

import keras
import numpy as np
from hyperopt import hp
from hyperopt.pyll import scope
from keras import Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from cv import parse_args
from cv.dnn import ObjectiveCV, CmdFactory
from stg1_data import load_train_data, load_test_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(777)


class _CV(ObjectiveCV):
    def fit(self, sp, X_tr, X_tr_aux, y_tr, X_val, X_val_aux, y_val):
        keras.backend.clear_session()

        print(X_tr.shape)

        input_img = Input(shape=(X_tr.shape[1:]))
        # We don't use it in this CNN.
        input_aux = Input(shape=[X_tr_aux.shape[1]])

        x = Conv2D(32, kernel_size=(3, 3))(input_img)
        x = LeakyReLU()(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = Dropout(0.1)(x)

        x = Conv2D(64, kernel_size=(3, 3))(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = Dropout(0.4)(x)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = LeakyReLU()(x)
        x = Dropout(sp['dropout1'])(x)

        x = Dense(10, activation='softmax')(x)

        model = Model([input_img, input_aux], x)
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=Adam(lr=sp['lr']),
                      metrics=['accuracy'])

        batch_size = sp['batch_size']

        generator = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=20,
            zoom_range=0.2,
            shear_range=0.2,
        )
        iter_ = generator.flow(
            X_tr,
            range(len(X_tr)),
            batch_size=batch_size,
            seed=123,
        )

        def datagen():
            while True:
                X_batch, idx = iter_.next()
                yield [X_batch, X_tr_aux[idx]], y_tr[idx]

        best_weights = '.best_weights.h5'
        model.fit_generator(
            generator=datagen(),
            steps_per_epoch=int(X_tr.shape[0] / batch_size),
            # epochs=50,
            epochs=1,
            verbose=2,
            validation_data=([X_val, X_val_aux], y_val),
            callbacks=[
                EarlyStopping(monitor='val_loss',
                              patience=10,
                              verbose=1),
                ModelCheckpoint(best_weights,
                                save_best_only=True,
                                monitor='val_loss'),
            ])
        model.load_weights(filepath=best_weights)

        return model


def main():
    base_name = 'stg1/01_cnn4l'
    seed = 101_000

    args = parse_args()
    cmd = CmdFactory(_CV)

    X, y = load_train_data()
    X_test, _ = load_test_data()

    if not os.path.exists('artifacts/stg1'):
        os.makedirs('artifacts/stg1')

    if args.type == 'hpt':
        space = {
            'lr': hp.loguniform('lr', np.log(1e-5), np.log(1e-3)),
            'batch_size': scope.int(hp.quniform('batch_size', 8, 64, 8)),
            'dropout1': hp.uniform('dropout1', 0, 1),
        }
        cmd.hpt(
            train_data=(X, None, y),
            space=space,
            seed=seed,
            trials_file='artifacts/{}_trials.pickle'.format(base_name),
            max_iter=args.max_iter,
            n_class=10,
        )
    elif args.type == 'pred':
        params = {
            'lr': 0.0004302720418964592,
            'batch_size': 64,
            'dropout1': 0.1,
        }
        cmd.pred(
            train_data=(X, None, y),
            test_data=(X_test, None),
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
