### External imports ###

import numpy as np
import os
import pickle
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, optimizers, Sequential
from tensorflow.keras.callbacks import EarlyStopping


### Internal imports ###

from brainsignals.utils import create_id, time_print, check_balance
from brainsignals.preprocess_class import Preprocessor


### Classes ###

class Model:

    def __init__(self):
        pass


    def initialize_model(self, preprocessor, learning_rate = 0.0005):

        self.dimensions = preprocessor.dimensions
        self.number_of_classes = preprocessor.number_of_classes
        self.diagnostics = preprocessor.diagnostics
        self.slicing_bot = preprocessor.slicing_bot
        self.slicing_top = preprocessor.slicing_top
        self.files_used = preprocessor.files_used
        self.learning_rate = learning_rate

        input_shape = list(self.dimensions).copy()
        input_shape.append(1)

        model = Sequential()

        model.add(layers.Conv3D(filters=16, kernel_size=3, activation="relu", input_shape=input_shape))
        model.add(layers.MaxPool3D(pool_size=2))

        model.add(layers.Conv3D(filters=32, kernel_size=2, activation="relu",))
        model.add(layers.MaxPool3D(pool_size=2))

        model.add(layers.Conv3D(filters=64, kernel_size=2, activation="relu",))

        model.add(layers.Flatten())
        model.add(layers.Dense(units=120, activation="relu"))
        model.add(layers.Dense(units=40, activation="relu"))
        model.add(layers.Dense(units=20, activation="relu"))
        model.add(layers.Dense(units=self.number_of_classes, activation="softmax"))

        adam_opt = optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(loss="categorical_crossentropy", optimizer=adam_opt, metrics=["accuracy"])

        self.model = model

        pass



    def train_model(self, X, y, epochs, patience,
                    monitor, batch_size,
                    validation_split = 0.3,
                    verbose = 2):

        start = time.perf_counter()

        try:
            self.dimensions # just to check if model has been initialized

            self.epochs = epochs
            self.patience = patience
            self.monitor = monitor
            self.batch_size = batch_size
            self.validation_split = validation_split

            es = EarlyStopping(patience = patience,
                               monitor = monitor,
                               verbose = verbose,
                               restore_best_weights = True)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_split, shuffle=True)

            history = self.model.fit(X_train, y_train,
                                     validation_data=(X_test,y_test), epochs=epochs,
                                     batch_size=batch_size, verbose=verbose,
                                     callbacks=[es], use_multiprocessing=True)

            if not es.best_epoch:
                self.best_epoch = max(history.epoch)
                print('No best epoch found, the model will use last epoch as best\n'\
                      'You should try to increase epochs or decrease patience')
            else:
                self.best_epoch = es.best_epoch

            end = time.perf_counter()

            if verbose > 0:
                print(f'Model has been trained in {time_print(start,end)}.')

        except AttributeError:
            print('ERROR : This model has not been initialized.')

        pass


    def save_model(self, creator_comment = None):

        try:
            self.best_epoch # just to check if the model has been trained

            self.creator_comment = creator_comment
            self.model_id = create_id()
            self.model_creator = os.environ.get('USER')
            model_path = os.path.join(os.environ.get('LOCAL_REGISTRY_PATH'), self.model_id)
            with open(model_path, 'wb') as f:
                pickle.dump(self, f)

        except AttributeError:
            print('ERROR : Model not saved.\n'\
                  'It seems this model has not been trained yet. You can only save trained models.')

        except FileNotFoundError:
            print('ERROR : Model not saved.\n'\
                  'Registry folder not found, make sure you updated LOCAL_REGISTRY_PATH in .env file.')

        pass



    def load_model(self, model_id):

        model_id = model_id[-20:]
        model_path = os.path.join(os.environ.get('LOCAL_REGISTRY_PATH'), model_id)

        try:
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            print(f'Model loaded from {model_path}')
            if loaded_model.creator_comment:
                print(loaded_model.creator_comment)
            return loaded_model

        except FileNotFoundError:
            print(f'ERROR : Model not found at {model_path}.\n'\
                  'Make sure you updated LOCAL_REGISTRY_PATH in .env file.')
            pass


    def predict(self, volume):
        vol_proc = Preprocessor().initialize_from_model(self).transform_vol(volume)
        vol_proc = np.array([vol_proc])

        y_pred = np.round(self.model.predict(vol_proc), 1)

        preds = {}
        for n in range(len(y_pred[0])):
            preds[self.diagnostics[n]] = y_pred[0][n]

        return preds


    def display_model(self):
        # display all the model attributes formated
        pass
