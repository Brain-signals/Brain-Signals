
from tensorflow.keras import layers, optimizers, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import (TrueNegatives,
                                TruePositives,
                                FalseNegatives,
                                FalsePositives)

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

#first model 3dCNN
def initialize_model(width, length, depth,number_of_class,learning_rate=0.001):
    """Build a 3D convolutional neural network model."""

    model = Sequential()

    model.add(layers.Conv3D(
        filters=16,
        kernel_size=3,
        activation="relu",
        input_shape=(width, length, depth, 1)))
    model.add(layers.MaxPool3D(pool_size=2))

    model.add(layers.Conv3D(
        filters=32,
        kernel_size=2,
        activation="relu",))
    model.add(layers.MaxPool3D(pool_size=2))

    model.add(layers.Conv3D(
        filters=64,
        kernel_size=2,
        activation="relu",))
    model.add(layers.MaxPool3D(pool_size=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(units=360, activation="relu"))
    model.add(layers.Dense(units=60, activation="relu"))
    model.add(layers.Dense(units=number_of_class, activation="softmax"))

    adam_opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
      loss="categorical_crossentropy",
      optimizer=adam_opt,
      metrics=["accuracy"], # or TrueNegatives(), TruePositives(), FalseNegatives(), FalsePositives()
    )

    return model

def initialize_model_alzheimer(width, length, depth,learning_rate=0.001):
    """Build a 3D convolutional neural network model."""

    model = Sequential()

    model.add(layers.Conv3D(
        filters=16,
        kernel_size=3,
        activation="relu",
        input_shape=(width, length, depth, 1)))
    model.add(layers.MaxPool3D(pool_size=2))

    model.add(layers.Conv3D(
        filters=32,
        kernel_size=2,
        activation="tanh",))
    model.add(layers.MaxPool3D(pool_size=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(units=50, activation="tanh"))
    model.add(layers.Dense(units=1, activation="linear"))

    adam_opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
      loss="mse",
      optimizer=adam_opt,
      metrics=["mae"],
    )

    return model


def train_model(model,
                X_train,
                y_train,
                patience,
                epochs,
                monitor,
                batch_size,
                validation_split,
                verbose):


    es = EarlyStopping(patience=patience,
                       monitor=monitor,
                       restore_best_weights=True,
                       verbose=verbose)

    # Train the model, doing validation at the end of each epoch
    history=model.fit(
        X_train, y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=[es],
    )
    if not es.best_epoch:
        best_epoch_index = max(history.epoch)
        print('No best epoch found, the model will use last epoch as best\n'\
              'You should try to increase epochs or decrease patience')
    else:
        best_epoch_index = es.best_epoch

    return model, history, best_epoch_index
