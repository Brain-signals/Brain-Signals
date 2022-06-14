
from tensorflow.keras import Model, layers, optimizers, Sequential
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.metrics import (TrueNegatives,
                                TruePositives,
                                FalseNegatives,
                                FalsePositives)

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
    model.add(layers.Dense(units=80, activation="relu"))
    model.add(layers.Dense(units=40, activation="relu"))
    model.add(layers.Dense(units=20, activation="relu"))
    model.add(layers.Dense(units=number_of_class, activation="softmax"))

    adam_opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
      loss="categorical_crossentropy",
      optimizer=adam_opt,
      metrics=["accuracy", TrueNegatives(), TruePositives(), FalseNegatives(), FalsePositives()],
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


def train_model(model: Model,
                X_train: np.ndarray,
                y_train: np.ndarray,
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

    return model, history
