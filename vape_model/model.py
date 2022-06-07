
from tensorflow.keras import Model, layers, Input, optimizers, Sequential
from tensorflow.keras.layers import Reshape
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.model import train_test_split

from colorama import Fore, Style

def split_train_test(X, y_encoded):
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.30)
    return X_train, X_test, y_train, y_test

def encoding_y(y):
    enc = OneHotEncoder(sparse = False)
    enc.fit(y[['diagnostic']])
    y_encoded = enc.transform(y[['diagnostic']]).astype('int8')

#first model 3dCNN
def initialize_model(length, width, height):
    """Build a 3D convolutional neural network model."""

    model = Sequential()

    model.add(layers.Conv3D(
        filters=32,
        kernel_size=3,
        activation="relu",
        input_shape=(width, length, depth, 1)))
    model.add(layers.MaxPool3D(pool_size=2))

    model.add(layers.Conv3D(
        filters=32,
        kernel_size=3,
        activation="relu",
        input_shape=(width, length, depth, 1)))
    model.add(layers.MaxPool3D(pool_size=2))

    model.add(layers.Conv3D(
        filters=32,
        kernel_size=3,
        activation="relu",))
    model.add(layers.MaxPool3D(pool_size=2))

    model.add(layers.Flatten())
    #model.add(layers.Dense(units=10, activation="relu"))
    model.add(layers.Dense(units=100, activation="relu"))
    model.add(layers.Dense(units=50, activation="relu"))
    model.add(layers.Dense(units=20, activation="relu"))
    model.add(layers.Dense(units=3, activation="softmax"))

    model.compile(
      loss="categorical_crossentropy",
      optimizer='adam',
      metrics=["accuracy"],
    )

    return model

def train_model(model: Model,
                X_train: np.ndarray,
                y_train: np.ndarray,
                validation_split=0.3,
                validation_data=None):


    es = EarlyStopping(patience=5, restore_best_weights=True, verbose=1)

    # Train the model, doing validation at the end of each epoch
    history=model.fit(
        X_train, y_train,
        validation_split=0.3,
        epochs=100,
        verbose=1,
        callbacks=[es],
    )
    return model, history

def evaluate_model(model: Model,
                   X_test: np.ndarray,
                   y_test: np.ndarray
                   ):
    """
    Evaluate trained model performance on dataset
    """

    print(Fore.BLUE + f"\nEvaluate model on {len(X_test)} rows..." + Style.RESET_ALL)

    metrics = model.evaluate(
        x=X_test,
        y=y_test,
        verbose=1,
        # callbacks=None,
        return_dict=True)

    loss = metrics["loss"]
    mae = metrics["mae"]

    print(f"\n✅ model evaluated: loss {round(loss, 2)} mae {round(mae, 2)}")

    return metrics
