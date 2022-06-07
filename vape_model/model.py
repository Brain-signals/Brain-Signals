
from tensorflow.keras import Model, layers, Input, optimizers, Sequential
from tensorflow.keras.layers import Reshape
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical


#first model 3dCNN
def initialize_model(width, length, depth):
    """Build a 3D convolutional neural network model."""

    model = Sequential()

    model.add(layers.Conv3D(
        filters=32,
        kernel_size=3,
        activation="relu",
        input_shape=(width, length, depth, 1)))
    model.add(layers.MaxPool3D(pool_size=2))

    model.add(layers.Conv3D(
        filters=64,
        kernel_size=3,
        activation="relu",))
    model.add(layers.MaxPool3D(pool_size=2))

    model.add(layers.Conv3D(
        filters=32,
        kernel_size=3,
        activation="relu",))
    model.add(layers.MaxPool3D(pool_size=2))

    model.add(layers.Flatten())
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
