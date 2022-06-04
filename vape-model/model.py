
from tensorflow.keras import Model, layers, Input, optimizers, Sequential
from tensorflow.keras.layers import Reshape
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical


#first model 3dCNN
def get_model(width=128, height=128, depth=128):
    """Build a 3D convolutional neural network model."""

    inputs = Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(filters=128, kernel_size=4, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = Model(inputs, outputs, name="3dcnn")
    return model


# Compile model.
def compile_mod(model,initial_learning_rate=0.0001):

  lr_schedule = schedules.ExponentialDecay(
      initial_learning_rate,
      decay_steps=100000,
      decay_rate=0.96,
      staircase=True)

  adam_opt = optimizers.Adam(learning_rate=lr_schedule)
  model.compile(
      loss="binary_crossentropy",
      optimizer= adam_opt,
      metrics=["accuracy"],
  )
  return model
