from vape_model.preprocess import crop_volume, resize_and_pad, normalize_vol
from vape_model.registry import load_model
import os
import numpy as np

def predict_from_volume(volume):

    model_name = os.environ.get("MLFLOW_MODEL_NAME")

    model = load_model(model_name=model_name)
    os.environ["TARGET_RES"] = str(model.layers[0].input_shape[1])

    volume = crop_volume(volume)
    volume = resize_and_pad(volume)
    X_processed = np.array(volume)
    X_processed = normalize_vol(X_processed)

    y_pred = model.predict(X_processed)

    return 'ok'
