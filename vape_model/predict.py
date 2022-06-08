from vape_model.preprocess import crop_volume, resize_and_pad, normalize_vol
from vape_model.registry import load_model
import os
import numpy as np

def predict_from_volume(volume):

    model_name = os.environ.get("MLFLOW_MODEL_NAME")

    model = load_model(model_name=model_name)
    os.environ["TARGET_RES"] = str(model.layers[0].input_shape[1])

    vol_crop = crop_volume(volume)
    vol_res = resize_and_pad(vol_crop)
    X_tmp = []
    X_tmp.append(vol_res)
    X = np.array(X_tmp)
    X_processed = normalize_vol(X)

    y_pred = model.predict(X_processed)

    return np.round(y_pred,3)
