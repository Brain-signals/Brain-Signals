from vape_model.preprocess import crop_volume, crop_volume_v2, resize_and_pad, normalize_vol
from vape_model.registry import load_model_from_local,load_model_from_local_alzheimer,load_model_from_mlflow
import os
import numpy as np

def predict_from_volume(volume):

    # model, diagnostics = load_model_from_mlflow()

    model, diagnostics, crop_volume_version = load_model_from_local()

    os.environ["TARGET_RES"] = str(model.layers[0].input_shape[1])
    # print(f'TARGET_RES is {os.environ["TARGET_RES"]}')

    if crop_volume_version == 2:
        vol_crop = crop_volume_v2(volume)
    else:
        vol_crop = crop_volume(volume)

    vol_res = resize_and_pad(vol_crop)
    X_tmp = []
    X_tmp.append(vol_res)
    X = np.array(X_tmp)
    X_processed = normalize_vol(X)

    y_pred = np.round(model.predict(X_processed),3)
    preds = {}
    for n in range(len(y_pred[0])):
        preds[diagnostics[n]] = y_pred[0][n]

    return preds


def predict_from_volume_alzheimer(volume):

    # model, diagnostics = load_model_from_mlflow()
    model = load_model_from_local_alzheimer()

    os.environ["TARGET_RES"] = str(model.layers[0].input_shape[1])

    vol_crop = crop_volume_v2(volume)
    vol_res = resize_and_pad(vol_crop)
    X_tmp = []
    X_tmp.append(vol_res)
    X = np.array(X_tmp)
    X_processed = normalize_vol(X)

    y_pred = np.round(model.predict(X_processed),3)

    return y_pred
