import mlflow
import os
from time import strftime
import pickle
import glob
import keras

from tensorflow.keras import Model


def model_to_mlflow(model,model_name:str, params:dict, metrics:dict):

    mlflow.set_tracking_uri('https://mlflow.lewagon.ai') #VARIABLE

    try:
        experiment_id = mlflow.create_experiment('VAPE_MRI')
    except:
        experiment_id = mlflow.get_experiment_by_name('VAPE_MRI').experiment_id

    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.keras.log_model(keras_model=model,
            artifact_path="model",
            keras_module="tensorflow.keras",
            registered_model_name=model_name)

    print("\ndata saved to mlflow")

    pass


def model_to_pickle(model, params:dict, metrics:dict):

    suffix = strftime("%Y%m%d-%H%M%S")

    # save params
    params_path = os.path.join(os.environ.get("LOCAL_REGISTRY_PATH"), "params", suffix + ".pickle")
    with open(params_path, 'wb') as f:
        pickle.dump(params, f)

    # save metrics
    metrics_path = os.path.join(os.environ.get("LOCAL_REGISTRY_PATH"), "metrics", suffix + ".pickle")
    with open(metrics_path, "wb") as f:
        pickle.dump(metrics, f)

    # save model
    model_path = os.path.join(os.environ.get("LOCAL_REGISTRY_PATH"), "models", suffix + ".pickle")
    model.save(model_path)

    print("\ndata saved locally")

    pass


def load_model_from_mlflow(stage="None") -> Model:
    """
    load the latest saved model.
    """
    mlflow.set_tracking_uri('https://mlflow.lewagon.ai')

    mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")
    model_uri = f"models:/{mlflow_model_name}/{stage}"

    model = mlflow.keras.load_model(model_uri=model_uri)
    print("\nmodel loaded from mlflow")

    return model


def load_model_from_local(model_id=''):

    model_directory = os.path.join(os.environ.get("LOCAL_REGISTRY_PATH"), "models")
    params_directory = os.path.join(os.environ.get("LOCAL_REGISTRY_PATH"), "params")

    if model_id == '':
        chosen_model_path = sorted(glob.glob(f"{model_directory}/*"))[-1]
        model_id = chosen_model_path[-22:]
    else:
        for model_path in glob.glob(f"{model_directory}/*"):
            if model_id in model_path:
                chosen_model_path = model_path

    params_path = os.path.join(params_directory, model_id)
    with open(params_path, "rb") as f:
        params = pickle.load(f)

    diagnostics = params['diagnostics']
    try:
        crop_volume_version = params['crop_volume_version']
    except:
        crop_volume_version = 1
    model = keras.models.load_model(chosen_model_path)

    return model, diagnostics, crop_volume_version


def load_model_from_local_alzheimer(model_id=''):

    model_directory = os.path.join(os.environ.get("LOCAL_REGISTRY_PATH"), "models")

    if model_id == '':
        chosen_model_path = sorted(glob.glob(f"{model_directory}/*"))[-1]
        model_id = chosen_model_path[-22:]
    else:
        for model_path in glob.glob(f"{model_directory}/*"):
            if model_id in model_path:
                chosen_model_path = model_path

    model = keras.models.load_model(chosen_model_path)

    return model
