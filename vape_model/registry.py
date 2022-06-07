import mlflow
from mlflow.tracking import MlflowClient

import glob
import os
from time import strftime
import pickle

from colorama import Fore, Style

from tensorflow import keras
from tensorflow.keras import Model

def model_to_mlflow(model:Model,model_name:str, params:dict, metrics:dict):
    mlflow.set_tracking_uri('https://mlflow.lewagon.ai')
    client=MlflowClient()

    try:
        experiment_id=client.create_experiment('VAPE_MRI')
    except:
        experiment_id=client.get_experiment_by_name('VAPE_MRI').experiment_id

    with client.start_run():

            # push parameters to mlflow
            if params is not None:
                for param_name, param in params.items():
                    client.log_param(param_name, param)

            # push metrics to mlflow
            if metrics is not None:
                for metric_name, metric in metrics.items():
                    client.log_metric(metric_name, metric)

            # push model to mlflow
            if model is not None:

                client.keras.log_model(keras_model=model,
                                       artifact_path="model",
                                       keras_module="tensorflow.keras",
                                       registered_model_name=model_name)

    print("\n✅ data saved to mlflow")


    print("\nSave model to local disk..." )

    suffix = strftime("%Y%m%d-%H%M%S")

    # save params
    if params is not None:
        params_path = os.path.join(os.environ.get("LOCAL_REGISTRY_PATH"), "params", suffix + ".pickle")

        print(f"- params path: {params_path}")

        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # save metrics
    if metrics is not None:
        metrics_path = os.path.join(os.environ.get("LOCAL_REGISTRY_PATH"), "metrics", suffix + ".pickle")

        print(f"- metrics path: {metrics_path}")

        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    # save model
    if model is not None:
        model_path = os.path.join(os.environ.get("LOCAL_REGISTRY_PATH"), "models", suffix + ".pickle")

        print(f"- model path: {model_path}")

        model.save(model_path)

    print("\n✅ data saved locally")

def load_model(stage="None", model_name:str) -> Model:
    """
    load the latest saved model
    """
    mlflow.set_tracking_uri('https://mlflow.lewagon.ai')
    client=MlflowClient()


    model_uri = f"models:/{model_name}/{stage}"
    print(f"- uri: {model_uri}")

    try:
        model = client.keras.load_model(model_uri=model_uri)
        print("\n✅ model loaded from mlflow")
    except:
        # raise exception if no model exists
        raise NameError(f"No {model_name} model in {stage} stage stored in mlflow")

    return model
