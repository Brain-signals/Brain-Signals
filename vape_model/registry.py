import mlflow
import os
from time import strftime
import pickle

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

    print("\n✅ data saved to mlflow")

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

    print("\n✅ data saved locally")

    pass


def load_model(model_name:str, stage="None") -> Model:
    """
    load the latest saved model.
    """
    mlflow.set_tracking_uri('https://mlflow.lewagon.ai')

    mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")
    model_uri = f"models:/{mlflow_model_name}/{stage}"
    print(f"- uri: {model_uri}")

    try:
        model = mlflow.keras.load_model(model_uri=model_uri)
        print("\n✅ model loaded from mlflow")
    except:
        # raise exception if no model exists
        raise NameError(f"No {model_name} model in {stage} stage stored in mlflow")

    return model
