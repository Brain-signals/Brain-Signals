
from vape_model.model import (initialize_model,
                              train_model,
                              evaluate_model,
                              split_train_test,
                              encoding_y)

from vape_model.preprocess import (compute_roi,
                                   get_brain_contour_nii,
                                   crop_volume,
                                   resize_and_pad,
                                   normalize_vol
                                   )
from vape_model.registry import (model_to_mlflow,load_model)
from vape_model.files import (scan_folder_for_nii,
                              NII_to_3Darray,
                              NII_to_layer,
                              open_dataset
                              )


import numpy as np
import pandas as pd
import os

from colorama import Fore, Style

def preprocess_and_train(
    first_row=0
    , stage="None"
):
    """
    Load data in memory, clean and preprocess it, train a Keras model on it,
    save the model, and finally compute & save a performance metric
    on a validation set holdout at the `model.fit()` level
    """
    datasets_path = os.environ.get("DATASETS_PATH")
    dataset_names = os.listdir(datasets_path)

    print("\n⭐️ use case: preprocess and train")
    X1,y1 = open_dataset('MRI_PD_vanicek_control', verbose=1)
    X2,y2 = open_dataset('MRI_PD_vanicek_parkinsons', verbose=1)
    X3,y3 = open_dataset('Wonderwall', verbose=1)
    X4,y4 = open_dataset('MRI_PD_1', verbose=1)

    X = np.concatenate((X1,X2,X3,X4))
    y = pd.concat((y1,y2,y3,y4),ignore_index=True)

    #encode the y
    y_encoded=encoding_y(y)

    #split the dataset
    X_train, X_test, y_train, y_test=split_train_test(X, y_encoded)

    #initialize model
    target_res=os.environ.get('TARGET_RES')
    model= initialize_model(target_res,target_res,target_res)

    #train model
    model, history= train_model(model, X_train, y_train,validation_split=0.3,
                validation_data=None)

    # model params
    #learning_rate = 0.001
    #batch_size = 256

    # compute val_metrics
    val_acc = np.max(history.history['val_accuracy'])
    metrics = dict(val_acc=val_acc)

    # save model
    params = dict(
        # hyper parameters
        #learning_rate=learning_rate,
        #batch_size=batch_size,
        # package behavior
        context="preprocess and train",
        # data source
        used_dataset=dataset_names,)

    model_to_mlflow(model=model, params=params, metrics=metrics)

    print(f"\n✅ model uploaded on mlflow")

    return val_acc
