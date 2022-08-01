### External imports ###

from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pandas as pd
import os
import time

### Internal imports ###

from vape_model.model import initialize_model, train_model
from vape_model.registry import model_to_mlflow, model_to_pickle
from vape_model.files import open_dataset
from vape_model.utils import time_print, check_balance
from vape_model.evaluate import evaluation



### Settings ###

# How to pick for the training set
# set 0 as limit to use entire dataset
chosen_datasets = [('Controls',20), # max = 63
                   ('MRI_PD_vanicek_control',10), # max = 21
                   ('MRI_PD1_control',10), # max = 15
                   ('Wonderwall_control',30), # max = 424

                   ('MRI_PD1_parkinsons',30), # max = 30
                   ('MRI_PD_vanicek_parkinsons',20), # max = 20

                   ('Wonderwall_alzheimers',60), # max = 197
    ] # ('MRI_MS',40) # max = 60

# crop_volume_version 1 or 2 ?
crop_volume_version = 2

# model params
patience = 15
validation_split = 0.3
learning_rate = 0.0005
batch_size = 16
epochs = 100
es_monitor = 'val_accuracy'

# load_dataset verbose option
verbose = 1
# check for dataset's balance
balance_check = 1
# training verbose
tr_verbose = 1

# run evaluation after training
model_eval = False

### Functions ###

def preprocess_and_train():
    """
    Load data in memory, clean and preprocess it, train a Keras model on it,
    save the model, and finally compute & save a performance metric
    on a validation set holdout at the `model.fit()` level
    """

    start = time.perf_counter()

    for dataset in chosen_datasets:
        if chosen_datasets.index(dataset) == 0:
            X,y = open_dataset(dataset[0],limit=dataset[1],
                            verbose=verbose,
                            crop_volume_version=crop_volume_version)
        else:
            X_tmp,y_tmp = open_dataset(dataset[0],limit=dataset[1],
                            verbose=verbose,
                            crop_volume_version=crop_volume_version)
            X = np.concatenate((X,X_tmp))
            y = pd.concat((y,y_tmp),ignore_index=True)

        print(f'{dataset[0]} added to current training dataset\n')

    end = time.perf_counter()
    print('Training dataset has been created and preprocessed',time_print(start,end))

    #encode the y
    enc = OneHotEncoder(sparse = False)
    y_encoded = enc.fit_transform(y[['diagnostic']]).astype('int8')
    number_of_class = len(enc.get_feature_names_out())
    diagnostics = enc.get_feature_names_out()

    X_train, X_test, y_train, y_test = train_test_split(X,y_encoded,
                                                        test_size=validation_split,
                                                        shuffle=True)

    #initialize model
    target_res = int(os.environ.get('TARGET_RES'))
    model = initialize_model(width=target_res,
                             length=target_res,
                             depth=target_res,
                             number_of_class=number_of_class,
                             learning_rate=learning_rate)

    if balance_check == 1:
        print(f'diagnostics are : {diagnostics}')
        print('for training')
        check_balance(y_train)
        print('for testing')
        check_balance(y_test)

    #train model
    model, history, best_epoch_index = train_model(model,
                                X_train, y_train,
                                X_test, y_test,
                                patience=patience,
                                monitor=es_monitor,
                                batch_size = batch_size,
                                epochs=epochs,
                                verbose=tr_verbose)

    # compute val_metrics
    metrics = {}
    for metric,score in history.history.items():
        metrics[metric] = score[best_epoch_index]

    # save model
    params = dict(
        # hyper parameters
        crop_volume_version=crop_volume_version,
        used_dataset=chosen_datasets,
        diagnostics=diagnostics,
        target_res=target_res,
        patience=patience,
        validation_split=validation_split,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        # package behavior
        context="preprocess and train")

    model_name = os.environ.get("MLFLOW_MODEL_NAME")

    model_id = model_to_pickle(model=model,
                               params=params,
                               metrics=metrics)

    # model_to_mlflow(model=model,
    #                 model_name=model_name,
    #                 params=params,
    #                 metrics=metrics)

    return model_id



### Launch ###

if __name__ == '__main__':
    start = time.perf_counter()
    model_id = preprocess_and_train()
    end = time.perf_counter()
    print('model has been trained in',time_print(start,end))
    if model_eval:
        evaluation(model_id)
