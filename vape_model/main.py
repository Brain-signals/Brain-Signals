from vape_model.model import initialize_model,train_model,encoding_y
from vape_model.registry import model_to_mlflow, model_to_pickle
from vape_model.files import open_dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pandas as pd
import os

def preprocess_and_train(eval=False):
    """
    Load data in memory, clean and preprocess it, train a Keras model on it,
    save the model, and finally compute & save a performance metric
    on a validation set holdout at the `model.fit()` level
    """

    chosen_datasets = [
        ('Controls',30),
        ('MRI_PD_vanicek_control',21),
        ('MRI_PD1_control',15),
        ('Wonderwall_control',54),
        ('MRI_PD1_parkinsons',30),
        ('MRI_PD_vanicek_parkinsons',20),
        ('Wonderwall_alzheimers',80),
    ]

    # unchosen_datasets :
    # ('MRI_MS',40)

    # model params
    patience = 10
    validation_split = 0.25
    learning_rate = 0.0005
    batch_size = 16
    epochs = 100
    es_monitor = 'val_accuracy'

    for dataset in chosen_datasets:
        if chosen_datasets.index(dataset) == 0:
            X,y = open_dataset(dataset[0],limit=dataset[1],verbose=1)
        else:
            X_tmp,y_tmp = open_dataset(dataset[0],limit=dataset[1],verbose=1)
            X = np.concatenate((X,X_tmp))
            y = pd.concat((y,y_tmp),ignore_index=True)

    #encode the y
    enc = OneHotEncoder(sparse = False)
    y_encoded = enc.fit_transform(y[['diagnostic']]).astype('int8')
    number_of_class = len(enc.get_feature_names_out())
    diagnostics = enc.get_feature_names_out()

    #split the dataset
    X_train, X_test, y_train, y_test=train_test_split(X,y_encoded,test_size=0.3)

    #initialize model
    target_res = int(os.environ.get('TARGET_RES'))
    model = initialize_model(width=target_res,
                             length=target_res,
                             depth=target_res,
                             number_of_class=number_of_class,
                             learning_rate=learning_rate)

    #train model
    model, history = train_model(model,
                                X_train, y_train,
                                patience=patience,
                                monitor=es_monitor,
                                validation_split = validation_split,
                                batch_size = batch_size,
                                epochs=epochs,
                                verbose=1)

    # compute val_metrics
    best_epoch = max(history.epoch) - patience
    metrics = {}
    for metric,score in history.history.items():
        metrics[metric] = score[best_epoch]

    # save model
    params = dict(
        # hyper parameters
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

    model_to_pickle(model=model,
                    params=params,
                    metrics=metrics)

    # model_to_mlflow(model=model,
    #                 model_name=model_name,
    #                 params=params,
    #                 metrics=metrics)

    if eval:
        metrics_eval = model.evaluate(x=X_test,y=y_test,verbose=1,return_dict=True)
        for metric,score in metrics_eval.items():
            print(f'{metric} is {score}')

    pass

if __name__ == '__main__':
    preprocess_and_train()
