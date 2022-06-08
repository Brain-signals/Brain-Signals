
from vape_model.model import evaluate_model, initialize_model,train_model,encoding_y
from vape_model.registry import model_to_mlflow
from vape_model.files import open_dataset

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import os

def preprocess_and_train(eval=True):
    """
    Load data in memory, clean and preprocess it, train a Keras model on it,
    save the model, and finally compute & save a performance metric
    on a validation set holdout at the `model.fit()` level
    """

    chosen_datasets = [
        ('Controls',40),
        ('Wonderwall_alzheimers',100),
        ('Wonderwall_control',24)
    ]

    # unchosen_datasets :
    # ('MRI_MS',0),
    # ('MRI_PD_vanicek_control',0),
    # ('MRI_PD_vanicek_parkinsons',0),
    # ('MRI_PD1_control',0),
    # ('MRI_PD1_parkinsons',0),

    # model params
    patience = 10
    validation_split = 0.3
    learning_rate = 0.001
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
    y_encoded=encoding_y(y)

    #split the dataset
    X_train, X_test, y_train, y_test=train_test_split(X,y_encoded,test_size=0.3)

    #initialize model
    target_res = int(os.environ.get('TARGET_RES'))
    model = initialize_model(width=target_res,
                             length=target_res,
                             depth=target_res,
                             learning_rate=learning_rate)

    #train model
    model, history= train_model(model,
                                X_train, y_train,
                                patience=patience,
                                monitor=es_monitor,
                                validation_split = validation_split,
                                batch_size = batch_size,
                                epochs=epochs,
                                verbose=1)

    # compute val_metrics
    metrics = history.history

    # save model
    params = dict(
        # hyper parameters
        used_dataset=chosen_datasets,
        target_res=target_res,
        patience=patience,
        validation_split=validation_split,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        # package behavior
        context="preprocess and train")

    model_to_mlflow(model=model, params=params, metrics=metrics)

    print(f"\nModel uploaded on mlflow")

    if eval:
        metrics_eval = evaluate_model(X_test,y_test,model=model)

    pass

if __name__ == '__main__':
    dqsd = 1562
    preprocess_and_train()
