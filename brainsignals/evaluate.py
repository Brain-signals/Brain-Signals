### External imports ###

import numpy as np
import pandas as pd
import os
import sys
import time
from sklearn.preprocessing import OneHotEncoder


### Internal imports ###

from brainsignals.model_class import Model
from brainsignals.preprocess_class import Preprocessor
from brainsignals.utils import time_print


### Settings ###

ctrl_datasets = [('Controls',2), # max = 63
                 ('MRI_PD_vanicek_control',2), # max = 21
                 ('MRI_PD1_control',2), # max = 15
                 ('Wonderwall_control',2)] # max = 424

park_datasets = [('MRI_PD1_parkinsons',4), # max = 30
                 ('MRI_PD_vanicek_parkinsons',4)] # max = 20

alz_datasets = [('Wonderwall_alzheimers',7)] # max = 197


### Functions ###

def evaluate_model(model_id, max_run = 20):

    start = time.perf_counter()

    model_id = model_id[-20:]
    model_path = os.path.join(os.environ.get('LOCAL_REGISTRY_PATH'), model_id)

    model = Model().load_model(model_path)

    run = 1
    results = []

    print('')

    while run <= max_run:
        print(' '*40, end='\r', flush=True)
        print(f'Evaluation : run {run} / {max_run}', end='\r', flush=True)
        results.append(score_model(model))
        run += 1

    print('Evaluation completed.', ' '*10, end="\r")
    print('\n')

    run = 0
    for result in results:
        print(f'for run {run+1} evalution was {result}')
        run += 1

    for key in results[0].keys():
        tot = 0
        for r in range(max_run):
            tot += results[r][key]
        tot /= max_run
        print(f'average {key} : {round(tot,3)}')

    end = time.perf_counter()
    print(f'\nModel {model_id} has been evaluated in {time_print(start,end)}')

    pass


def score_model(model, dataset_verbose=0, score_verbose=0):

    preproc = Preprocessor().initialize_from_model(model)

    if 'diagnostic_Healthy' in model.diagnostics:

        for dataset in ctrl_datasets:
            if ctrl_datasets.index(dataset) == 0:
                X_c,y_c = preproc.open_subdataset(dataset[0], verbose=dataset_verbose, limit=dataset[1])

            else:
                X_tmp,y_tmp = preproc.open_subdataset(dataset[0], verbose=dataset_verbose, limit=dataset[1])
                X_c = np.concatenate((X_c,X_tmp))
                y_c = pd.concat((y_c,y_tmp), ignore_index=True)

    else:
        X_c = np.array(None)

    if 'diagnostic_Parkinson' in model.diagnostics:

        for dataset in park_datasets:
            if park_datasets.index(dataset) == 0:
                X_p,y_p = preproc.open_subdataset(dataset[0], verbose=dataset_verbose, limit=dataset[1])

            else:
                X_tmp,y_tmp = preproc.open_subdataset(dataset[0], verbose=dataset_verbose, limit=dataset[1])
                X_p = np.concatenate((X_p,X_tmp))
                y_p = pd.concat((y_p,y_tmp), ignore_index=True)

    else:
        X_p = np.array(None)

    if 'diagnostic_Alzheimer' in model.diagnostics:

        for dataset in alz_datasets:
            if alz_datasets.index(dataset) == 0:
                X_a,y_a = preproc.open_subdataset(dataset[0], verbose=dataset_verbose, limit=dataset[1])

            else:
                X_tmp,y_tmp = preproc.open_subdataset(dataset[0], verbose=dataset_verbose, limit=dataset[1])
                X_a = np.concatenate((X_a,X_tmp))
                y_a = pd.concat((y_a,y_tmp), ignore_index=True)

    else:
        X_a = np.array(None)

    if X_c.any() and X_a.any() and X_p.any():
        X = np.concatenate((X_c, X_a, X_p))
        y = pd.concat((y_c, y_a, y_p), ignore_index=True)

    elif X_c.any() and X_a.any():
        X = np.concatenate((X_c, X_a))
        y = pd.concat((y_c, y_a), ignore_index=True)

    elif X_a.any() and X_p.any():
        X = np.concatenate((X_a, X_p))
        y = pd.concat((y_a, y_p), ignore_index=True)

    elif X_c.any() and X_p.any():
        X = np.concatenate((X_c, X_p))
        y = pd.concat((y_c, y_p), ignore_index=True)

    else:
        if X_c.any():
            return X_c, y_c
        elif X_a.any():
            return X_a, y_a
        else:
            return X_p, y_p

    enc = OneHotEncoder(sparse = False)
    y_encoded = enc.fit_transform(y[['diagnostic']]).astype('int8')

    metrics_eval = model.model.evaluate(x=X, y=y_encoded, verbose=score_verbose, return_dict=True)

    return metrics_eval


### Launch ###

if __name__ == '__main__':
    model_id = sys.argv[1]
    evaluate_model(model_id, max_run=25)
