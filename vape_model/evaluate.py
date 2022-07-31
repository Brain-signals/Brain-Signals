### External imports ###

from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pandas as pd
import os
import time
import sys

### Internal imports ###

from vape_model.registry import load_model_from_local
from vape_model.files import open_dataset
from vape_model.utils import time_print, display_model



### Experimental variables ###

# How many evaluation runs ? (dataset picks change every run)
max_run = 20

# How to pick for each evaluation set ?
ctrl_datasets = [('Controls',4), # max = 63
                 ('MRI_PD_vanicek_control',4), # max = 21
                 ('MRI_PD1_control',3), # max = 15
                 ('Wonderwall_control',4) # max = 424
]

park_datasets = [('MRI_PD1_parkinsons',7), # max = 30
                 ('MRI_PD_vanicek_parkinsons',8) # max = 20
]

alz_datasets = [('Wonderwall_alzheimers',15) # max = 197
    ]

# load_dataset verbose option
verbose = 0
# .evaluate verbose option
ev_verbose = 0
# display the model
ml_verbose = 0



### Functions ###

def evaluate_model(model_id):

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    model_id = model_id.strip('.pickle')[-15:]+'.pickle'
    model_path = f'{os.environ.get("LOCAL_REGISTRY_PATH")}/models/{model_id}'

    try:
        model, diagnostics, crop_volume_version = load_model_from_local(model_id)

    except FileNotFoundError:
        print(f'\nModel with id {model_id} has not been found in :')
        print(f'{model_path}.\n')
        return False

    if ml_verbose:
        display_model(model)

    run = 0
    results = []
    print(f'Model succefully loaded from :\n{model_path}\n')
    while run < max_run:
        print(f'Evaluation : run {run+1} / {max_run}...',end='\r')
        results.append(score_model(model,diagnostics,
                                   crop_volume_version,
                                   verbose=verbose,
                                   ev_verbose=ev_verbose))
        run += 1

    print('Evaluation completed.',end="\r")
    print('\n')

    if verbose or not ev_verbose:
        run = 0
        for result in results:
            print(f'for run {run+1} evalution was {result}')
            run += 1

    print('')

    for key in results[0].keys():
        tot = 0
        for r in range(max_run):
            tot += results[r][key]
        tot /= max_run
        print(f'average {key} : {tot}')

    print(f'\nModel # {model_id} has been evaluated after {max_run} runs')

    return True


def score_model(model,diagnostics,crop_volume_version,verbose=0,ev_verbose=1):

    os.environ["TARGET_RES"] = str(model.layers[0].input_shape[1])

    if 'diagnostic_Healthy' in diagnostics:

        for dataset in ctrl_datasets:
            if ctrl_datasets.index(dataset) == 0:
                X_c,y_c = open_dataset(dataset[0],limit=dataset[1],
                                       verbose=verbose,
                                       crop_volume_version=crop_volume_version)
            else:
                X_tmp,y_tmp = open_dataset(dataset[0],limit=dataset[1],
                                           verbose=verbose,
                                           crop_volume_version=crop_volume_version)
                X_c = np.concatenate((X_c,X_tmp))
                y_c = pd.concat((y_c,y_tmp),ignore_index=True)

    else:
        X_c = np.array([])
        y_c = pd.Series()


    if 'diagnostic_Parkinson' in diagnostics:

        for dataset in park_datasets:
            if park_datasets.index(dataset) == 0:
                X_p,y_p = open_dataset(dataset[0],limit=dataset[1],
                                       verbose=verbose,
                                       crop_volume_version=crop_volume_version)
            else:
                X_tmp,y_tmp = open_dataset(dataset[0],limit=dataset[1],
                                           verbose=verbose,
                                           crop_volume_version=crop_volume_version)
                X_p = np.concatenate((X_p,X_tmp))
                y_p = pd.concat((y_p,y_tmp),ignore_index=True)

    else:
        X_p = np.array([])
        y_p = pd.Series()



    if 'diagnostic_Alzheimer' in diagnostics:

        for dataset in alz_datasets:
            if alz_datasets.index(dataset) == 0:
                X_a,y_a = open_dataset(dataset[0],limit=dataset[1],
                                       verbose=verbose,
                                       crop_volume_version=crop_volume_version)
            else:
                X_tmp,y_tmp = open_dataset(dataset[0],limit=dataset[1],
                                           verbose=verbose,
                                           crop_volume_version=crop_volume_version)
                X_a = np.concatenate((X_a,X_tmp))
                y_a = pd.concat((y_a,y_tmp),ignore_index=True)

    else:
        X_a = np.array([])
        y_a = pd.Series()

    X = np.concatenate((X_c, X_a, X_p))
    y = pd.concat((y_c, y_a, y_p),ignore_index=True)

    enc = OneHotEncoder(sparse = False)
    y_encoded = enc.fit_transform(y[['diagnostic']]).astype('int8')

    metrics_eval = model.evaluate(x=X,y=y_encoded,verbose=ev_verbose,return_dict=True)

    print('')

    return metrics_eval



### Launch ###

if __name__ == '__main__':
    start = time.perf_counter()
    valid_model = evaluate_model(sys.argv[1])
    if valid_model:
        end = time.perf_counter()
        print('And',time_print(start,end))

def evaluation(model_id):
    start = time.perf_counter()
    valid_model = evaluate_model(model_id)
    if valid_model:
        end = time.perf_counter()
        print('And \033[94m',time_print(start,end))
