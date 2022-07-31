import nibabel as nib
import numpy as np
import os
import pandas as pd
from vape_model.preprocess import crop_volume, resize_and_pad, normalize_vol, crop_volume_v2
from vape_model.utils import time_print
import time


def scan_folder_for_nii(path):

    file_names = []
    all_file_names = os.listdir(path)
    for file_name in all_file_names:
        if '.nii' in file_name:
            file_names.append(file_name)
    return file_names



def NII_to_3Darray(path):
    NII = nib.load(path).get_fdata()
    return NII



def NII_to_layer(path,slicing=0.6):
    NII = nib.load(path).get_fdata()
    layer = NII[int(NII.shape[0]*slicing),:,:]
    return np.array(layer)



def open_dataset(dataset_name,verbose=0,limit=0,crop_volume_version=2):

    # will fetch the infos.csv in the specified dataset's folder

    start = time.perf_counter()

    datasets_path = os.environ.get("DATASETS_PATH")

    path = os.path.join(datasets_path, dataset_name)
    info_path = os.path.join(path, 'infos/')
    csv_path = os.path.join(info_path, dataset_name+'.csv')

    file_names = pd.read_csv(csv_path)

    if limit != 0 :
        # file_names = file_names.sample(frac=1).head(limit)
        file_names = file_names.sample(n=limit)
        if verbose:
            print(f"Opening {dataset_name} dataset with a limit of {limit} files.")

    elif limit == 0:
        if verbose:
            print(f'Opening {dataset_name} dataset with no limit of files.')

    # put every .nii transformed in a list of array
    X_tmp = []
    n = 1
    for file_name in file_names['file_name']:
        if verbose == 1:
            print(f'processing file {n}/{len(file_names["file_name"])} : {file_name}',
                  end='\r')
            print('')
            n += 1

        file_path = os.path.join(path, file_name)
        volume = NII_to_3Darray(file_path)
        if crop_volume_version == 2:
            volume = crop_volume_v2(volume)
        else:
            volume = crop_volume(volume)
        volume = resize_and_pad(volume)

        X_tmp.append(volume)

    # transform that list in an array and normalize it
    if verbose == 1:
        print('Nifti files processed and compiled to X.')
    X = np.array(X_tmp)
    X = normalize_vol(X)

    # create an array of the diagnostics
    y = file_names[['diagnostic']]

    end = time.perf_counter()

    if verbose == 1:
        print('Diagnostics processed.')
        print(f"Dataset \033[94m{dataset_name}\033[0m processed in {time_print(start,end)}")


    return X,y



def open_dataset_alzheimer(dataset_name,verbose=0,limit=0):

    # will fetch the infos.csv in the specified dataset's folder

    start = time.perf_counter()

    datasets_path = os.environ.get("DATASETS_PATH")
    #datasets_path = '/Users/lison/code/Elise-L/VAPE-MRI/Jupyter_notebook/'

    path = os.path.join(datasets_path, dataset_name)
    info_path = os.path.join(path, 'infos/')
    csv_path = os.path.join(info_path, dataset_name+'_mmse_cdr.csv')

    file_names = pd.read_csv(csv_path)
    if limit != 0 :
        file_names = file_names.sample(frac=1).head(limit)

    # put every .nii transformed in a list of array
    X_tmp = []
    n = 1
    for file_name in file_names['file_name']:
        if verbose == 1:
            print(f'processing file {n}/{len(file_names["file_name"])} : {file_name}')
            n += 1

        file_path = os.path.join(path, file_name)
        volume = NII_to_layer(file_path)
        # volume = crop_volume(volume)
        volume = resize_and_pad(volume)

        X_tmp.append(volume)

    # transform that list in an array and normalize it
    if verbose == 1:
        print('.nii files processed. Compiling to X (might take a moment)')
    X = np.array(X_tmp)
    X = normalize_vol(X)

    # create an array of the diagnostics
    if verbose == 1:
            print('Processing diagnostics...')


    y = file_names[['diagnostic']]
    y_mmse = file_names[['mmse']]
    y_cdr = file_names[['cdr']]

    end = time.perf_counter()

    if verbose == 1:
        print('Diagnostics processed')
        print(f'Dataset {dataset_name} processed in {time_print(start,end)} secs',end = '')

    return X,y_mmse
