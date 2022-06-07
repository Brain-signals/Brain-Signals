import nibabel as nib
import numpy as np
import os
import pandas as pd
from vape_model.preprocess import crop_volume, resize_and_pad, normalize_vol
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
    layer = NII[:,:,int(NII.shape[2]*slicing)]
    return np.array(layer)



def open_dataset(dataset_name,verbose=0,limit=0):

    # will fetch the infos.csv in the specified dataset's folder

    start = time.perf_counter()

    datasets_path = os.environ.get("DATASETS_PATH")
    # datasets_path = '/content/drive/MyDrive/6- Bootcamp/VAPE - Brain/Datasets for 3D/'

    path = datasets_path+dataset_name+'/'
    info_path = path+'infos/'

    file_names = pd.read_csv(info_path+dataset_name+'.csv')
    if limit != 0 :
        file_names = file_names.sample(frac=1).head(limit)

    # put every .nii transformed in a list of array
    X_tmp = []
    n = 1
    for file_name in file_names['file_name']:
        if verbose == 1:
            print(f'processing file {n}/{len(file_names["file_name"])} : {file_name}')
            n += 1

        volume = NII_to_3Darray(path+file_name)
        volume = crop_volume(volume)
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

    end = time.perf_counter()

    if verbose == 1:
        print('Diagnostics processed')
        print(f'Dataset {dataset_name} processed in {round(end - start, 2)} secs\n')


    return X,y
