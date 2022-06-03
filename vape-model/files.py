import nibabel as nib
import numpy as np
import os
import pandas as pd

# def scan_folder(path):

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

def open_dataset(dataset_name,verbose=0):

    # will fetch the infos.csv in the specified dataset's folder
    datasets_path = os.environ.get("DATASETS_PATH")
    path = datasets_path+dataset_name+'/'
    info_path = path+'infos/'

    file_names = pd.read_csv(info_path+dataset_name+'.csv')

    # put every .nii transformed in a list of array
    X_tmp = []
    for file_name in file_names['file_name']:
        if verbose == 1:
            print(f'processing file : {file_name}')
        X_tmp.append(NII_to_3Darray(path+file_name))

    # transform that list in an array
    if verbose == 1:
        print('.nii files processed. Compiling to X (might take a moment)')
    X = np.array(X_tmp)

    # create an array of the diagnostics
    if verbose == 1:
            print('Processing diagnostics...')
    y = pd.read_csv(info_path+dataset_name+'.csv')['diagnostic']

    if verbose == 1:
            print('Diagnostics processed')
    return X,y
