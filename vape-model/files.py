import nibabel as nib
import numpy as np
import os
import pandas as pd
from scipy import ndimage

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

def resize_volume(volume):

    # Get current shape
    current_width = volume.shape[0]
    current_length = volume.shape[1]
    current_depth = volume.shape[2]

    # Compute shape factor
    width = current_width / int(os.environ.get("TARGET_WIDTH"))
    length = current_length / int(os.environ.get("TARGET_LENGTH"))
    depth = current_depth / int(os.environ.get("TARGET_DEPTH"))
    width_factor = 1 / width
    length_factor = 1 / length
    depth_factor = 1 / depth

    # Rotate
    volume = ndimage.rotate(volume, 90, reshape=False)

    # Resize across z-axis
    volume = ndimage.zoom(volume, (width_factor,
                             length_factor,
                             depth_factor), order=1)
    return volume

def normalize(volume):
    """Normalize the volume"""
    min = 0
    max = 2**16
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def open_dataset(dataset_name,verbose=0):

    # will fetch the infos.csv in the specified dataset's folder

    datasets_path = os.environ.get("DATASETS_PATH")
    # datasets_path = '/content/drive/MyDrive/6- Bootcamp/VAPE - Brain/Datasets for 3D/'

    path = datasets_path+dataset_name+'/'
    info_path = path+'infos/'

    file_names = pd.read_csv(info_path+dataset_name+'.csv')

    # put every .nii transformed in a list of array
    X_tmp = []
    for file_name in file_names['file_name']:
        if verbose == 1:
            print(f'processing file : {file_name}')

        volume = NII_to_3Darray(path+file_name)
        volume = normalize(volume)
        volume = resize_volume(volume)

        X_tmp.append(volume)

    # transform that list in an array
    if verbose == 1:
        print('.nii files processed. Compiling to X (might take a moment)')
    X = np.array(X_tmp, dtype='object')

    # create an array of the diagnostics
    if verbose == 1:
            print('Processing diagnostics...')
    y = pd.read_csv(info_path+dataset_name+'.csv',index_col=0)

    if verbose == 1:
            print('Diagnostics processed')
    return X,y
