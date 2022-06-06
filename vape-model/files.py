import nibabel as nib
import numpy as np
import os
import pandas as pd
from scipy import ndimage
import cv2



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



target_res= int(os.environ.get("TARGET_RES"))

def resize_and_pad(volume):

    # Get current shape
    current_width = volume.shape[0]
    current_length = volume.shape[1]
    current_depth = volume.shape[2]

    # Compute shape factor
    width_factor = 1 / ( current_width / target_res )
    length_factor = 1 / ( current_length / target_res )
    depth_factor = 1 / ( current_depth / target_res )

    factor = min(width_factor,length_factor,depth_factor)

    # Zoom to the target, based on the biggest axis
    volume = ndimage.zoom(volume, (factor, factor, factor))

    # and pad zeros until wanted shape

    def get_padding(axis_shape):
        zeros_to_add = target_res-axis_shape
        if zeros_to_add%2 == 0:
            padding = (int(zeros_to_add/2),int(zeros_to_add/2))
        else:
            padding = (int(zeros_to_add//2),int(zeros_to_add//2+1))
        return padding

    pad_width = get_padding(volume.shape[0])
    pad_length = get_padding(volume.shape[1])
    pad_depth = get_padding(volume.shape[2])

    volume = np.pad(volume, (pad_width, pad_length, pad_depth), mode='minimum')

    return volume



def normalize(X):
    """Normalize the volume"""
    cv2.normalize(X, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)



def open_dataset(dataset_name,verbose=0):

    # will fetch the infos.csv in the specified dataset's folder

    datasets_path = os.environ.get("DATASETS_PATH")
    # datasets_path = '/content/drive/MyDrive/6- Bootcamp/VAPE - Brain/Datasets for 3D/'

    path = datasets_path+dataset_name+'/'
    info_path = path+'infos/'

    file_names = pd.read_csv(info_path+dataset_name+'.csv')

    # put every .nii transformed in a list of array
    X_tmp = []
    n = 1
    for file_name in file_names['file_name']:
        if verbose == 1:
            print(f'processing file {n}/{len(file_names["file_name"])} : {file_name}')
            n += 1

        volume = NII_to_3Darray(path+file_name)
        volume = resize_and_pad(volume)

        X_tmp.append(volume)

    # transform that list in an array and normalize it
    if verbose == 1:
        print('.nii files processed. Compiling to X (might take a moment)')
    X = np.array(X_tmp)
    X = cv2.normalize(X, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # create an array of the diagnostics
    if verbose == 1:
            print('Processing diagnostics...')
    y = pd.read_csv(info_path+dataset_name+'.csv',index_col=0)

    if verbose == 1:
            print('Diagnostics processed')
    return X,y
