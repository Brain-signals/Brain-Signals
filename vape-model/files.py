import nibabel as nib
import numpy as np
import os

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
