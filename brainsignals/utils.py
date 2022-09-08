### External imports ###

import matplotlib.pyplot as plt
import nibabel as nib
import random
import string
from time import strftime


### Functions ###

def check_balance(y_encoded):
    y_encoded = y_encoded.tolist()

    diags = {}
    for key in range(len(y_encoded[0])):
        diags[key] = 0
    for diag in y_encoded:
        key = diag.index(1)
        diags[key] +=1
    total = 0
    for key in range(len(diags)):
        total += diags[key]
    for key,value in diags.items():
        print(f'diagnostic {key} represents {round(value*100/total,2)} % of the dataset')
    pass


def create_id():
    tag = ''.join(random.choices(list(string.ascii_lowercase), k=4)).upper()
    return strftime("%Y%m%d_%H%M%S_") + tag


def NII_to_3Darray(path):
    NII = nib.load(path).get_fdata()
    return NII


def show_nii_2D(volume):
    slice_x = int(volume.shape[0]/2)
    slice_y = int(volume.shape[1]/2)
    slice_z = int(volume.shape[2]/2)

    fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(10,4))
    ax1.imshow(volume[slice_x,:,:],cmap='gray')
    ax1.set_ylabel('Y axis')
    ax1.set_xlabel('Z axis')
    ax1.tick_params(bottom=False,left=False,labelbottom=False,labelleft=False)
    ax2.imshow(volume[:,slice_y,:],cmap='gray')
    ax2.set_ylabel('X axis')
    ax2.set_xlabel('Z axis')
    ax2.tick_params(bottom=False,left=False,labelbottom=False,labelleft=False)
    ax3.imshow(volume[:,:,slice_z],cmap='gray')
    ax3.set_ylabel('X axis')
    ax3.set_xlabel('Y axis')
    ax3.tick_params(bottom=False,left=False,labelbottom=False,labelleft=False)
    return fig


def time_print(start,end):
    total_sec = round(end - start, 2)
    if total_sec > 7200:
        time_prompt = f'\033[1m{total_sec} secs ({round(total_sec/3600,3)} hours)\033[0m'
    elif total_sec > 120:
        time_prompt = f'\033[1m{total_sec} secs ({round(total_sec/60,2)} minutes)\033[0m'
    else:
        time_prompt = f'\033[1m{total_sec} secs\033[0m'
    return time_prompt
