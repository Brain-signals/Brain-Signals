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

def load_Nii(file_name, path):

  """Read and load volume"""
    # Read file
  scan = nib.load(path+file_name).get_fdata()
  return scan

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
    X = np.array(X_tmp, dtype='object')

    # create an array of the diagnostics
    if verbose == 1:
            print('Processing diagnostics...')
    y = pd.read_csv(info_path+dataset_name+'.csv')['diagnostic']

    if verbose == 1:
            print('Diagnostics processed')
    return X,y


#Preprocessing images functions
def resize_volume(img,TARGET_DEPTH,TARGET_WIDTH,TARGET_LENGTH):
    """Resize nii images across z-axis"""

    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_length = img.shape[1]
    # Compute depth factor
    depth = current_depth / TARGET_DEPTH
    width = current_width / TARGET_WIDTH
    length = current_length / TARGET_LENGTH
    depth_factor = 1 / depth
    width_factor = 1 / width
    length_factor = 1 / length
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor,
                             length_factor,
                             depth_factor), order=1)
    return img

def normalize(volume):
    """Normalize the volume of the images to convert to float32 to
    reduce the size of the volume"""
    min = 0
    max = 2**16
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def process_scan(path,file):
    """Read and resize volume"""
    # Read scan
    volume = load_Nii(file, path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume

# remove non desired file names and check lenght of the folder
def remove_unwanted(file_names):
  if '.DS_Store' in file_names:
    file_names.remove('.DS_Store')
  if '._.DS_Store' in file_names:
    file_names.remove('._.DS_Store')
  if 'infos' in file_names:
    file_names.remove('infos')
  return file_names
