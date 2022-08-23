### External imports ###

import cv2
import numpy as np
import os
import pandas as pd
import time
from scipy import ndimage
from sklearn.preprocessing import OneHotEncoder


### Internal imports ###

from brainsignals.utils import NII_to_3Darray, time_print


### Classes ###

class Preprocessor:

    def __init__(self):
        pass


    def initialize_preprocessor(self, target_res, slicing_bot=0.3, slicing_top=0.15):
        self.slicing_bot = slicing_bot
        self.slicing_top = slicing_top
        self.target_res = target_res
        pass



    def create_dataset(self, chosen_datasets, verbose=1):

        try:
            start = time.perf_counter()
            self.files_used = []
            print('\nCreating training dataset...')

            for dataset in chosen_datasets:
                if chosen_datasets.index(dataset) == 0:
                    X,y = self.open_subdataset(dataset[0],limit=dataset[1],
                                        verbose=verbose)

                else:
                    X_tmp,y_tmp = self.open_subdataset(dataset[0],limit=dataset[1],
                                                verbose=verbose)

                    X = np.concatenate((X,X_tmp))
                    y = pd.concat((y,y_tmp),ignore_index=True)

            enc = OneHotEncoder(sparse = False)
            y_encoded = enc.fit_transform(y[['diagnostic']]).astype('int8')
            self.number_of_classes = len(enc.get_feature_names_out())
            self.diagnostics = enc.get_feature_names_out()
            self.dimensions = compute_shape(self.target_res, self.slicing_bot, self.slicing_top)

            end = time.perf_counter()
            if verbose > 0:
                print(f'\nTraining dataset created in {time_print(start,end)}.\n')

            return X, y_encoded

        except AttributeError:
            print('ERROR : This preprocessor has not been initialized.')

        pass


    def open_subdataset(self, dataset_name, verbose=0, limit=0):

        datasets_path = os.environ.get("DATASETS_PATH")

        path = os.path.join(datasets_path, dataset_name)
        info_path = os.path.join(path, 'infos/')
        csv_path = os.path.join(info_path, dataset_name+'.csv')

        file_names = pd.read_csv(csv_path)
        if limit != 0 :
            file_names = file_names.sample(n=limit)

        if verbose > 0:
            print(f'\nOpening {dataset_name} dataset...')
        X_tmp = []
        n = 1
        for file_name in file_names['file_name']:

            self.files_used.append(file_name)

            if verbose > 0:
                print(' '*60, end='\r', flush=True)
                print(f'processing file {n}/{len(file_names["file_name"])} : {file_name}', end='\r', flush=True)
                n += 1

            file_path = os.path.join(path, file_name)
            volume = NII_to_3Darray(file_path)
            volume = crop_volume(volume)
            volume = resize_and_pad(volume, self.target_res)
            volume = slice_volume(volume, self.slicing_bot, self.slicing_top)
            volume = normalize_vol(volume)

            X_tmp.append(volume)
        X = np.array(X_tmp)


        y = file_names[['diagnostic']]

        if verbose > 0:
            print(f"{dataset_name} dataset processed with a limit of {limit} files.")

        return X, y


    def initialize_from_model(self, model):

        self.slicing_top = model.slicing_top
        self.slicing_bot = model.slicing_bot
        self.target_res = model.dimensions[0]
        self.files_used = []

        return self


    def transform_vol(self, volume):

        volume = crop_volume(volume)
        volume = resize_and_pad(volume, self.target_res)
        volume = slice_volume(volume, self.slicing_bot, self.slicing_top)
        volume = normalize_vol(volume)

        return volume


### Functions ###

def compute_roi(contour):
    l = [dots.tolist()[0] for dots in contour]
    xs, ys = zip(*l)
    return np.min(xs), np.min(ys), np.max(xs), np.max(ys)


def compute_shape(target_res, slicing_bot=0.3, slicing_top=0.15):
    Z_dim = (target_res-int(target_res*slicing_top))-int(target_res*slicing_bot)-1
    return (target_res, target_res, Z_dim)


def crop_volume(volume):
    left_conts = []
    right_conts = []
    bottom_conts = []
    top_conts = []


    first_layer_checked = int(volume.shape[2] * 0.2)
    last_layer_checked = volume.shape[2]
    norm_vol = cv2.normalize(volume, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

    for layer in range(first_layer_checked, last_layer_checked):
        if np.max(norm_vol[:,:,layer]) > 40:
            left, bottom, right, top = compute_roi(get_brain_contour_nii(norm_vol[:,:,layer]))
            left_conts.append(left)
            right_conts.append(right)
            bottom_conts.append(bottom)
            top_conts.append(top)

    min_left = np.min(left_conts)
    max_right = np.max(right_conts)
    min_bottom = np.min(bottom_conts)
    max_top = np.max(top_conts)

    #crop along X and Y axis
    volume = volume[min_bottom : max_top, min_left : max_right, :]

    zs_conts = []
    for layer in range(volume.shape[1]):
        if np.max(norm_vol[:,layer,:]) > 40:
            zs_conts.append(compute_roi(get_brain_contour_nii(volume[:,layer,:]))[2])

    max_zs = np.max(zs_conts)
    return volume[:, :, 0:max_zs]


def get_brain_contour_nii(img):

    # convert nifti slice to cv2 image
    img = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # compute maximum area
    max_area = img.shape[0] * img.shape[1]

    tresh_params, thresh = cv2.threshold(img, 10, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    areas = []
    for c in contours:
        # retrieve bounding box
        left, bottom, right, top = compute_roi(c)
        # print(left,right,bottom,top)

        # compute ROI area => the biggest wins
        area = (right-left) * (top-bottom)
        if area < max_area:
            areas.append(area)

    return contours[np.argmax(areas, axis=0)]


def get_padding(axis_shape, target_res):
    zeros_to_add = target_res-axis_shape
    if zeros_to_add%2 == 0:
        padding = (int(zeros_to_add/2),int(zeros_to_add/2))
    else:
        padding = (int(zeros_to_add//2),int(zeros_to_add//2+1))
    return padding


def get_padding_Z(axis_shape, target_res):
    zeros_to_add = target_res-axis_shape
    padding = (zeros_to_add,0)
    return padding


def normalize_vol(volume):
    return cv2.normalize(volume, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)


def resize_and_pad(volume, target_res):

    # Get current shape
    Xaxis_shape = volume.shape[0]
    Yaxis_shape = volume.shape[1]
    Zaxis_shape = volume.shape[2]

    # Compute shape factor
    Xaxis_factor = 1 / ( Xaxis_shape / target_res )
    Yaxis_factor = 1 / ( Yaxis_shape / target_res )
    Zaxis_factor = 1 / ( Zaxis_shape / target_res )

    factor = min(Xaxis_factor, Yaxis_factor, Zaxis_factor)

    # Zoom to the target, based on the biggest axis
    volume = ndimage.zoom(volume, (factor, factor, factor))


    # and pad zeros until wanted shape
    Xaxis_padding = get_padding(volume.shape[0], target_res)
    Yaxis_padding = get_padding(volume.shape[1], target_res)
    Zaxis_padding = get_padding_Z(volume.shape[2], target_res)

    volume = np.pad(volume, (Xaxis_padding, Yaxis_padding, Zaxis_padding), mode='constant')

    return volume


def slice_volume(volume, slicing_bot=0.4, slicing_top=0.2):
    bot_i = int(volume.shape[2] * slicing_bot)
    top_i = int(volume.shape[2] - volume.shape[2] * slicing_top)
    return volume[:,:,bot_i:top_i]
