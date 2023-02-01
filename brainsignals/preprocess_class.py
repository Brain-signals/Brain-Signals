### External imports ###

import numpy as np
import os
import pandas as pd
import time
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder


### Internal imports ###

from brainsignals.utils import NII_to_3Darray, time_print
from brainsignals.preprocess_fncts import (compute_shape, crop_volume,
                                           resize_and_pad, slice_volume,
                                           normalize_vol, rotate_volume)


### Classes ###

class Preprocessor:

    def __init__(self):
        pass


    def initialize_preprocessor(self, target_res, slicing_bot=0.3, slicing_top=0.15):
        self.slicing_bot = slicing_bot
        self.slicing_top = slicing_top
        self.target_res = target_res
        self.files_used = []
        return self



    def create_dataset(self, chosen_datasets, verbose=1):

        try:
            start = time.perf_counter()

            if verbose > 0:
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
        orientation_path = os.path.join(datasets_path,'orientation.csv')

        file_names = pd.read_csv(csv_path).merge(pd.read_csv(orientation_path), on='file_name')
        if limit != 0 :
            file_names = file_names.sample(n=limit)

        print(f'\nOpening {dataset_name} dataset...')
        X_tmp = []
        n = 1
        for index, row in tqdm(file_names.iterrows(), total=len(file_names), ncols=100):

            self.files_used.append(row.file_name)

            if verbose > 0:
                print(' '*70, end='\r', flush=True)
                print(f'processing file {n}/{len(file_names["file_name"])} : {row.file_name}', end='\r', flush=True)
                n += 1

            file_path = os.path.join(path, row.file_name)
            volume = NII_to_3Darray(file_path)
            volume = rotate_volume(volume, input_orient=row.orientation)
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
