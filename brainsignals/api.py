### External imports ###

import numpy as np
import os
from fastapi import FastAPI, UploadFile, File


### Internal imports ###

from brainsignals.utils import NII_to_3Darray
from brainsignals.model_class import Model
from brainsignals.preprocess_class import Preprocessor


### Functions ###

app = FastAPI()

@app.get("/")
def root():
    return {'message':'API is running'}


@app.get("/model/")
def test(model_id):
    my_modele = Model().load_model(model_id)
    return {'Model_id': my_modele.model_id,
            'Classes' : my_modele.number_of_classes,
            'Dimensions' : my_modele.dimensions,
            'Learning_rate' : my_modele.learning_rate,
            'Slicing_bot':my_modele.slicing_bot,
            'Slicing_top':my_modele.slicing_top,
            'Epochs': my_modele.epochs,
            'Best_epoch': my_modele.best_epoch,
            'Patience': my_modele.patience,
            'Monitor': my_modele.monitor,
            'Batch_size': my_modele.batch_size,
            'Validation_split':my_modele.validation_split,
            'Model_creator':my_modele.model_creator,
            'Creator_comment':my_modele.creator_comment
            }


@app.post("/predict/")
async def upload_nii(model_id,nii_file: UploadFile=File(...)):
    try:
        path = os.path.join('brainsignals', 'tmp_files', nii_file.filename)
        with open(path, 'wb') as f:
            nii_content = await nii_file.read()
            f.write(nii_content)

        vol = NII_to_3Darray(path)

        my_modele = Model().load_model(model_id)
        y_pred = my_modele.predict(vol)
        print(y_pred)

    except Exception:
        return {'message':'There was an error uploading the file'}

    finally:
        await nii_file.close()
        os.remove(path)

    return y_pred
