from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pandas import to_datetime
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from vape_model import predict
import nibabel as nib
import shutil


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


def read_imagefile(file) -> Image.Image:
    image = Image.open(io.BytesIO(file))
    return image


@app.get("/")
def root():
    # YOUR CODE HERE
    return {'greeting': 'Hello'}

@app.get("/test")
def root():
    # YOUR CODE HERE
    return {'greeting': 'Hello'}


@app.post("/predict")
def predict_api(file: UploadFile = File(...)):


    with open("file.nii", 'wb') as f:
        shutil.copyfileobj(file.file, f)
    #vol=nib.load(file.filename).get_fdata()
    vol=nib.load("file.nii").get_fdata()

    pred = predict.predict_from_volume(vol)

    #_return = {"Start of answer":file}
    # _return['ls'] = os.listdir()
    #_return['filename'] = file.filename
    #_return["shape"] = vol.shape
    # _return["filetype"] = vol.shape
    #_return['predict'] = str(pred[0])
    return str(pred[0])

    #extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png", "nii")
    #if not extension:
    #    return "Image must be jpg or png format!"
    #image = read_imagefile(file.read())
    #return "merci"
