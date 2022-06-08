
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pandas import to_datetime
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

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
    return "merci pour l'upload"

    #extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png", "nii")
    #if not extension:
    #    return "Image must be jpg or png format!"
    #image = read_imagefile(file.read())
    #return "merci"
