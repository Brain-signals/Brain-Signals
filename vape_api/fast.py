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
import numpy as np
from pydantic import BaseModel
import base64

class Item(BaseModel):
    image: str
    size: int
    height: int
    width: int
    channel: int

def image_from_dict(api_dict, dtype='uint8', encoding='utf-8'):
    '''
    Convert an item representing a batch of images into a ndarray
    item is an instance of an Item class,
    inheriting from BaseModel pydantic class
    ----------
    Parameters
    ----------
    api_dict: an item(image, height, width, channel) representing an image
    dtype: target data type for ndarray
    encoding: encoding used for image string
    ----------
    Returns
    ----------
    ndarray of shape (size, height, width, channel)
    '''
    # Decode image string
    img = base64.b64decode(bytes(api_dict.image, encoding))
    # Convert to np.ndarray and ensure dtype
    img = np.frombuffer(img, dtype=dtype)
    # Reshape to original shape
    img = img.reshape((api_dict.size,
                       api_dict.height,
                       api_dict.width,
                       api_dict.channel))

    return img

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
#def predict_api(file: UploadFile = File(...)):
async def predict_api(file: UploadFile = File(...)):
    file.filename = "file"
    content_img = await file.read()

    # img=image_from_dict(item)
    pred = predict.predict_from_volume(content_img)


    return {'pred':pred}

    #extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png", "nii")
    #if not extension:
    #    return "Image must be jpg or png format!"
    #image = read_imagefile(file.read())
    #return "merci"

    #image = np.array(Image.open(file.file))

    #file.filename='file'
    #file_img= await file.read()
    #vol=file_img.get_fdata()
    #vol=nib.load(file_img).get_fdata()

    #f = open("file.nii", 'wb')
    #f.write('test')
    #shutil.copyfileobj(file.file, f)

    # with open("file.nii", 'wb') as f:
    #     shutil.copyfileobj(file.file, f)
    #     return {'check':'check'}

    #vol=nib.load(file.filename).get_fdata()

    #_return = {"Start of answer":file}
    # _return['ls'] = os.listdir()
    #_return['filename'] = file.filename
    #_return["shape"] = vol.shape
    # _return["filetype"] = vol.shape
    #_return['predict'] = str(pred[0])
