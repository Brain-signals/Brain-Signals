from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import io
from vape_model.predict import predict_from_volume
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
    return {'greeting': 'Hello'}

@app.post("/upload_file/")
async def create_upload_file(file: UploadFile):

    with open("tmp.nii", 'wb') as f:
         shutil.copyfileobj(file.file, f)

    vol=nib.load(file).get_fdata()
    return 'ok'
