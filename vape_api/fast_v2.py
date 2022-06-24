from fastapi import FastAPI, UploadFile, File
from vape_model.files import NII_to_3Darray
from vape_model.predict import predict_from_volume
import os

app = FastAPI()

@app.get("/")
def root():
    return {'greeting': 'Hello la team'}



@app.post("/predict")
async def upload_nii(nii_file: UploadFile=File(...)):
    try:
        path = os.path.join('vape_api', 'tmp_files', nii_file.filename)
        with open(path, 'wb') as f:
            nii_content = await nii_file.read()
            f.write(nii_content)

        vol = NII_to_3Darray(path)
        y_pred = predict_from_volume(vol)
        pred = 'unknow'
        for key,value in y_pred.items():
            if int(round(value)) == 1:
                pred = key

    except Exception:
        return {"message": "There was an error uploading the file"}

    finally:
        await nii_file.close()
        os.remove(path)

    return {'pred':pred}
