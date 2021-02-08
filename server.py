# import flask import Flask, request, jsonify
from torch_utils import transformImage, getPrediction
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowedFile(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # 1. load image
    # 2. image -> tensor
    # 3. prediction
    # 4. return result as json

    # ----------------IMPLEMENTATION----------------#
    if file is None or file.filename == '':
        return JSONResponse({'error': 'no file'})

    if not allowedFile(file.filename):
        return jsonify({'error': 'format not supported'})
        
    try:
        imgBytes = await file.read()
        tensor = transformImage(imgBytes)
        prediction = getPrediction(tensor)
        data = {
                'prediction': prediction.item(),
                'class_name': str(prediction.item())
                }
        return JSONResponse(data)
    except:
        return JSONResponse({'error': 'error during prediction'})


