from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
from src.schemas import  AgePrediction, AgePredictionRequest, AgePredicted, AgePredictionResult
from PIL import Image
import io
import requests
import torch
from src.predictor import AgePredictor

app = FastAPI(
    title="Predicting_age_by_photo"
)

DEVICE='cuda:0' if torch.cuda.is_available() else 'cpu'

model = AgePredictor(root_dir = 'model_store', device=DEVICE)

@app.post('/predict', response_model=AgePredictionResult)
async def predict(request: AgePredictionRequest):
    results = []
    for age_prediction in request.images: 
        response = requests.get(age_prediction.image_url)
        image = Image.open(io.BytesIO(response.content))
        image = model.transform_image(image)
        age = model.get_age_by_image(image)
        result = AgePredicted(id=age_prediction.id, age=age)
        results.append(result)
    return AgePredictionResult(result=results)
