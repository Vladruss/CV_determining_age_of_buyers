# Predicting age by photo
This project is an example of using machine learning to predict age by photo.
The best trained model was integrated in Api using FastAPI. The web application was packaged in a Docker container.

## Requirements:
* Python 3.10+

## Training and tuning models
Training and tuning model in jupyter [predicting_age_by_photo.ipynb](https://github.com/Vladruss/TelecomCustomerOutflow/blob/main/telecom_customer_outflow.ipynb)

## Running the model locally in fastapi
1. Activate the environment and install dependencies
```
source /path/to/venv/bin/activate
pip install -r requirements.txt
```

2. Launch the service
```
uvicorn main:app --reload
```

## Deployment with Docker
1. Build the Docker image
```
docker build -t fastapi .
```
3. Running the Docker image
```
docker run -d -p 8000:8000 fastapi
```
