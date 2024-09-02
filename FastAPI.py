from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load pre-trained models
scaler = joblib.load('scaler.joblib')
kmeans = joblib.load('kmeans.joblib')

# City-to-integer mapping
city_mapping = {
    'Al Khobar': 0, 'AlUla': 1, 'Dammam': 2, 'Dhahran': 3, 'Jazan': 4, 
    'Jeddah': 5, 'Madinah': 6, 'Makkah': 7, 'Rabigh': 8, 'Riyadh': 9, 
    'Taif': 10, 'The Riyadh Province': 11
}

# Define FastAPI app
app = FastAPI()

# Define the request model
class RestaurantData(BaseModel):
    city: str
    rating: float
    num_reviewers: int

# Define the prediction endpoint
@app.post("/predict_cluster/")
async def predict_cluster(data: RestaurantData):
    # Validate input
    if data.city not in city_mapping:
        raise HTTPException(status_code=400, detail="Invalid city name")

    # Preprocess data: scale rating and number of reviewers
    city_encoded = city_mapping[data.city]
    input_data = np.array([[data.rating, data.num_reviewers]])
    data_scaled = scaler.transform(input_data)
    data_final = np.append(data_scaled, [[city_encoded]], axis=1)

    # Predict the cluster
    cluster = kmeans.predict(data_final)[0]

    return {"city": data.city, "rating": data.rating, "num_reviewers": data.num_reviewers, "predicted_cluster": int(cluster)}