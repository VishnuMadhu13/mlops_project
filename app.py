from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Fetch latest model directly from MLflow Model Registry
MODEL_URI = "models:/IrisRandomForest/latest"
model = mlflow.sklearn.load_model(MODEL_URI)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(features: IrisInput):
    # Prepare input dataframe
    data_df = pd.DataFrame([features.dict().values()], columns=[
        "sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"
    ])
    
    # Inference
    prediction = model.predict(data_df)
    return {"class_id": int(prediction[0])}