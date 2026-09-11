from fastapi import FastAPI, HTTPException
import mlflow
import mlflow.sklearn
import pandas as pd
from pydantic import BaseModel


app = FastAPI(
    title="Iris MLOps Level 2 API"
)


MODEL_NAME = "IrisRandomForestModel"

# Model starts as unavailable.
# It will be loaded during application startup.
model = None


def load_model():
    """
    Load the latest registered model from MLflow.

    If the model is unavailable, keep the API running
    and allow the /health endpoint to report model_loaded=False.
    """

    global model

    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

        model_uri = f"models:/{MODEL_NAME}/latest"

        model = mlflow.sklearn.load_model(model_uri)

        print(
            f"Successfully loaded model: {MODEL_NAME}"
        )

    except Exception as e:
        print(
            f"Model loading warning: {e}"
        )
        model = None


@app.on_event("startup")
def startup_event():
    load_model()


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(features: IrisInput):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact unavailable."
        )

    data_df = pd.DataFrame(
        [[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )

    prediction = model.predict(data_df)

    return {
        "prediction": int(prediction[0])
    }