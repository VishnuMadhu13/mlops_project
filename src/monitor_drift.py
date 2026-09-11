import pandas as pd
from scipy.stats import ks_2samp
from sklearn.datasets import load_iris
from evidently import Report
from evidently.presets import DataDriftPreset
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Iris MLOps Level 2 API")

MODEL_NAME = "IrisRandomForestModel"

# Load latest model version from MLflow Registry
def load_latest_model():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model_uri = f"models:/{MODEL_NAME}/latest"
    try:
        return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"Registry load warning ({e}); loading local fallback artifact...")
        return mlflow.sklearn.load_model("models/model.pkl")

model = load_latest_model()

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(data: IrisInput):
    features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}


def check_data_drift():
    # 1. Load reference data (Iris baseline)
    iris = load_iris(as_frame=True)
    reference = iris.frame.sample(n=100, random_state=42)

    # 2. Simulate current production data with artificial drift
    current = iris.frame.sample(n=100, random_state=99)
    current['sepal length (cm)'] = current['sepal length (cm)'] * 1.5

    # 3. Create Report object
    drift_report = Report(metrics=[DataDriftPreset()])
    
    # 4. Run analysis (returns Snapshot)
    snapshot = drift_report.run(reference_data=reference, current_data=current)
    
    # 5. Export HTML from the snapshot safely
    if hasattr(snapshot, "save_html"):
        snapshot.save_html("drift_report.html")
    elif hasattr(snapshot, "get_html"):
        with open("drift_report.html", "w", encoding="utf-8") as f:
            f.write(snapshot.get_html())

    # 6. Statistical KS Test for pipeline execution logic
    drifted_cols = sum(
        1 for col in iris.feature_names 
        if ks_2samp(reference[col], current[col]).pvalue < 0.05
    )

    dataset_drift = drifted_cols >= (len(iris.feature_names) / 2)
    print(f"Dataset Drift Detected: {dataset_drift}")
    
    return dataset_drift

if __name__ == "__main__":
    check_data_drift()