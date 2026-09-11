import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from monitor_drift import check_data_drift

# Configure local tracking database
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris-continuous-retraining")

MODEL_NAME = "IrisRandomForestModel"

def train_and_register_model():
    print("Initiating retraining process...")
    
    # 1. Load fresh dataset (incorporating new production records)
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    with mlflow.start_run() as run:
        # 2. Train model hyper-parameters
        n_estimators = 100
        max_depth = 5
        
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # 3. Evaluate candidate model
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        # 4. Log params and metrics to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)

        print(f"Candidate Model Trained. Accuracy: {acc:.4f}")

        # 5. Log and register the candidate model if it meets quality gate
        if acc >= 0.90:
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=MODEL_NAME
            )
            print(f"Model successfully logged and registered under URI: {model_info.model_uri}")
        else:
            print("Candidate model failed accuracy threshold (0.90). Promotion skipped.")

def run_pipeline():
    print("Checking production data for drift...")
    drift_detected = check_data_drift()

    if drift_detected:
        print("Data drift confirmed! Triggering retraining pipeline...")
        train_and_register_model()
    else:
        print("No significant drift detected. Model operating within normal bounds.")

if _name_ == "_main_":
    run_pipeline()