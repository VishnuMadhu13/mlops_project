import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Enable MLflow tracking
mlflow.set_experiment("iris_classification")

# 2. Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

n_estimators = 50
max_depth = 3

with mlflow.start_run() as run:
    # 3. Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluate metrics
    accuracy = model.score(X_test, y_test)
    
    # 5. Log parameters, metrics, and artifact model to MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)
    
    # Register the model version into MLflow Model Registry
    mlflow.sklearn.log_model(model, "model", registered_model_name="IrisRandomForest")
    
    print(f"Model logged with Run ID: {run.info.run_id}")
    print(f"Test Accuracy: {accuracy:.4f}")