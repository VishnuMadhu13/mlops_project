import os
import sqlite3
import pytest
import mlflow
from src.monitor_drift import check_data_drift
from src.retrain_pipeline import train_and_register_model, run_pipeline, MODEL_NAME


def test_data_drift_detection():
    """Verify drift detection output interface."""
    drift_status = check_data_drift()
    assert isinstance(drift_status, bool)
    assert drift_status is True


def test_train_and_register_model_execution():
    """Test model training and verification of MLflow run logging."""
    # Ensure tracking URI matches pipeline setup
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # Execute retraining step
    train_and_register_model()
    
    # Query MLflow tracking server for the experiment
    experiment = mlflow.get_experiment_by_name("iris-continuous-retraining")
    assert experiment is not None
    
    # Fetch runs associated with the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) > 0
    
    # Verify logged metrics
    latest_run = runs.iloc[0]
    assert "metrics.accuracy" in latest_run
    assert latest_run["metrics.accuracy"] >= 0.90


def test_full_retrain_pipeline_trigger(capsys):
    """Verify end-to-end execution flow of the retrain pipeline."""
    run_pipeline()
    captured = capsys.readouterr()
    
    assert "Checking production data for drift..." in captured.out
    assert "Data drift confirmed! Triggering retraining pipeline..." in captured.out