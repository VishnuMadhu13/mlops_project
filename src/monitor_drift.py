import pandas as pd
from scipy.stats import ks_2samp
from sklearn.datasets import load_iris
from evidently import Report
from evidently.presets import DataDriftPreset

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