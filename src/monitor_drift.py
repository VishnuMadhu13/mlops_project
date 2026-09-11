from scipy.stats import ks_2samp
from sklearn.datasets import load_iris
from evidently import Report
from evidently.presets import DataDriftPreset


def check_data_drift():
    """
    Check for data drift between reference Iris data
    and simulated production data.
    """

    # 1. Load reference data
    iris = load_iris(as_frame=True)

    reference = iris.frame.sample(
        n=100,
        random_state=42
    )

    # 2. Simulate current production data with artificial drift
    current = iris.frame.sample(
        n=100,
        random_state=99
    ).copy()

    current["sepal length (cm)"] = (
        current["sepal length (cm)"] * 1.5
    )

    # 3. Create Evidently report
    drift_report = Report(
        metrics=[DataDriftPreset()]
    )

    # 4. Run drift analysis
    snapshot = drift_report.run(
        reference_data=reference,
        current_data=current
    )

    # 5. Save HTML report
    if hasattr(snapshot, "save_html"):
        snapshot.save_html("drift_report.html")

    elif hasattr(snapshot, "get_html"):
        with open(
            "drift_report.html",
            "w",
            encoding="utf-8"
        ) as f:
            f.write(snapshot.get_html())

    # 6. KS statistical test
    drifted_cols = sum(
        1
        for col in iris.feature_names
        if ks_2samp(
            reference[col],
            current[col]
        ).pvalue < 0.05
    )

    # Consider the dataset drifted when at least
    # half of the features show significant drift.
    dataset_drift = drifted_cols >= (
        len(iris.feature_names) / 2
    )

    print(
        f"Dataset Drift Detected: {dataset_drift}"
    )

    return dataset_drift


if __name__ == "__main__":
    check_data_drift()