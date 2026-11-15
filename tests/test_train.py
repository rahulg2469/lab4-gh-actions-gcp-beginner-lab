import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_and_save_model as t  # noqa: E402


def test_data_shape():
    X, y = t.load_data()
    assert X.shape == (569, 30)
    assert set(y.unique()) == {0, 1}


def test_split_is_stratified_and_sized():
    X, y = t.load_data()
    X_train, X_test, y_train, y_test = t.split_data(X, y)
    assert len(X_train) == 455 and len(X_test) == 114
    # malignant share should be about the same in both splits
    assert abs((y_train == 0).mean() - (y_test == 0).mean()) < 0.02


def test_metrics_in_range():
    X, y = t.load_data()
    X_train, X_test, y_train, y_test = t.split_data(X, y)
    model = t.train_model(X_train, y_train, n_estimators=20)
    m = t.evaluate(model, X_test, y_test)
    for key in ["accuracy", "recall_malignant", "precision_malignant", "f1_malignant", "roc_auc"]:
        assert 0.0 <= m[key] <= 1.0


def test_gate_checks_both_metrics():
    assert t.check_gate({"accuracy": 0.95, "recall_malignant": 0.90}, 0.90, 0.85) == []
    assert len(t.check_gate({"accuracy": 0.95, "recall_malignant": 0.50}, 0.90, 0.85)) == 1
    assert len(t.check_gate({"accuracy": 0.50, "recall_malignant": 0.50}, 0.90, 0.85)) == 2


def test_main_writes_artifacts(tmp_path):
    code = t.main(["--n-estimators", "20", "--min-accuracy", "0.5", "--min-recall", "0.5",
                   "--output-dir", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "model.joblib").exists()
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["passed_gate"] is True


def test_main_fails_when_gate_impossible(tmp_path):
    assert t.main(["--min-accuracy", "1.01", "--output-dir", str(tmp_path)]) == 1
