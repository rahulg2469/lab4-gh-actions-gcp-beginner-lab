"""Train a Gradient Boosting classifier on Breast Cancer Wisconsin,
evaluate it, and enforce a quality gate.

Writes artifacts/model.joblib and artifacts/metrics.json locally.
Uploading to GCS is done by the workflow's upload job, so this script
needs no cloud credentials and runs the same locally and in CI.

Exit code 0 = passed the gate, 1 = failed the gate.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

MALIGNANT = 0  # in sklearn's breast cancer data: 0 = malignant, 1 = benign


def load_data():
    data = load_breast_cancer(as_frame=True)
    return data.data, data.target


def split_data(X, y, test_size=0.2, seed=42):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def train_model(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=3, seed=42):
    model = GradientBoostingClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=seed
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    proba_benign = model.predict_proba(X_test)[:, 1]
    r = lambda v: round(float(v), 4)
    return {
        "accuracy": r(accuracy_score(y_test, preds)),
        "recall_malignant": r(recall_score(y_test, preds, pos_label=MALIGNANT)),
        "precision_malignant": r(precision_score(y_test, preds, pos_label=MALIGNANT, zero_division=0)),
        "f1_malignant": r(f1_score(y_test, preds, pos_label=MALIGNANT, zero_division=0)),
        "roc_auc": r(roc_auc_score(y_test, proba_benign)),
    }


def check_gate(metrics, min_accuracy, min_recall):
    """Return a list of failure reasons (empty list = passed)."""
    failures = []
    if metrics["accuracy"] < min_accuracy:
        failures.append(f"accuracy {metrics['accuracy']} < {min_accuracy}")
    if metrics["recall_malignant"] < min_recall:
        failures.append(f"malignant recall {metrics['recall_malignant']} < {min_recall}")
    return failures


def save_artifacts(model, metrics, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return out / "model.joblib", out / "metrics.json"


def parse_args(argv=None):
    # CLI flags override env vars; env vars are how the workflow passes its inputs
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-estimators", type=int, default=int(os.getenv("N_ESTIMATORS", "100")))
    p.add_argument("--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "0.1")))
    p.add_argument("--max-depth", type=int, default=int(os.getenv("MAX_DEPTH", "3")))
    p.add_argument("--min-accuracy", type=float, default=float(os.getenv("MIN_ACCURACY", "0.90")))
    p.add_argument("--min-recall", type=float, default=float(os.getenv("MIN_RECALL", "0.85")))
    p.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "artifacts"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train, args.n_estimators, args.learning_rate, args.max_depth)
    metrics = evaluate(model, X_test, y_test)

    failures = check_gate(metrics, args.min_accuracy, args.min_recall)
    metrics.update({
        "thresholds": {"min_accuracy": args.min_accuracy, "min_recall_malignant": args.min_recall},
        "passed_gate": not failures,
        "params": {"n_estimators": args.n_estimators, "learning_rate": args.learning_rate,
                   "max_depth": args.max_depth},
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "run_number": os.getenv("GITHUB_RUN_NUMBER", "local"),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })

    model_path, metrics_path = save_artifacts(model, metrics, args.output_dir)
    print(json.dumps(metrics, indent=2))
    print(f"Saved {model_path} and {metrics_path}")

    if failures:
        print("❌ Quality gate FAILED: " + "; ".join(failures))
        return 1
    print("✅ Quality gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
