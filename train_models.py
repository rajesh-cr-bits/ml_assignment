"""Train and evaluate the Assignment 2 classification models."""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
DATA_PATH = ROOT / "test_data.csv"
RANDOM_STATE = 42


def load_dataset():
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data.copy()
    target = dataset.target.rename("target")
    # Use readable class labels while retaining a binary numeric target for models.
    target = target.map({0: "malignant", 1: "benign"})
    return features, target


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "kNN": Pipeline([
            ("scale", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=7)),
        ]),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def evaluate(model, x_test, y_test):
    predictions = model.predict(x_test)
    benign_index = list(model.classes_).index("benign")
    probabilities = model.predict_proba(x_test)[:, benign_index]
    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score((y_test == "benign").astype(int), probabilities),
        "Precision": precision_score(y_test, predictions, pos_label="benign"),
        "Recall": recall_score(y_test, predictions, pos_label="benign"),
        "F1": f1_score(y_test, predictions, pos_label="benign"),
        "MCC": matthews_corrcoef(y_test, predictions),
    }


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    x, y = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    metrics = {}
    for name, model in build_models().items():
        model.fit(x_train, y_train)
        metrics[name] = evaluate(model, x_test, y_test)
        joblib.dump(model, MODEL_DIR / (name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"))

    test_data = x_test.copy()
    test_data["target"] = y_test
    test_data.to_csv(DATA_PATH, index=False)
    (MODEL_DIR / "feature_names.json").write_text(json.dumps(list(x.columns), indent=2))
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Dataset: {len(x)} rows, {x.shape[1]} features")
    print(f"Test data written to: {DATA_PATH}")
    print(pd.DataFrame(metrics).T.round(4).to_string())


if __name__ == "__main__":
    main()