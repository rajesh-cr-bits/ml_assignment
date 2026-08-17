"""Interactive Streamlit demonstration for Machine Learning Assignment 2."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"

st.set_page_config(page_title="Diagnostic Classifier Lab", page_icon="◎", layout="wide")


@st.cache_data
def dataset_info():
    dataset = load_breast_cancer(as_frame=True)
    return list(dataset.data.columns), dataset.target_names.tolist()


@st.cache_resource
def load_models():
    models = {}
    for path in MODEL_DIR.glob("*.joblib"):
        models[path.stem] = joblib.load(path)
    return models


def key_for_model(label):
    return label.lower().replace(" ", "_").replace("(", "").replace(")", "")


def main():
    feature_names, _ = dataset_info()
    models = load_models()
    st.title("Diagnostic Classifier Lab")
    st.caption("Five supervised classifiers evaluated on the UCI Breast Cancer Wisconsin Diagnostic dataset")

    with st.sidebar:
        st.header("Run an evaluation")
        uploaded = st.file_uploader("Upload labelled test data (CSV)", type="csv")
        model_label = st.selectbox("Choose a model", [
            "Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest (Ensemble)"
        ])
        st.divider()
        st.write("The CSV must contain the 30 dataset features and a `target` column.")

    data = pd.read_csv(uploaded) if uploaded else pd.read_csv(ROOT / "test_data.csv")
    missing = [column for column in feature_names if column not in data.columns]
    if missing or "target" not in data.columns:
        st.error("This file is missing the required feature columns or `target`. Use the included test_data.csv as the template.")
        st.stop()

    x = data[feature_names]
    y = data["target"].astype(str)
    model = models.get(key_for_model(model_label))
    predictions = model.predict(x)
    benign_index = list(model.classes_).index("benign")
    probabilities = model.predict_proba(x)[:, benign_index]
    metric_values = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": roc_auc_score((y == "benign").astype(int), probabilities),
        "Precision": precision_score(y, predictions, pos_label="benign", zero_division=0),
        "Recall": recall_score(y, predictions, pos_label="benign", zero_division=0),
        "F1": f1_score(y, predictions, pos_label="benign", zero_division=0),
        "MCC": matthews_corrcoef(y, predictions),
    }

    st.subheader(f"{model_label} results")
    columns = st.columns(6)
    for column, (name, value) in zip(columns, metric_values.items()):
        column.metric(name, f"{value:.3f}")

    left, right = st.columns([1, 1.35])
    with left:
        st.subheader("Confusion matrix")
        matrix = confusion_matrix(y, predictions, labels=["malignant", "benign"])
        figure, axis = plt.subplots(figsize=(4.5, 3.5))
        axis.imshow(matrix, cmap="Blues")
        axis.set_xticks([0, 1], ["malignant", "benign"])
        axis.set_yticks([0, 1], ["malignant", "benign"])
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("Actual label")
        for row in range(2):
            for col in range(2):
                axis.text(col, row, matrix[row, col], ha="center", va="center")
        figure.tight_layout()
        st.pyplot(figure)
    with right:
        st.subheader("Classification report")
        report = classification_report(y, predictions, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    st.subheader("Uploaded test data")
    st.write(f"{len(data)} rows evaluated")
    st.dataframe(data.head(10), use_container_width=True)


if __name__ == "__main__":
    main()