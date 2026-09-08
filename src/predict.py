from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer


MODEL_PATH = Path("models") / "model.joblib"


def load_model():
    return joblib.load(MODEL_PATH)


def predict_sample():
    data = load_breast_cancer()

    sample = data.data[0].reshape(1, -1)

    model = load_model()

    prediction = model.predict(sample)[0]

    class_name = data.target_names[prediction]

    print(f"Prediction: {prediction}")
    print(f"Class: {class_name}")


if __name__ == "__main__":
    predict_sample()