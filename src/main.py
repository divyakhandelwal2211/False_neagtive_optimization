# src/train.py

from data_load import load_data
from preprocess import preprocess_data
from model import build_model
from evaluate import evaluate_model
import joblib


def main():

    # Load Data
    df = load_data("data/disease_prediction_dataset.csv")

    # Preprocess
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df, "Disease")

    # Build Model
    model = build_model()

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Save model
    joblib.dump(model, "models/trained_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("Model saved successfully.")


if __name__ == "__main__":
    main()