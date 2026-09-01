import os
import pandas as pd
import pickle
import numpy as np
from main import load_and_preprocess_data, custom_train_test_split, LinearSVM, LogisticRegression


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data", "diabetes_dataset.csv")
df = pd.read_csv(data_path)


#This script trains our models that we implemented ourselves! This is from main but it makes it more clear and distinct here :)
def save_trained_models():
    # Load preprocessed features (no manual intercept column). Each model adds
    # its own intercept internally, so the bias term is handled exactly once.
    X, y, feature_names, scaler = load_and_preprocess_data(data_path)
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42)

    # The learned weight vector is [intercept, *feature_names].
    weight_names = ['intercept'] + feature_names

    lr = LogisticRegression(learning_rate=0.1, max_iter=10000, tol=1e-8)
    lr.fit(X_train, y_train)
    svm = LinearSVM(learning_rate=0.01, lambda_param=0.0005, max_iter=10000, tol=1e-8)
    svm.fit(X_train, y_train)

    # Save model, weight-vector names, and the fitted scaler together so the app
    # can scale new inputs identically to training.
    with open(os.path.join(BASE_DIR, "logistic_regression_model.pkl"), "wb") as f:
        pickle.dump((lr, weight_names, scaler), f)

    with open(os.path.join(BASE_DIR, "svm_model.pkl"), "wb") as f:
        pickle.dump((svm, weight_names, scaler), f)

    print("Saved logistic_regression_model.pkl and svm_model.pkl")


if __name__ == "__main__":
    save_trained_models()