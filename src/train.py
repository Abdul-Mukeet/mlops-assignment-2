import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Define global paths
DATA_PATH = 'data/dataset.csv'
MODEL_OUTPUT_PATH = 'models/model.pkl'


def load_data(file_path):
    """Loads the CSV file and returns a DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def split_data(df):
    """Splits data into X and y."""
    # Assuming last column is target. Adjust if needed.
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


def train_model(X, y):
    """Trains a simple Logistic Regression model."""
    model = LogisticRegression()
    model.fit(X, y)
    return model


def save_model(model, output_path):
    """Saves the trained model to a file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)


def main():
    print(f"Loading data from {DATA_PATH}...")
    df = load_data(DATA_PATH)

    print("Preprocessing data...")
    X, y = split_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    model = train_model(X_train, y_train)

    print(f"Saving model to {MODEL_OUTPUT_PATH}...")
    save_model(model, MODEL_OUTPUT_PATH)
    print("Training complete!")


if __name__ == "__main__":
    main()