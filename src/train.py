import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Define file paths (Must match your DVC command)
DATA_PATH = 'data/dataset.csv'
MODEL_OUTPUT_PATH = 'models/model.pkl'


def train():
    # 1. Load the dataset
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: File not found at {DATA_PATH}. Please make sure data/dataset.csv exists.")
        return

    # 2. Preprocessing (Assuming the last column is the target)
    # Adjust 'iloc' if your target column is different
    X = df.iloc[:, :-1]  # All columns except the last one
    y = df.iloc[:, -1]  # The last column is the target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train the model
    print("Training Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 4. Save the model
    print(f"Saving model to {MODEL_OUTPUT_PATH}...")

    # Ensure the 'models' directory exists
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)

    with open(MODEL_OUTPUT_PATH, 'wb') as f:
        pickle.dump(model, f)

    print("Training complete!")


if __name__ == "__main__":
    train()