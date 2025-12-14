import pytest
import pandas as pd
import os
from src.train import load_data, split_data, train_model


# Create a dummy CSV file for testing
@pytest.fixture
def dummy_data(tmp_path):
    # Create a temporary CSV file with random data
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1],
        'target': [0, 1, 0, 1, 0]
    })
    file_path = tmp_path / "dataset.csv"
    df.to_csv(file_path, index=False)
    return file_path


def test_load_data(dummy_data):
    """Test if data loads correctly."""
    df = load_data(dummy_data)
    assert not df.empty
    assert df.shape == (5, 3)


def test_shape_validation(dummy_data):
    """Test if data splitting works correctly."""
    df = load_data(dummy_data)
    X, y = split_data(df)

    # Check shapes
    assert X.shape == (5, 2)  # 2 features
    assert len(y) == 5  # 5 targets


def test_model_training(dummy_data):
    """Test if the model trains without error."""
    df = load_data(dummy_data)
    X, y = split_data(df)

    # Train model
    model = train_model(X, y)

    # Check if model object is created
    assert model is not None
    # Basic check to see if it has the 'predict' method
    assert hasattr(model, "predict")