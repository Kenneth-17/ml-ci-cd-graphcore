# tests/test_model.py
import pytest
import torch
from src.model import SimpleCNN
from src.train import train_model

@pytest.fixture
def model():
    return SimpleCNN()

def test_model_forward(model):
    model.eval()
    input_tensor = torch.randn(1, 3, 32, 32)
    output = model(input_tensor)
    assert output.shape == (1, 10), "Output shape mismatch"

def test_training():
    # This test ensures that training runs without errors
    try:
        train_model(epochs=1, batch_size=8, learning_rate=0.001)
    except Exception as e:
        pytest.fail(f"Training failed with exception: {e}")