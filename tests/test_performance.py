# tests/test_performance.py
import pytest
import time
import torch
from src.model import SimpleCNN

def test_inference_performance():
    model = SimpleCNN()
    model.eval()
    input_tensor = torch.randn(1, 3, 32, 32)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            output = model(input_tensor)
    end_time = time.time()
    avg_inference_time = (end_time - start_time) / 100
    assert avg_inference_time < 0.05, f"Inference time too slow: {avg_inference_time}s"