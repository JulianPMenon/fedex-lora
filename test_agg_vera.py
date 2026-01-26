import torch
import types
import pytest
from fed_agg import aggregate_models_ours_vera

class DummyModule(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.vera_A = torch.nn.Parameter(torch.randn(size, size))
        self.vera_B = torch.nn.Parameter(torch.randn(size, size))
        self.register_parameter('vera_A.default.weight', self.vera_A)
        self.register_parameter('vera_B.default.weight', self.vera_B)
        self.vera_lambda_b = torch.nn.Parameter(torch.diag(torch.ones(size)))
        self.vera_lambda_d = torch.nn.Parameter(torch.diag(torch.ones(size)))
        self.register_parameter('vera_lambda_b.default.weight', self.vera_lambda_b)
        self.register_parameter('vera_lambda_d.default.weight', self.vera_lambda_d)
        self.base_layer = torch.nn.Parameter(torch.zeros(size, size))
        self.register_parameter('base_layer.weight', self.base_layer)
        self.classifier = torch.nn.Parameter(torch.ones(size, size))
        self.register_parameter('classifier.weight', self.classifier)

class Args:
    lora_alpha = 1.0
    lora_r = 1.0
    rslora = False

@pytest.fixture
def dummy_models():
    size = 4
    global_model = DummyModule(size)
    client_models = [DummyModule(size) for _ in range(3)]
    # Give each client different lambda_b and lambda_d
    for i, cm in enumerate(client_models):
        cm.vera_lambda_b.data = torch.diag(torch.ones(size) * (i+1))
        cm.vera_lambda_d.data = torch.diag(torch.ones(size) * (i+2))
    return global_model, client_models

def test_aggregate_models_ours_vera(dummy_models):
    global_model, client_models = dummy_models
    args = Args()
    updated_model = aggregate_models_ours_vera(global_model, client_models, args)
    # Check that global_model's lambda_b and lambda_d are the mean of clients
    expected_b = torch.stack([cm.vera_lambda_b.data for cm in client_models]).mean(0)
    expected_d = torch.stack([cm.vera_lambda_d.data for cm in client_models]).mean(0)
    assert torch.allclose(updated_model.vera_lambda_b.data, expected_b)
    assert torch.allclose(updated_model.vera_lambda_d.data, expected_d)
    # Check that base_layer was updated (not all zeros)
    assert not torch.all(updated_model.base_layer.data == 0)
    # Check that classifier was aggregated
    expected_classifier = torch.stack([cm.classifier.data for cm in client_models]).mean(0)
    assert torch.allclose(updated_model.classifier.data, expected_classifier)
