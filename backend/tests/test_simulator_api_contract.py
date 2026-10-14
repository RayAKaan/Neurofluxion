"""HTTP contracts for the inspectable NumPy simulator path."""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from app import app

    with TestClient(app) as test_client:
        yield test_client


def test_simulator_build_forward_backward_share_graph_id(client):
    architecture = {
        "layers": [
            {"type": "input", "neurons": 2, "input_shape": [2]},
            {"type": "dense", "neurons": 3, "activation": "tanh"},
            {"type": "output", "neurons": 2, "activation": "linear"},
        ]
    }
    built = client.post("/api/simulator/architecture/build", json=architecture)
    assert built.status_code == 200, built.text
    graph_id = built.json()["graph_id"]
    assert graph_id
    assert len(built.json()["weights"]) == 2

    forward = client.post(
        "/api/simulator/forward/full",
        json={"graph_id": graph_id, "input": [0.4, -0.7]},
    )
    assert forward.status_code == 200, forward.text
    forward_data = forward.json()
    assert forward_data["steps"]
    assert len(forward_data["final_output"]) == 2
    assert np.isfinite(forward_data["final_output"]).all()

    backward = client.post(
        "/api/simulator/backward/full",
        json={
            "graph_id": graph_id,
            "input": [0.4, -0.7],
            "target": [0.2, -0.1],
            "loss_function": "mse",
        },
    )
    assert backward.status_code == 200, backward.text
    backward_data = backward.json()
    assert np.isfinite(backward_data["loss_value"])
    assert len(backward_data["gradients_W"]) == 2
    assert len(backward_data["gradients_b"]) == 2
    assert all(np.isfinite(values).all() for values in backward_data["gradients_W"])
    assert all(np.isfinite(values).all() for values in backward_data["gradients_b"])


def test_simulator_forward_unknown_graph_returns_404(client):
    response = client.post(
        "/api/simulator/forward/full",
        json={"graph_id": "missing-graph-id", "input": [0.0, 1.0]},
    )
    assert response.status_code == 404
