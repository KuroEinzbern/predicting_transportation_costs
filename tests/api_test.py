from fastapi.testclient import TestClient
import pytest
from predicting_transportation_costs.api import api_endpoints


client = TestClient(api_endpoints.app)

@pytest.fixture
def client():
    with TestClient(api_endpoints.app) as client:
        yield client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict(client):
    payload = {
        "load_id": "TR-000001",
        "pickup": "Richmond",
        "delivery": "Baltimore",
        "pickup_lat": 38.09122,
        "pickup_lon": -76.78906,
        "delivery_lat": 38.16908,
        "delivery_lon": -72.74564,
        "distance": 274.3,
        "equipment": "Dry Van",
        "weight": 30658.0,
        "date": "2025-01-01",
        "market_index": 0.95684,
        "quote_signal": 2.39595,
        "posted_rate": 645.41
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()