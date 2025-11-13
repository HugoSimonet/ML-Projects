"""Comprehensive API endpoint tests using FastAPI TestClient."""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import MagicMock, patch

import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after path setup
from fastapi.testclient import TestClient
from src.api.app import app, loaded_models, model_metadata


class MockRecommender:
    """Mock recommender for testing API endpoints."""

    def __init__(self, name: str = "mock_model"):
        self.name = name
        self.is_trained = True
        self._user_items = {
            1: [101, 102, 103],
            2: [201, 202, 203],
            3: [301, 302, 303],
        }

    def recommend(self, user_id: int, n_items: int = 10, exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """Mock recommendation method."""
        # Return mock recommendations with scores
        base_items = self._user_items.get(user_id, [1, 2, 3])
        recommendations = [
            (item_id, 5.0 - i * 0.1) for i, item_id in enumerate(base_items[:n_items])
        ]
        return recommendations

    def predict(self, user_id: int, item_id: int) -> float:
        """Mock prediction method."""
        # Return mock rating prediction
        return 4.5

    def batch_recommend(self, user_ids: List[int], n_items: int = 10) -> Dict[int, List[Tuple[int, float]]]:
        """Mock batch recommendation method."""
        results = {}
        for user_id in user_ids:
            results[user_id] = self.recommend(user_id, n_items)
        return results


@pytest.fixture(scope="function")
def client():
    """Create test client with mock models."""
    # Clear any existing models
    loaded_models.clear()
    model_metadata.clear()

    # Add mock models
    mock_model = MockRecommender("test_model")
    loaded_models["test_model"] = mock_model
    model_metadata["test_model"] = {
        'name': 'test_model',
        'type': 'MockRecommender',
        'is_trained': True,
        'path': '/mock/path/test_model.pkl'
    }

    # Add second model for testing multiple models
    mock_model2 = MockRecommender("test_model_2")
    loaded_models["test_model_2"] = mock_model2
    model_metadata["test_model_2"] = {
        'name': 'test_model_2',
        'type': 'MockRecommender',
        'is_trained': True,
        'path': '/mock/path/test_model_2.pkl'
    }

    yield TestClient(app)

    # Cleanup
    loaded_models.clear()
    model_metadata.clear()


@pytest.fixture(scope="function")
def empty_client():
    """Create test client without any loaded models."""
    # Clear any existing models
    loaded_models.clear()
    model_metadata.clear()

    yield TestClient(app)

    # Cleanup
    loaded_models.clear()
    model_metadata.clear()


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns welcome message."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Welcome" in data["message"]
        assert "version" in data
        assert "docs" in data

    def test_root_endpoint_structure(self, client):
        """Test root endpoint response structure."""
        response = client.get("/")
        data = response.json()

        assert isinstance(data, dict)
        assert data["version"] == "1.0.0"
        assert data["docs"] == "/docs"


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check_with_models(self, client):
        """Test health check with loaded models."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["n_models_loaded"] == 2
        assert "test_model" in data["models"]
        assert "test_model_2" in data["models"]

    def test_health_check_without_models(self, empty_client):
        """Test health check without loaded models."""
        response = empty_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["n_models_loaded"] == 0
        assert data["models"] == []


class TestModelsEndpoint:
    """Tests for models listing endpoint."""

    def test_list_models(self, client):
        """Test listing all loaded models."""
        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

        # Check model info structure
        model_info = data[0]
        assert "name" in model_info
        assert "type" in model_info
        assert "is_trained" in model_info

    def test_list_models_empty(self, empty_client):
        """Test listing models when none are loaded."""
        response = empty_client.get("/models")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0


class TestRecommendEndpoint:
    """Tests for recommendation endpoint."""

    def test_recommend_basic(self, client):
        """Test basic recommendation request."""
        request_data = {
            "user_id": 1,
            "n_items": 5,
            "exclude_seen": True
        }

        response = client.post("/recommend", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == 1
        assert len(data["recommendations"]) <= 5
        assert data["model_used"] in ["test_model", "test_model_2"]
        assert data["n_recommendations"] == len(data["recommendations"])

    def test_recommend_with_specific_model(self, client):
        """Test recommendation with specific model."""
        request_data = {
            "user_id": 2,
            "n_items": 10,
            "model_name": "test_model_2"
        }

        response = client.post("/recommend", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["model_used"] == "test_model_2"
        assert data["user_id"] == 2

    def test_recommend_structure(self, client):
        """Test recommendation response structure."""
        request_data = {
            "user_id": 1,
            "n_items": 3
        }

        response = client.post("/recommend", json=request_data)
        data = response.json()

        # Check recommendation items structure
        for rec in data["recommendations"]:
            assert "item_id" in rec
            assert "score" in rec
            assert "rank" in rec
            assert isinstance(rec["item_id"], int)
            assert isinstance(rec["score"], float)
            assert isinstance(rec["rank"], int)

    def test_recommend_ranking_order(self, client):
        """Test that recommendations are properly ranked."""
        request_data = {
            "user_id": 1,
            "n_items": 3
        }

        response = client.post("/recommend", json=request_data)
        data = response.json()

        recommendations = data["recommendations"]
        # Check ranks are sequential
        for i, rec in enumerate(recommendations):
            assert rec["rank"] == i + 1

        # Check scores are in descending order
        scores = [rec["score"] for rec in recommendations]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_without_models(self, empty_client):
        """Test recommendation when no models are loaded."""
        request_data = {
            "user_id": 1,
            "n_items": 5
        }

        response = empty_client.post("/recommend", json=request_data)

        assert response.status_code == 503
        assert "No models loaded" in response.json()["detail"]

    def test_recommend_with_invalid_model(self, client):
        """Test recommendation with non-existent model."""
        request_data = {
            "user_id": 1,
            "n_items": 5,
            "model_name": "non_existent_model"
        }

        response = client.post("/recommend", json=request_data)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_recommend_validation(self, client):
        """Test request validation for recommendations."""
        # Missing required user_id
        response = client.post("/recommend", json={"n_items": 5})
        assert response.status_code == 422

        # Invalid n_items (too large)
        response = client.post("/recommend", json={"user_id": 1, "n_items": 150})
        assert response.status_code == 422

        # Invalid n_items (negative)
        response = client.post("/recommend", json={"user_id": 1, "n_items": -1})
        assert response.status_code == 422

    def test_recommend_model_error_handling(self, client):
        """Test error handling when model raises exception."""
        # Mock model to raise exception
        def raise_error(*args, **kwargs):
            raise ValueError("Model error")

        loaded_models["test_model"].recommend = raise_error

        request_data = {
            "user_id": 1,
            "n_items": 5
        }

        response = client.post("/recommend", json=request_data)

        assert response.status_code == 500
        assert "Model error" in response.json()["detail"]


class TestPredictEndpoint:
    """Tests for rating prediction endpoint."""

    def test_predict_basic(self, client):
        """Test basic rating prediction."""
        request_data = {
            "user_id": 1,
            "item_id": 101
        }

        response = client.post("/predict", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == 1
        assert data["item_id"] == 101
        assert isinstance(data["predicted_rating"], float)
        assert data["model_used"] in ["test_model", "test_model_2"]

    def test_predict_with_specific_model(self, client):
        """Test prediction with specific model."""
        request_data = {
            "user_id": 2,
            "item_id": 202,
            "model_name": "test_model_2"
        }

        response = client.post("/predict", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["model_used"] == "test_model_2"

    def test_predict_without_models(self, empty_client):
        """Test prediction when no models are loaded."""
        request_data = {
            "user_id": 1,
            "item_id": 101
        }

        response = empty_client.post("/predict", json=request_data)

        assert response.status_code == 503
        assert "No models loaded" in response.json()["detail"]

    def test_predict_with_invalid_model(self, client):
        """Test prediction with non-existent model."""
        request_data = {
            "user_id": 1,
            "item_id": 101,
            "model_name": "non_existent_model"
        }

        response = client.post("/predict", json=request_data)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_predict_validation(self, client):
        """Test request validation for predictions."""
        # Missing user_id
        response = client.post("/predict", json={"item_id": 101})
        assert response.status_code == 422

        # Missing item_id
        response = client.post("/predict", json={"user_id": 1})
        assert response.status_code == 422

    def test_predict_model_error_handling(self, client):
        """Test error handling when model raises exception."""
        # Mock model to raise exception
        def raise_error(*args, **kwargs):
            raise RuntimeError("Prediction failed")

        loaded_models["test_model"].predict = raise_error

        request_data = {
            "user_id": 1,
            "item_id": 101
        }

        response = client.post("/predict", json=request_data)

        assert response.status_code == 500
        assert "Prediction failed" in response.json()["detail"]


class TestSimpleRecommendEndpoint:
    """Tests for simple GET recommendation endpoint."""

    def test_simple_recommend_basic(self, client):
        """Test simple GET recommendation endpoint."""
        response = client.get("/recommend/1")

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == 1
        assert len(data["recommendations"]) <= 10  # Default

    def test_simple_recommend_with_params(self, client):
        """Test simple recommendation with query parameters."""
        response = client.get("/recommend/2?n=5&model=test_model_2")

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == 2
        assert len(data["recommendations"]) <= 5
        assert data["model_used"] == "test_model_2"

    def test_simple_recommend_validation(self, client):
        """Test validation for simple recommendation endpoint."""
        # Invalid n (too large)
        response = client.get("/recommend/1?n=150")
        assert response.status_code == 422

        # Invalid n (negative)
        response = client.get("/recommend/1?n=-5")
        assert response.status_code == 422


class TestBatchRecommendEndpoint:
    """Tests for batch recommendation endpoint."""

    def test_batch_recommend_basic(self, client):
        """Test basic batch recommendation."""
        request_data = {
            "user_ids": [1, 2, 3],
            "n_items": 5
        }

        response = client.post("/batch_recommend", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["n_users"] == 3
        assert len(data["results"]) == 3
        assert data["model_used"] in ["test_model", "test_model_2"]

    def test_batch_recommend_structure(self, client):
        """Test batch recommendation response structure."""
        request_data = {
            "user_ids": [1, 2],
            "n_items": 3
        }

        response = client.post("/batch_recommend", json=request_data)
        data = response.json()

        # Check each user's recommendations
        for result in data["results"]:
            assert "user_id" in result
            assert "recommendations" in result
            assert "n_recommendations" in result

            # Check recommendation structure
            for rec in result["recommendations"]:
                assert "item_id" in rec
                assert "score" in rec
                assert "rank" in rec

    def test_batch_recommend_with_model(self, client):
        """Test batch recommendation with specific model."""
        request_data = {
            "user_ids": [1, 2],
            "n_items": 5,
            "model_name": "test_model_2"
        }

        response = client.post("/batch_recommend", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["model_used"] == "test_model_2"

    def test_batch_recommend_empty_list(self, client):
        """Test batch recommendation with empty user list."""
        request_data = {
            "user_ids": [],
            "n_items": 5
        }

        response = client.post("/batch_recommend", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["n_users"] == 0
        assert len(data["results"]) == 0

    def test_batch_recommend_without_models(self, empty_client):
        """Test batch recommendation when no models are loaded."""
        request_data = {
            "user_ids": [1, 2],
            "n_items": 5
        }

        response = empty_client.post("/batch_recommend", json=request_data)

        assert response.status_code == 503
        assert "No models loaded" in response.json()["detail"]

    def test_batch_recommend_with_invalid_model(self, client):
        """Test batch recommendation with non-existent model."""
        request_data = {
            "user_ids": [1, 2],
            "n_items": 5,
            "model_name": "non_existent_model"
        }

        response = client.post("/batch_recommend", json=request_data)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_batch_recommend_model_error(self, client):
        """Test error handling in batch recommendations."""
        # Mock model to raise exception
        def raise_error(*args, **kwargs):
            raise Exception("Batch processing error")

        loaded_models["test_model"].batch_recommend = raise_error

        request_data = {
            "user_ids": [1, 2],
            "n_items": 5
        }

        response = client.post("/batch_recommend", json=request_data)

        assert response.status_code == 500
        assert "Batch processing error" in response.json()["detail"]


class TestLoadModelEndpoint:
    """Tests for model loading endpoint."""

    @patch('joblib.load')
    def test_load_model_success(self, mock_joblib_load, client):
        """Test successful model loading."""
        # Mock joblib.load to return our mock model
        mock_model = MockRecommender("new_model")
        mock_joblib_load.return_value = mock_model

        response = client.post(
            "/load_model",
            json={"model_path": "/path/to/new_model.pkl"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "Model 'new_model' loaded successfully" in data["message"]
        assert "model_info" in data
        assert data["model_info"]["name"] == "new_model"

    @patch('joblib.load')
    def test_load_model_error(self, mock_joblib_load, client):
        """Test model loading error handling."""
        # Mock joblib.load to raise exception
        mock_joblib_load.side_effect = Exception("File not found")

        response = client.post(
            "/load_model",
            json={"model_path": "/invalid/path.pkl"}
        )

        assert response.status_code == 500
        assert "File not found" in response.json()["detail"]


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_large_n_items_clamped(self, client):
        """Test that n_items > 100 is rejected."""
        request_data = {
            "user_id": 1,
            "n_items": 1000
        }

        response = client.post("/recommend", json=request_data)
        assert response.status_code == 422

    def test_zero_n_items_rejected(self, client):
        """Test that n_items = 0 is rejected."""
        request_data = {
            "user_id": 1,
            "n_items": 0
        }

        response = client.post("/recommend", json=request_data)
        assert response.status_code == 422

    def test_negative_user_id(self, client):
        """Test handling of negative user_id."""
        request_data = {
            "user_id": -1,
            "n_items": 5
        }

        # API should accept but model might handle differently
        response = client.post("/recommend", json=request_data)
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 400, 500]

    def test_concurrent_requests(self, client):
        """Test that API can handle multiple requests."""
        responses = []

        for user_id in [1, 2, 3]:
            response = client.post(
                "/recommend",
                json={"user_id": user_id, "n_items": 5}
            )
            responses.append(response)

        # All requests should succeed
        for response in responses:
            assert response.status_code == 200


class TestDocumentation:
    """Tests for API documentation endpoints."""

    def test_openapi_schema_exists(self, client):
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data

    def test_docs_redirect(self, client):
        """Test that /docs endpoint is accessible."""
        response = client.get("/docs", follow_redirects=False)
        # Should either succeed or redirect
        assert response.status_code in [200, 307]


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
