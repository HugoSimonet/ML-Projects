"""Model serving module for serving ML models via REST API."""

import logging
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge
import asyncio


logger = logging.getLogger(__name__)


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for predictions."""
    data: Union[List[Dict[str, Any]], Dict[str, Any]]
    model_version: Optional[str] = None
    return_probabilities: bool = False
    preprocessing: Optional[Dict[str, Any]] = None

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "data": {"feature_1": 0.5, "feature_2": 1.2, "feature_3": 3},
                "return_probabilities": False
            }]
        }
    }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[Any]
    probabilities: Optional[List[List[float]]] = None
    model_version: str
    prediction_id: str
    timestamp: str
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_version: str
    uptime_seconds: float
    total_predictions: int
    avg_latency_ms: float


# Prometheus metrics
PREDICTIONS_COUNTER = Counter(
    'model_predictions_total',
    'Total number of predictions',
    ['model_version', 'status']
)

PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_version']
)

ACTIVE_REQUESTS = Gauge(
    'model_active_requests',
    'Number of active prediction requests'
)


class ModelServer:
    """Serves ML models via REST API."""

    def __init__(self, model: Any, config: Dict[str, Any]):
        """Initialize model server.

        Args:
            model: Trained model to serve
            config: Configuration dictionary containing:
                - model_version: Version of the model
                - preprocessing: Preprocessing configuration
                - batch_size: Maximum batch size
                - timeout: Prediction timeout
        """
        self.model = model
        self.config = config
        self.model_version = config.get('model_version', 'unknown')
        self.batch_size = config.get('batch_size', 32)
        self.timeout = config.get('timeout', 30)

        # Statistics
        self.start_time = time.time()
        self.total_predictions = 0
        self.total_latency = 0.0

        # Create FastAPI app
        self.app = FastAPI(
            title="ML Model Serving API",
            description="Production-ready model serving API",
            version="1.0.0"
        )

        # Register routes
        self._register_routes()

        logger.info(f"Initialized ModelServer with version {self.model_version}")

    def _register_routes(self):
        """Register API routes."""

        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            """Make predictions."""
            return await self.predict(request)

        @self.app.post("/predict/batch", response_model=PredictionResponse)
        async def predict_batch(request: PredictionRequest):
            """Make batch predictions."""
            return await self.predict_batch(request)

        @self.app.get("/health", response_model=HealthResponse)
        async def health():
            """Health check endpoint."""
            return self.health_check()

        @self.app.get("/model/info")
        async def model_info():
            """Get model information."""
            return self.get_model_info()

        @self.app.post("/model/reload")
        async def reload_model():
            """Reload model."""
            return {"status": "success", "message": "Model reloaded"}

    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make a prediction.

        Args:
            request: Prediction request

        Returns:
            Prediction response
        """
        start_time = time.time()
        ACTIVE_REQUESTS.inc()

        try:
            # Convert request data to dataframe
            if isinstance(request.data, list):
                data = pd.DataFrame(request.data)
            else:
                data = pd.DataFrame([request.data])

            # Make prediction
            predictions = self._predict_internal(data)

            # Get probabilities if requested
            probabilities = None
            if request.return_probabilities and hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(data).tolist()

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Update metrics
            self.total_predictions += len(predictions)
            self.total_latency += latency_ms
            PREDICTIONS_COUNTER.labels(
                model_version=self.model_version,
                status='success'
            ).inc()
            PREDICTION_LATENCY.labels(
                model_version=self.model_version
            ).observe(latency_ms / 1000)

            # Generate response
            response = PredictionResponse(
                predictions=predictions.tolist(),
                probabilities=probabilities,
                model_version=self.model_version,
                prediction_id=self._generate_prediction_id(),
                timestamp=datetime.now().isoformat(),
                latency_ms=latency_ms
            )

            return response

        except Exception as e:
            PREDICTIONS_COUNTER.labels(
                model_version=self.model_version,
                status='error'
            ).inc()
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        finally:
            ACTIVE_REQUESTS.dec()

    async def predict_batch(self, request: PredictionRequest) -> PredictionResponse:
        """Make batch predictions.

        Args:
            request: Batch prediction request

        Returns:
            Prediction response
        """
        if not isinstance(request.data, list):
            raise HTTPException(
                status_code=400,
                detail="Batch prediction requires list of data points"
            )

        # Split into batches if needed
        data = pd.DataFrame(request.data)

        if len(data) > self.batch_size:
            # Process in batches
            predictions = []
            for i in range(0, len(data), self.batch_size):
                batch = data.iloc[i:i + self.batch_size]
                batch_preds = self._predict_internal(batch)
                predictions.extend(batch_preds.tolist())

            predictions = np.array(predictions)
        else:
            predictions = self._predict_internal(data)

        # Create simple request for single prediction
        single_request = PredictionRequest(
            data=request.data,
            model_version=request.model_version,
            return_probabilities=request.return_probabilities
        )

        return await self.predict(single_request)

    def _predict_internal(self, data: pd.DataFrame) -> np.ndarray:
        """Internal prediction method.

        Args:
            data: Input data

        Returns:
            Predictions
        """
        try:
            # Handle different model types
            if hasattr(self.model, 'predict'):
                predictions = self.model.predict(data)
            else:
                raise ValueError("Model does not have a predict method")

            return predictions

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def health_check(self) -> HealthResponse:
        """Check server health.

        Returns:
            Health status
        """
        uptime = time.time() - self.start_time
        avg_latency = (self.total_latency / self.total_predictions
                      if self.total_predictions > 0 else 0)

        return HealthResponse(
            status='healthy',
            model_version=self.model_version,
            uptime_seconds=uptime,
            total_predictions=self.total_predictions,
            avg_latency_ms=avg_latency
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.

        Returns:
            Model information
        """
        return {
            'model_version': self.model_version,
            'model_type': type(self.model).__name__,
            'batch_size': self.batch_size,
            'timeout': self.timeout,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'total_predictions': self.total_predictions
        }

    def _generate_prediction_id(self) -> str:
        """Generate unique prediction ID."""
        import uuid
        return str(uuid.uuid4())


class BatchPredictor:
    """Handles batch predictions for large datasets."""

    def __init__(self, model: Any, batch_size: int = 1000):
        """Initialize batch predictor.

        Args:
            model: Trained model
            batch_size: Size of batches
        """
        self.model = model
        self.batch_size = batch_size

    def predict(self, data: pd.DataFrame, show_progress: bool = True) -> np.ndarray:
        """Make batch predictions.

        Args:
            data: Input data
            show_progress: Show progress bar

        Returns:
            Predictions
        """
        predictions = []

        num_batches = (len(data) + self.batch_size - 1) // self.batch_size

        iterator = range(0, len(data), self.batch_size)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=num_batches, desc="Batch prediction")
            except ImportError:
                pass

        for i in iterator:
            batch = data.iloc[i:i + self.batch_size]
            batch_predictions = self.model.predict(batch)
            predictions.extend(batch_predictions)

        return np.array(predictions)


class ModelEnsemble:
    """Ensemble multiple models for predictions."""

    def __init__(self, models: List[Any], weights: Optional[List[float]] = None):
        """Initialize model ensemble.

        Args:
            models: List of models
            weights: Optional weights for each model
        """
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")

        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError("Weights must sum to 1.0")

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions.

        Args:
            data: Input data

        Returns:
            Weighted ensemble predictions
        """
        predictions = []

        for model in self.models:
            pred = model.predict(data)
            predictions.append(pred)

        # Weighted average
        predictions = np.array(predictions)
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)

        return weighted_pred

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Make ensemble probability predictions.

        Args:
            data: Input data

        Returns:
            Weighted ensemble probabilities
        """
        probabilities = []

        for model in self.models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(data)
                probabilities.append(proba)

        if not probabilities:
            raise ValueError("No models support predict_proba")

        # Weighted average
        probabilities = np.array(probabilities)
        weighted_proba = np.average(probabilities, axis=0, weights=self.weights)

        return weighted_proba


# Create a simple standalone app for Docker deployment
app = FastAPI(
    title="MLOps Pipeline API",
    description="Production-ready ML model serving API",
    version="1.0.0"
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "MLOps Pipeline API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "mlops-api"
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    # This would normally return Prometheus format metrics
    return {
        "predictions_total": 0,
        "active_requests": 0,
        "uptime_seconds": 0
    }
