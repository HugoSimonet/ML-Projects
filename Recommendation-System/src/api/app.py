"""FastAPI application for serving recommendations."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from ..models.base import BaseRecommender
from ..utils.config_loader import get_models_path, load_config
from ..utils.logger import setup_logger

# Setup logger
setup_logger(log_file="logs/api.log")

# Initialize FastAPI app
app = FastAPI(
    title="Recommendation System API",
    description="API for serving personalized recommendations across multiple domains",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
loaded_models: Dict[str, BaseRecommender] = {}
model_metadata: Dict[str, Dict] = {}


# Pydantic models for request/response
class RecommendationRequest(BaseModel):
    """Request for recommendations."""
    user_id: int = Field(..., description="User identifier")
    n_items: int = Field(10, description="Number of items to recommend", ge=1, le=100)
    exclude_seen: bool = Field(True, description="Whether to exclude items user has already seen")
    model_name: Optional[str] = Field(None, description="Specific model to use (optional)")


class PredictionRequest(BaseModel):
    """Request for rating prediction."""
    user_id: int = Field(..., description="User identifier")
    item_id: int = Field(..., description="Item identifier")
    model_name: Optional[str] = Field(None, description="Specific model to use (optional)")


class RecommendationResponse(BaseModel):
    """Response with recommendations."""
    user_id: int
    recommendations: List[Dict[str, Any]]
    model_used: str
    n_recommendations: int


class PredictionResponse(BaseModel):
    """Response with rating prediction."""
    user_id: int
    item_id: int
    predicted_rating: float
    model_used: str


class ModelInfo(BaseModel):
    """Information about a loaded model."""
    name: str
    type: str
    is_trained: bool


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    n_models_loaded: int
    models: List[str]


class BatchRecommendationRequest(BaseModel):
    """Request for batch recommendations."""
    user_ids: List[int] = Field(..., description="List of user identifiers")
    n_items: int = Field(10, description="Number of items to recommend", ge=1, le=100)
    model_name: Optional[str] = Field(None, description="Specific model to use (optional)")


class LoadModelRequest(BaseModel):
    """Request to load a model."""
    model_path: str = Field(..., description="Path to the saved model file")


# API endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    logger.info("Starting Recommendation System API...")

    # Try to load default models
    models_path = get_models_path()
    models_path.mkdir(parents=True, exist_ok=True)

    # Look for saved models
    model_files = list(models_path.glob("*.pkl")) + list(models_path.glob("*.joblib"))

    for model_file in model_files:
        try:
            model_name = model_file.stem
            logger.info(f"Loading model: {model_name}")

            model = joblib.load(model_file)
            loaded_models[model_name] = model

            model_metadata[model_name] = {
                'name': model_name,
                'type': type(model).__name__,
                'is_trained': model.is_trained,
                'path': str(model_file)
            }

            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model {model_file}: {e}")

    if not loaded_models:
        logger.warning("No models loaded. Please train and save models first.")
    else:
        logger.info(f"Loaded {len(loaded_models)} models: {list(loaded_models.keys())}")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Recommendation System API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        n_models_loaded=len(loaded_models),
        models=list(loaded_models.keys())
    )


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all loaded models."""
    return [
        ModelInfo(
            name=name,
            type=metadata['type'],
            is_trained=metadata['is_trained']
        )
        for name, metadata in model_metadata.items()
    ]


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get personalized recommendations for a user.

    Returns top-N item recommendations with predicted scores.
    """
    if not loaded_models:
        raise HTTPException(status_code=503, detail="No models loaded")

    # Select model
    if request.model_name:
        if request.model_name not in loaded_models:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model_name}' not found"
            )
        model = loaded_models[request.model_name]
        model_name = request.model_name
    else:
        # Use first available model
        model_name = list(loaded_models.keys())[0]
        model = loaded_models[model_name]

    try:
        # Get recommendations
        recommendations = model.recommend(
            user_id=request.user_id,
            n_items=request.n_items,
            exclude_seen=request.exclude_seen
        )

        # Format response
        formatted_recs = [
            {
                "item_id": int(item_id),  # Convert numpy.int64 to Python int
                "score": float(score),
                "rank": i + 1
            }
            for i, (item_id, score) in enumerate(recommendations)
        ]

        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=formatted_recs,
            model_used=model_name,
            n_recommendations=len(formatted_recs)
        )

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict_rating(request: PredictionRequest):
    """
    Predict rating for a user-item pair.

    Returns predicted rating score.
    """
    if not loaded_models:
        raise HTTPException(status_code=503, detail="No models loaded")

    # Select model
    if request.model_name:
        if request.model_name not in loaded_models:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model_name}' not found"
            )
        model = loaded_models[request.model_name]
        model_name = request.model_name
    else:
        # Use first available model
        model_name = list(loaded_models.keys())[0]
        model = loaded_models[model_name]

    try:
        # Get prediction
        predicted_rating = model.predict(
            user_id=request.user_id,
            item_id=request.item_id
        )

        return PredictionResponse(
            user_id=request.user_id,
            item_id=request.item_id,
            predicted_rating=float(predicted_rating),
            model_used=model_name
        )

    except Exception as e:
        logger.error(f"Error predicting rating: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/{user_id}")
async def get_recommendations_simple(
    user_id: int,
    n: int = Query(10, description="Number of recommendations", ge=1, le=100),
    model: Optional[str] = Query(None, description="Model name")
):
    """
    Simple GET endpoint for recommendations.

    Alternative to POST endpoint for easier testing.
    """
    request = RecommendationRequest(
        user_id=user_id,
        n_items=n,
        model_name=model
    )
    return await get_recommendations(request)


@app.post("/batch_recommend")
async def batch_recommendations(request: BatchRecommendationRequest):
    """
    Get recommendations for multiple users in batch.

    More efficient than making individual requests.
    """
    if not loaded_models:
        raise HTTPException(status_code=503, detail="No models loaded")

    # Select model
    if request.model_name:
        if request.model_name not in loaded_models:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model_name}' not found"
            )
        model = loaded_models[request.model_name]
        model_name = request.model_name
    else:
        model_name = list(loaded_models.keys())[0]
        model = loaded_models[model_name]

    try:
        # Get batch recommendations
        batch_recs = model.batch_recommend(request.user_ids, request.n_items)

        # Format response
        results = []
        for user_id, recommendations in batch_recs.items():
            formatted_recs = [
                {
                    "item_id": int(item_id),  # Convert numpy.int64 to Python int
                    "score": float(score),
                    "rank": i + 1
                }
                for i, (item_id, score) in enumerate(recommendations)
            ]

            results.append({
                "user_id": int(user_id),  # Convert numpy.int64 to Python int
                "recommendations": formatted_recs,
                "n_recommendations": len(formatted_recs)
            })

        return {
            "model_used": model_name,
            "n_users": len(results),
            "results": results
        }

    except Exception as e:
        logger.error(f"Error generating batch recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load_model")
async def load_model(request: LoadModelRequest):
    """
    Load a model from disk.

    Args:
        request: Request containing the model path.
    """
    try:
        model = joblib.load(request.model_path)
        model_name = Path(request.model_path).stem

        loaded_models[model_name] = model
        model_metadata[model_name] = {
            'name': model_name,
            'type': type(model).__name__,
            'is_trained': model.is_trained,
            'path': request.model_path
        }

        logger.info(f"Loaded model: {model_name}")

        return {
            "message": f"Model '{model_name}' loaded successfully",
            "model_info": model_metadata[model_name]
        }

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the API server."""
    import uvicorn

    config = load_config()
    api_config = config["api"]

    uvicorn.run(
        "src.api.app:app",
        host=api_config["host"],
        port=api_config["port"],
        workers=api_config["workers"],
        reload=api_config["reload"]
    )


if __name__ == "__main__":
    main()
