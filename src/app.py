import logging
from typing import Optional, List, Dict, Any, Hashable

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from m4_forecasting.config import PipelineConfig
from m4_forecasting.trainer import ForecastingEngine
from m4_forecasting.utils import setup_logging


# ---------------------------------------------------------
# Pydantic Models for Input/Output Validation
# ---------------------------------------------------------
class PredictionRequest(BaseModel):
    """
    Configuration payload for generating predictions.
    """
    unique_id: Optional[str] = Field(
        default=None,
        description="The specific ID to forecast. If None, forecasts all IDs."
    )
    horizon: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Forecasting horizon (number of steps ahead)."
    )


class PredictionResponse(BaseModel):
    """
    Standardized response format.
    """
    unique_id: str
    forecast: List[Dict[Hashable, Any]]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# Define Global Variables
logger = setup_logging()
logger.info("API Startup: Initializing Forecasting Engine...")

# --------------------
# ------- SETUP ------
# --------------------
models_loaded: bool = False

# Initialize Config
config = PipelineConfig()

# Initialize Engine
engine = ForecastingEngine(config)
try:

    # Load the trained Model
    engine.load_train_model()
    logger.info("API Startup: Train model loaded successfully.")
    models_loaded: bool = True

except Exception as e:
    # We don't raise here to allow the app to start (and fail health checks instead),
    # which is often better for debugging in container orchestrators.
    logger.error(f"API Startup Failed: Could not load model. Error: {e}")

# ======================
# Application Definition
# ======================
app = FastAPI(
    title="M4 Forecasting API",
    description="Inference API for M4 Time Series Forecasting",
    version="1.0.0",
    docs_url="/docs",  # Default value
    redoc_url=None  # Disables ReDoc documentation
)


# ======================
# Endpoints
# ======================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Checks the health of the service and model status.
    """
    is_loaded: bool = models_loaded
    health_status: str = "up" if is_loaded else "down"

    return HealthResponse(
        status=health_status,
        model_loaded=is_loaded
    )


@app.post("/predict", response_model=None, tags=["Inference"])
async def generate_forecast(
        request: PredictionRequest,
):
    """
    Generates time series forecasts based on the provided configuration.
    """
    if not models_loaded:
        return JSONResponse(
            {"status": "FAILURE",
             "message": f"Error message: Model not loaded."},
        )

    try:
        # 1. Determine Horizon
        # Use request horizon if provided, otherwise fallback to model default config
        h = request.horizon if request.horizon else engine.config.horizon

        # Determine IDs
        # If unique_id is provided, filter for it. Otherwise, predict for all.
        ids = [request.unique_id] if request.unique_id else None

        logger.info(f"Generating predictions for ids: {ids}, and horizon: {h}")
        # 3. Generate Predictions
        preds_df = engine.predict(horizon=h, ids=ids)
        # logger.info(preds_df)

        # =======================
        # Format Response
        # =======================
        # Convert DataFrame to the List[PredictionResponse] format
        results: List[PredictionResponse] = []

        # Group by unique_id to format the output
        for uid, group in preds_df.groupby('unique_id'):
            # Convert the forecast rows to a list of dicts
            forecast_data = group.drop(columns=['unique_id']).to_dict(orient='records')

            results.append(PredictionResponse(
                unique_id=str(uid),
                forecast=forecast_data
            ))

        return results

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
