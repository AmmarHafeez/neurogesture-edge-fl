"""Minimal FastAPI application for local inference integration."""

from fastapi import FastAPI

from src.api.gesture_commands import command_for_label
from src.api.schemas import HealthResponse, PredictionRequest, PredictionResponse


app = FastAPI(title="neurogesture-edge-fl")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return basic service status."""
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Return a placeholder prediction until model inference is implemented."""
    label = 0
    return PredictionResponse(label=label, command=command_for_label(label))
