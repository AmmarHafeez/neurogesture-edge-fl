"""API request and response schemas."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class PredictionRequest(BaseModel):
    emg_window: list[list[float]] = Field(..., description="Window of EMG channel values.")


class PredictionResponse(BaseModel):
    label: int
    command: str
