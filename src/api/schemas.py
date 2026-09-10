from pydantic import BaseModel


class WindowPrediction(BaseModel):
    window_index: int
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]
    reconstruction_error: float
    is_anomaly: bool
    physics_context: dict | None = None


class PredictionResponse(BaseModel):
    n_windows: int
    predicted_class: str  # majority vote across windows
    confidence: float  # mean confidence for the majority-vote class
    anomaly_rate: float  # fraction of windows flagged anomalous
    is_anomaly: bool
    windows: list[WindowPrediction]


class HealthResponse(BaseModel):
    status: str
    classes: list[str]
    window_size: int
