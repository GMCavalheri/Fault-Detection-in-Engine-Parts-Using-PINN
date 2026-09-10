"""FastAPI inference service - Phase 6 of the project plan.

Run directly:      uv run uvicorn src.api.main:app --reload
Interactive docs:   http://localhost:8000/docs
"""

from collections import Counter

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.api.inference import FaultDetector
from src.api.schemas import HealthResponse, PredictionResponse, WindowPrediction
from src.api.signal_parsing import parse_signal_upload
from src.data.cwru import window_signal

app = FastAPI(
    title="Fault Detection in Engine Parts",
    description="CWRU bearing fault classification + anomaly detection, with an optional "
    "physics-informed defect-frequency readout when RPM is supplied.",
    version="1.0.0",
)

# dashboard runs as a separate service/origin (see docker/docker-compose.yml) - allow it to call this API
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

detector = FaultDetector()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", classes=detector.class_names, window_size=detector.window_size)


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), rpm: float | None = Form(default=None)) -> PredictionResponse:
    data = await file.read()
    try:
        signal = parse_signal_upload(file.filename, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if len(signal) < detector.window_size:
        raise HTTPException(
            status_code=400,
            detail=f"Signal has {len(signal)} samples, need at least {detector.window_size} "
            f"(the model's window size at {detector.sample_rate_hz:.0f} Hz).",
        )

    windows = window_signal(signal, detector.window_size)
    window_results = [detector.analyze(w, rpm) for w in windows]

    predictions = [
        WindowPrediction(
            window_index=i,
            predicted_class=r["predicted_class"],
            confidence=r["confidence"],
            class_probabilities=r["class_probabilities"],
            reconstruction_error=r["reconstruction_error"],
            is_anomaly=r["is_anomaly"],
            physics_context=r["physics_context"],
        )
        for i, r in enumerate(window_results)
    ]

    vote_counts = Counter(p.predicted_class for p in predictions)
    majority_class, _ = vote_counts.most_common(1)[0]
    majority_confidences = [p.confidence for p in predictions if p.predicted_class == majority_class]
    anomaly_rate = sum(p.is_anomaly for p in predictions) / len(predictions)

    return PredictionResponse(
        n_windows=len(predictions),
        predicted_class=majority_class,
        confidence=sum(majority_confidences) / len(majority_confidences),
        anomaly_rate=anomaly_rate,
        is_anomaly=anomaly_rate > 0.5,
        windows=predictions,
    )
