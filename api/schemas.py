from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class PredictResponse(BaseModel):
    features: Dict[str, float]
    predictions: Dict[str, float]
    p_index: Optional[int]
    sampling_rate: Optional[float]
    p_window: Optional[List[float]]

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
