from pydantic import BaseModel
from typing import Optional

# ============================================
# MODÈLES PYDANTIC
# ============================================

class UserQuery(BaseModel):
    query: str

class AgentDocuments(BaseModel):
    documents: list[dict]  # Liste de 5 documents/articles

class LSTMPrediction(BaseModel):
    predicted_price: float
    prediction_date: str

class AnalysisRequest(BaseModel):
    user_query: str
    documents: list[dict]
    lstm_prediction: float
    lstm_prediction_date: str

class AnalysisResponse(BaseModel):
    predicted_price_10d: float
    explanation: str
    confidence: str
    timestamp: str