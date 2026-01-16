from pydantic import BaseModel
from typing import Optional


class UserInput(BaseModel):
    text: str

class UserQuery(BaseModel):
    query: str


class AgentDocuments(BaseModel):
    documents: str


class LSTMPrediction(BaseModel):
    predicted_price: float
    prediction_date: str


class AnalysisRequest(BaseModel):
    user_query: str
    lstm_prediction: float
    lstm_prediction_date: str


class AnalysisResponse(BaseModel):
    predicted_price_10d: float
    explanation: str
    confidence: str
    key_factors: list[str]
    timestamp: str
    market_data_summary: Optional[dict] = None
    lstm_input: Optional[float] = None