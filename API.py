from fastapi import FastAPI,HTTPException
from Pipeline_Data.test_rag_copy import create_rag, get_answer
from datetime import datetime
from typing import Optional

from formatage_pydantic import UserInput,UserQuery, AgentDocuments, LSTMPrediction, AnalysisRequest
from Version_Finale_Agent_Explicateur.functions import aggregate_market_data, analyze_market


rag = create_rag()
app = FastAPI()

# Stockage en mémoire (POC)
memory_store = {
    "user_query": None,
    "documents": None,
    "lstm_prediction": None,
    "lstm_prediction_date": None
}


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: UserInput):
    answer = get_answer(data.text, rag)
    return {"text": answer}


@app.post("/user-query")
def receive_user_query(query: UserQuery):
    """Reçoit la query de l'utilisateur depuis Streamlit."""
    memory_store["user_query"] = query.query
    print(f"📝 Query reçue: {query.query}")
    return {"status": "received", "query": query.query}


@app.post("/agent-documents")
def receive_agent_documents(docs: AgentDocuments):
    """Reçoit les documents du premier agent."""
    memory_store["documents"] = docs.documents
    print(f"📄 {len(docs.documents)} documents reçus")
    return {"status": "received", "count": len(docs.documents)}


@app.post("/lstm-prediction")
def receive_lstm_prediction(pred: LSTMPrediction):
    """Reçoit la prédiction LSTM."""
    memory_store["lstm_prediction"] = pred.predicted_price
    memory_store["lstm_prediction_date"] = pred.prediction_date
    print(f"🔮 Prédiction LSTM reçue: ${pred.predicted_price:.2f}")
    return {"status": "received", "prediction": pred.predicted_price}


@app.post("/analyze")
def run_analysis(request: Optional[AnalysisRequest] = None):
    """Lance l'analyse complète."""

    if request:
        user_query = request.user_query
        lstm_pred = request.lstm_prediction
        lstm_date = request.lstm_prediction_date
    else:
        user_query = memory_store.get("user_query")
        lstm_pred = memory_store.get("lstm_prediction")
        lstm_date = memory_store.get("lstm_prediction_date")
    
    documents=get_answer(user_query, rag)

    if not user_query:
        raise HTTPException(status_code=400, detail="User query manquante")
    if lstm_pred is None:
        raise HTTPException(status_code=400, detail="LSTM prediction manquante")
    
    result = analyze_market(
        user_query=user_query,
        documents=documents,
        lstm_prediction=lstm_pred,
        lstm_prediction_date=lstm_date or datetime.now().isoformat()
    )
    
    return result


@app.get("/market-data")
def get_market_data():
    """Récupère les données de marché."""
    return aggregate_market_data()




