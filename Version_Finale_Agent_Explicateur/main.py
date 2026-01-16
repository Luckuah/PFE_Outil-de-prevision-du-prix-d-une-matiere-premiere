import uvicorn
from fastapi import FastAPI, HTTPException
from typing import Optional
from datetime import datetime

# Import des modules locaux
from config import OLLAMA_MODEL, SYMBOL
from models import UserQuery, AgentDocuments, LSTMPrediction, AnalysisRequest, AnalysisResponse
from data_provider import aggregate_market_data
from core_agent import analyze_market

# ============================================
# API FASTAPI
# ============================================

app = FastAPI(
    title="Market Analysis Agent POC",
    description="POC pour l'analyse de marché CL=F avec LLM",
    version="0.1.0"
)

# Stockage en mémoire (POC)
memory_store = {
    "user_query": None,
    "documents": [],
    "lstm_prediction": None,
    "lstm_prediction_date": None
}

@app.get("/")
def root():
    return {"status": "running", "message": "Market Analysis Agent POC"}

@app.get("/health")
def health():
    return {"status": "healthy", "ollama_model": OLLAMA_MODEL}

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
    """
    Lance l'analyse complète.
    Peut recevoir toutes les données en une fois OU utiliser les données en mémoire.
    """
    if request:
        # Utiliser les données de la requête
        user_query = request.user_query
        documents = request.documents
        lstm_pred = request.lstm_prediction
        lstm_date = request.lstm_prediction_date
    else:
        # Utiliser les données en mémoire
        user_query = memory_store.get("user_query")
        documents = memory_store.get("documents", [])
        lstm_pred = memory_store.get("lstm_prediction")
        lstm_date = memory_store.get("lstm_prediction_date")
    
    # Validation
    if not user_query:
        raise HTTPException(status_code=400, detail="User query manquante")
    if lstm_pred is None:
        raise HTTPException(status_code=400, detail="LSTM prediction manquante")
    
    # Lancer l'analyse
    result = analyze_market(
        user_query=user_query,
        documents=documents or [],
        lstm_prediction=lstm_pred,
        lstm_prediction_date=lstm_date or datetime.now().isoformat()
    )
    
    return result

@app.get("/market-data")
def get_market_data():
    """Endpoint pour récupérer uniquement les données de marché."""
    return aggregate_market_data()

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("🚀 Démarrage du serveur FastAPI...")
    print(f"📡 Ollama Model: {OLLAMA_MODEL}")
    print(f"🛢️  Symbol: {SYMBOL}")
    print("-" * 40)
    uvicorn.run(app, host="0.0.0.0", port=8000)