"""
POC - Market Analysis Agent pour CL=F (Crude Oil Futures)
Réplique du workflow N8N avec:
- Données Yahoo Finance (15min, 4h, 1D)
- Query utilisateur via Streamlit
- Documents d'un premier agent
- Prédiction LSTM
- Analyse via Qwen/Ollama
"""

import requests
import json
from datetime import datetime, timedelta
from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yfinance as yf
import pandas as pd

# ============================================
# CONFIGURATION
# ============================================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:latest"  # Ajuste selon ton modèle Qwen installé
SYMBOL = "CL=F"  # Crude Oil Futures

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

# ============================================
# FONCTIONS DE RÉCUPÉRATION DES DONNÉES
# ============================================

def fetch_yahoo_data_15min(days: int = 60) -> pd.DataFrame:
    """
    Récupère les données 15min de Yahoo Finance.
    Note: Yahoo limite les données intraday à ~60 jours max pour 15min.
    """
    ticker = yf.Ticker(SYMBOL)
    # Pour 15min, Yahoo permet max 60 jours
    df = ticker.history(period=f"{min(days, 60)}d", interval="15m")
    df['timeframe'] = '15min'
    return df

def fetch_yahoo_data_4h(days: int = 60) -> pd.DataFrame:
    """
    Récupère les données 4h de Yahoo Finance.
    Note: Yahoo n'a pas d'interval 4h natif, on resample depuis 1h.
    """
    ticker = yf.Ticker(SYMBOL)
    # Récupérer en 1h puis resample en 4h
    df = ticker.history(period=f"{days}d", interval="1h")
    
    if df.empty:
        return pd.DataFrame()
    
    # Resample en 4h
    df_4h = df.resample('4h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    df_4h['timeframe'] = '4h'
    return df_4h

def fetch_yahoo_data_daily(days: int = 90) -> pd.DataFrame:
    """
    Récupère les données daily de Yahoo Finance.
    """
    ticker = yf.Ticker(SYMBOL)
    df = ticker.history(period=f"{days}d", interval="1d")
    df['timeframe'] = '1D'
    return df

def aggregate_market_data() -> dict:
    """
    Agrège les données de tous les timeframes.
    """
    print("📊 Récupération des données de marché...")
    
    try:
        df_15min = fetch_yahoo_data_15min(60)
        df_4h = fetch_yahoo_data_4h(60)
        df_daily = fetch_yahoo_data_daily(90)
        
        # Statistiques résumées pour chaque timeframe
        def summarize_df(df: pd.DataFrame, name: str) -> dict:
            if df.empty:
                return {"timeframe": name, "error": "No data available"}
            
            recent = df.tail(20)  # 20 dernières périodes
            return {
                "timeframe": name,
                "latest_close": float(df['Close'].iloc[-1]) if not df.empty else None,
                "latest_date": str(df.index[-1]) if not df.empty else None,
                "period_high": float(recent['High'].max()),
                "period_low": float(recent['Low'].min()),
                "period_avg": float(recent['Close'].mean()),
                "trend": "UP" if recent['Close'].iloc[-1] > recent['Close'].iloc[0] else "DOWN",
                "volatility": float(recent['Close'].std()),
                "total_records": len(df),
                "recent_closes": recent['Close'].tail(5).tolist()
            }
        
        return {
            "symbol": SYMBOL,
            "fetch_time": datetime.now().isoformat(),
            "timeframes": {
                "15min": summarize_df(df_15min, "15min"),
                "4h": summarize_df(df_4h, "4h"),
                "daily": summarize_df(df_daily, "daily")
            }
        }
    except Exception as e:
        print(f"❌ Erreur lors de la récupération: {e}")
        return {"error": str(e)}

# ============================================
# FONCTION D'APPEL À OLLAMA/QWEN
# ============================================

def call_ollama(prompt: str, system_prompt: str = "") -> str:
    """
    Appelle le modèle Qwen via Ollama.
    """
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 2000
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur Ollama: {e}")
        return f"Erreur lors de l'appel au LLM: {e}"

# ============================================
# AGENT D'ANALYSE PRINCIPAL
# ============================================

def analyze_market(
    user_query: str,
    documents: list[dict],
    lstm_prediction: float,
    lstm_prediction_date: str
) -> dict:
    """
    Agent principal qui analyse le marché et génère une prédiction à 10 jours.
    """
    print("🤖 Démarrage de l'analyse...")
    
    # 1. Récupérer les données de marché
    market_data = aggregate_market_data()
    
    # 2. Formater les documents
    docs_text = "\n".join([
        f"Document {i+1}: {doc.get('title', 'Sans titre')}\n{doc.get('content', doc.get('summary', 'Pas de contenu'))}\n"
        for i, doc in enumerate(documents[:5])
    ])
    
    # 3. Construire le prompt
    system_prompt = """Tu es un analyste financier expert spécialisé dans les marchés pétroliers (Crude Oil Futures - CL=F).
Ta mission est d'analyser les données de marché multi-timeframe, les actualités pertinentes, et une prédiction LSTM 
pour fournir une prédiction de prix à 10 jours avec une explication détaillée.

Réponds TOUJOURS en JSON avec ce format exact:
{
    "predicted_price_10d": <float>,
    "confidence": "HIGH" | "MEDIUM" | "LOW",
    "explanation": "<explication détaillée en français>",
    "key_factors": ["facteur1", "facteur2", "facteur3"]
}"""

    analysis_prompt = f"""
## Question de l'utilisateur:
{user_query}

## Données de marché actuelles (CL=F - Crude Oil Futures):
{json.dumps(market_data, indent=2, default=str)}

## Prédiction du modèle LSTM:
- Prix prédit: ${lstm_prediction:.2f}
- Date de prédiction: {lstm_prediction_date}

## Documents/Articles pertinents fournis par l'agent de recherche:
{docs_text}

## Ta mission:
1. Analyse les tendances sur les 3 timeframes (15min, 4h, daily)
2. Prends en compte la prédiction LSTM comme un indicateur parmi d'autres
3. Intègre les informations des documents pour contextualiser
4. Réponds à la question de l'utilisateur
5. Fournis ta prédiction de prix à 10 jours avec explication

Réponds en JSON valide.
"""
    
    # 4. Appeler le LLM
    print("💭 Appel au LLM Qwen...")
    raw_response = call_ollama(analysis_prompt, system_prompt)
    
    # 5. Parser la réponse
    try:
        # Essayer d'extraire le JSON de la réponse
        json_start = raw_response.find('{')
        json_end = raw_response.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = raw_response[json_start:json_end]
            parsed_response = json.loads(json_str)
        else:
            parsed_response = {
                "predicted_price_10d": lstm_prediction,  # Fallback sur LSTM
                "confidence": "LOW",
                "explanation": raw_response,
                "key_factors": ["Parsing error - raw response returned"]
            }
    except json.JSONDecodeError:
        parsed_response = {
            "predicted_price_10d": lstm_prediction,
            "confidence": "LOW", 
            "explanation": raw_response,
            "key_factors": ["JSON parsing failed"]
        }
    
    # 6. Ajouter métadonnées
    parsed_response["timestamp"] = datetime.now().isoformat()
    parsed_response["market_data_summary"] = market_data.get("timeframes", {})
    parsed_response["lstm_input"] = lstm_prediction
    
    return parsed_response

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
# MODE STANDALONE (TEST)
# ============================================

def test_standalone():
    """
    Test du système en mode standalone sans FastAPI.
    """
    print("=" * 60)
    print("🚀 TEST STANDALONE - Market Analysis Agent POC")
    print("=" * 60)
    
    # Données de test
    test_query = "Quelle est ta prédiction pour le prix du pétrole dans les 10 prochains jours? Quels facteurs surveiller?"
    
    test_documents = [
        {
            "title": "OPEC+ maintains production cuts",
            "content": "OPEC+ has decided to maintain current production cuts through Q1, supporting oil prices amid global demand concerns."
        },
        {
            "title": "US Crude Inventories Rise",
            "content": "EIA reports unexpected build in US crude inventories, adding 2.1 million barrels vs expected draw of 1.5 million."
        },
        {
            "title": "China Economic Data Mixed",
            "content": "China's latest PMI data shows manufacturing activity stabilizing but services sector slowing, creating mixed outlook for oil demand."
        },
        {
            "title": "Geopolitical Tensions Middle East",
            "content": "Rising tensions in the Middle East continue to support risk premium in oil prices, with traders monitoring shipping routes."
        },
        {
            "title": "US Dollar Weakness",
            "content": "The US dollar index fell to multi-week lows, typically supportive for dollar-denominated commodities like crude oil."
        }
    ]
    
    test_lstm_prediction = 72.50  # Prix prédit par LSTM
    test_lstm_date = (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")
    
    # Lancer l'analyse
    result = analyze_market(
        user_query=test_query,
        documents=test_documents,
        lstm_prediction=test_lstm_prediction,
        lstm_prediction_date=test_lstm_date
    )
    
    print("\n" + "=" * 60)
    print("📊 RÉSULTAT DE L'ANALYSE:")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    
    return result

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Mode test standalone
        test_standalone()
    else:
        # Mode API
        print("🚀 Démarrage du serveur FastAPI...")
        print(f"📡 Ollama Model: {OLLAMA_MODEL}")
        print(f"🛢️  Symbol: {SYMBOL}")
        print("-" * 40)
        uvicorn.run(app, host="0.0.0.0", port=8000)
