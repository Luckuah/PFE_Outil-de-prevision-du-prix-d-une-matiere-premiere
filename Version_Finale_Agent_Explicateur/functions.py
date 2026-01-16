"""
Toutes les fonctions métier
- LLM (llama-cpp)
- Yahoo Finance
- Analyse de marché
"""

import json
from datetime import datetime
import yfinance as yf
import pandas as pd
from llama_cpp import Llama

LLM_REPO_ID = "Qwen/Qwen2.5-3B-Instruct-GGUF"
LLM_FILENAME = "*q4_k_m.gguf"  # Quantisation 4-bit
LLM_N_CTX = 4096              # Context window
LLM_N_GPU_LAYERS = -1         # -1 = utilise tout le GPU disponible
SYMBOL = "CL=F"  # Crude Oil Futures



# ============================================
# LLM (llama-cpp) - Singleton pour éviter de recharger
# ============================================

_llm_instance = None


def get_llm() -> Llama:
    """
    Retourne l'instance du LLM (singleton).
    Le modèle est chargé une seule fois.
    """
    global _llm_instance
    
    if _llm_instance is None:
        print(f"🤖 Chargement du LLM: {LLM_REPO_ID}...")
        _llm_instance = Llama.from_pretrained(
            repo_id=LLM_REPO_ID,
            filename=LLM_FILENAME,
            n_gpu_layers=LLM_N_GPU_LAYERS,
            n_ctx=LLM_N_CTX,
            verbose=False
        )
        print("✅ LLM chargé avec succès")
    
    return _llm_instance


def call_llm(prompt: str, system_prompt: str = "") -> str:
    """
    Appelle le LLM via llama-cpp.
    
    Args:
        prompt: Le prompt utilisateur
        system_prompt: Le prompt système
    
    Returns:
        La réponse du modèle
    """
    llm = get_llm()
    
    messages = []
    
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    try:
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        return response['choices'][0]['message']['content']
    
    except Exception as e:
        print(f"❌ Erreur LLM: {e}")
        return f"Erreur lors de l'appel au LLM: {e}"


# ============================================
# YAHOO FINANCE
# ============================================

def fetch_yahoo_data_15min(days: int = 60) -> pd.DataFrame:
    """Récupère les données 15min."""
    ticker = yf.Ticker(SYMBOL)
    df = ticker.history(period=f"{min(days, 60)}d", interval="15m")
    df['timeframe'] = '15min'
    return df


def fetch_yahoo_data_4h(days: int = 60) -> pd.DataFrame:
    """Récupère les données 4h (resample depuis 1h)."""
    ticker = yf.Ticker(SYMBOL)
    df = ticker.history(period=f"{days}d", interval="1h")
    
    if df.empty:
        return pd.DataFrame()
    
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
    """Récupère les données daily."""
    ticker = yf.Ticker(SYMBOL)
    df = ticker.history(period=f"{days}d", interval="1d")
    df['timeframe'] = '1D'
    return df


def aggregate_market_data() -> dict:
    """Agrège les données de tous les timeframes."""
    print("📊 Récupération des données de marché...")
    
    try:
        df_15min = fetch_yahoo_data_15min(60)
        df_4h = fetch_yahoo_data_4h(60)
        df_daily = fetch_yahoo_data_daily(90)
        
        def summarize_df(df: pd.DataFrame, name: str) -> dict:
            if df.empty:
                return {"timeframe": name, "error": "No data available"}
            
            recent = df.tail(20)
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
# ANALYSE DE MARCHÉ
# ============================================

def analyze_market(
    user_query: str,
    documents: str,
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
    docs_text = documents
    
    # 3. Construire les prompts
    system_prompt = """Tu es un analyste financier expert spécialisé dans les marchés pétroliers (Crude Oil Futures - CL=F).
Ta mission est d'analyser les données de marché multi-timeframe, les actualités pertinentes, et une prédiction LSTM 
pour fournir une prédiction de prix à 10 jours avec une explication détaillée.
Pour l'explication géopolitique, oublie toute ta based de connaissance , et focus toi uniquement sur les documents que tu recevras.  

Réponds TOUJOURS en JSON avec ce format exact:
{
    "predicted_price_10d": <float>,
    "confidence": "HIGH" | "MEDIUM" | "LOW",
    "explanation": "<explication détaillée en français>",
    "key_factors": ["facteur1", "facteur2", "facteur3"]
    "sources": ["source1", "source2", "source3"]
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
1. Analyse les tendances sur les 3 timeframes (15min, 4h, daily) et utilise les pour fair ta prédiction
2. Prends en compte la prédiction LSTM comme un indicateur parmi d'autres
3. Intègre les informations des documents pour contextualiser et oublie tout ce qu'on t'a donné comme info en données d'entrainements concernant le petrole 
4. Réponds à la question de l'utilisateur ( en citant les documents mentionnées, url, titre, content)
5. Fournis ta prédiction de prix à 10 jours avec explication

Réponds en JSON valide.
"""
    
    # 4. Appeler le LLM
    print("💭 Appel au LLM...")
    raw_response = call_llm(analysis_prompt, system_prompt)
    
    # 5. Parser la réponse
    try:
        json_start = raw_response.find('{')
        json_end = raw_response.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = raw_response[json_start:json_end]
            parsed_response = json.loads(json_str)
        else:
            parsed_response = {
                "predicted_price_10d": lstm_prediction,
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
    
    print("✅ Analyse terminée")
    return parsed_response