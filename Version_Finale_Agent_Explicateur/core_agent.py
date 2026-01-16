import json
from datetime import datetime
from data_provider import aggregate_market_data
from llm_client import call_ollama

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