import json
from datetime import datetime, timedelta
from core_agent import analyze_market

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

if __name__ == "__main__":
    test_standalone()