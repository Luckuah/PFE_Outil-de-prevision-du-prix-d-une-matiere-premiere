import requests
from config import OLLAMA_URL, OLLAMA_MODEL

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