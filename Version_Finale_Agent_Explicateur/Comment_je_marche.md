# 🛢️ Market Analysis Agent - POC (CL=F)

Ce projet est un Proof of Concept (POC) d'un agent d'analyse financière autonome pour le Pétrole Brut (Crude Oil Futures). Il réplique un workflow complexe (N8N) en une architecture Python modulaire.

Il combine :
- **Yahoo Finance** pour les données de marché (15min, 4h, 1D).
- **Ollama (Qwen 2.5)** pour l'analyse fondamentale et le raisonnement.
- **FastAPI** pour exposer l'agent via une API REST.
- **Pydantic** pour la validation stricte des données.

---

## 📂 Structure du Projet

Le code a été découpé pour être simple à maintenir. Voici le rôle de chaque fichier :

### 1. Configuration & Données
- **`config.py`** : 
  - C'est le centre de contrôle. Contient les constantes globales (URL d'Ollama, Symbole boursier, nom du modèle).
  - *Pourquoi ?* Pour ne pas avoir de valeurs "en dur" éparpillées partout.

- **`models.py`** : 
  - Définit la "forme" des données qui circulent (les schémas Pydantic).
  - *Pourquoi ?* Assure que si on attend un prix (float), on reçoit bien un float. Indispensable pour FastAPI.

### 2. Services Externes (I/O)
- **`data_provider.py`** : 
  - Gère toute la connexion avec Yahoo Finance. Il récupère, nettoie et formate les dataframes Pandas.
  - *Pourquoi ?* Si demain tu veux remplacer Yahoo par AlphaVantage, tu modifies uniquement ce fichier.

- **`llm_client.py`** : 
  - Gère la communication technique avec Ollama.
  - *Pourquoi ?* Sépare la technique (requête HTTP vers l'IA) de la logique métier (le prompt).

### 3. Cerveau & Logique
- **`core_agent.py`** : 
  - C'est le cœur du système. Il orchestre tout : il appelle Yahoo, formate le Prompt, appelle Ollama et nettoie la réponse JSON.
  - *Pourquoi ?* C'est ici que réside "l'intelligence" de l'analyste.

### 4. Interfaces
- **`main.py`** : 
  - Le point d'entrée de l'API. Il crée les routes (`/analyze`, `/market-data`, etc.) et gère la mémoire temporaire du POC.
  - *Pourquoi ?* Pour connecter ce système à Streamlit, N8N ou un Frontend web.

- **`test_standalone.py`** : 
  - Un script pour tester la logique SANS lancer le serveur web.
  - *Pourquoi ?* Pour le débogage rapide et le développement de prompts.

### 5. Gestion des dépendances
- **`pyproject.toml`** : 
  - Liste les librairies nécessaires (FastAPI, Pandas, etc.) et configure le projet.
  - *Pourquoi ?* Remplace `requirements.txt` pour une installation plus moderne et propre.

---

## 🚀 Installation & Démarrage

### 1. Prérequis
- Python 3.9+
- [Ollama](https://ollama.ai/) installé et tournant en fond (`ollama serve`).
- Modèle Qwen récupéré : `ollama pull qwen2.5:latest`

### 2. Installation
```bash
# Si tu utilises pip standard
pip install .

# OU si tu utilises Poetry (recommandé)
poetry install