import numpy as np
import pandas as pd
import faiss
import pickle
from pathlib import Path
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama  # <--- Ajout pour la génération

from ..utils.logger import get_logger
from ..utils.config_loader import get_config

logger = get_logger(__name__)

class RAGAgent:
    """Agent RAG hybride : Recherche FAISS + Génération Qwen (llama-cpp)"""
    
    def __init__(self):
        self.config = get_config()
        
        # 1. Modèle d'embeddings (Sentence-BERT)
        emb_model_name = self.config.get('models.embeddings.name', 'paraphrase-multilingual-mpnet-base-v2')
        logger.info(f"🔢 Loading embedding model: {emb_model_name}")
        self.embedding_model = SentenceTransformer(emb_model_name)
        self.dimension = self.config.get('models.embeddings.dimension', 768)
        
        # 2. Modèle de Génération (Qwen via llama-cpp)
        # On charge le 0.5B ou le 7B selon ton choix dans la config
        repo_id = self.config.get('models.rag.repo_id', "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
        filename = self.config.get('models.rag.filename', "*q8_0.gguf")
        
        logger.info(f"🤖 Loading LLM for RAG: {repo_id}")
        self.llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_gpu_layers=-1, # Priorité GPU
            n_ctx=4096,      # Augmenté pour gérer les documents récupérés
            verbose=False
        )
        
        # FAISS setup
        self.index = None
        self.documents_df = None
        self.index_path = Path(self.config.get('paths.faiss_index'))
        self.index_file = self.index_path / 'faiss_index.bin'
        self.docs_file = self.index_path / 'documents.pkl'

    def search(self, query: str, k: int = 3) -> pd.DataFrame:
        """Recherche sémantique pure via FAISS"""
        if self.index is None:
            self.load_index()
            
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        top_docs = self.documents_df.iloc[indices[0]].copy()
        top_docs['similarity_score'] = distances[0]
        return top_docs

    def answer_with_context(self, query: str, k: int = 3) -> str:
        """Boucle RAG complète : Recherche + Synthèse par le LLM"""
        
        # 1. Retrieval (Récupération)
        docs = self.search(query, k=k)
        
        # Construction du contexte textuel
        context_text = "\n\n".join([
            f"Doc {i+1} (Titre: {row['article_title']}): {row['article_content'][:600]}"
            for i, (_, row) in enumerate(docs.iterrows())
        ])
        
        # 2. Generation (Augmentation)
        prompt = f"""Utilise les documents suivants pour répondre à la question. 
Si la réponse n'est pas dans le contexte, dis que tu ne sais pas.

CONTEXTE:
{context_text}

QUESTION:
{query}

RÉPONSE:"""

        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "Tu es un analyste expert du marché de l'énergie."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        
        return response['choices'][0]['message']['content']

    # --- MÉTHODES DE GESTION D'INDEX (Inchangées ou presque) ---
    def load_index(self):
        try:
            self.index = faiss.read_index(str(self.index_file))
            with open(self.docs_file, 'rb') as f:
                self.documents_df = pickle.load(f)
            return True
        except: return False