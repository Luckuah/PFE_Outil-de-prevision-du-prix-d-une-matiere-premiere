"""
RAG Agent for semantic search and answer generation
"""
import numpy as np
import pandas as pd
import faiss
import pickle
from pathlib import Path
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import get_logger
from utils.config_loader import get_config

logger = get_logger(__name__)


class RAGAgent:
    """RAG Agent: FAISS Semantic Search + Qwen Generation"""
    
    def __init__(self):
        """Initialize RAG Agent"""
        self.config = get_config()
        
        # 1. Embedding Model (Sentence-BERT)
        emb_model_name = self.config.get(
            'models.embeddings.name', 
            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
        )
        logger.info(f"🔢 Loading embedding model: {emb_model_name}")
        self.embedding_model = SentenceTransformer(emb_model_name)
        self.dimension = self.config.get('models.embeddings.dimension', 768)
        
        # 2. LLM for Generation (Qwen via llama-cpp)
        repo_id = self.config.get('models.rag.repo_id', "Qwen/Qwen2.5-3B-Instruct-GGUF")
        filename = self.config.get('models.rag.filename', "*q4_k_m.gguf") # Quantisation 4-bit
        n_ctx = self.config.get('models.rag.n_ctx', 4096)
        
        logger.info(f"🤖 Loading LLM for RAG: {repo_id}")
        self.llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_gpu_layers=-1,  # Use GPU if available
            n_ctx=n_ctx,
            verbose=False
        )
        logger.info("✅ LLM loaded successfully")
        
        # FAISS setup
        self.index = None
        self.documents_df = None
        
        # Paths
        self.index_path = Path(self.config.get('paths.faiss_index', './data/faiss_index'))
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.index_path / 'faiss_index.bin'
        self.docs_file = self.index_path / 'documents.pkl'
        
        logger.info("✅ RAG Agent initialized")
    
    def build_index(self, df: pd.DataFrame, 
                   text_column: str = 'article_content',
                   batch_size: int = 32):
        """
        Build FAISS index from DataFrame
        
        Args:
            df: DataFrame with articles
            text_column: Column containing text to embed
            batch_size: Batch size for embedding
        """
        logger.info(f"📚 Building FAISS index from {len(df)} documents...")
        
        if df.empty:
            logger.warning("⚠️ Empty DataFrame provided")
            return
        
        # Vérifier que la colonne existe
        if text_column not in df.columns:
            logger.error(f"❌ Column '{text_column}' not found in DataFrame")
            logger.error(f"   Available columns: {df.columns.tolist()}")
            return
        
        # Préparer les textes
        texts = df[text_column].fillna('').tolist()
        
        # Générer les embeddings par batch
        logger.info("🔢 Generating embeddings...")
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        # Créer l'index FAISS
        logger.info("🔍 Creating FAISS index...")
        embeddings = embeddings.astype('float32')
        
        # Utiliser IndexFlatIP pour Inner Product (cosine similarity)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        
        # Sauvegarder les documents
        self.documents_df = df.copy()
        
        # Sauvegarder sur disque
        self.save_index()
        
        logger.info(f"✅ FAISS index built with {self.index.ntotal} vectors")
        logger.info(f"   Dimension: {self.dimension}")
        logger.info(f"   Saved to: {self.index_file}")
    
    def save_index(self):
        """Save FAISS index and documents to disk"""
        try:
            # Sauvegarder l'index FAISS
            faiss.write_index(self.index, str(self.index_file))
            
            # Sauvegarder le DataFrame
            with open(self.docs_file, 'wb') as f:
                pickle.dump(self.documents_df, f)
            
            logger.info(f"💾 Index saved to {self.index_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save index: {e}")
    
    def load_index(self) -> bool:
        """
        Load FAISS index from disk
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.index_file.exists() or not self.docs_file.exists():
                logger.warning("⚠️ Index files not found")
                return False
            
            logger.info(f"📂 Loading FAISS index from {self.index_file}")
            
            # Charger l'index FAISS
            self.index = faiss.read_index(str(self.index_file))
            
            # Charger le DataFrame
            with open(self.docs_file, 'rb') as f:
                self.documents_df = pickle.load(f)
            
            logger.info(f"✅ Loaded index with {self.index.ntotal} vectors")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load index: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> pd.DataFrame:
        """
        Semantic search using FAISS
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            DataFrame with top-k documents and similarity scores
        """
        if self.index is None:
            logger.warning("⚠️ Index not loaded, attempting to load...")
            if not self.load_index():
                logger.error("❌ No index available. Build index first.")
                return pd.DataFrame()
        
        # Générer embedding de la requête
        query_embedding = self.embedding_model.encode(
            [query], 
            normalize_embeddings=True
        )
        
        # Rechercher dans FAISS
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            k
        )
        
        # Récupérer les documents
        if self.documents_df is None or len(indices[0]) == 0:
            return pd.DataFrame()
        
        top_docs = self.documents_df.iloc[indices[0]].copy()
        top_docs['similarity_score'] = distances[0]
        
        return top_docs
    
    def answer_with_context(self, query: str, k: int = 5) -> str:
        """
        RAG: Retrieve documents + Generate answer
        
        Args:
            query: Question to answer
            k: Number of documents to retrieve
            
        Returns:
            Generated answer
        """
        logger.info(f"🔍 RAG Query: {query}")
        
        # 1. Retrieval - Recherche sémantique
        docs = self.search(query, k=k)
        
        if docs.empty:
            return "I couldn't find relevant documents to answer this question."
        
        # 2. Construction du contexte
        context_text = "\n\n".join([
        f"Document {i+1}:\n"
        f"Title: {row.get('article_title', 'N/A')}\n"
        f"URL: {row.get('source_url', 'N/A')}\n"  # <--- On ajoute l'URL ici
        f"Content: {row.get('article_content', '')[:800]}" # Augmenté un peu pour plus de contexte
        for i, (_, row) in enumerate(docs.iterrows())
        ])
        
        # 3. Generation - Réponse par le LLM
        prompt = f"""

            You are an expert oil market and geopolitics analyst.

            RULES:
            - Use ONLY the information from the CONTEXT below.
            - Do NOT rely on prior knowledge for god sake.
            - Do NOT invent titles, sources, or facts.
            - If the answer is not explicitly supported by the context, respond exactly with:
            "I don't know based on the provided documents."

            TASK:
            Answer the question focusing on:
            - oil supply and demand
            - geopolitics and sanctions
            - production and transport
            - OPEC decisions
            - market expectations

            CONTEXT:
            {context_text}

            QUESTION:
            {query}

            OUTPUT REQUIREMENTS:
            - Your answer MUST include:
            - full_tittle
            - These fields must come from the CONTEXT.
            - If multiple documents are relevant, list them all.
            - If no document supports the answer, use the fallback response rule above.

            OUTPUT FORMAT:
            Answer:
            <clear and concise analytical answer>

            Sources:
            - article_title: "<title from context>"
            source_url: "<url from context>"
            
            """

        
        logger.info("🤖 Generating answer with LLM...")
        
        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert analyst on oil markets and geopolique news."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=500
            )
            
            answer = response['choices'][0]['message']['content']
            logger.info("✅ Answer generated")
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ Failed to generate answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def get_index_stats(self) -> dict:
        """
        Get statistics about the current index
        
        Returns:
            Dictionary with index statistics
        """
        if self.index is None:
            return {
                'status': 'No index loaded',
                'num_vectors': 0,
                'dimension': self.dimension
            }
        
        return {
            'status': 'Index loaded',
            'num_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'num_documents': len(self.documents_df) if self.documents_df is not None else 0,
            'index_file': str(self.index_file),
            'docs_file': str(self.docs_file)
        }