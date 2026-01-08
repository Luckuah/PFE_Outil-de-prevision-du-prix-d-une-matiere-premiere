import re
import pandas as pd
from typing import Tuple
from tqdm import tqdm
from llama_cpp import Llama

from ..utils.logger import get_logger
from ..utils.config_loader import get_config

logger = get_logger(__name__)

class TrieurAgent:
    """Agent pour scorer la pertinence des articles avec llama-cpp (Qwen2.5-0.5B GGUF)"""
    
    PROMPT_TEMPLATE = """Tu es un expert en analyse pétrolière. Évalue cet article pour prédire les prix.
Format de réponse strict:
Score: [0-4]
Justification: [Texte court]

Article:
Titre: {title}
Contenu: {content}"""

    def __init__(self):
        self.config = get_config()
        
        # Récupération des paramètres GGUF
        # Exemple: repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF", filename="*q8_0.gguf"
        repo_id = self.config.get('models.trieur.repo_id', "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
        filename = self.config.get('models.trieur.filename', "*q8_0.gguf")
        
        logger.info(f"🤖 Loading llama-cpp model: {repo_id}")

        # Initialisation du moteur llama-cpp
        self.llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_gpu_layers=-1,    # Charge tout sur tes 6GB de VRAM
            n_ctx=2048,         # Fenêtre de contexte
            verbose=False       # Désactive les logs techniques lourds
        )
        
        self.temperature = self.config.get('models.trieur.temperature', 0.1)
        logger.info("✅ Agent Trieur prêt sur GPU via CUDA")

    def score_article(self, title: str, content: str) -> Tuple[int, str]:
        content_truncated = content[:800] if content else ""
        prompt = self.PROMPT_TEMPLATE.format(
            title=title or "Sans titre",
            content=content_truncated
        )

        try:
            # Utilisation de l'API de chat (format Qwen géré par llama-cpp)
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "Tu es un analyste financier précis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=150
            )

            text_output = response['choices'][0]['message']['content']
            return self._parse_response(text_output)
            
        except Exception as e:
            logger.error(f"❌ Erreur llama-cpp: {e}")
            return 1, "Erreur lors de l'inférence."

    def _parse_response(self, response: str) -> Tuple[int, str]:
        # Extraction du score
        score_match = re.search(r'Score:\s*(\d)', response)
        score = int(score_match.group(1)) if score_match else 1
        score = max(0, min(4, score))

        # Extraction de la justification
        just_match = re.search(r'Justification:\s*(.+)', response, re.DOTALL | re.IGNORECASE)
        justification = just_match.group(1).strip() if just_match else "Pas de justification."
        
        return score, justification

    def score_articles_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Version optimisée pour DataFrame"""
        logger.info(f"🎯 Scoring de {len(df)} articles...")
        results = [self.score_article(row['article_title'], row['article_content']) 
                   for _, row in tqdm(df.iterrows(), total=len(df))]
        
        df['llm_score'], df['llm_justification'] = zip(*results)
        return df