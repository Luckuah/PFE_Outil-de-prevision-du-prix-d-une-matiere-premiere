
import re
import pandas as pd
from typing import Tuple, List
from tqdm import tqdm
from llama_cpp import Llama
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import get_logger
from utils.config_loader import get_config

logger = get_logger(__name__)


def load_prompt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8")

class TrieurAgent:
    """Agent pour scorer la pertinence des articles avec llama-cpp (Qwen2.5-0.5B GGUF)"""
    
    PROMPT_TEMPLATE = """Tu es un expert en analyse pétrolière et géopolitique. Évalue cet article pour estimer son impact potentiel sur les prix du pétrole à court terme (jours à semaines).

Critères d'évaluation :
- Impact direct ou indirect sur l'offre ou la demande mondiale de pétrole
- Événements géopolitiques affectant des pays producteurs, routes énergétiques ou sanctions
- Annonces économiques ou décisions institutionnelles influençant le marché pétrolier
- Crises, conflits ou tensions susceptibles de perturber la production ou le transport
- Décisions politiques ou stratégiques (OPEP, États, sanctions, régulation)

Échelle de notation :
0 = Non pertinent pour le marché pétrolier
1 = Faible pertinence (impact indirect ou marginal)
2 = Pertinence modérée (impact possible mais limité)
3 = Forte pertinence (impact probable sur les prix)
4 = Pertinence extrême (choc majeur ou immédiat sur les prix)

Format de réponse STRICT :
Score: [0-4]
Justification: [1-2 phrases mentionnant explicitement le mécanisme pétrolier]

Article à évaluer :
Titre: {title}
Contenu: {content}

Ta réponse :
:"""

    def __init__(self):
        """Initialize Trieur Agent with Qwen model via llama-cpp"""
        self.config = get_config()
        
        # Récupération des paramètres GGUF
        repo_id = self.config.get('models.trieur.repo_id', "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
        filename = self.config.get('models.trieur.filename', "*q8_0.gguf")
        
        logger.info(f"🤖 Loading llama-cpp model: {repo_id}/{filename}")
        
        try:
            # Initialisation du moteur llama-cpp
            self.llm = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_gpu_layers=-1,    # Charge tout sur GPU si disponible
                n_ctx=2048,         # Fenêtre de contexte
                verbose=False,      # Désactive les logs techniques
                download_kwargs={'timeout': 600}  # 10 minutes
            )
            
            self.temperature = self.config.get('models.trieur.temperature', 0.1)
            self.max_tokens = self.config.get('models.trieur.max_new_tokens', 150)
            
            # Récupérer les poids de scoring
            self.weights = self.config.get('scoring.weights', {
                'llm': 0.6,
                'goldstein': 0.1,
                'mentions': 0.1,
                'tone': 0.1,
                'oil_country': 0.1
            })
            
            # Liste des pays producteurs
            self.oil_countries = self.config.get('filtering.oil_producing_countries', [
                'SAU', 'RUS', 'USA', 'IRQ', 'IRN', 'CAN', 'CHN', 'UAE', 
                'KWT', 'VEN', 'NGA', 'NOR', 'MEX', 'QAT', 'DZA', 'BRA', 'KAZ', 'GBR'
            ])
            
            logger.info("✅ Agent Trieur prêt (llama-cpp + GGUF)")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def score_article(self, title: str, content: str) -> Tuple[int, str]:
        """
        Score a single article using LLM
        
        Args:
            title: Article title
            content: Article content
            
        Returns:
            Tuple of (score 0-4, justification)
        """
        # Tronquer le contenu
        content_truncated = content[:800] if content else ""
        
        # Créer le prompt
        prompt = self.PROMPT_TEMPLATE.format(
            title=title or "Sans titre",
            content=content_truncated
        )
        
        try:
            # Appel au modèle via llama-cpp
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "Tu es un analyste financier expert en marchés pétroliers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extraire la réponse
            text_output = response['choices'][0]['message']['content']
            
            # Parser la réponse
            return self._parse_response(text_output)
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du scoring LLM: {e}")
            return 1, "Erreur lors de l'inférence."

    def _parse_response(self, response: str) -> Tuple[int, str]:
        """
        Parse LLM response to extract score and justification
        
        Args:
            response: Raw LLM output
            
        Returns:
            Tuple of (score, justification)
        """
        # Extraction du score
        score_match = re.search(r'Score:\s*(\d)', response, re.IGNORECASE)
        if score_match:
            score = int(score_match.group(1))
            score = max(0, min(4, score))  # Clamp entre 0 et 4
        else:
            score = 1
            logger.debug(f"⚠️ Score non trouvé dans: {response[:100]}")
        
        # Extraction de la justification
        just_match = re.search(r'Justification:\s*(.+?)(?:\n|$)', response, re.DOTALL | re.IGNORECASE)
        if just_match:
            justification = just_match.group(1).strip()
            # Limiter la longueur
            justification = justification[:500]
        else:
            justification = "Pas de justification fournie."
        
        return score, justification

    def compute_final_score(self, row: pd.Series) -> float:
        """
        Compute final score combining LLM score with GDELT features
        
        Formula:
        final_score = (
            0.6 * (llm_score/4 * 100) +
            0.1 * ((abs(goldstein) + 10) / 20 * 100) +
            0.1 * (min(mentions/10, 1) * 100) +
            0.1 * (abs(tone)/100 * 100) +
            0.1 * (100 if is_oil_country else 0)
        )
        
        Args:
            row: DataFrame row with features
            
        Returns:
            Final score (0-100)
        """
        try:
            # Score LLM normalisé (0-4 → 0-100)
            llm_score = row.get('llm_score', 0)
            llm_component = (llm_score / 4.0) * 100
            
            # Goldstein scale normalisé (-10 à +10 → 0-100)
            goldstein = row.get('GoldsteinScale', 0)
            goldstein_component = ((abs(goldstein) + 10) / 20.0) * 100
            
            # Mentions normalisé (0-10+ → 0-100)
            mentions = row.get('NumMentions', 0)
            mentions_component = min(mentions / 10.0, 1.0) * 100
            
            # Tone normalisé (abs value, -100 à +100 → 0-100)
            tone = row.get('AvgTone', 0)
            tone_component = (abs(tone) / 100.0) * 100
            
            # Pays producteur (boolean → 0 ou 100)
            is_oil_country = row.get('is_oil_country', False)
            country_component = 100 if is_oil_country else 0
            
            # Calcul du score final
            final_score = (
                self.weights['llm'] * llm_component +
                self.weights['goldstein'] * goldstein_component +
                self.weights['mentions'] * mentions_component +
                self.weights['tone'] * tone_component +
                self.weights['oil_country'] * country_component
            )
            
            # Arrondir et clamper entre 0 et 100
            final_score = round(max(0, min(100, final_score)), 2)
            
            return final_score
            
        except Exception as e:
            logger.error(f"❌ Error computing final score: {e}")
            return 0.0

    def score_articles_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score multiple articles in batch
        
        Args:
            df: DataFrame with articles (must have article_title, article_content)
            
        Returns:
            DataFrame with llm_score, llm_justification, final_score columns
        """
        if df.empty:
            logger.warning("⚠️ Empty DataFrame provided for scoring")
            return df
        
        logger.info(f"🎯 Scoring {len(df)} articles with LLM...")
        
        scores = []
        justifications = []
        
        # Score chaque article avec barre de progression
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring articles"):
            title = row.get('article_title', '')
            content = row.get('article_content', '')
            
            score, justification = self.score_article(title, content)
            scores.append(score)
            justifications.append(justification)
        
        # Ajouter les scores LLM
        df['llm_score'] = scores
        df['llm_justification'] = justifications
        
        logger.info(f"✅ LLM scoring complete. Average score: {sum(scores)/len(scores):.2f}")
        
        # Calculer les scores finaux
        logger.info("📊 Computing final scores...")
        df['final_score'] = df.apply(self.compute_final_score, axis=1)
        
        avg_final = df['final_score'].mean()
        logger.info(f"✅ Final scores computed. Average: {avg_final:.2f}")
        
        return df

    def filter_top_articles(self, df: pd.DataFrame, min_score: float = 50, 
                           max_articles: int = 100) -> pd.DataFrame:
        """
        Filter articles by score
        
        Args:
            df: DataFrame with final_score column
            min_score: Minimum score threshold
            max_articles: Maximum number of articles to keep
            
        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df
        
        # Filtrer par score minimum
        df_filtered = df[df['final_score'] >= min_score].copy()
        
        logger.info(f"📊 Articles with score >= {min_score}: {len(df_filtered)}")
        
        # Garder les top N
        if len(df_filtered) > max_articles:
            df_filtered = df_filtered.nlargest(max_articles, 'final_score')
            logger.info(f"🔝 Keeping top {max_articles} articles")
        
        return df_filtered


# Standalone function for quick testing
def score_single_article(title: str, content: str) -> Tuple[int, str, float]:
    """
    Quick function to score a single article
    
    Returns:
        Tuple of (llm_score, justification, final_score)
    """
    agent = TrieurAgent()
    
    # Create a dummy row for final score computation
    llm_score, justification = agent.score_article(title, content)
    
    row = pd.Series({
        'llm_score': llm_score,
        'GoldsteinScale': 0,
        'NumMentions': 5,
        'AvgTone': -5.0,
        'is_oil_country': True
    })
    
    final_score = agent.compute_final_score(row)
    
    return llm_score, justification, final_score