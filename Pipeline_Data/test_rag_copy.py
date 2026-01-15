"""
Test script for RAG Agent
"""
import sys
sys.path.append('.')

from Pipeline_Data.models.rag_agent import RAGAgent
from database.mysql_connector import MySQLConnector
from utils.logger import get_logger
import pandas as pd

logger = get_logger(__name__)


def verif_db():
    # ==================== STEP 1: Vérifier la Base de Données ====================
    print("\n📊 STEP 1: Vérification de la base de données")
    print("-"*80)
    
    db = MySQLConnector()
    
    # Tester la connexion
    if not db.test_connection():
        print("❌ Impossible de se connecter à la base de données")
        return
    
    # Compter les articles
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM gdelt_articles_scored")
    count = cursor.fetchone()[0]
    cursor.close()
    
    print(f"✅ Base de données connectée")
    print(f"📈 Nombre d'articles en base: {count}")
    
    if count == 0:
        print("\n⚠️ Aucun article en base de données!")
        print("   Veuillez d'abord exécuter le pipeline pour insérer des articles.")
        print("   Commande: poetry run python le_test.py")
        return
    
    # Afficher quelques articles
    cursor = conn.cursor()
    cursor.execute("""
        SELECT article_title, final_score, day 
        FROM gdelt_articles_scored 
        ORDER BY final_score DESC 
        LIMIT 5
    """)
    top_articles = cursor.fetchall()
    cursor.close()
    
    print(f"\n🔝 Top 5 articles par score:")
    for i, (title, score, day) in enumerate(top_articles, 1):
        print(f"   {i}. [{score:.1f}] {title[:70]}... ({day})")

def init_rag():
    # ==================== STEP 2: Initialiser RAG ====================
    print("\n\n🔧 STEP 2: Initialisation du RAG Agent")
    print("-"*80)
    
    try:
        rag = RAGAgent()
        print("✅ RAG Agent initialisé")
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        return
    
    return rag
    
    

def create_rag():

    rag=init_rag()

    db = MySQLConnector()

    try:
        # Charger les articles depuis la base
        df_articles = db.get_articles(limit=1000, min_score=0, order_by='day DESC')
        
        if df_articles.empty:
            print("❌ Aucun article récupéré")
            return
        
        print(f"✅ {len(df_articles)} articles chargés depuis la base")
        
        # Construire l'index
        rag.build_index(df_articles)
        print(f"✅ Index FAISS construit avec {len(df_articles)} documents")
        
    except Exception as e:
        print(f"❌ Erreur lors de la construction de l'index: {e}")
        import traceback
        traceback.print_exc()
        return
    
    return rag


def get_answer(rag_query:str,rag:RAGAgent):
    print(f"\n💬 Question: {rag_query}")
    print("-"*80)
    
    try:
        answer = rag.answer_with_context(rag_query, k=5)
        print(f"\n📝 Réponse générée par le LLM:\n")
        print(answer)
        
    except Exception as e:
        print(f"❌ Erreur lors de la génération: {e}")
        import traceback
        traceback.print_exc()
    
    return answer


def unit_test(rag:RAGAgent):
        # ==================== STEP 4: Tests de Recherche ====================
    print("\n\n🔍 STEP 4: Tests de recherche sémantique")
    print("-"*80)
    #--------------------------------------------------------------------------------------------------------
    # Liste de requêtes de test
    test_queries = [
        "OPEC production cuts and oil prices",
        "Russia energy sanctions",
        "Saudi Arabia oil policy",
        "US petroleum reserves",
        "Ukraine conflict impact on gas"
    ]
    
    print("\n📋 Requêtes de test:")
    for i, q in enumerate(test_queries, 1):
        print(f"   {i}. {q}")
    
    print("\n" + "="*80)
    
    for query_num, query in enumerate(test_queries, 1):
        print(f"\n🔎 Requête {query_num}/{len(test_queries)}: '{query}'")
        print("-"*80)
        
        try:
            # Recherche sémantique
            results = rag.search(query, k=3)
            
            if results.empty:
                print("   ⚠️ Aucun résultat trouvé")
                continue
            
            print(f"   ✅ {len(results)} résultats trouvés\n")
            
            for i, (idx, row) in enumerate(results.iterrows(), 1):
                similarity = row.get('similarity_score', 0)
                title = row.get('article_title', 'N/A')
                score = row.get('final_score', 0)
                day = row.get('day', 'N/A')
                
                print(f"   {i}. [Similarité: {similarity:.3f}] [Score: {score:.1f}]")
                print(f"      Titre: {title[:80]}...")
                print(f"      Date: {day}")
                
                # Afficher un extrait du contenu
                content = row.get('article_content', '')
                if content:
                    excerpt = content[:150].replace('\n', ' ')
                    print(f"      Extrait: {excerpt}...")
                print()
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            import traceback
            traceback.print_exc()

