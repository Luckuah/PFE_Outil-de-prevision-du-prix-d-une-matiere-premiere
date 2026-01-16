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


def main():
    print("="*80)
    print("🤖 TEST DU SYSTÈME RAG")
    print("="*80)
    
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
    
    # ==================== STEP 2: Initialiser RAG ====================
    print("\n\n🔧 STEP 2: Initialisation du RAG Agent")
    print("-"*80)
    
    try:
        rag = RAGAgent()
        print("✅ RAG Agent initialisé")
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        return
    
    # ==================== STEP 3: Construire l'Index ====================
    print("\n\n📚 STEP 3: Construction de l'index FAISS")
    print("-"*80)
    
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
    
    # ==================== STEP 5: Test RAG Complet (Recherche + Génération) ====================
    print("\n\n🤖 STEP 5: Test RAG complet (Recherche + Génération)")
    print("="*80)
    
    rag_query = "which contries have United state sanctions ?"
    
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
    
    # ==================== STEP 6: Mode Interactif ====================
    print("\n\n💡 STEP 6: Mode interactif")
    print("="*80)
    print("\nVoulez-vous tester d'autres requêtes? (y/n)")
    
    try:
        choice = input("Votre choix: ").strip().lower()
        
        if choice == 'y':
            print("\n🎯 Mode interactif activé")
            print("   Tapez vos questions (ou 'quit' pour quitter)\n")
            
            while True:
                query = input("🔍 Votre question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Au revoir!")
                    break
                
                if not query:
                    continue
                
                print("\n📚 Recherche...")
                results = rag.search(query, k=5)
                
                if not results.empty:
                    print(f"✅ {len(results)} documents trouvés\n")
                    for i, (_, row) in enumerate(results.iterrows(), 1):
                        print(f"{i}. [{row['similarity_score']:.3f}] {row['article_title'][:70]}...")
                
                print("\n🤖 Génération de la réponse...")
                answer = rag.answer_with_context(query, k=3)
                print(f"\n📝 Réponse:\n{answer}\n")
                print("-"*80 + "\n")
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
    
    print("\n" + "="*80)
    print("✅ Tests terminés!")
    print("="*80)


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()