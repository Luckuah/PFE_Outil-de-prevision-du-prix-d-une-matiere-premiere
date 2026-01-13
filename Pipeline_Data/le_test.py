import sys
import os
sys.path.append('.')

from Pipeline_Data.pipeline import GDELTPipeline

def main():
    # Initialiser le pipeline
    pipeline = GDELTPipeline()

    # Tester 1 jour
    print("🚀 Test du pipeline pour hier...")
    
    try:
        df = pipeline.run_single_day('2025-01-05')

        if df is not None and not df.empty:
            print(f"\n✅ {len(df)} articles traités")
            
            # Vérification sécurisée de la colonne
            if 'final_score' in df.columns:
                print(f"Score moyen : {df['final_score'].mean():.2f}")
                print("\n🔝 Top 3 articles :")
                print(df.nlargest(3, 'final_score')[['article_title', 'final_score']])
            else:
                print("⚠️ Attention : La colonne 'final_score' est absente du DataFrame.")
                print("Colonnes disponibles :", df.columns.tolist())
        else:
            print("\n⚠️ Aucun article n'a été récupéré ou traité pour cette date.")

    except Exception as e:
        print(f"\n❌ Une erreur est survenue pendant l'exécution : {e}")

# CRUCIAL : Protection pour le multiprocessing sur Windows
if __name__ == '__main__':
    main()