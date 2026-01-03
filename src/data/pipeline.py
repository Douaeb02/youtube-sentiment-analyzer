"""
Pipeline complet de préparation des données
Exécute toutes les étapes dans l'ordre
"""
import sys
from pathlib import Path

# Ajouter le dossier parent au path pour les imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.download_data import download_reddit_dataset
from src.data.clean_data import clean_dataset
from src.data.exploratory_analysis import analyze_dataset
from src.data.prepare_train_test import create_train_test_split

def run_data_pipeline():
    """
    Exécute le pipeline complet de préparation des données
    """
    print("\n" + "🚀 "*25)
    print("DÉMARRAGE DU PIPELINE DE DONNÉES")
    print("🚀 "*25 + "\n")
    
    try:
        # Étape 1: Télécharger les données
        print("\n📥 ÉTAPE 1/4: Téléchargement du dataset")
        print("-"*70)
        raw_data_path = download_reddit_dataset()
        
        if raw_data_path is None:
            print("❌ Échec du téléchargement. Arrêt du pipeline.")
            return False
        
        input("\n⏸️  Appuyez sur Entrée pour continuer vers le nettoyage...")
        
        # Étape 2: Nettoyer les données
        print("\n\n🧹 ÉTAPE 2/4: Nettoyage des données")
        print("-"*70)
        clean_data_path = "data/processed/reddit_clean.csv"
        clean_dataset(raw_data_path, clean_data_path)
        
        input("\n⏸️  Appuyez sur Entrée pour continuer vers l'analyse exploratoire...")
        
        # Étape 3: Analyse exploratoire
        print("\n\n🔍 ÉTAPE 3/4: Analyse exploratoire des données")
        print("-"*70)
        analyze_dataset(clean_data_path)
        
        input("\n⏸️  Appuyez sur Entrée pour continuer vers le split train/test...")
        
        # Étape 4: Créer le split train/test
        print("\n\n✂️  ÉTAPE 4/4: Création du split train/test")
        print("-"*70)
        create_train_test_split(
            input_path=clean_data_path,
            output_dir="data/processed",
            test_size=0.2,
            random_state=42
        )
        
        # Résumé final
        print("\n" + "🎉 "*25)
        print("PIPELINE DE DONNÉES TERMINÉ AVEC SUCCÈS!")
        print("🎉 "*25 + "\n")
        
        print("📁 Fichiers créés:")
        print("  ✅ data/raw/reddit.csv               (données brutes)")
        print("  ✅ data/processed/reddit_clean.csv   (données nettoyées)")
        print("  ✅ data/processed/train.csv          (ensemble d'entraînement)")
        print("  ✅ data/processed/test.csv           (ensemble de test)")
        
        print("\n🎯 Prochaine étape:")
        print("  → Phase 3: Développement et entraînement du modèle")
        print("  → Exécutez: python src/models/train_model.py")
        
        return True
        
    except Exception as e:
        print(f"\n Erreur dans le pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_data_pipeline()
    sys.exit(0 if success else 1)