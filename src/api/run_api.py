"""
Script pour démarrer l'API FastAPI
"""
import uvicorn
import sys
from pathlib import Path

def check_requirements():
    """Vérifie que les fichiers nécessaires existent"""
    print("🔍 Vérification des prérequis...")
    
    required_files = [
        "models/sentiment_model.joblib",
        "models/tfidf_vectorizer.joblib"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("\n❌ Fichiers manquants:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 Entraînez d'abord le modèle:")
        print("   python src/models/train_model.py")
        return False
    
    print("✅ Tous les fichiers sont présents\n")
    return True

def main():
    """Fonction principale"""
    print("\n" + "🚀 "*35)
    print("DÉMARRAGE DE L'API YOUTUBE SENTIMENT ANALYZER")
    print("🚀 "*35 + "\n")
    
    # Vérifier les prérequis
    if not check_requirements():
        sys.exit(1)
    
    print("📡 Démarrage du serveur FastAPI...")
    print("="*70)
    print("📍 URL: http://localhost:8000")
    print("📚 Documentation: http://localhost:8000/docs")
    print("📖 ReDoc: http://localhost:8000/redoc")
    print("💚 Health Check: http://localhost:8000/health")
    print("="*70)
    print("\n⌨️  Appuyez sur Ctrl+C pour arrêter le serveur\n")
    
    # Démarrer le serveur
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Recharge automatiquement lors des modifications
        log_level="info"
    )

if __name__ == "__main__":
    main()