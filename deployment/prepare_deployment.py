"""
Script pour préparer le déploiement sur Hugging Face Spaces
"""

import os
import shutil
from pathlib import Path

def prepare_deployment():
    """Prépare les fichiers pour le déploiement"""
    
    print("\n" + "🚀 " * 35)
    print("PRÉPARATION DU DÉPLOIEMENT SUR HUGGING FACE SPACES")
    print("🚀 " * 35 + "\n")
    
    # Chemins
    project_root = Path(__file__).parent.parent
    deployment_dir = project_root / "deployment"
    models_source = project_root / "models"
    models_dest = deployment_dir / "models"
    
    print("📂 Chemins:")
    print(f"   Projet: {project_root}")
    print(f"   Déploiement: {deployment_dir}")
    print(f"   Modèles source: {models_source}")
    print(f"   Modèles destination: {models_dest}\n")
    
    # Vérifications
    print("🔍 Vérification des prérequis...")
    
    checks = {
        "Dossier deployment": deployment_dir.exists(),
        "Dockerfile": (deployment_dir / "Dockerfile").exists(),
        "app_api.py": (deployment_dir / "app_api.py").exists(),
        "requirements_production.txt": (deployment_dir / "requirements_production.txt").exists(),
        "Modèle entraîné": (models_source / "sentiment_model.joblib").exists(),
        "Vectoriseur": (models_source / "tfidf_vectorizer.joblib").exists(),
    }
    
    all_ok = True
    for check, status in checks.items():
        icon = "✅" if status else "❌"
        print(f"   {icon} {check}")
        if not status:
            all_ok = False
    
    if not all_ok:
        print("\n❌ Certains fichiers sont manquants!")
        print("   Assurez-vous d'avoir:")
        print("   1. Créé le dossier deployment/")
        print("   2. Copié tous les fichiers dans deployment/")
        print("   3. Entraîné votre modèle (models/*.joblib)")
        return False
    
    print("\n✅ Tous les prérequis sont satisfaits!\n")
    
    # Copier les modèles
    print("📦 Copie des modèles...")
    
    # Supprimer l'ancien dossier models s'il existe
    if models_dest.exists():
        print(f"   🗑️  Suppression de l'ancien dossier: {models_dest}")
        shutil.rmtree(models_dest)
    
    # Créer le nouveau dossier
    print(f"   📁 Création du dossier: {models_dest}")
    models_dest.mkdir(parents=True, exist_ok=True)
    
    # Copier les fichiers
    files_to_copy = [
        "sentiment_model.joblib",
        "tfidf_vectorizer.joblib"
    ]
    
    for filename in files_to_copy:
        source = models_source / filename
        dest = models_dest / filename
        
        if source.exists():
            print(f"   📄 Copie: {filename}")
            shutil.copy2(source, dest)
            
            # Afficher la taille du fichier
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"      Taille: {size_mb:.2f} MB")
        else:
            print(f"   ❌ Fichier non trouvé: {filename}")
            return False
    
    print("\n✅ Modèles copiés avec succès!\n")
    
    # Créer un fichier README pour HF Spaces
    print("📝 Création du README.md pour Hugging Face Spaces...")
    
    readme_content = """---
title: YouTube Sentiment Analyzer API
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 🎬 YouTube Sentiment Analyzer API

API d'analyse de sentiment pour les commentaires YouTube utilisant un modèle de Machine Learning.

## 🚀 Endpoints

- `GET /` - Page d'accueil
- `GET /health` - Vérification de l'état de l'API
- `POST /predict` - Analyse d'un seul commentaire
- `POST /predict_batch` - Analyse de plusieurs commentaires
- `GET /docs` - Documentation interactive (Swagger UI)

## 📊 Exemple d'utilisation

### Analyser un commentaire

```bash
curl -X POST "https://YOUR-SPACE-NAME.hf.space/predict" \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Cette vidéo est incroyable!"}'
```

### Analyser plusieurs commentaires

```bash
curl -X POST "https://YOUR-SPACE-NAME.hf.space/predict_batch" \\
  -H "Content-Type: application/json" \\
  -d '{"comments": ["Super vidéo!", "Pas terrible", "Intéressant"]}'
```

## 🛠️ Technologies

- **FastAPI** - Framework web moderne et rapide
- **scikit-learn** - Machine Learning
- **Docker** - Containerisation
- **Hugging Face Spaces** - Hébergement

## 📦 Modèle

Le modèle a été entraîné sur des commentaires YouTube en français pour classifier le sentiment en trois catégories :
- 😊 Positif
- 😐 Neutre
- 😞 Négatif

## 🔗 Extension Chrome

Cette API est utilisée par l'extension Chrome "YouTube Sentiment Analyzer" pour analyser automatiquement les commentaires YouTube.
"""
    
    readme_path = deployment_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"   ✅ README.md créé: {readme_path}\n")
    
    # Résumé
    print("=" * 70)
    print("✅ PRÉPARATION TERMINÉE AVEC SUCCÈS!")
    print("=" * 70)
    print("\n📋 Fichiers prêts pour le déploiement:")
    print(f"   📁 {deployment_dir}/")
    print("      ├── 🐳 Dockerfile")
    print("      ├── 🐍 app_api.py")
    print("      ├── 📦 requirements_production.txt")
    print("      ├── 📝 README.md")
    print("      └── 📂 models/")
    print("          ├── sentiment_model.joblib")
    print("          └── tfidf_vectorizer.joblib")
    
    print("\n🎯 PROCHAINES ÉTAPES:")
    print("\n1️⃣  Créer un compte sur Hugging Face (si pas déjà fait)")
    print("   👉 https://huggingface.co/join")
    
    print("\n2️⃣  Créer un nouveau Space")
    print("   👉 https://huggingface.co/new-space")
    print("   - Nom: youtube-sentiment-analyzer")
    print("   - SDK: Docker")
    print("   - Public ou Private: à votre choix")
    
    print("\n3️⃣  Cloner le Space localement")
    print("   git clone https://huggingface.co/spaces/YOUR-USERNAME/youtube-sentiment-analyzer")
    
    print("\n4️⃣  Copier les fichiers du dossier deployment/")
    print("   Copiez tout le contenu de deployment/ dans le dossier du Space")
    
    print("\n5️⃣  Push vers Hugging Face")
    print("   cd youtube-sentiment-analyzer")
    print("   git add .")
    print("   git commit -m 'Initial deployment'")
    print("   git push")
    
    print("\n6️⃣  Attendre le déploiement (2-3 minutes)")
    print("   Suivez les logs sur la page de votre Space")
    
    print("\n7️⃣  Tester l'API déployée")
    print("   python deployment/test_deployment.py")
    
    print("\n8️⃣  Mettre à jour l'extension Chrome")
    print("   Modifier API_CONFIG.production dans extension/background.js")
    
    print("\n" + "🎉 " * 35)
    print("BONNE CHANCE AVEC VOTRE DÉPLOIEMENT!")
    print("🎉 " * 35 + "\n")
    
    return True

if __name__ == "__main__":
    success = prepare_deployment()
    exit(0 if success else 1)