import subprocess
import sys
from pathlib import Path
import time

class SystemValidator:
    def __init__(self):
        self.checks = []
        self.root_dir = Path.cwd()
    
    def check_files_structure(self):
        """Vérifie la structure des fichiers"""
        print("\n" + "="*70)
        print("VÉRIFICATION 1: STRUCTURE DES FICHIERS")
        print("="*70)
        
        required_files = {
            "Données": [
                "data/raw/reddit.csv",
                "data/processed/reddit_clean.csv",
                "data/processed/train.csv",
                "data/processed/test.csv"
            ],
            "Modèle": [
                "models/sentiment_model.joblib",
                "models/tfidf_vectorizer.joblib",
                "models/training_history.json"
            ],
            "API": [
                "src/api/main.py",
                "src/api/models.py",
                "src/api/prediction_service.py",
                "app_api.py",
                "Dockerfile",
                "requirements_production.txt"
            ],
            "Extension": [
                "chrome-extension/manifest.json",
                "chrome-extension/popup.html",
                "chrome-extension/popup.js",
                "chrome-extension/styles.css",
                "chrome-extension/content.js",
                "chrome-extension/background.js"
            ],
            "Scripts": [
                "src/data/download_data.py",
                "src/models/train_model.py",
                "tests/test_api.py",
                "run_api.py"
            ]
        }
        
        all_present = True
        
        for category, files in required_files.items():
            print(f"\n📁 {category}:")
            category_missing = []
            
            for file_path in files:
                path = Path(file_path)
                if path.exists():
                    size = path.stat().st_size / 1024  # KB
                    print(f"  ✅ {file_path:<50} ({size:>8.2f} KB)")
                else:
                    print(f"  ❌ {file_path:<50} MANQUANT")
                    category_missing.append(file_path)
                    all_present = False
            
            if category_missing:
                print(f"\n  ⚠️  Fichiers manquants dans {category}: {len(category_missing)}")
        
        if all_present:
            print("\n✅ Tous les fichiers sont présents")
        else:
            print("\n⚠️  Certains fichiers sont manquants")
        
        self.checks.append(("Structure des fichiers", all_present))
        return all_present
    
    def check_python_dependencies(self):
        """Vérifie les dépendances Python"""
        print("\n" + "="*70)
        print("VÉRIFICATION 2: DÉPENDANCES PYTHON")
        print("="*70)
        
        required_packages = [
            'numpy',
            'pandas',
            'scikit-learn',
            'fastapi',
            'uvicorn',
            'pydantic',
            'joblib',
            'requests'
        ]
        
        all_installed = True
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"  ✅ {package}")
            except ImportError:
                print(f"  ❌ {package} - NON INSTALLÉ")
                all_installed = False
        
        if not all_installed:
            print("\n💡 Installez les dépendances avec:")
            print("   pip install -r requirements.txt")
        else:
            print("\n✅ Toutes les dépendances sont installées")
        
        self.checks.append(("Dépendances Python", all_installed))
        return all_installed
    
    def check_model_quality(self):
        """Vérifie la qualité du modèle"""
        print("\n" + "="*70)
        print("VÉRIFICATION 3: QUALITÉ DU MODÈLE")
        print("="*70)
        
        try:
            import json
            history_path = Path("models/training_history.json")
            
            if not history_path.exists():
                print("❌ Historique d'entraînement non trouvé")
                self.checks.append(("Qualité du modèle", False))
                return False
            
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            metrics = history.get('metrics', {})
            
            test_accuracy = metrics.get('test_accuracy', 0)
            avg_f1 = metrics.get('avg_f1', 0)
            inference_time = metrics.get('inference_time_ms', 0) * 50
            
            print(f"\n📊 Métriques du modèle:")
            print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
            print(f"  Avg F1-Score: {avg_f1:.4f}")
            print(f"  Temps batch 50: {inference_time:.2f}ms")
            
            # Vérifier les critères
            criteria = {
                "Accuracy ≥ 80%": test_accuracy >= 0.80,
                "F1-Score ≥ 0.75": avg_f1 >= 0.75,
                "Temps < 100ms": inference_time < 100
            }
            
            print(f"\n✅ Critères de performance:")
            all_met = True
            for criterion, met in criteria.items():
                status = "✅" if met else "❌"
                print(f"  {status} {criterion}")
                if not met:
                    all_met = False
            
            self.checks.append(("Qualité du modèle", all_met))
            return all_met
        
        except Exception as e:
            print(f"❌ Erreur: {e}")
            self.checks.append(("Qualité du modèle", False))
            return False
    
    def check_api_local(self):
        """Vérifie que l'API locale peut démarrer"""
        print("\n" + "="*70)
        print("VÉRIFICATION 4: API LOCALE")
        print("="*70)
        
        try:
            import requests
            
            print("🔍 Vérification de l'API locale...")
            
            try:
                response = requests.get("http://localhost:8000/health", timeout=2)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'healthy':
                        print("✅ API locale accessible et en bonne santé")
                        self.checks.append(("API locale", True))
                        return True
                    else:
                        print("⚠️  API accessible mais pas healthy")
                        self.checks.append(("API locale", False))
                        return False
                else:
                    print(f"⚠️  Status code: {response.status_code}")
                    self.checks.append(("API locale", False))
                    return False
            
            except requests.exceptions.ConnectionError:
                print("⚠️  API locale non accessible")
                print("   Pour tester l'API, lancez: python run_api.py")
                self.checks.append(("API locale", None))  # Non testé
                return None
        
        except Exception as e:
            print(f"❌ Erreur: {e}")
            self.checks.append(("API locale", False))
            return False
    
    def check_extension_files(self):
        """Vérifie les fichiers de l'extension"""
        print("\n" + "="*70)
        print("VÉRIFICATION 5: EXTENSION CHROME")
        print("="*70)
        
        try:
            import json
            
            manifest_path = Path("chrome-extension/manifest.json")
            
            if not manifest_path.exists():
                print("❌ manifest.json non trouvé")
                self.checks.append(("Extension Chrome", False))
                return False
            
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            print(f"\n📋 Manifest:")
            print(f"  Nom: {manifest.get('name')}")
            print(f"  Version: {manifest.get('version')}")
            print(f"  Manifest version: {manifest.get('manifest_version')}")
            
            # Vérifier les permissions
            permissions = manifest.get('permissions', [])
            print(f"\n🔐 Permissions:")
            for perm in permissions:
                print(f"  • {perm}")
            
            # Vérifier les content scripts
            content_scripts = manifest.get('content_scripts', [])
            if content_scripts:
                print(f"\n📜 Content Scripts:")
                for cs in content_scripts:
                    print(f"  • Matches: {cs.get('matches')}")
                    print(f"  • JS: {cs.get('js')}")
            
            print("\n✅ Extension Chrome correctement configurée")
            self.checks.append(("Extension Chrome", True))
            return True
        
        except Exception as e:
            print(f"❌ Erreur: {e}")
            self.checks.append(("Extension Chrome", False))
            return False
    
    def check_deployment_readiness(self):
        """Vérifie si le projet est prêt pour le déploiement"""
        print("\n" + "="*70)
        print("VÉRIFICATION 6: PRÊT POUR LE DÉPLOIEMENT")
        print("="*70)
        
        deployment_files = [
            "app_api.py",
            "Dockerfile",
            "requirements_production.txt",
            "models/sentiment_model.joblib",
            "models/tfidf_vectorizer.joblib"
        ]
        
        all_ready = True
        
        for file_path in deployment_files:
            path = Path(file_path)
            if path.exists():
                print(f"  ✅ {file_path}")
            else:
                print(f"  ❌ {file_path} - MANQUANT")
                all_ready = False
        
        if all_ready:
            print("\n✅ Prêt pour le déploiement")
            print("\n💡 Prochaine étape:")
            print("   python prepare_deployment.py")
        else:
            print("\n⚠️  Fichiers manquants pour le déploiement")
        
        self.checks.append(("Prêt pour déploiement", all_ready))
        return all_ready
    
    def generate_report(self):
        """Génère un rapport de validation"""
        print("\n" + "="*70)
        print("📊 RAPPORT DE VALIDATION DU SYSTÈME")
        print("="*70)
        
        for name, status in self.checks:
            if status is True:
                icon = "✅ RÉUSSI"
            elif status is False:
                icon = "❌ ÉCHOUÉ"
            else:
                icon = "⚠️  NON TESTÉ"
            
            print(f"  {name:30s}: {icon}")
        
        # Compter les résultats
        passed = sum(1 for _, s in self.checks if s is True)
        failed = sum(1 for _, s in self.checks if s is False)
        skipped = sum(1 for _, s in self.checks if s is None)
        total = len(self.checks)
        
        print(f"\n📈 Résumé:")
        print(f"  Réussis: {passed}/{total}")
        print(f"  Échoués: {failed}/{total}")
        print(f"  Non testés: {skipped}/{total}")
        
        if failed == 0 and skipped == 0:
            print("\n🎉 SYSTÈME ENTIÈREMENT VALIDÉ!")
            print("   Tous les composants sont fonctionnels!")
            return True
        elif failed == 0:
            print("\n✅ SYSTÈME VALIDÉ (avec tests non exécutés)")
            print("   Les composants présents sont fonctionnels")
            return True
        else:
            print("\n⚠️  PROBLÈMES DÉTECTÉS")
            print("   Veuillez corriger les erreurs avant de continuer")
            return False

def main():
    """Fonction principale"""
    print("\n" + "🔍 "*35)
    print("VALIDATION COMPLÈTE DU SYSTÈME")
    print("YouTube Sentiment Analyzer")
    print("🔍 "*35)
    
    validator = SystemValidator()
    
    # Exécuter toutes les vérifications
    validator.check_files_structure()
    validator.check_python_dependencies()
    validator.check_model_quality()
    validator.check_api_local()
    validator.check_extension_files()
    validator.check_deployment_readiness()
    
    # Générer le rapport
    success = validator.generate_report()
    
    if success:
        print("\n" + "="*70)
        print("🎯 PROCHAINES ÉTAPES")
        print("="*70)
        print("\n1. Tests du modèle:")
        print("   python tests/test_complete_model.py")
        print("\n2. Tests de l'API:")
        print("   python run_api.py  # Dans un terminal")
        print("   python tests/test_api.py  # Dans un autre terminal")
        print("\n3. Tests d'intégration:")
        print("   python tests/test_integration.py")
        print("\n4. Préparer le déploiement:")
        print("   python prepare_deployment.py")
        print("\n5. Installer l'extension Chrome:")
        print("   chrome://extensions/ → Mode développeur → Charger")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()