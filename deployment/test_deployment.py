"""
Script pour tester l'API déployée sur Hugging Face Spaces
"""

import requests
import json
import sys

def test_deployed_api(api_url):
    """
    Teste l'API déployée
    
    Args:
        api_url: URL de l'API déployée (sans le / final)
    """
    
    print("\n" + "🧪 " * 35)
    print(f"TEST DE L'API DÉPLOYÉE: {api_url}")
    print("🧪 " * 35 + "\n")
    
    # Test 1: Health Check
    print("1️⃣  Test du Health Check...")
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("   ✅ Health check réussi!")
            print(f"   📊 Status: {data.get('status')}")
            print(f"   🤖 Modèle chargé: {data.get('model_loaded')}")
            print(f"   🔧 Vectoriseur chargé: {data.get('vectorizer_loaded')}")
            print(f"   📌 Version: {data.get('version')}\n")
        else:
            print(f"   ❌ Échec: Status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur: {str(e)}\n")
        return False
    
    # Test 2: Prédiction unique
    print("2️⃣  Test de prédiction unique...")
    try:
        test_comment = "Cette vidéo est absolument incroyable! J'adore!"
        
        response = requests.post(
            f"{api_url}/predict",
            json={"text": test_comment},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print("   ✅ Prédiction réussie!")
            print(f"   💬 Commentaire: '{test_comment}'")
            print(f"   😊 Sentiment: {data.get('sentiment')}")
            print(f"   🎯 Confiance: {data.get('confidence'):.2%}")
            print(f"   📊 Probabilités:")
            for sentiment, prob in data.get('probabilities', {}).items():
                print(f"      - {sentiment}: {prob:.2%}\n")
        else:
            print(f"   ❌ Échec: Status {response.status_code}")
            print(f"   Réponse: {response.text}\n")
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur: {str(e)}\n")
        return False
    
    # Test 3: Prédiction batch
    print("3️⃣  Test de prédiction batch...")
    try:
        test_comments = [
            "Super vidéo! Merci pour le partage!",
            "Pas terrible, je suis déçu...",
            "C'est intéressant mais pas exceptionnel",
            "J'adore votre contenu, continuez comme ça!",
            "Nul, je n'ai pas aimé du tout"
        ]
        
        response = requests.post(
            f"{api_url}/predict_batch",
            json={"comments": test_comments},
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            print("   ✅ Prédiction batch réussie!")
            print(f"   📊 {len(test_comments)} commentaires analysés\n")
            
            # Afficher les statistiques
            stats = data.get('statistics', {})
            print("   📈 Statistiques:")
            print(f"      Total: {stats.get('total')}")
            print(f"      😊 Positif: {stats.get('positive')} ({stats.get('positive_percent')}%)")
            print(f"      😐 Neutre: {stats.get('neutral')} ({stats.get('neutral_percent')}%)")
            print(f"      😞 Négatif: {stats.get('negative')} ({stats.get('negative_percent')}%)")
            print(f"      🎯 Confiance moyenne: {stats.get('avg_confidence'):.2%}\n")
            
            # Afficher quelques prédictions
            print("   💬 Exemples de prédictions:")
            for i, (comment, pred) in enumerate(zip(test_comments[:3], data.get('predictions', [])[:3]), 1):
                print(f"      {i}. '{comment}'")
                print(f"         → {pred.get('sentiment')} ({pred.get('confidence'):.2%})\n")
        else:
            print(f"   ❌ Échec: Status {response.status_code}")
            print(f"   Réponse: {response.text}\n")
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur: {str(e)}\n")
        return False
    
    # Test 4: Documentation
    print("4️⃣  Test de la documentation...")
    try:
        response = requests.get(f"{api_url}/docs", timeout=10)
        
        if response.status_code == 200:
            print("   ✅ Documentation accessible!")
            print(f"   📖 URL: {api_url}/docs\n")
        else:
            print(f"   ⚠️  Documentation non accessible (status {response.status_code})\n")
            
    except Exception as e:
        print(f"   ⚠️  Erreur lors de l'accès à la documentation: {str(e)}\n")
    
    # Résumé
    print("=" * 70)
    print("✅ TOUS LES TESTS SONT PASSÉS!")
    print("=" * 70)
    print("\n🎉 Votre API est opérationnelle sur Hugging Face Spaces!")
    print(f"\n🔗 URL de l'API: {api_url}")
    print(f"📖 Documentation: {api_url}/docs")
    print(f"🏥 Health Check: {api_url}/health")
    
    print("\n📋 PROCHAINE ÉTAPE:")
    print("   Mettez à jour l'extension Chrome avec cette URL:")
    print(f"   API_CONFIG.production = '{api_url}'")
    print("   (dans extension/background.js)\n")
    
    return True

def main():
    """Point d'entrée principal"""
    
    print("\n🎯 Test de l'API YouTube Sentiment Analyzer")
    print("=" * 70)
    
    # Demander l'URL de l'API
    if len(sys.argv) > 1:
        api_url = sys.argv[1]
    else:
        print("\n💡 Entrez l'URL de votre Space Hugging Face")
        print("   Exemple: https://your-username-youtube-sentiment-analyzer.hf.space")
        print("   (sans le / à la fin)")
        api_url = input("\n🔗 URL de l'API: ").strip().rstrip('/')
    
    if not api_url:
        print("\n❌ URL invalide!")
        return 1
    
    # Valider l'URL
    if not (api_url.startswith('http://') or api_url.startswith('https://')):
        print("\n❌ L'URL doit commencer par http:// ou https://")
        return 1
    
    # Tester l'API
    success = test_deployed_api(api_url)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())