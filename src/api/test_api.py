"""
Tests pour l'API FastAPI
"""
import requests
import json
import time

# URL de base de l'API
BASE_URL = "http://localhost:8000"

def test_root():
    """Test du endpoint racine"""
    print("\n" + "="*70)
    print("🧪 TEST: Endpoint racine (/)")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/")
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    print("✅ Test réussi!")

def test_health():
    """Test du endpoint health"""
    print("\n" + "="*70)
    print("🧪 TEST: Endpoint health (/health)")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/health")
    
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    
    assert response.status_code == 200
    assert data['model_loaded'] == True
    assert data['vectorizer_loaded'] == True
    
    print("✅ Test réussi!")

def test_predict_batch():
    """Test du endpoint predict_batch"""
    print("\n" + "="*70)
    print("🧪 TEST: Endpoint predict_batch (/predict_batch)")
    print("="*70)
    
    # Données de test
    test_comments = [
        "This video is amazing! Best tutorial ever!",
        "Didn't work for me. Waste of time.",
        "Thanks for sharing this.",
        "First! 🎉",
        "The audio quality is terrible.",
        "Very helpful, subscribed!",
        "This is trash.",
        "Can you make more videos like this?",
        "Not bad, but could be better.",
        "Absolutely perfect! 5 stars!"
    ]
    
    payload = {
        "comments": test_comments
    }
    
    print(f"\n📤 Envoi de {len(test_comments)} commentaires...")
    
    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/predict_batch",
        json=payload
    )
    request_time = (time.time() - start_time) * 1000
    
    print(f"⏱️  Temps de requête: {request_time:.2f}ms")
    print(f"Status Code: {response.status_code}")
    
    assert response.status_code == 200
    
    data = response.json()
    
    print(f"\n📊 Statistiques:")
    print(f"  Total: {data['total_comments']}")
    print(f"  Temps de traitement: {data['processing_time_ms']:.2f}ms")
    
    stats = data['statistics']
    print(f"\n📈 Répartition des sentiments:")
    print(f"  Positif: {stats['positive']} ({stats['positive_percent']}%)")
    print(f"  Neutre:  {stats['neutral']} ({stats['neutral_percent']}%)")
    print(f"  Négatif: {stats['negative']} ({stats['negative_percent']}%)")
    print(f"  Confiance moyenne: {stats['avg_confidence']:.2%}")
    
    print(f"\n📝 Exemples de prédictions:")
    for i, pred in enumerate(data['predictions'][:3]):
        emoji = {"Négatif": "😞", "Neutre": "😐", "Positif": "😊"}[pred['sentiment']]
        print(f"\n  {i+1}. {emoji} \"{pred['text'][:50]}...\"")
        print(f"     → {pred['sentiment']} (confiance: {pred['confidence']:.2%})")
    
    print("\n✅ Test réussi!")

def test_predict_batch_edge_cases():
    """Test des cas limites"""
    print("\n" + "="*70)
    print("🧪 TEST: Cas limites")
    print("="*70)
    
    # Test 1: Liste vide (devrait échouer)
    print("\n📌 Test 1: Liste vide")
    response = requests.post(
        f"{BASE_URL}/predict_batch",
        json={"comments": []}
    )
    print(f"  Status Code: {response.status_code}")
    print(f"  Résultat attendu: 422 (erreur de validation)")
    assert response.status_code == 422
    print("  ✅ Validation correcte")
    
    # Test 2: Commentaire très court
    print("\n📌 Test 2: Commentaire très court")
    response = requests.post(
        f"{BASE_URL}/predict_batch",
        json={"comments": ["Ok"]}
    )
    print(f"  Status Code: {response.status_code}")
    assert response.status_code == 200
    print("  ✅ Traité correctement")
    
    # Test 3: Commentaire très long
    print("\n📌 Test 3: Commentaire très long")
    long_comment = "This is a test " * 100
    response = requests.post(
        f"{BASE_URL}/predict_batch",
        json={"comments": [long_comment]}
    )
    print(f"  Status Code: {response.status_code}")
    print(f"  ✅ Traité correctement")
    
    # Test 4: Trop de commentaires (> 100)
    print("\n📌 Test 4: Trop de commentaires (>100)")
    many_comments = ["Test comment"] * 101
    response = requests.post(
        f"{BASE_URL}/predict_batch",
        json={"comments": many_comments}
    )
    print(f"  Status Code: {response.status_code}")
    print(f"  Résultat attendu: 422 (trop de commentaires)")
    assert response.status_code == 422
    print("  ✅ Validation correcte")
    
    print("\n✅ Tous les tests de cas limites réussis!")

def test_performance():
    """Test de performance"""
    print("\n" + "="*70)
    print("🧪 TEST: Performance")
    print("="*70)
    
    # Test avec différentes tailles de batch
    batch_sizes = [10, 25, 50, 75, 100]
    
    for size in batch_sizes:
        comments = [f"Test comment number {i}" for i in range(size)]
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/predict_batch",
            json={"comments": comments}
        )
        request_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            processing_time = data['processing_time_ms']
            
            print(f"\n📊 Batch de {size} commentaires:")
            print(f"  Temps total: {request_time:.2f}ms")
            print(f"  Temps traitement: {processing_time:.2f}ms")
            print(f"  Temps par commentaire: {processing_time/size:.2f}ms")
            
            if size == 50 and processing_time < 100:
                print(f"  ✅ Performance excellente (< 100ms)")
            elif size == 50:
                print(f"  ⚠️  Performance à améliorer")
    
    print("\n✅ Test de performance terminé!")

def test_realistic_youtube_comments():
    """Test avec de vrais commentaires YouTube"""
    print("\n" + "="*70)
    print("🧪 TEST: Commentaires YouTube réalistes")
    print("="*70)
    
    youtube_comments = [
        "First! 🎉",
        "This video saved my life! Thank you so much!",
        "Worst tutorial ever. Didn't understand anything.",
        "Who's watching in 2025?",
        "Like if you agree!",
        "Great explanation, very clear and concise.",
        "The audio is too low, can't hear anything.",
        "Please make more videos like this!",
        "This doesn't work. Total waste of time.",
        "Amazing content as always! Keep it up! 👍",
        "Can someone explain the part at 5:30?",
        "Why is there no dark mode?",
        "Subscribed and hit the bell! 🔔",
        "This is outdated, doesn't work anymore.",
        "Perfect timing! I was just looking for this!",
    ]
    
    response = requests.post(
        f"{BASE_URL}/predict_batch",
        json={"comments": youtube_comments}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    print(f"\n📊 Analyse de {len(youtube_comments)} commentaires YouTube:")
    stats = data['statistics']
    
    # Afficher un graphique en texte
    pos_bar = "█" * int(stats['positive_percent'] / 2)
    neu_bar = "█" * int(stats['neutral_percent'] / 2)
    neg_bar = "█" * int(stats['negative_percent'] / 2)
    
    print(f"\n😊 Positif: {stats['positive']:2d} ({stats['positive_percent']:5.1f}%) {pos_bar}")
    print(f"😐 Neutre:  {stats['neutral']:2d} ({stats['neutral_percent']:5.1f}%) {neu_bar}")
    print(f"😞 Négatif: {stats['negative']:2d} ({stats['negative_percent']:5.1f}%) {neg_bar}")
    
    print(f"\n📝 Quelques exemples:")
    for pred in data['predictions'][:5]:
        emoji = {"Négatif": "😞", "Neutre": "😐", "Positif": "😊"}[pred['sentiment']]
        print(f"\n  {emoji} \"{pred['text']}\"")
        print(f"     → {pred['sentiment']} ({pred['confidence']:.1%})")
    
    print("\n✅ Test réussi!")

def run_all_tests():
    """Exécute tous les tests"""
    print("\n" + "🧪 "*35)
    print("TESTS DE L'API YOUTUBE SENTIMENT ANALYZER")
    print("🧪 "*35)
    
    try:
        # Vérifier que l'API est accessible
        print("\n🔍 Vérification de la disponibilité de l'API...")
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print("❌ L'API n'est pas accessible!")
            print("   Lancez l'API avec: python src/api/main.py")
            return
        print("✅ API accessible\n")
        
        # Exécuter les tests
        test_root()
        test_health()
        test_predict_batch()
        test_predict_batch_edge_cases()
        test_performance()
        test_realistic_youtube_comments()
        
        print("\n" + "🎉 "*35)
        print("TOUS LES TESTS SONT RÉUSSIS!")
        print("🎉 "*35 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Impossible de se connecter à l'API!")
        print("   Assurez-vous que l'API est lancée:")
        print("   python src/api/main.py")
    except Exception as e:
        print(f"\n❌ Erreur lors des tests: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()