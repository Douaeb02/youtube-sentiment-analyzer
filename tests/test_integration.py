"""
Tests d'intégration end-to-end
Teste le flux complet: Données → Modèle → API → Résultats
"""
import requests
import json
import time
import subprocess
import sys
from pathlib import Path

class IntegrationTester:
    """Classe pour les tests d'intégration"""
    
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.test_comments = [
            "This is absolutely amazing! Best content ever!",
            "Terrible video, complete waste of time.",
            "Thanks for the information, very helpful.",
            "First! 🎉",
            "Can you make a tutorial about Python?",
            "Worst explanation I've ever seen.",
            "Great job! Keep up the excellent work!",
            "The audio quality is really bad.",
            "Subscribed! You deserve more views!",
            "This doesn't work at all."
        ]
    
    def test_api_availability(self):
        """Test 1: Disponibilité de l'API"""
        print("\n" + "="*70)
        print("TEST 1: DISPONIBILITÉ DE L'API")
        print("="*70)
        
        try:
            print(f"🔍 Vérification de {self.api_url}...")
            response = requests.get(f"{self.api_url}/", timeout=5)
            
            if response.status_code == 200:
                print("✅ API accessible")
                data = response.json()
                print(f"   Version: {data.get('version', 'N/A')}")
                return True
            else:
                print(f"❌ Status code: {response.status_code}")
                return False
        
        except requests.exceptions.ConnectionError:
            print("❌ Impossible de se connecter à l'API")
            print("   Lancez l'API avec: python run_api.py")
            return False
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return False
    
    def test_health_endpoint(self):
        """Test 2: Endpoint /health"""
        print("\n" + "="*70)
        print("TEST 2: HEALTH CHECK")
        print("="*70)
        
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"Status: {data.get('status')}")
                print(f"Modèle chargé: {data.get('model_loaded')}")
                print(f"Vectoriseur chargé: {data.get('vectorizer_loaded')}")
                
                if data.get('status') == 'healthy':
                    print("✅ API en bonne santé")
                    return True
                else:
                    print("❌ API pas healthy")
                    return False
            else:
                print(f"❌ Status code: {response.status_code}")
                return False
        
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return False
    
    def test_single_prediction(self):
        """Test 3: Prédiction simple"""
        print("\n" + "="*70)
        print("TEST 3: PRÉDICTION SIMPLE")
        print("="*70)
        
        test_comment = "This is a great tutorial!"
        
        try:
            print(f"📤 Envoi du commentaire: \"{test_comment}\"")
            
            response = requests.post(
                f"{self.api_url}/predict",
                params={"comment": test_comment},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"\n📥 Réponse:")
                print(f"   Sentiment: {data['sentiment']}")
                print(f"   Label: {data['label']}")
                print(f"   Confiance: {data['confidence']:.2%}")
                
                # Vérifications
                assert 'sentiment' in data
                assert 'label' in data
                assert 'confidence' in data
                assert data['label'] in [-1, 0, 1]
                assert 0 <= data['confidence'] <= 1
                
                print("\n✅ Prédiction simple réussie")
                return True
            else:
                print(f"❌ Status code: {response.status_code}")
                return False
        
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return False
    
    def test_batch_prediction(self):
        """Test 4: Prédiction batch"""
        print("\n" + "="*70)
        print("TEST 4: PRÉDICTION BATCH")
        print("="*70)
        
        try:
            print(f"📤 Envoi de {len(self.test_comments)} commentaires...")
            
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/predict_batch",
                json={"comments": self.test_comments},
                timeout=30
            )
            request_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"\n📥 Réponse reçue en {request_time:.2f}ms")
                print(f"   Temps de traitement API: {data['processing_time_ms']:.2f}ms")
                print(f"   Commentaires traités: {data['total_comments']}")
                
                stats = data['statistics']
                print(f"\n📊 Statistiques:")
                print(f"   Positifs: {stats['positive']} ({stats['positive_percent']}%)")
                print(f"   Neutres: {stats['neutral']} ({stats['neutral_percent']}%)")
                print(f"   Négatifs: {stats['negative']} ({stats['negative_percent']}%)")
                
                # Vérifications
                assert len(data['predictions']) == len(self.test_comments)
                assert data['total_comments'] == len(self.test_comments)
                assert stats['total'] == len(self.test_comments)
                
                # Vérifier chaque prédiction
                for pred in data['predictions']:
                    assert 'sentiment' in pred
                    assert 'label' in pred
                    assert 'confidence' in pred
                    assert pred['label'] in [-1, 0, 1]
                
                print("\n✅ Prédiction batch réussie")
                return True, data
            else:
                print(f"❌ Status code: {response.status_code}")
                return False, None
        
        except Exception as e:
            print(f"❌ Erreur: {e}")
            import traceback
            traceback.print_exc()
            return False, None
    
    def test_error_handling(self):
        """Test 5: Gestion des erreurs"""
        print("\n" + "="*70)
        print("TEST 5: GESTION DES ERREURS")
        print("="*70)
        
        test_cases = [
            ({"comments": []}, 422, "Liste vide"),
            ({"comments": [""]}, 422, "Commentaire vide"),
            ({"comments": ["test"] * 101}, 422, "Trop de commentaires (>100)"),
        ]
        
        passed = 0
        for payload, expected_status, description in test_cases:
            try:
                print(f"\n🧪 Test: {description}")
                response = requests.post(
                    f"{self.api_url}/predict_batch",
                    json=payload,
                    timeout=10
                )
                
                if response.status_code == expected_status:
                    print(f"   ✅ Status {response.status_code} (attendu)")
                    passed += 1
                else:
                    print(f"   ⚠️  Status {response.status_code} (attendu: {expected_status})")
            
            except Exception as e:
                print(f"   ❌ Erreur: {e}")
        
        print(f"\n📊 {passed}/{len(test_cases)} tests d'erreur réussis")
        return passed == len(test_cases)
    
    def test_performance(self):
        """Test 6: Performance"""
        print("\n" + "="*70)
        print("TEST 6: PERFORMANCE")
        print("="*70)
        
        batch_sizes = [10, 25, 50]
        
        for size in batch_sizes:
            comments = [f"Test comment number {i}" for i in range(size)]
            
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.api_url}/predict_batch",
                    json={"comments": comments},
                    timeout=30
                )
                total_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    processing_time = data['processing_time_ms']
                    
                    print(f"\n📊 Batch de {size} commentaires:")
                    print(f"   Temps total (avec réseau): {total_time:.2f}ms")
                    print(f"   Temps traitement API: {processing_time:.2f}ms")
                    print(f"   Temps par commentaire: {processing_time/size:.2f}ms")
                    
                    if size == 50 and processing_time < 100:
                        print(f"   ✅ Performance excellente (< 100ms)")
                    elif size == 50 and processing_time < 200:
                        print(f"   ✅ Performance acceptable (< 200ms)")
            
            except Exception as e:
                print(f"\n❌ Erreur avec batch {size}: {e}")
                return False
        
        return True
    
    def test_consistency(self):
        """Test 7: Cohérence des résultats"""
        print("\n" + "="*70)
        print("TEST 7: COHÉRENCE DES RÉSULTATS")
        print("="*70)
        
        test_comment = ["This is an excellent video!"]
        
        print("🔄 Envoi du même commentaire 5 fois...")
        
        predictions = []
        for i in range(5):
            try:
                response = requests.post(
                    f"{self.api_url}/predict_batch",
                    json={"comments": test_comment},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    sentiment = data['predictions'][0]['sentiment']
                    predictions.append(sentiment)
                    print(f"   Run {i+1}: {sentiment}")
            
            except Exception as e:
                print(f"   ❌ Run {i+1}: Erreur - {e}")
                return False
        
        # Vérifier la cohérence
        unique_predictions = set(predictions)
        if len(unique_predictions) == 1:
            print(f"\n✅ Résultats cohérents (toutes les prédictions identiques)")
            return True
        else:
            print(f"\n⚠️  Résultats incohérents: {len(unique_predictions)} prédictions différentes")
            return False

def run_integration_tests(api_url="http://localhost:8000"):
    """Exécute tous les tests d'intégration"""
    print("\n" + "🔗 "*35)
    print("TESTS D'INTÉGRATION END-TO-END")
    print("🔗 "*35)
    print(f"\nAPI URL: {api_url}\n")
    
    tester = IntegrationTester(api_url)
    
    results = []
    
    # Test 1: Disponibilité
    if not tester.test_api_availability():
        print("\n❌ API non accessible. Arrêt des tests.")
        return False
    results.append(("Disponibilité API", True))
    
    # Test 2: Health
    results.append(("Health check", tester.test_health_endpoint()))
    
    # Test 3: Prédiction simple
    results.append(("Prédiction simple", tester.test_single_prediction()))
    
    # Test 4: Prédiction batch
    batch_result, batch_data = tester.test_batch_prediction()
    results.append(("Prédiction batch", batch_result))
    
    # Test 5: Gestion des erreurs
    results.append(("Gestion des erreurs", tester.test_error_handling()))
    
    # Test 6: Performance
    results.append(("Performance", tester.test_performance()))
    
    # Test 7: Cohérence
    results.append(("Cohérence", tester.test_consistency()))
    
    # Résumé
    print("\n" + "="*70)
    print("📊 RÉSUMÉ DES TESTS D'INTÉGRATION")
    print("="*70)
    
    for name, passed in results:
        status = "✅ RÉUSSI" if passed else "❌ ÉCHOUÉ"
        print(f"  {name:25s}: {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\n📈 Total: {total_passed}/{total_tests} tests réussis ({total_passed/total_tests*100:.0f}%)")
    
    if total_passed == total_tests:
        print("\n🎉 TOUS LES TESTS D'INTÉGRATION SONT RÉUSSIS!")
        return True
    else:
        print("\n⚠️  Certains tests ont échoué")
        return False

if __name__ == "__main__":
    import sys
    
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    success = run_integration_tests(api_url)
    
    sys.exit(0 if success else 1)