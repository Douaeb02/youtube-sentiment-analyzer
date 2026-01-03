"""
Script de test et validation du modèle entraîné
"""
import joblib
import numpy as np
import time
from pathlib import Path

class SentimentPredictor:
    """Classe pour charger et utiliser le modèle entraîné"""
    
    def __init__(self, model_path='models/sentiment_model.joblib', 
                 vectorizer_path='models/tfidf_vectorizer.joblib'):
        """
        Charge le modèle et le vectoriseur
        
        Args:
            model_path: Chemin vers le modèle
            vectorizer_path: Chemin vers le vectoriseur
        """
        print("📂 Chargement du modèle...")
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        print("✅ Modèle chargé avec succès!")
        
        self.sentiment_map = {
            -1: "Négatif",
            0: "Neutre",
            1: "Positif"
        }
    
    def predict(self, text):
        """
        Prédit le sentiment d'un texte
        
        Args:
            text: Texte à analyser
            
        Returns:
            dict avec label, sentiment et confidence
        """
        # Vectoriser
        text_vec = self.vectorizer.transform([text])
        
        # Prédire
        label = self.model.predict(text_vec)[0]
        
        # Obtenir les probabilités
        if hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(text_vec)[0]
            confidence = float(probas.max())
        else:
            confidence = 1.0
        
        return {
            'label': int(label),
            'sentiment': self.sentiment_map[label],
            'confidence': confidence
        }
    
    def predict_batch(self, texts):
        """
        Prédit le sentiment de plusieurs textes
        
        Args:
            texts: Liste de textes
            
        Returns:
            Liste de dict avec predictions
        """
        # Vectoriser
        texts_vec = self.vectorizer.transform(texts)
        
        # Prédire
        labels = self.model.predict(texts_vec)
        
        # Obtenir les probabilités
        if hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(texts_vec)
            confidences = probas.max(axis=1)
        else:
            confidences = np.ones(len(texts))
        
        results = []
        for i, (label, confidence) in enumerate(zip(labels, confidences)):
            results.append({
                'text': texts[i],
                'label': int(label),
                'sentiment': self.sentiment_map[label],
                'confidence': float(confidence)
            })
        
        return results

def test_edge_cases():
    """Teste le modèle sur des cas limites"""
    print("\n" + "="*70)
    print("🧪 TESTS SUR CAS LIMITES")
    print("="*70)
    
    predictor = SentimentPredictor()
    
    test_cases = [
        # Textes très courts
        ("Great!", "Court positif"),
        ("Bad", "Court négatif"),
        ("Ok", "Court neutre"),
        
        # Textes avec emojis
        ("I love this! 😊❤️", "Avec emojis positifs"),
        ("This is terrible 😡😠", "Avec emojis négatifs"),
        
        # Textes ambigus
        ("It's not bad", "Négation (not bad)"),
        ("Could be better", "Mitigé"),
        
        # Textes longs
        ("This is an absolutely amazing product that exceeded all my expectations. I highly recommend it to everyone!", "Long positif"),
        ("I'm extremely disappointed with this purchase. It doesn't work as advertised and customer service was unhelpful.", "Long négatif"),
        
        # Mélange de langues (même si le modèle est anglais)
        ("This is great bon excellent", "Mélange de langues"),
        
        # Texte neutre informatif
        ("The meeting is scheduled for tomorrow at 3pm", "Informatif neutre"),
        
        # Sarcasme (difficile)
        ("Oh great, another bug to fix", "Sarcasme"),
    ]
    
    print("\n📝 Résultats des tests:")
    print("-"*70)
    
    for text, description in test_cases:
        result = predictor.predict(text)
        emoji = {"Négatif": "😞", "Neutre": "😐", "Positif": "😊"}[result['sentiment']]
        
        print(f"\n{description}")
        print(f"  Texte: \"{text}\"")
        print(f"  {emoji} Sentiment: {result['sentiment']} (confiance: {result['confidence']:.2%})")

def test_inference_speed():
    """Teste la vitesse d'inférence"""
    print("\n" + "="*70)
    print("⏱️  TEST DE PERFORMANCE D'INFÉRENCE")
    print("="*70)
    
    predictor = SentimentPredictor()
    
    # Textes de test
    test_texts = [
        "This is a great product!",
        "I'm not satisfied with the quality.",
        "The delivery was on time.",
    ] * 20  # 60 commentaires
    
    print(f"\n📊 Test avec {len(test_texts)} commentaires...")
    
    # Test batch
    start_time = time.time()
    results = predictor.predict_batch(test_texts)
    batch_time = time.time() - start_time
    
    print(f"\n⚡ Résultats:")
    print(f"  Temps total: {batch_time*1000:.2f}ms")
    print(f"  Temps par commentaire: {(batch_time/len(test_texts))*1000:.2f}ms")
    print(f"  Commentaires par seconde: {len(test_texts)/batch_time:.1f}")
    
    # Test batch de 50
    test_50 = test_texts[:50]
    start_time = time.time()
    results_50 = predictor.predict_batch(test_50)
    time_50 = time.time() - start_time
    
    print(f"\n⚡ Batch de 50 commentaires:")
    print(f"  Temps: {time_50*1000:.2f}ms")
    
    if time_50 * 1000 < 100:
        print(f"  ✅ EXCELLENT (< 100ms)")
    elif time_50 * 1000 < 200:
        print(f"  ✅ BON (< 200ms)")
    else:
        print(f"  ⚠️  À améliorer (> 200ms)")

def interactive_test():
    """Mode de test interactif"""
    print("\n" + "="*70)
    print("🎮 MODE INTERACTIF")
    print("="*70)
    
    predictor = SentimentPredictor()
    
    print("\nEntrez des commentaires pour tester le modèle.")
    print("Tapez 'quit' pour quitter.\n")
    
    while True:
        text = input("💬 Votre commentaire: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("👋 Au revoir!")
            break
        
        if not text:
            continue
        
        result = predictor.predict(text)
        emoji = {"Négatif": "😞", "Neutre": "😐", "Positif": "😊"}[result['sentiment']]
        
        print(f"   {emoji} Sentiment: {result['sentiment']}")
        print(f"   📊 Confiance: {result['confidence']:.2%}\n")

def test_with_real_comments():
    """Teste avec des commentaires réalistes"""
    print("\n" + "="*70)
    print("🎬 TEST AVEC COMMENTAIRES YOUTUBE RÉALISTES")
    print("="*70)
    
    predictor = SentimentPredictor()
    
    # Commentaires typiques YouTube
    youtube_comments = [
        "First! 🎉",
        "This video is amazing! Best tutorial ever!",
        "Didn't work for me. Waste of time.",
        "Thanks for sharing this. Very helpful.",
        "Who's watching in 2025?",
        "Like if you agree!",
        "This is trash. Dislike.",
        "Can you make a video about Python?",
        "Your channel is growing so fast! Congrats! 🎊",
        "Worst explanation ever. Confusing.",
        "Subbed! Keep up the good work!",
        "The audio quality is terrible.",
    ]
    
    results = predictor.predict_batch(youtube_comments)
    
    print("\n📝 Analyse des commentaires:")
    print("-"*70)
    
    for result in results:
        emoji = {"Négatif": "😞", "Neutre": "😐", "Positif": "😊"}[result['sentiment']]
        
        print(f"\n{emoji} \"{result['text']}\"")
        print(f"   → {result['sentiment']} ({result['confidence']:.1%})")
    
    # Statistiques
    sentiments = [r['sentiment'] for r in results]
    print("\n📊 Statistiques:")
    print(f"  Positifs: {sentiments.count('Positif')}/{len(sentiments)}")
    print(f"  Neutres:  {sentiments.count('Neutre')}/{len(sentiments)}")
    print(f"  Négatifs: {sentiments.count('Négatif')}/{len(sentiments)}")

def main():
    """Fonction principale"""
    print("\n" + "🧪 "*25)
    print("TESTS ET VALIDATION DU MODÈLE")
    print("🧪 "*25 + "\n")
    
    # Vérifier que le modèle existe
    if not Path('models/sentiment_model.joblib').exists():
        print("❌ Erreur: Le modèle n'existe pas!")
        print("   Exécutez d'abord: python src/models/train_model.py")
        return
    
    # Exécuter tous les tests
    test_edge_cases()
    test_inference_speed()
    test_with_real_comments()
    
    # Mode interactif
    print("\n" + "="*70)
    response = input("\n🎮 Voulez-vous tester en mode interactif? (o/n): ")
    if response.lower() in ['o', 'oui', 'y', 'yes']:
        interactive_test()
    
    print("\n✅ Tests terminés!")

if __name__ == "__main__":
    main()