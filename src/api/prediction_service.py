"""
Service de prédiction de sentiment
"""
import joblib
import numpy as np
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class SentimentPredictionService:
    """Service pour gérer les prédictions de sentiment"""
    
    def __init__(self, model_path: str = "models/sentiment_model.joblib",
                 vectorizer_path: str = "models/tfidf_vectorizer.joblib"):
        """
        Initialise le service de prédiction
        
        Args:
            model_path: Chemin vers le modèle
            vectorizer_path: Chemin vers le vectoriseur
        """
        self.model = None
        self.vectorizer = None
        self.sentiment_map = {
            -1: "Négatif",
            0: "Neutre",
            1: "Positif"
        }
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        
        # Charger le modèle au démarrage
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle et le vectoriseur"""
        try:
            logger.info(f"Chargement du modèle depuis {self.model_path}")
            self.model = joblib.load(self.model_path)
            
            logger.info(f"Chargement du vectoriseur depuis {self.vectorizer_path}")
            self.vectorizer = joblib.load(self.vectorizer_path)
            
            logger.info("✅ Modèle et vectoriseur chargés avec succès")
            
        except FileNotFoundError as e:
            logger.error(f"❌ Fichier non trouvé: {e}")
            raise RuntimeError(f"Impossible de charger le modèle: {e}")
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement: {e}")
            raise RuntimeError(f"Erreur de chargement du modèle: {e}")
    
    def is_loaded(self) -> bool:
        """Vérifie si le modèle est chargé"""
        return self.model is not None and self.vectorizer is not None
    
    def predict_single(self, text: str) -> Dict:
        """
        Prédit le sentiment d'un seul texte
        
        Args:
            text: Texte à analyser
            
        Returns:
            Dict avec label, sentiment et confidence
        """
        if not self.is_loaded():
            raise RuntimeError("Modèle non chargé")
        
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
            'text': text,
            'label': int(label),
            'sentiment': self.sentiment_map[label],
            'confidence': confidence
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Prédit le sentiment de plusieurs textes
        
        Args:
            texts: Liste de textes
            
        Returns:
            Liste de dict avec predictions
        """
        if not self.is_loaded():
            raise RuntimeError("Modèle non chargé")
        
        if not texts:
            return []
        
        # Vectoriser tous les textes
        texts_vec = self.vectorizer.transform(texts)
        
        # Prédire tous les labels
        labels = self.model.predict(texts_vec)
        
        # Obtenir les probabilités
        if hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(texts_vec)
            confidences = probas.max(axis=1)
        else:
            confidences = np.ones(len(texts))
        
        # Construire les résultats
        results = []
        for i, (text, label, confidence) in enumerate(zip(texts, labels, confidences)):
            results.append({
                'text': text,
                'label': int(label),
                'sentiment': self.sentiment_map[label],
                'confidence': float(confidence)
            })
        
        return results
    
    def calculate_statistics(self, predictions: List[Dict]) -> Dict:
        """
        Calcule les statistiques sur un batch de prédictions
        
        Args:
            predictions: Liste de prédictions
            
        Returns:
            Dict avec statistiques
        """
        if not predictions:
            return {
                'total': 0,
                'positive': 0,
                'neutral': 0,
                'negative': 0,
                'positive_percent': 0,
                'neutral_percent': 0,
                'negative_percent': 0,
                'avg_confidence': 0
            }
        
        total = len(predictions)
        sentiments = [p['sentiment'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        
        positive_count = sentiments.count('Positif')
        neutral_count = sentiments.count('Neutre')
        negative_count = sentiments.count('Négatif')
        
        return {
            'total': total,
            'positive': positive_count,
            'neutral': neutral_count,
            'negative': negative_count,
            'positive_percent': round((positive_count / total) * 100, 2),
            'neutral_percent': round((neutral_count / total) * 100, 2),
            'negative_percent': round((negative_count / total) * 100, 2),
            'avg_confidence': round(np.mean(confidences), 4)
        }

# Instance globale du service (singleton)
_prediction_service = None

def get_prediction_service() -> SentimentPredictionService:
    """Retourne l'instance du service de prédiction (singleton)"""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = SentimentPredictionService()
    return _prediction_service