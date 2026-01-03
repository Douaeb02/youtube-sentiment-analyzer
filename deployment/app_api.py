"""
API FastAPI optimisée pour le déploiement sur Hugging Face Spaces
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import joblib
import os
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration de l'application"""
    MODEL_PATH = "models/sentiment_model.joblib"
    VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"
    MAX_COMMENTS = 1000  # Limite pour éviter les surcharges
    VERSION = "1.0.0"

# ============================================================================
# Modèles Pydantic
# ============================================================================

class CommentRequest(BaseModel):
    """Requête pour analyser un seul commentaire"""
    text: str = Field(..., min_length=1, max_length=5000)
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Le texte ne peut pas être vide')
        return v.strip()

class BatchRequest(BaseModel):
    """Requête pour analyser plusieurs commentaires"""
    comments: List[str] = Field(..., min_items=1)
    
    @validator('comments')
    def validate_comments(cls, v):
        if len(v) > Config.MAX_COMMENTS:
            raise ValueError(f'Maximum {Config.MAX_COMMENTS} commentaires autorisés')
        # Nettoyer les commentaires vides
        cleaned = [c.strip() for c in v if c.strip()]
        if not cleaned:
            raise ValueError('Aucun commentaire valide trouvé')
        return cleaned

class PredictionResponse(BaseModel):
    """Réponse pour une prédiction"""
    sentiment: str
    confidence: float
    probabilities: dict

class BatchResponse(BaseModel):
    """Réponse pour un batch de prédictions"""
    predictions: List[PredictionResponse]
    statistics: dict

class HealthResponse(BaseModel):
    """Réponse du health check"""
    status: str
    model_loaded: bool
    vectorizer_loaded: bool
    version: str
    message: str

# ============================================================================
# Service de Prédiction
# ============================================================================

class PredictionService:
    """Service pour charger le modèle et faire des prédictions"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_mapping = {0: 'Négatif', 1: 'Neutre', 2: 'Positif'}
        
    def load_models(self):
        """Charge le modèle et le vectoriseur"""
        try:
            # Vérifier que les fichiers existent
            if not os.path.exists(Config.MODEL_PATH):
                raise FileNotFoundError(f"Modèle non trouvé: {Config.MODEL_PATH}")
            if not os.path.exists(Config.VECTORIZER_PATH):
                raise FileNotFoundError(f"Vectoriseur non trouvé: {Config.VECTORIZER_PATH}")
            
            logger.info(f"Chargement du modèle depuis {Config.MODEL_PATH}")
            self.model = joblib.load(Config.MODEL_PATH)
            
            logger.info(f"Chargement du vectoriseur depuis {Config.VECTORIZER_PATH}")
            self.vectorizer = joblib.load(Config.VECTORIZER_PATH)
            
            logger.info("✅ Modèle et vectoriseur chargés avec succès")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement: {str(e)}")
            raise
    
    def predict_single(self, text: str) -> dict:
        """
        Prédit le sentiment d'un seul commentaire
        
        Args:
            text: Texte du commentaire
            
        Returns:
            dict avec sentiment, confidence et probabilities
        """
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Modèle non chargé")
        
        # Vectoriser le texte
        X = self.vectorizer.transform([text])
        
        # Prédiction
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        sentiment = self.label_mapping[prediction]
        confidence = float(probabilities[prediction])
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {
                'Positif': float(probabilities[2]),
                'Neutre': float(probabilities[1]),
                'Négatif': float(probabilities[0])
            }
        }
    
    def predict_batch(self, comments: List[str]) -> dict:
        """
        Prédit le sentiment de plusieurs commentaires
        
        Args:
            comments: Liste de commentaires
            
        Returns:
            dict avec predictions et statistics
        """
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Modèle non chargé")
        
        # Vectoriser tous les commentaires
        X = self.vectorizer.transform(comments)
        
        # Prédictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Créer les résultats
        results = []
        sentiment_counts = {'Positif': 0, 'Neutre': 0, 'Négatif': 0}
        total_confidence = 0
        
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            sentiment = self.label_mapping[pred]
            confidence = float(probs[pred])
            
            results.append({
                'sentiment': sentiment,
                'confidence': confidence,
                'probabilities': {
                    'Positif': float(probs[2]),
                    'Neutre': float(probs[1]),
                    'Négatif': float(probs[0])
                }
            })
            
            sentiment_counts[sentiment] += 1
            total_confidence += confidence
        
        # Calculer les statistiques
        total = len(comments)
        statistics = {
            'total': total,
            'positive': sentiment_counts['Positif'],
            'neutral': sentiment_counts['Neutre'],
            'negative': sentiment_counts['Négatif'],
            'positive_percent': round((sentiment_counts['Positif'] / total) * 100, 1),
            'neutral_percent': round((sentiment_counts['Neutre'] / total) * 100, 1),
            'negative_percent': round((sentiment_counts['Négatif'] / total) * 100, 1),
            'avg_confidence': round(total_confidence / total, 3)
        }
        
        return {
            'predictions': results,
            'statistics': statistics
        }

# ============================================================================
# Application FastAPI
# ============================================================================

app = FastAPI(
    title="YouTube Sentiment Analyzer API",
    description="API d'analyse de sentiment pour les commentaires YouTube",
    version=Config.VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS pour permettre les requêtes depuis l'extension Chrome
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifiez les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialiser le service de prédiction
prediction_service = PredictionService()

# ============================================================================
# Event Handlers
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Chargement des modèles au démarrage"""
    logger.info("🚀 Démarrage de l'API YouTube Sentiment Analyzer")
    try:
        prediction_service.load_models()
        logger.info("✅ Service de prédiction initialisé avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Nettoyage lors de l'arrêt"""
    logger.info("👋 Arrêt de l'API")

# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Page d'accueil de l'API"""
    return {
        "name": "YouTube Sentiment Analyzer API",
        "version": Config.VERSION,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict_batch",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Vérifie l'état de santé de l'API
    """
    model_loaded = prediction_service.model is not None
    vectorizer_loaded = prediction_service.vectorizer is not None
    healthy = model_loaded and vectorizer_loaded
    
    return HealthResponse(
        status="healthy" if healthy else "unhealthy",
        model_loaded=model_loaded,
        vectorizer_loaded=vectorizer_loaded,
        version=Config.VERSION,
        message="API et modèle opérationnels" if healthy else "Modèle non chargé"
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_sentiment(request: CommentRequest):
    """
    Prédit le sentiment d'un seul commentaire
    
    - **text**: Texte du commentaire à analyser
    
    Returns:
    - **sentiment**: Positif, Neutre ou Négatif
    - **confidence**: Niveau de confiance (0-1)
    - **probabilities**: Probabilités pour chaque classe
    """
    try:
        result = prediction_service.predict_single(request.text)
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )

@app.post("/predict_batch", response_model=BatchResponse, tags=["Prediction"])
async def predict_batch(request: BatchRequest):
    """
    Prédit le sentiment de plusieurs commentaires
    
    - **comments**: Liste de commentaires à analyser
    
    Returns:
    - **predictions**: Liste des prédictions pour chaque commentaire
    - **statistics**: Statistiques globales (nombre, pourcentages, etc.)
    """
    try:
        result = prediction_service.predict_batch(request.comments)
        return BatchResponse(**result)
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction batch: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )

# ============================================================================
# Handler d'erreurs global
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Gestionnaire d'erreurs global"""
    logger.error(f"Erreur non gérée: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Erreur interne du serveur"}
    )

# ============================================================================
# Point d'entrée
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)