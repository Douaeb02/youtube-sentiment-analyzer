"""
API FastAPI pour l'analyse de sentiment YouTube
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging
from typing import List

from src.api.models import (
    CommentBatch,
    SentimentPrediction,
    BatchPredictionResponse,
    HealthResponse,
    ErrorResponse
)
from src.api.prediction_service import get_prediction_service

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Créer l'application FastAPI
app = FastAPI(
    title="YouTube Sentiment Analyzer API",
    description="API REST pour l'analyse de sentiment des commentaires YouTube",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS pour autoriser les requêtes depuis l'extension Chrome
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifiez les origines autorisées
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le service de prédiction au démarrage
@app.on_event("startup")
async def startup_event():
    """Événement exécuté au démarrage de l'API"""
    logger.info("🚀 Démarrage de l'API YouTube Sentiment Analyzer")
    try:
        service = get_prediction_service()
        if service.is_loaded():
            logger.info("✅ Service de prédiction initialisé avec succès")
        else:
            logger.error("❌ Échec de l'initialisation du service")
    except Exception as e:
        logger.error(f"❌ Erreur au démarrage: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Événement exécuté à l'arrêt de l'API"""
    logger.info("👋 Arrêt de l'API YouTube Sentiment Analyzer")

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Endpoint racine avec informations sur l'API"""
    return {
        "message": "YouTube Sentiment Analyzer API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict_batch": "/predict_batch",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Endpoint de santé pour vérifier l'état de l'API et du modèle
    
    Returns:
        HealthResponse avec l'état du système
    """
    try:
        service = get_prediction_service()
        
        model_loaded = service.model is not None
        vectorizer_loaded = service.vectorizer is not None
        
        if model_loaded and vectorizer_loaded:
            return HealthResponse(
                status="healthy",
                model_loaded=True,
                vectorizer_loaded=True,
                message="API et modèle opérationnels"
            )
        else:
            return HealthResponse(
                status="unhealthy",
                model_loaded=model_loaded,
                vectorizer_loaded=vectorizer_loaded,
                message="Modèle ou vectoriseur non chargé"
            )
    
    except Exception as e:
        logger.error(f"Erreur health check: {e}")
        return HealthResponse(
            status="error",
            model_loaded=False,
            vectorizer_loaded=False,
            message=f"Erreur: {str(e)}"
        )

@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch: CommentBatch):
    """
    Analyse le sentiment d'un batch de commentaires
    
    Args:
        batch: Batch de commentaires à analyser
        
    Returns:
        BatchPredictionResponse avec prédictions et statistiques
    """
    start_time = time.time()
    
    try:
        # Récupérer le service
        service = get_prediction_service()
        
        if not service.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service de prédiction non disponible"
            )
        
        # Valider le nombre de commentaires
        if len(batch.comments) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 100 commentaires par batch"
            )
        
        logger.info(f"📥 Réception de {len(batch.comments)} commentaires")
        
        # Faire les prédictions
        predictions = service.predict_batch(batch.comments)
        
        # Calculer les statistiques
        statistics = service.calculate_statistics(predictions)
        
        # Temps de traitement
        processing_time = (time.time() - start_time) * 1000  # en ms
        
        logger.info(f"✅ Traitement terminé en {processing_time:.2f}ms")
        
        # Convertir en modèles Pydantic
        prediction_models = [
            SentimentPrediction(**pred) for pred in predictions
        ]
        
        return BatchPredictionResponse(
            predictions=prediction_models,
            statistics=statistics,
            total_comments=len(predictions),
            processing_time_ms=round(processing_time, 2)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur lors de la prédiction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne: {str(e)}"
        )

@app.post("/predict", response_model=SentimentPrediction, tags=["Prediction"])
async def predict_single(comment: str):
    """
    Analyse le sentiment d'un seul commentaire
    
    Args:
        comment: Commentaire à analyser
        
    Returns:
        SentimentPrediction avec le résultat
    """
    try:
        service = get_prediction_service()
        
        if not service.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service de prédiction non disponible"
            )
        
        if not comment or not comment.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Le commentaire ne peut pas être vide"
            )
        
        prediction = service.predict_single(comment.strip())
        
        return SentimentPrediction(**prediction)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur lors de la prédiction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne: {str(e)}"
        )

# Gestionnaire d'erreurs global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Gestionnaire global des exceptions"""
    logger.error(f"Exception non gérée: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )