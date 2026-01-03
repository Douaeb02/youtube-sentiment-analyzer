"""
Modèles Pydantic pour la validation des données de l'API
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum

class SentimentLabel(int, Enum):
    """Enum pour les labels de sentiment"""
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1

class Comment(BaseModel):
    """Modèle pour un commentaire individuel"""
    text: str = Field(..., min_length=1, max_length=5000, description="Texte du commentaire")
    
    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Le commentaire ne peut pas être vide')
        return v.strip()

class CommentBatch(BaseModel):
    """Modèle pour un batch de commentaires"""
    comments: List[str] = Field(..., min_items=1, max_items=100, description="Liste de commentaires")
    
    @validator('comments')
    def validate_comments(cls, v):
        if not v:
            raise ValueError('La liste de commentaires ne peut pas être vide')
        
        # Nettoyer et valider chaque commentaire
        cleaned = []
        for comment in v:
            if isinstance(comment, str) and comment.strip():
                cleaned.append(comment.strip())
        
        if not cleaned:
            raise ValueError('Aucun commentaire valide dans la liste')
        
        return cleaned

class SentimentPrediction(BaseModel):
    """Modèle pour une prédiction de sentiment"""
    text: str = Field(..., description="Texte du commentaire")
    label: int = Field(..., description="Label numérique (-1, 0, 1)")
    sentiment: str = Field(..., description="Sentiment en texte (Négatif, Neutre, Positif)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Niveau de confiance (0-1)")

class BatchPredictionResponse(BaseModel):
    """Modèle pour la réponse d'un batch de prédictions"""
    predictions: List[SentimentPrediction]
    statistics: dict
    total_comments: int
    processing_time_ms: float

class HealthResponse(BaseModel):
    """Modèle pour la réponse du endpoint health"""
    status: str
    model_loaded: bool
    vectorizer_loaded: bool
    version: str = "1.0.0"
    message: Optional[str] = None

class ErrorResponse(BaseModel):
    """Modèle pour les réponses d'erreur"""
    error: str
    detail: Optional[str] = None
    status_code: int