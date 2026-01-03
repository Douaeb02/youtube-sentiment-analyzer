# 📡 API Documentation - YouTube Sentiment Analyzer

## 🌐 URL de base
```
http://localhost:8000
```

## 📚 Documentation interactive
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🔌 Endpoints

### 1. **GET /** - Info de l'API
Retourne les informations générales sur l'API.

**Exemple de requête:**
```bash
curl http://localhost:8000/
```

**Réponse:**
```json
{
  "message": "YouTube Sentiment Analyzer API",
  "version": "1.0.0",
  "status": "running",
  "endpoints": {
    "health": "/health",
    "predict_batch": "/predict_batch",
    "docs": "/docs"
  }
}
```

---

### 2. **GET /health** - Health Check
Vérifie l'état de l'API et du modèle.

**Exemple de requête:**
```bash
curl http://localhost:8000/health
```

**Réponse:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "vectorizer_loaded": true,
  "version": "1.0.0",
  "message": "API et modèle opérationnels"
}
```

---

### 3. **POST /predict_batch** - Analyse de sentiment (batch)
Analyse le sentiment d'un batch de commentaires.

**Paramètres:**
| Champ | Type | Description | Requis |
|-------|------|-------------|--------|
| comments | array[string] | Liste de commentaires (1-100) | Oui |

**Exemple de requête:**
```bash
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "comments": [
      "This video is amazing!",
      "Didn't work for me.",
      "Thanks for sharing."
    ]
  }'
```

**Réponse:**
```json
{
  "predictions": [
    {
      "text": "This video is amazing!",
      "label": 1,
      "sentiment": "Positif",
      "confidence": 0.92
    },
    {
      "text": "Didn't work for me.",
      "label": -1,
      "sentiment": "Négatif",
      "confidence": 0.85
    },
    {
      "text": "Thanks for sharing.",
      "label": 1,
      "sentiment": "Positif",
      "confidence": 0.78
    }
  ],
  "statistics": {
    "total": 3,
    "positive": 2,
    "neutral": 0,
    "negative": 1,
    "positive_percent": 66.67,
    "neutral_percent": 0.0,
    "negative_percent": 33.33,
    "avg_confidence": 0.85
  },
  "total_comments": 3,
  "processing_time_ms": 45.23
}
```

---

### 4. **POST /predict** - Analyse d'un seul commentaire
Analyse le sentiment d'un commentaire unique.

**Paramètres:**
| Champ | Type | Description | Requis |
|-------|------|-------------|--------|
| comment | string | Commentaire à analyser | Oui |

**Exemple de requête:**
```bash
curl -X POST "http://localhost:8000/predict?comment=This%20is%20great!"
```

**Réponse:**
```json
{
  "text": "This is great!",
  "label": 1,
  "sentiment": "Positif",
  "confidence": 0.89
}
```

---

## 📊 Modèles de données

### SentimentPrediction
```typescript
{
  text: string,          // Texte du commentaire
  label: int,            // -1 (Négatif), 0 (Neutre), 1 (Positif)
  sentiment: string,     // "Négatif", "Neutre", "Positif"
  confidence: float      // 0.0 - 1.0
}
```

### Statistics
```typescript
{
  total: int,
  positive: int,
  neutral: int,
  negative: int,
  positive_percent: float,
  neutral_percent: float,
  negative_percent: float,
  avg_confidence: float
}
```

---

## ⚠️ Codes d'erreur

| Code | Description |
|------|-------------|
| 200 | Succès |
| 400 | Requête invalide (commentaire vide, format incorrect) |
| 422 | Erreur de validation (trop de commentaires, etc.) |
| 500 | Erreur serveur interne |
| 503 | Service indisponible (modèle non chargé) |

---

## 🔧 Configuration CORS

L'API accepte les requêtes depuis **toutes les origines** (`allow_origins: ["*"]`).

Pour la production, il est recommandé de restreindre les origines:
```python
allow_origins=[
    "chrome-extension://your-extension-id",
    "https://your-domain.com"
]
```

---

## 🚀 Utilisation depuis JavaScript

### Exemple avec fetch:
```javascript
async function analyzeSentiment(comments) {
  const response = await fetch('http://localhost:8000/predict_batch', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ comments: comments })
  });
  
  const data = await response.json();
  return data;
}

// Utilisation
const comments = [
  "Great video!",
  "Didn't like it",
  "Thanks for sharing"
];

analyzeSentiment(comments)
  .then(result => {
    console.log('Statistics:', result.statistics);
    console.log('Predictions:', result.predictions);
  });
```

---

## 📝 Limitations

- **Batch size**: Maximum 100 commentaires par requête
- **Texte**: Maximum 5000 caractères par commentaire
- **Rate limiting**: Aucune limite actuellement (à implémenter en production)

---

## 🧪 Tests

Testez l'API avec le script fourni:
```bash
python tests/test_api.py
```

---

## 🔍 Monitoring

### Métriques à surveiller:
- Temps de réponse moyen
- Taux d'erreur
- Distribution des sentiments prédits
- Confiance moyenne des prédictions

### Logs:
Les logs sont configurés avec le format:
```
%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

---

## 💡 Bonnes pratiques

1. **Toujours vérifier /health** avant d'utiliser l'API
2. **Gérer les erreurs** côté client
3. **Limiter la taille des batchs** pour de meilleures performances
4. **Nettoyer les commentaires** avant de les envoyer (supprimer les espaces excessifs)
5. **Utiliser des timeouts** côté client

---

## 🆘 Dépannage

### L'API ne démarre pas
- Vérifiez que le modèle est entraîné: `ls models/`
- Vérifiez les dépendances: `pip install -r requirements.txt`

### Erreur 503 (Service Unavailable)
- Le modèle n'est pas chargé
- Vérifiez les logs au démarrage

### Performance lente
- Réduisez la taille des batchs
- Vérifiez les ressources système (CPU, RAM)