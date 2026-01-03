---
title: YouTube Sentiment Analyzer API
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 🎬 YouTube Sentiment Analyzer API

API d'analyse de sentiment pour les commentaires YouTube utilisant un modèle de Machine Learning.

## 🚀 Endpoints

- `GET /` - Page d'accueil
- `GET /health` - Vérification de l'état de l'API
- `POST /predict` - Analyse d'un seul commentaire
- `POST /predict_batch` - Analyse de plusieurs commentaires
- `GET /docs` - Documentation interactive (Swagger UI)

## 📊 Exemple d'utilisation

### Analyser un commentaire

```bash
curl -X POST "https://YOUR-SPACE-NAME.hf.space/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Cette vidéo est incroyable!"}'
```

### Analyser plusieurs commentaires

```bash
curl -X POST "https://YOUR-SPACE-NAME.hf.space/predict_batch" \
  -H "Content-Type: application/json" \
  -d '{"comments": ["Super vidéo!", "Pas terrible", "Intéressant"]}'
```

## 🛠️ Technologies

- **FastAPI** - Framework web moderne et rapide
- **scikit-learn** - Machine Learning
- **Docker** - Containerisation
- **Hugging Face Spaces** - Hébergement

## 📦 Modèle

Le modèle a été entraîné sur des commentaires YouTube en français pour classifier le sentiment en trois catégories :
- 😊 Positif
- 😐 Neutre
- 😞 Négatif

## 🔗 Extension Chrome

Cette API est utilisée par l'extension Chrome "YouTube Sentiment Analyzer" pour analyser automatiquement les commentaires YouTube.
