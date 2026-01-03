# Utiliser une image Python légère
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier requirements en premier (pour le cache Docker)
COPY requirements_production.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements_production.txt

# Copier les fichiers de l'application
COPY app_api.py .
COPY models/ ./models/

# Exposer le port 7860 (port par défaut de Hugging Face Spaces)
EXPOSE 7860

# Variable d'environnement pour le port
ENV PORT=7860

# Commande pour démarrer l'application
CMD ["uvicorn", "app_api:app", "--host", "0.0.0.0", "--port", "7860"]