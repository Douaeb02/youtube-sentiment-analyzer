import os
import requests
import pandas as pd
from pathlib import Path

def download_reddit_dataset():
    """
    Télécharge le dataset Reddit depuis GitHub et affiche les statistiques
    """
    # URL du dataset
    url = "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
    
    # Créer le dossier data/raw s'il n'existe pas
    raw_data_dir = Path("data/raw")
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = raw_data_dir / "reddit.csv"
    
    print("📥 Téléchargement du dataset Reddit Sentiment Analysis...")
    print(f"URL: {url}")
    
    try:
        # Télécharger le fichier
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Sauvegarder le fichier
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f" Dataset téléchargé avec succès: {output_path}")
        
        # Charger et afficher les statistiques
        df = pd.read_csv(output_path)
        
        print("\n" + "="*60)
        print(" STATISTIQUES DU DATASET")
        print("="*60)
        
        print(f"\n Nombre total de commentaires: {len(df)}")
        print(f" Colonnes disponibles: {list(df.columns)}")
        
        # Distribution des labels
        print("\n Distribution des sentiments:")
        sentiment_counts = df['category'].value_counts().sort_index()
        for label, count in sentiment_counts.items():
            sentiment_name = {-1: "Négatif", 0: "Neutre", 1: "Positif"}.get(label, "Inconnu")
            percentage = (count / len(df)) * 100
            print(f"  {sentiment_name:10s} ({label:2d}): {count:5d} commentaires ({percentage:.1f}%)")
        
        # Aperçu des données
        print("\n Aperçu des premières lignes:")
        print(df.head())
        
        # Informations sur les données manquantes
        print("\n Données manquantes:")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            print("   Aucune donnée manquante")
        else:
            print(missing)
        
        # Statistiques sur la longueur des textes
        df['text_length'] = df['clean_comment'].astype(str).str.len()
        print(f"\n Longueur des commentaires:")
        print(f"  Moyenne: {df['text_length'].mean():.1f} caractères")
        print(f"  Minimum: {df['text_length'].min()} caractères")
        print(f"  Maximum: {df['text_length'].max()} caractères")
        print(f"  Médiane: {df['text_length'].median():.1f} caractères")
        
        print("\n" + "="*60)
        print(" Dataset prêt à être utilisé!")
        print("="*60)
        
        return output_path
        
    except requests.exceptions.RequestException as e:
        print(f" Erreur lors du téléchargement: {e}")
        return None
    except Exception as e:
        print(f" Erreur inattendue: {e}")
        return None

if __name__ == "__main__":
    download_reddit_dataset()