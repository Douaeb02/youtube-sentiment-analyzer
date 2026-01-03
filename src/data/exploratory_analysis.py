import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import re

def analyze_dataset(data_path):
    """
    Effectue une analyse exploratoire complète du dataset
    
    Args:
        data_path: Chemin vers le fichier CSV nettoyé
    """
    print(" ANALYSE EXPLORATOIRE DES DONNÉES")
    print("="*70)
    
    # Charger les données
    df = pd.read_csv(data_path)
    print(f"\n Dataset chargé: {len(df)} commentaires")
    
    # ========== 1. DISTRIBUTION DES CLASSES ==========
    print("\n" + "="*70)
    print(" 1. DISTRIBUTION DES CLASSES")
    print("="*70)
    
    label_counts = df['label'].value_counts().sort_index()
    total = len(df)
    
    for label, count in label_counts.items():
        sentiment_name = {-1: "Négatif", 0: "Neutre", 1: "Positif"}.get(label, "Inconnu")
        percentage = (count / total) * 100
        bar = "" * int(percentage / 2)
        print(f"{sentiment_name:10s} ({label:2d}): {count:5d} ({percentage:5.1f}%) {bar}")
    
    # Vérifier le déséquilibre
    max_class = label_counts.max()
    min_class = label_counts.min()
    imbalance_ratio = max_class / min_class
    
    print(f"\n  Ratio de déséquilibre: {imbalance_ratio:.2f}:1")
    if imbalance_ratio > 2:
        print("  Dataset déséquilibré détecté! Considérez:")
        print("   - Utiliser class_weight='balanced' dans le modèle")
        print("   - Appliquer SMOTE ou sous-échantillonnage")
    else:
        print(" Dataset relativement équilibré")
    
    # ========== 2. LONGUEUR DES TEXTES ==========
    print("\n" + "="*70)
    print(" 2. ANALYSE DE LA LONGUEUR DES TEXTES")
    print("="*70)
    
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    
    print(f"\nLongueur en caractères:")
    print(f"  Moyenne:  {df['text_length'].mean():.1f}")
    print(f"  Médiane:  {df['text_length'].median():.1f}")
    print(f"  Min:      {df['text_length'].min()}")
    print(f"  Max:      {df['text_length'].max()}")
    print(f"  Std:      {df['text_length'].std():.1f}")
    
    print(f"\nNombre de mots:")
    print(f"  Moyenne:  {df['word_count'].mean():.1f}")
    print(f"  Médiane:  {df['word_count'].median():.1f}")
    print(f"  Min:      {df['word_count'].min()}")
    print(f"  Max:      {df['word_count'].max()}")
    
    # Longueur par sentiment
    print(f"\n Longueur moyenne par sentiment:")
    for label in sorted(df['label'].unique()):
        sentiment_name = {-1: "Négatif", 0: "Neutre", 1: "Positif"}.get(label, "Inconnu")
        avg_length = df[df['label'] == label]['text_length'].mean()
        avg_words = df[df['label'] == label]['word_count'].mean()
        print(f"  {sentiment_name:10s}: {avg_length:.1f} caractères, {avg_words:.1f} mots")
    
    # ========== 3. MOTS LES PLUS FRÉQUENTS ==========
    print("\n" + "="*70)
    print(" 3. MOTS LES PLUS FRÉQUENTS")
    print("="*70)
    
    # Mots communs à exclure (stop words simples)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                  'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'should', 'could', 'may', 'might', 'must', 'can', 'it', 'its',
                  'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they'}
    
    all_words = []
    for text in df['text']:
        words = re.findall(r'\b[a-z]+\b', text.lower())
        all_words.extend([w for w in words if w not in stop_words and len(w) > 2])
    
    word_freq = Counter(all_words)
    print(f"\nTop 20 mots les plus fréquents:")
    for word, count in word_freq.most_common(20):
        print(f"  {word:15s}: {count:5d}")
    
    # ========== 4. MOTS PAR SENTIMENT ==========
    print("\n" + "="*70)
    print(" 4. MOTS CARACTÉRISTIQUES PAR SENTIMENT")
    print("="*70)
    
    for label in sorted(df['label'].unique()):
        sentiment_name = {-1: "Négatif", 0: "Neutre", 1: "Positif"}.get(label, "Inconnu")
        sentiment_texts = df[df['label'] == label]['text']
        
        words = []
        for text in sentiment_texts:
            text_words = re.findall(r'\b[a-z]+\b', text.lower())
            words.extend([w for w in text_words if w not in stop_words and len(w) > 2])
        
        word_freq = Counter(words)
        print(f"\nTop 10 mots pour [{sentiment_name}]:")
        for word, count in word_freq.most_common(10):
            print(f"  {word:15s}: {count:5d}")
    
    # ========== 5. EXEMPLES PAR CLASSE ==========
    print("\n" + "="*70)
    print(" 5. EXEMPLES DE COMMENTAIRES PAR CLASSE")
    print("="*70)
    
    for label in sorted(df['label'].unique()):
        sentiment_name = {-1: "Négatif", 0: "Neutre", 1: "Positif"}.get(label, "Inconnu")
        print(f"\n[{sentiment_name}] - Exemples:")
        samples = df[df['label'] == label].sample(min(3, len(df[df['label'] == label])))
        for idx, row in samples.iterrows():
            print(f"  • {row['text'][:100]}...")
    
    # ========== 6. RECOMMANDATIONS ==========
    print("\n" + "="*70)
    print(" 6. RECOMMANDATIONS POUR L'ENTRAÎNEMENT")
    print("="*70)
    
    print("\n Taille du dataset:")
    if len(df) < 500:
        print(" Dataset petit (<500). Envisagez la data augmentation.")
    elif len(df) < 2000:
        print(" Dataset de taille acceptable (500-2000).")
    else:
        print(" Dataset de bonne taille (>2000).")
    
    print("\n Split train/test recommandé:")
    print(f"  Train: {int(len(df) * 0.8)} commentaires (80%)")
    print(f"  Test:  {int(len(df) * 0.2)} commentaires (20%)")
    
    print("\n Hyperparamètres TF-IDF suggérés:")
    avg_words = df['word_count'].mean()
    if avg_words < 10:
        print("  max_features: 3000-5000")
        print("  ngram_range: (1, 2)")
    else:
        print("  max_features: 5000-10000")
        print("  ngram_range: (1, 3)")
    
    print("\n" + "="*70)
    print(" Analyse exploratoire terminée!")
    print("="*70)

if __name__ == "__main__":
    data_file = "data/processed/reddit_clean.csv"
    analyze_dataset(data_file)