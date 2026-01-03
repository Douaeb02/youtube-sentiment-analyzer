"""
Script pour créer un split train/test stratifié et reproductible
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def create_train_test_split(input_path, output_dir, test_size=0.2, random_state=42):
    """
    Crée un split train/test stratifié
    
    Args:
        input_path: Chemin vers le fichier CSV nettoyé
        output_dir: Dossier de sortie pour les fichiers train/test
        test_size: Proportion du test set (défaut: 0.2)
        random_state: Seed pour la reproductibilité (défaut: 42)
    """
    print("✂️  CRÉATION DU SPLIT TRAIN/TEST")
    print("="*70)
    
    # Charger les données
    df = pd.read_csv(input_path)
    print(f"📊 Dataset chargé: {len(df)} commentaires")
    
    # Vérifier la distribution des classes
    print("\n📈 Distribution des classes dans le dataset complet:")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        sentiment_name = {-1: "Négatif", 0: "Neutre", 1: "Positif"}.get(label, "Inconnu")
        percentage = (count / len(df)) * 100
        print(f"  {sentiment_name:10s}: {count:5d} ({percentage:.1f}%)")
    
    # Créer le split stratifié
    print(f"\n✂️  Création du split avec test_size={test_size}, random_state={random_state}")
    
    X = df['text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintient les proportions des classes
    )
    
    # Créer les DataFrames
    train_df = pd.DataFrame({'text': X_train, 'label': y_train})
    test_df = pd.DataFrame({'text': X_test, 'label': y_test})
    
    # Créer le dossier de sortie s'il n'existe pas
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder les fichiers
    train_path = output_path / "train.csv"
    test_path = output_path / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n💾 Fichiers sauvegardés:")
    print(f"  Train: {train_path} ({len(train_df)} commentaires)")
    print(f"  Test:  {test_path} ({len(test_df)} commentaires)")
    
    # Vérifier que le split est stratifié
    print("\n✅ VÉRIFICATION DU SPLIT STRATIFIÉ")
    print("-"*70)
    
    print("\n📊 Distribution dans le TRAIN set:")
    train_counts = train_df['label'].value_counts().sort_index()
    for label, count in train_counts.items():
        sentiment_name = {-1: "Négatif", 0: "Neutre", 1: "Positif"}.get(label, "Inconnu")
        percentage = (count / len(train_df)) * 100
        print(f"  {sentiment_name:10s}: {count:5d} ({percentage:.1f}%)")
    
    print("\n📊 Distribution dans le TEST set:")
    test_counts = test_df['label'].value_counts().sort_index()
    for label, count in test_counts.items():
        sentiment_name = {-1: "Négatif", 0: "Neutre", 1: "Positif"}.get(label, "Inconnu")
        percentage = (count / len(test_df)) * 100
        print(f"  {sentiment_name:10s}: {count:5d} ({percentage:.1f}%)")
    
    # Comparaison des distributions
    print("\n📈 COMPARAISON DES DISTRIBUTIONS:")
    print(f"{'Classe':<15} {'Original %':<12} {'Train %':<12} {'Test %':<12} {'Différence'}")
    print("-"*70)
    
    for label in sorted(df['label'].unique()):
        sentiment_name = {-1: "Négatif", 0: "Neutre", 1: "Positif"}.get(label, "Inconnu")
        orig_pct = (label_counts[label] / len(df)) * 100
        train_pct = (train_counts[label] / len(train_df)) * 100
        test_pct = (test_counts[label] / len(test_df)) * 100
        diff = abs(orig_pct - train_pct)
        
        status = "✓" if diff < 1.0 else "⚠"
        print(f"{sentiment_name:<15} {orig_pct:>10.1f}% {train_pct:>10.1f}% {test_pct:>10.1f}% {status} {diff:.2f}%")
    
    # Statistiques supplémentaires
    print("\n" + "="*70)
    print("📊 RÉSUMÉ DU SPLIT")
    print("="*70)
    print(f"Total de commentaires:     {len(df)}")
    print(f"Train set:                 {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Test set:                  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"Random state:              {random_state}")
    print(f"Stratification:            ✅ Activée")
    
    print("\n✅ Split train/test créé avec succès!")
    print("="*70)
    
    return train_df, test_df

if __name__ == "__main__":
    input_file = "data/processed/reddit_clean.csv"
    output_directory = "data/processed"
    
    create_train_test_split(input_file, output_directory)