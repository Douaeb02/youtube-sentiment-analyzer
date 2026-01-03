import re
import pandas as pd
from pathlib import Path

class TextCleaner:
    """Classe pour nettoyer et préprocesser les textes"""
    
    @staticmethod
    def remove_urls(text):
        """Supprime les URLs du texte"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    @staticmethod
    def remove_mentions(text):
        """Supprime les mentions @username"""
        return re.sub(r'@\w+', '', text)
    
    @staticmethod
    def remove_hashtags(text):
        """Supprime les hashtags mais garde le texte"""
        return re.sub(r'#(\w+)', r'\1', text)
    
    @staticmethod
    def remove_special_chars(text):
        """Supprime les caractères spéciaux mais garde les emojis et la ponctuation de base"""
        # Garde les lettres, chiffres, espaces, ponctuation de base et emojis
        text = re.sub(r'[^\w\s.,!?;\-\'\"]+', ' ', text)
        return text
    
    @staticmethod
    def remove_extra_whitespace(text):
        """Supprime les espaces multiples"""
        return re.sub(r'\s+', ' ', text).strip()
    
    @staticmethod
    def clean_text(text):
        """Pipeline complet de nettoyage"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        text = text.lower()  # Mettre en minuscules
        text = TextCleaner.remove_urls(text)
        text = TextCleaner.remove_mentions(text)
        text = TextCleaner.remove_hashtags(text)
        text = TextCleaner.remove_special_chars(text)
        text = TextCleaner.remove_extra_whitespace(text)
        
        return text

def clean_dataset(input_path, output_path):
    """
    Nettoie le dataset et sauvegarde le résultat
    
    Args:
        input_path: Chemin vers le fichier CSV brut
        output_path: Chemin pour sauvegarder le fichier nettoyé
    """
    print(" Démarrage du nettoyage des données...")
    print(f" Input: {input_path}")
    
    # Charger les données
    df = pd.read_csv(input_path)
    initial_count = len(df)
    print(f" {initial_count} commentaires chargés")
    
    # Utiliser la colonne 'clean_comment' qui existe déjà
    # Mais on va quand même appliquer notre nettoyage supplémentaire
    print("\n Application du nettoyage...")
    
    # Créer une nouvelle colonne avec le texte nettoyé
    df['text'] = df['clean_comment'].apply(TextCleaner.clean_text)
    
    # Renommer la colonne category en label pour plus de clarté
    df['label'] = df['category']
    
    # Supprimer les lignes avec du texte vide après nettoyage
    df = df[df['text'].str.strip() != '']
    print(f" {len(df)} commentaires après suppression des textes vides")
    
    # Supprimer les lignes avec des textes très courts (< 3 caractères)
    df = df[df['text'].str.len() >= 3]
    print(f" {len(df)} commentaires après suppression des textes trop courts")
    
    # Supprimer les doublons exacts
    df = df.drop_duplicates(subset=['text'])
    print(f" {len(df)} commentaires après suppression des doublons")
    
    # Sélectionner uniquement les colonnes nécessaires
    df_clean = df[['text', 'label']].copy()
    
    # Créer le dossier de sortie s'il n'existe pas
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder les données nettoyées
    df_clean.to_csv(output_path, index=False)
    
    print(f"\n Données nettoyées sauvegardées: {output_path}")
    
    # Afficher les statistiques finales
    print("\n" + "="*60)
    print(" STATISTIQUES APRÈS NETTOYAGE")
    print("="*60)
    print(f"Commentaires initiaux: {initial_count}")
    print(f"Commentaires finaux: {len(df_clean)}")
    print(f"Commentaires supprimés: {initial_count - len(df_clean)} ({(initial_count - len(df_clean))/initial_count*100:.1f}%)")
    
    print("\n Distribution des labels:")
    label_counts = df_clean['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        sentiment_name = {-1: "Négatif", 0: "Neutre", 1: "Positif"}.get(label, "Inconnu")
        percentage = (count / len(df_clean)) * 100
        print(f"  {sentiment_name:10s} ({label:2d}): {count:5d} ({percentage:.1f}%)")
    
    print("\n Exemples de commentaires nettoyés:")
    for idx, row in df_clean.sample(3).iterrows():
        sentiment_name = {-1: "Négatif", 0: "Neutre", 1: "Positif"}.get(row['label'], "Inconnu")
        print(f"\n  [{sentiment_name}]: {row['text'][:100]}...")
    
    print("\n Nettoyage terminé avec succès!")
    
    return df_clean

if __name__ == "__main__":
    input_file = "data/raw/reddit.csv"
    output_file = "data/processed/reddit_clean.csv"
    
    clean_dataset(input_file, output_file)