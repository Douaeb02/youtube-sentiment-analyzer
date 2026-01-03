"""
Script d'entraînement du modèle de classification de sentiment
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import joblib
from pathlib import Path
import time
import json

class SentimentModelTrainer:
    """Classe pour entraîner et évaluer le modèle de sentiment"""
    
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.training_history = {}
    
    def load_data(self, train_path, test_path):
        """Charge les données train/test"""
        print("📂 Chargement des données...")
        
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        print(f"  Train: {len(self.train_df)} commentaires")
        print(f"  Test:  {len(self.test_df)} commentaires")
        
        self.X_train = self.train_df['text']
        self.y_train = self.train_df['label']
        self.X_test = self.test_df['text']
        self.y_test = self.test_df['label']
    
    def create_vectorizer(self, max_features=5000, ngram_range=(1, 2)):
        """
        Crée et entraîne le vectoriseur TF-IDF
        
        Args:
            max_features: Nombre maximum de features
            ngram_range: Range des n-grams à utiliser
        """
        print(f"\n🔤 Création du vectoriseur TF-IDF...")
        print(f"  max_features: {max_features}")
        print(f"  ngram_range: {ngram_range}")
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,  # Ignore les termes qui apparaissent dans moins de 2 documents
            max_df=0.95,  # Ignore les termes qui apparaissent dans plus de 95% des documents
            strip_accents='unicode',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{2,}\b'  # Mots de 2+ lettres
        )
        
        print("  Entraînement du vectoriseur...")
        start_time = time.time()
        self.X_train_vec = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vec = self.vectorizer.transform(self.X_test)
        vectorize_time = time.time() - start_time
        
        print(f"  ✅ Vectorisation terminée en {vectorize_time:.2f}s")
        print(f"  Shape train: {self.X_train_vec.shape}")
        print(f"  Shape test:  {self.X_test_vec.shape}")
        
        return self.vectorizer
    
    def train_logistic_regression(self, C=1.0, max_iter=1000):
        """
        Entraîne un modèle Logistic Regression
        
        Args:
            C: Paramètre de régularisation inverse
            max_iter: Nombre max d'itérations
        """
        print(f"\n🎯 Entraînement: Logistic Regression")
        print(f"  C: {C}")
        print(f"  max_iter: {max_iter}")
        
        start_time = time.time()
        
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=42,
            class_weight='balanced',  # Gère le déséquilibre des classes
            solver='lbfgs',
            multi_class='multinomial'
        )
        
        self.model.fit(self.X_train_vec, self.y_train)
        train_time = time.time() - start_time
        
        print(f"  ✅ Entraînement terminé en {train_time:.2f}s")
        
        self.training_history['model_type'] = 'LogisticRegression'
        self.training_history['train_time'] = train_time
        self.training_history['params'] = {'C': C, 'max_iter': max_iter}
        
        return self.model
    
    def train_random_forest(self, n_estimators=100, max_depth=None):
        """Entraîne un modèle Random Forest"""
        print(f"\n🌳 Entraînement: Random Forest")
        print(f"  n_estimators: {n_estimators}")
        print(f"  max_depth: {max_depth}")
        
        start_time = time.time()
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        self.model.fit(self.X_train_vec, self.y_train)
        train_time = time.time() - start_time
        
        print(f"  ✅ Entraînement terminé en {train_time:.2f}s")
        
        self.training_history['model_type'] = 'RandomForest'
        self.training_history['train_time'] = train_time
        
        return self.model
    
    def train_svm(self, C=1.0, kernel='rbf'):
        """Entraîne un modèle SVM"""
        print(f"\n⚙️  Entraînement: SVM")
        print(f"  C: {C}")
        print(f"  kernel: {kernel}")
        
        start_time = time.time()
        
        self.model = SVC(
            C=C,
            kernel=kernel,
            random_state=42,
            class_weight='balanced',
            probability=True  # Pour obtenir les probabilités
        )
        
        self.model.fit(self.X_train_vec, self.y_train)
        train_time = time.time() - start_time
        
        print(f"  ✅ Entraînement terminé en {train_time:.2f}s")
        
        self.training_history['model_type'] = 'SVM'
        self.training_history['train_time'] = train_time
        
        return self.model
    
    def evaluate(self):
        """Évalue le modèle sur le test set"""
        print("\n" + "="*70)
        print("📊 ÉVALUATION DU MODÈLE")
        print("="*70)
        
        # Prédictions
        start_time = time.time()
        y_pred_train = self.model.predict(self.X_train_vec)
        y_pred_test = self.model.predict(self.X_test_vec)
        inference_time = (time.time() - start_time) / len(self.X_test)
        
        # Accuracy
        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        
        print(f"\n🎯 ACCURACY:")
        print(f"  Train: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"  Test:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Classification report
        print(f"\n📈 CLASSIFICATION REPORT (Test Set):")
        print("-"*70)
        
        target_names = ['Négatif (-1)', 'Neutre (0)', 'Positif (1)']
        report = classification_report(
            self.y_test, 
            y_pred_test,
            target_names=target_names,
            digits=4
        )
        print(report)
        
        # Métriques par classe
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_test, 
            y_pred_test,
            average=None
        )
        
        print(f"\n📊 MÉTRIQUES DÉTAILLÉES PAR CLASSE:")
        print(f"{'Classe':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
        print("-"*70)
        
        for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
            class_name = target_names[i]
            status = "✅" if f >= 0.75 else "⚠️"
            print(f"{class_name:<15} {p:>10.4f}  {r:>10.4f}  {f:>10.4f}  {status} {int(s):>6d}")
        
        # Moyenne
        avg_f1 = f1.mean()
        print(f"\n{'Moyenne':<15} {precision.mean():>10.4f}  {recall.mean():>10.4f}  {avg_f1:>10.4f}")
        
        # Confusion Matrix
        print(f"\n🔢 MATRICE DE CONFUSION:")
        print("-"*70)
        cm = confusion_matrix(self.y_test, y_pred_test)
        
        print(f"\n{'':>15} {'Pred Neg':<12} {'Pred Neutre':<15} {'Pred Pos':<12}")
        print("-"*70)
        class_labels = ['Vrai Neg', 'Vrai Neutre', 'Vrai Pos']
        for i, label in enumerate(class_labels):
            print(f"{label:<15} {cm[i][0]:>10d}  {cm[i][1]:>14d}  {cm[i][2]:>12d}")
        
        # Temps d'inférence
        print(f"\n⏱️  PERFORMANCE D'INFÉRENCE:")
        print(f"  Temps par commentaire: {inference_time*1000:.2f}ms")
        print(f"  Temps pour 50 commentaires: {inference_time*50*1000:.2f}ms")
        
        batch_50_time = inference_time * 50 * 1000
        if batch_50_time < 100:
            print(f"  ✅ Performance EXCELLENTE (< 100ms)")
        elif batch_50_time < 200:
            print(f"  ✅ Performance BONNE (< 200ms)")
        else:
            print(f"  ⚠️  Performance à améliorer (> 200ms)")
        
        # Critères de performance
        print(f"\n✅ CRITÈRES DE PERFORMANCE:")
        print(f"  Accuracy minimale (80%):     {'✅ ATTEINT' if test_accuracy >= 0.80 else '❌ NON ATTEINT'} ({test_accuracy*100:.1f}%)")
        print(f"  F1-score moyen (>0.75):      {'✅ ATTEINT' if avg_f1 >= 0.75 else '❌ NON ATTEINT'} ({avg_f1:.3f})")
        print(f"  Temps batch 50 (<100ms):     {'✅ ATTEINT' if batch_50_time < 100 else '❌ NON ATTEINT'} ({batch_50_time:.1f}ms)")
        
        # Sauvegarder les métriques
        self.training_history['metrics'] = {
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_score': f1.tolist(),
            'avg_f1': float(avg_f1),
            'inference_time_ms': float(inference_time * 1000),
            'confusion_matrix': cm.tolist()
        }
        
        return {
            'accuracy': test_accuracy,
            'f1_score': avg_f1,
            'inference_time': inference_time
        }
    
    def save_model(self, model_dir='models'):
        """Sauvegarde le modèle et le vectoriseur"""
        print(f"\n💾 Sauvegarde du modèle...")
        
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder le modèle
        model_file = model_path / 'sentiment_model.joblib'
        joblib.dump(self.model, model_file)
        print(f"  ✅ Modèle sauvegardé: {model_file}")
        
        # Sauvegarder le vectoriseur
        vectorizer_file = model_path / 'tfidf_vectorizer.joblib'
        joblib.dump(self.vectorizer, vectorizer_file)
        print(f"  ✅ Vectoriseur sauvegardé: {vectorizer_file}")
        
        # Sauvegarder l'historique d'entraînement
        history_file = model_path / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"  ✅ Historique sauvegardé: {history_file}")
        
        return model_file, vectorizer_file

def main():
    """Fonction principale d'entraînement"""
    print("\n" + "🤖 "*25)
    print("ENTRAÎNEMENT DU MODÈLE DE SENTIMENT")
    print("🤖 "*25 + "\n")
    
    # Initialiser le trainer
    trainer = SentimentModelTrainer()
    
    # Charger les données
    trainer.load_data('data/processed/train.csv', 'data/processed/test.csv')
    
    # Créer le vectoriseur
    trainer.create_vectorizer(max_features=5000, ngram_range=(1, 2))
    
    # Entraîner le modèle Logistic Regression
    trainer.train_logistic_regression(C=1.0, max_iter=1000)
    
    # Évaluer
    results = trainer.evaluate()
    
    # Sauvegarder
    trainer.save_model()
    
    print("\n" + "🎉 "*25)
    print("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("🎉 "*25 + "\n")
    
    print("📁 Fichiers créés:")
    print("  ✅ models/sentiment_model.joblib")
    print("  ✅ models/tfidf_vectorizer.joblib")
    print("  ✅ models/training_history.json")
    
    print("\n🎯 Prochaine étape:")
    print("  → Phase 4: Développement de l'API REST")
    print("  → Créez le fichier: src/api/main.py")

if __name__ == "__main__":
    main()