"""
Script d'optimisation des hyperparamètres avec GridSearchCV
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
import time

def tune_logistic_regression(X_train, y_train):
    """
    Optimise les hyperparamètres de Logistic Regression
    """
    print("🔧 OPTIMISATION: Logistic Regression")
    print("="*70)
    
    # Paramètres à tester
    param_grid = {
        'C': [0.1, 0.5, 1.0, 2.0, 5.0],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [500, 1000, 2000]
    }
    
    print(f"Espace de recherche:")
    print(f"  C: {param_grid['C']}")
    print(f"  solver: {param_grid['solver']}")
    print(f"  max_iter: {param_grid['max_iter']}")
    print(f"  Total de combinaisons: {len(param_grid['C']) * len(param_grid['solver']) * len(param_grid['max_iter'])}")
    
    # Modèle de base
    base_model = LogisticRegression(
        random_state=42,
        class_weight='balanced',
        multi_class='multinomial'
    )
    
    # GridSearch avec validation croisée
    scorer = make_scorer(f1_score, average='weighted')
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring=scorer,
        n_jobs=-1,
        verbose=1
    )
    
    print("\n🔍 Démarrage de la recherche...")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    print(f"\n✅ Recherche terminée en {search_time:.2f}s")
    print(f"\n🏆 MEILLEURS PARAMÈTRES:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\n📊 Score F1 (CV): {grid_search.best_score_:.4f}")
    
    # Top 5 des configurations
    print(f"\n📈 TOP 5 DES CONFIGURATIONS:")
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')
    
    for i in range(min(5, len(results_df))):
        row = results_df.iloc[i]
        print(f"\n  Rang {i+1}:")
        print(f"    Score: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
        print(f"    Params: {row['params']}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def tune_tfidf_and_model(X_train, y_train):
    """
    Optimise conjointement TF-IDF et le modèle
    """
    print("\n🔧 OPTIMISATION CONJOINTE: TF-IDF + Logistic Regression")
    print("="*70)
    
    from sklearn.pipeline import Pipeline
    
    # Créer un pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(random_state=42, class_weight='balanced'))
    ])
    
    # Paramètres à tester
    param_grid = {
        'tfidf__max_features': [3000, 5000, 7000],
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__min_df': [2, 3],
        'tfidf__max_df': [0.9, 0.95],
        'clf__C': [0.5, 1.0, 2.0],
        'clf__solver': ['lbfgs']
    }
    
    print(f"Espace de recherche:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    print(f"  Total de combinaisons: {total_combinations}")
    
    # RandomizedSearchCV pour gagner du temps
    print("\n🎲 Utilisation de RandomizedSearchCV (20 itérations)...")
    
    scorer = make_scorer(f1_score, average='weighted')
    
    random_search = RandomizedSearchCV(
        pipeline,
        param_grid,
        n_iter=20,
        cv=3,
        scoring=scorer,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    print("\n🔍 Démarrage de la recherche...")
    start_time = time.time()
    random_search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    print(f"\n✅ Recherche terminée en {search_time:.2f}s ({search_time/60:.1f} minutes)")
    print(f"\n🏆 MEILLEURS PARAMÈTRES:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\n📊 Score F1 (CV): {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_

def compare_models(X_train, y_train, X_test, y_test):
    """
    Compare différents algorithmes
    """
    print("\n⚔️  COMPARAISON DE MODÈLES")
    print("="*70)
    
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, f1_score
    
    # Vectoriser les données
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0, max_iter=1000, random_state=42, 
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=42,
            class_weight='balanced', n_jobs=-1
        ),
        'SVM (linear)': SVC(
            C=1.0, kernel='linear', random_state=42,
            class_weight='balanced'
        )
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\n🔹 Entraînement: {name}")
        
        start_time = time.time()
        model.fit(X_train_vec, y_train)
        train_time = time.time() - start_time
        
        # Prédictions
        y_pred = model.predict(X_test_vec)
        
        # Métriques
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Temps d'inférence
        start_time = time.time()
        _ = model.predict(X_test_vec[:50])
        inference_time = (time.time() - start_time) * 1000  # en ms
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'F1-Score': f1,
            'Train Time (s)': train_time,
            'Inference Time (ms)': inference_time
        })
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Train Time: {train_time:.2f}s")
        print(f"  Inference Time (50 samples): {inference_time:.2f}ms")
    
    # Afficher le tableau de comparaison
    print("\n" + "="*70)
    print("📊 TABLEAU COMPARATIF")
    print("="*70)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('F1-Score', ascending=False)
    
    print("\n" + results_df.to_string(index=False))
    
    print("\n🏆 Meilleur modèle:", results_df.iloc[0]['Model'])
    
    return results_df

def main():
    """Fonction principale"""
    print("\n" + "🔧 "*25)
    print("OPTIMISATION DES HYPERPARAMÈTRES")
    print("🔧 "*25 + "\n")
    
    # Charger les données
    print("📂 Chargement des données...")
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    X_train = train_df['text']
    y_train = train_df['label']
    X_test = test_df['text']
    y_test = test_df['label']
    
    print(f"  Train: {len(X_train)} commentaires")
    print(f"  Test:  {len(X_test)} commentaires")
    
    # Option 1: Comparer les modèles
    print("\n" + "="*70)
    print("OPTION 1: COMPARAISON RAPIDE DES MODÈLES")
    print("="*70)
    compare_models(X_train, y_train, X_test, y_test)
    
    # Option 2: Optimisation détaillée (commentez si trop long)
    # print("\n" + "="*70)
    # print("OPTION 2: OPTIMISATION DÉTAILLÉE")
    # print("="*70)
    # best_model, best_params = tune_tfidf_and_model(X_train, y_train)
    
    print("\n✅ Optimisation terminée!")

if __name__ == "__main__":
    main()