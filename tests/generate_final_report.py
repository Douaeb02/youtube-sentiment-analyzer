"""
Génère un rapport final complet du projet
"""
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

def generate_final_report():
    """Génère le rapport final du projet"""
    
    report = []
    
    # En-tête
    report.append("="*80)
    report.append("RAPPORT FINAL - YOUTUBE SENTIMENT ANALYZER")
    report.append("Projet MLOps Complet : Du Développement au Déploiement Cloud")
    report.append("="*80)
    report.append(f"\nDate: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    report.append(f"Auteur: [Votre Nom]")
    report.append("\n")
    
    # 1. Résumé du projet
    report.append("="*80)
    report.append("1. RÉSUMÉ DU PROJET")
    report.append("="*80)
    report.append("\nObjectif:")
    report.append("  Développer un système MLOps complet pour l'analyse automatique du sentiment")
    report.append("  des commentaires YouTube en temps réel.")
    report.append("\nComposants:")
    report.append("  • Modèle ML (TF-IDF + Logistic Regression)")
    report.append("  • API REST (FastAPI)")
    report.append("  • Extension Chrome")
    report.append("  • Déploiement Docker sur Hugging Face Spaces")
    report.append("\n")
    
    # 2. Architecture
    report.append("="*80)
    report.append("2. ARCHITECTURE TECHNIQUE")
    report.append("="*80)
    report.append("\nFlux de données:")
    report.append("  1. Utilisateur visite une vidéo YouTube")
    report.append("  2. Extension Chrome extrait les commentaires")
    report.append("  3. Commentaires envoyés à l'API cloud (batch)")
    report.append("  4. API traite avec le modèle ML")
    report.append("  5. Prédictions retournées")
    report.append("  6. Extension affiche les résultats avec visualisations")
    report.append("\n")
    
    # 3. Données
    report.append("="*80)
    report.append("3. DONNÉES")
    report.append("="*80)
    
    try:
        # Données train
        train_path = Path("data/processed/train.csv")
        test_path = Path("data/processed/test.csv")
        
        if train_path.exists() and test_path.exists():
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            report.append(f"\nDataset source: Reddit Sentiment Analysis")
            report.append(f"Train set: {len(train_df)} commentaires")
            report.append(f"Test set:  {len(test_df)} commentaires")
            report.append(f"Total:     {len(train_df) + len(test_df)} commentaires")
            
            # Distribution
            train_dist = train_df['label'].value_counts().sort_index()
            report.append("\nDistribution des classes (train):")
            for label, count in train_dist.items():
                sentiment = {-1: "Négatif", 0: "Neutre", 1: "Positif"}[label]
                pct = (count / len(train_df)) * 100
                report.append(f"  {sentiment:10s}: {count:5d} ({pct:5.1f}%)")
        else:
            report.append("\n⚠️  Fichiers de données non trouvés")
    
    except Exception as e:
        report.append(f"\n⚠️  Erreur lors de la lecture des données: {e}")
    
    report.append("\n")
    
    # 4. Modèle
    report.append("="*80)
    report.append("4. MODÈLE ML")
    report.append("="*80)
    
    try:
        history_path = Path("models/training_history.json")
        
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            report.append("\nAlgorithme: Logistic Regression")
            report.append("Vectorisation: TF-IDF")
            
            params = history.get('params', {})
            report.append(f"\nHyperparamètres:")
            for key, value in params.items():
                report.append(f"  {key}: {value}")
            
            metrics = history.get('metrics', {})
            
            report.append(f"\nPerformance:")
            report.append(f"  Train Accuracy:  {metrics.get('train_accuracy', 0):.4f}")
            report.append(f"  Test Accuracy:   {metrics.get('test_accuracy', 0):.4f}")
            report.append(f"  Avg F1-Score:    {metrics.get('avg_f1', 0):.4f}")
            
            report.append(f"\nMétriques par classe:")
            classes = ['Négatif', 'Neutre', 'Positif']
            precision = metrics.get('precision', [])
            recall = metrics.get('recall', [])
            f1 = metrics.get('f1_score', [])
            
            report.append(f"  {'Classe':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
            report.append(f"  {'-'*48}")
            for i, cls in enumerate(classes):
                report.append(f"  {cls:<12} {precision[i]:>10.4f}  {recall[i]:>10.4f}  {f1[i]:>10.4f}")
            
            # Performance d'inférence
            inf_time = metrics.get('inference_time_ms', 0)
            report.append(f"\nPerformance d'inférence:")
            report.append(f"  Temps par commentaire: {inf_time:.2f}ms")
            report.append(f"  Temps batch 50:        {inf_time*50:.2f}ms")
            
            # Critères
            report.append(f"\nCritères de performance:")
            test_acc = metrics.get('test_accuracy', 0)
            avg_f1_val = metrics.get('avg_f1', 0)
            batch_time = inf_time * 50
            
            report.append(f"  {'Critère':<30} {'Requis':<15} {'Atteint':<15} {'Status'}")
            report.append(f"  {'-'*70}")
            
            acc_status = "✅" if test_acc >= 0.80 else "❌"
            report.append(f"  {'Accuracy':<30} {'≥ 80%':<15} {f'{test_acc*100:.1f}%':<15} {acc_status}")
            
            f1_status = "✅" if avg_f1_val >= 0.75 else "❌"
            report.append(f"  {'F1-Score moyen':<30} {'≥ 0.75':<15} {f'{avg_f1_val:.3f}':<15} {f1_status}")
            
            time_status = "✅" if batch_time < 100 else "⚠️"
            report.append(f"  {'Temps batch 50':<30} {'< 100ms':<15} {f'{batch_time:.1f}ms':<15} {time_status}")
        
        else:
            report.append("\n⚠️  Historique d'entraînement non trouvé")
    
    except Exception as e:
        report.append(f"\n⚠️  Erreur lors de la lecture de l'historique: {e}")
    
    report.append("\n")
    
    # 5. API
    report.append("="*80)
    report.append("5. API REST")
    report.append("="*80)
    report.append("\nFramework: FastAPI")
    report.append("Port: 8000 (local) / 7860 (Hugging Face)")
    report.append("\nEndpoints:")
    report.append("  GET  /          - Info de l'API")
    report.append("  GET  /health    - Health check")
    report.append("  POST /predict_batch - Analyse batch (1-100 commentaires)")
    report.append("  POST /predict   - Analyse d'un seul commentaire")
    report.append("  GET  /docs      - Documentation Swagger")
    report.append("\nFonctionnalités:")
    report.append("  • Validation Pydantic")
    report.append("  • CORS configuré pour extension Chrome")
    report.append("  • Gestion d'erreurs robuste")
    report.append("  • Logging détaillé")
    report.append("\n")
    
    # 6. Extension Chrome
    report.append("="*80)
    report.append("6. EXTENSION CHROME")
    report.append("="*80)
    report.append("\nFonctionnalités:")
    report.append("  • Extraction automatique des commentaires YouTube")
    report.append("  • Analyse en temps réel avec l'API")
    report.append("  • Statistiques détaillées avec graphiques")
    report.append("  • Filtres par sentiment (positif/neutre/négatif)")
    report.append("  • Mode sombre/clair")
    report.append("  • Copie des résultats")
    report.append("  • Configuration API (locale/production)")
    report.append("\nTechnologies:")
    report.append("  • Manifest V3")
    report.append("  • Content Scripts")
    report.append("  • Service Worker")
    report.append("  • HTML/CSS/JavaScript moderne")
    report.append("\n")
    
    # 7. Déploiement
    report.append("="*80)
    report.append("7. DÉPLOIEMENT")
    report.append("="*80)
    report.append("\nPlateforme: Hugging Face Spaces")
    report.append("Containerisation: Docker")
    report.append("Image de base: python:3.10-slim")
    report.append("\nOptimisations:")
    report.append("  • Image légère (slim)")
    report.append("  • Requirements minimaux (production)")
    report.append("  • Healthcheck intégré")
    report.append("  • Multi-stage non utilisé (simplicité)")
    report.append("\nURL de production:")
    report.append("  https://YOUR-USERNAME-youtube-sentiment-analyzer.hf.space")
    report.append("\n")
    
    # 8. Tests
    report.append("="*80)
    report.append("8. TESTS ET VALIDATION")
    report.append("="*80)
    report.append("\nTypes de tests:")
    report.append("  1. Tests du modèle")
    report.append("     • Chargement")
    report.append("     • Performance sur test set")
    report.append("     • Cas limites")
    report.append("     • Cohérence")
    report.append("     • Exemples réalistes")
    report.append("\n  2. Tests de l'API")
    report.append("     • Health check")
    report.append("     • Endpoints")
    report.append("     • Validation des données")
    report.append("     • Gestion d'erreurs")
    report.append("     • Performance")
    report.append("\n  3. Tests d'intégration")
    report.append("     • Flux complet end-to-end")
    report.append("     • API locale et déployée")
    report.append("     • Extension Chrome")
    report.append("\n")
    
    # 9. Difficultés rencontrées
    report.append("="*80)
    report.append("9. DIFFICULTÉS RENCONTRÉES ET SOLUTIONS")
    report.append("="*80)
    report.append("\nDifficulté 1: Déséquilibre des classes")
    report.append("  Solution: Utilisation de class_weight='balanced'")
    report.append("\nDifficulté 2: Extraction des commentaires YouTube dynamiques")
    report.append("  Solution: Défilement progressif et attente du chargement")
    report.append("\nDifficulté 3: Taille de l'image Docker")
    report.append("  Solution: Image slim + requirements minimaux")
    report.append("\nDifficulté 4: CORS pour l'extension Chrome")
    report.append("  Solution: Configuration allow_origins=['*']")
    report.append("\n")
    
    # 10. Améliorations futures
    report.append("="*80)
    report.append("10. AMÉLIORATIONS FUTURES POSSIBLES")
    report.append("="*80)
    report.append("\nModèle:")
    report.append("  • Support multi-langue (français, arabe...)")
    report.append("  • Modèles transformers (BERT, RoBERTa)")
    report.append("  • Détection du sarcasme")
    report.append("  • Analyse des emojis avancée")
    report.append("\nAPI:")
    report.append("  • Rate limiting")
    report.append("  • Cache des prédictions")
    report.append("  • Monitoring avec Prometheus")
    report.append("  • Base de données pour historique")
    report.append("\nExtension:")
    report.append("  • Analyse des réponses aux commentaires")
    report.append("  • Export CSV/PDF")
    report.append("  • Comparaison entre vidéos")
    report.append("  • Notifications pour commentaires négatifs")
    report.append("  • Analyse temporelle des sentiments")
    report.append("\n")
    
    # 11. Conclusion
    report.append("="*80)
    report.append("11. CONCLUSION")
    report.append("="*80)
    report.append("\nCe projet démontre une maîtrise complète du cycle MLOps:")
    report.append("  ✅ Préparation des données et EDA")
    report.append("  ✅ Entraînement et optimisation du modèle")
    report.append("  ✅ Développement d'une API REST professionnelle")
    report.append("  ✅ Création d'une interface utilisateur moderne")
    report.append("  ✅ Containerisation avec Docker")
    report.append("  ✅ Déploiement cloud sur Hugging Face")
    report.append("  ✅ Tests et validation complète")
    report.append("\nLe système est fonctionnel, performant et prêt pour la production.")
    report.append("\n")
    report.append("="*80)
    
    # Sauvegarder le rapport
    report_text = "\n".join(report)
    
    output_path = Path("RAPPORT_FINAL.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✅ Rapport sauvegardé: {output_path}")
    
    return output_path

if __name__ == "__main__":
    generate_final_report()
    