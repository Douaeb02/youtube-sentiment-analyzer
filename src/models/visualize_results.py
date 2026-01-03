"""
Script pour visualiser les résultats du modèle
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_training_history():
    """Charge l'historique d'entraînement"""
    history_file = Path('models/training_history.json')
    
    if not history_file.exists():
        print("❌ Fichier training_history.json introuvable!")
        return None
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    return history

def plot_confusion_matrix(history):
    """Affiche la matrice de confusion"""
    if 'metrics' not in history or 'confusion_matrix' not in history['metrics']:
        print("⚠️  Matrice de confusion non disponible")
        return
    
    cm = np.array(history['metrics']['confusion_matrix'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Négatif', 'Neutre', 'Positif'],
        yticklabels=['Négatif', 'Neutre', 'Positif'],
        cbar_kws={'label': 'Nombre de prédictions'}
    )
    
    plt.title('Matrice de Confusion', fontsize=16, fontweight='bold')
    plt.ylabel('Vraie Classe', fontsize=12)
    plt.xlabel('Classe Prédite', fontsize=12)
    plt.tight_layout()
    
    # Sauvegarder
    output_path = Path('models/confusion_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Matrice de confusion sauvegardée: {output_path}")
    
    plt.show()

def plot_metrics_by_class(history):
    """Affiche les métriques par classe"""
    if 'metrics' not in history:
        print("⚠️  Métriques non disponibles")
        return
    
    metrics = history['metrics']
    classes = ['Négatif', 'Neutre', 'Positif']
    
    precision = metrics.get('precision', [])
    recall = metrics.get('recall', [])
    f1_score = metrics.get('f1_score', [])
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, f1_score, width, label='F1-Score', color='#e74c3c')
    
    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Métriques par Classe', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    def add_values(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    add_values(bars1)
    add_values(bars2)
    add_values(bars3)
    
    plt.tight_layout()
    
    # Sauvegarder
    output_path = Path('models/metrics_by_class.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Métriques par classe sauvegardées: {output_path}")
    
    plt.show()

def plot_performance_summary(history):
    """Affiche un résumé des performances"""
    if 'metrics' not in history:
        print("⚠️  Métriques non disponibles")
        return
    
    metrics = history['metrics']
    
    # Préparer les données
    data = {
        'Train Accuracy': metrics.get('train_accuracy', 0),
        'Test Accuracy': metrics.get('test_accuracy', 0),
        'Avg F1-Score': metrics.get('avg_f1', 0)
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Graphique 1: Accuracy
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = ax1.bar(data.keys(), data.values(), color=colors, alpha=0.7)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Performance Globale du Modèle', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1.1])
    ax1.grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Graphique 2: Temps d'inférence
    inference_time = metrics.get('inference_time_ms', 0)
    batch_50_time = inference_time * 50
    
    times = [inference_time, batch_50_time]
    labels = ['1 commentaire', '50 commentaires']
    
    bars2 = ax2.bar(labels, times, color=['#9b59b6', '#f39c12'], alpha=0.7)
    ax2.set_ylabel('Temps (ms)', fontsize=12)
    ax2.set_title('Performance d\'Inférence', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}ms',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Ligne de référence à 100ms pour batch de 50
    ax2.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Objectif (100ms)')
    ax2.legend()
    
    plt.tight_layout()
    
    # Sauvegarder
    output_path = Path('models/performance_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Résumé des performances sauvegardé: {output_path}")
    
    plt.show()

def generate_report():
    """Génère un rapport texte"""
    history = load_training_history()
    
    if not history:
        return
    
    report_path = Path('models/performance_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("RAPPORT DE PERFORMANCE DU MODÈLE DE SENTIMENT\n")
        f.write("="*70 + "\n\n")
        
        # Informations générales
        f.write(f"Type de modèle: {history.get('model_type', 'N/A')}\n")
        f.write(f"Temps d'entraînement: {history.get('train_time', 0):.2f}s\n\n")
        
        # Métriques
        if 'metrics' in history:
            metrics = history['metrics']
            
            f.write("MÉTRIQUES DE PERFORMANCE\n")
            f.write("-"*70 + "\n")
            f.write(f"Train Accuracy:  {metrics.get('train_accuracy', 0):.4f} ({metrics.get('train_accuracy', 0)*100:.2f}%)\n")
            f.write(f"Test Accuracy:   {metrics.get('test_accuracy', 0):.4f} ({metrics.get('test_accuracy', 0)*100:.2f}%)\n")
            f.write(f"Avg F1-Score:    {metrics.get('avg_f1', 0):.4f}\n\n")
            
            f.write("MÉTRIQUES PAR CLASSE\n")
            f.write("-"*70 + "\n")
            classes = ['Négatif (-1)', 'Neutre (0)', 'Positif (1)']
            
            precision = metrics.get('precision', [])
            recall = metrics.get('recall', [])
            f1 = metrics.get('f1_score', [])
            
            f.write(f"{'Classe':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
            f.write("-"*70 + "\n")
            
            for i, class_name in enumerate(classes):
                f.write(f"{class_name:<15} {precision[i]:>10.4f}  {recall[i]:>10.4f}  {f1[i]:>10.4f}\n")
            
            f.write("\n")
            f.write("PERFORMANCE D'INFÉRENCE\n")
            f.write("-"*70 + "\n")
            inf_time = metrics.get('inference_time_ms', 0)
            f.write(f"Temps par commentaire: {inf_time:.2f}ms\n")
            f.write(f"Temps pour 50 commentaires: {inf_time*50:.2f}ms\n")
            
            status = "✅ ATTEINT" if inf_time*50 < 100 else "❌ NON ATTEINT"
            f.write(f"Objectif (<100ms): {status}\n")
    
    print(f"✅ Rapport textuel sauvegardé: {report_path}")

def main():
    """Fonction principale"""
    print("\n" + "📊 "*25)
    print("VISUALISATION DES RÉSULTATS")
    print("📊 "*25 + "\n")
    
    # Charger l'historique
    history = load_training_history()
    
    if not history:
        print("❌ Impossible de charger l'historique d'entraînement")
        return
    
    print("✅ Historique chargé\n")
    
    # Générer les visualisations
    print("📈 Génération des graphiques...\n")
    
    plot_confusion_matrix(history)
    plot_metrics_by_class(history)
    plot_performance_summary(history)
    
    # Générer le rapport
    print("\n📝 Génération du rapport...\n")
    generate_report()
    
    print("\n" + "="*70)
    print("✅ Toutes les visualisations ont été générées!")
    print("="*70)
    print("\n📁 Fichiers créés dans le dossier models/:")
    print("  • confusion_matrix.png")
    print("  • metrics_by_class.png")
    print("  • performance_summary.png")
    print("  • performance_report.txt")

if __name__ == "__main__":
    # Configuration matplotlib
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    main()