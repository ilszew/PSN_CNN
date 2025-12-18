"""
Moduł do wizualizacji wyników treningu i ewaluacji.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch

from config import CLASS_NAMES, RESULTS_DIR


def ensure_results_dir():
    """Upewnia się, że katalog results istnieje."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_training_history(history, experiment_name='model'):
    """
    Rysuje krzywe uczenia (loss i accuracy).
    
    Args:
        history: Słownik z historią treningu
        experiment_name: Nazwa eksperymentu do zapisu pliku
    """
    ensure_results_dir()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Wykres Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoka', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Funkcja straty w trakcie treningu', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Wykres Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoka', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Dokładność w trakcie treningu', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Zapisz wykres
    filepath = os.path.join(RESULTS_DIR, f'{experiment_name}_training_curves.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Zapisano wykres krzywych uczenia: {filepath}")


def plot_confusion_matrix(y_true, y_pred, experiment_name='model'):
    """
    Rysuje macierz pomyłek.
    
    Args:
        y_true: Prawdziwe etykiety
        y_pred: Przewidziane etykiety
        experiment_name: Nazwa eksperymentu do zapisu pliku
    """
    ensure_results_dir()
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.xlabel('Przewidziana klasa', fontsize=12)
    plt.ylabel('Prawdziwa klasa', fontsize=12)
    plt.title('Macierz pomyłek (Confusion Matrix)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Zapisz wykres
    filepath = os.path.join(RESULTS_DIR, f'{experiment_name}_confusion_matrix.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Zapisano macierz pomyłek: {filepath}")


def plot_sample_predictions(model, test_loader, device, experiment_name='model', num_samples=25):
    """
    Rysuje siatkę przykładowych predykcji.
    
    Args:
        model: Wytrenowany model
        test_loader: DataLoader z danymi testowymi
        device: Urządzenie (CPU/GPU)
        experiment_name: Nazwa eksperymentu
        num_samples: Liczba przykładów do wyświetlenia
    """
    ensure_results_dir()
    
    model.eval()
    
    # Pobierz przykładowe dane
    images_list = []
    labels_list = []
    preds_list = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            images_list.append(images.cpu())
            labels_list.append(labels)
            preds_list.append(predicted.cpu())
            
            if len(torch.cat(images_list)) >= num_samples:
                break
    
    images = torch.cat(images_list)[:num_samples]
    labels = torch.cat(labels_list)[:num_samples]
    preds = torch.cat(preds_list)[:num_samples]
    
    # Rysowanie siatki
    n_cols = 5
    n_rows = (num_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten()
    
    for idx in range(num_samples):
        ax = axes[idx]
        
        # Denormalizacja obrazu
        img = images[idx].squeeze().numpy()
        img = img * 0.5 + 0.5  # Odwrócenie normalizacji
        
        ax.imshow(img, cmap='gray')
        
        true_label = CLASS_NAMES[labels[idx]]
        pred_label = CLASS_NAMES[preds[idx]]
        
        # Kolor tytułu: zielony jeśli poprawne, czerwony jeśli błędne
        color = 'green' if labels[idx] == preds[idx] else 'red'
        ax.set_title(f'Pred: {pred_label}\nTrue: {true_label}', color=color, fontsize=9)
        ax.axis('off')
    
    # Ukryj puste subploty
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Przykładowe predykcje modelu', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Zapisz wykres
    filepath = os.path.join(RESULTS_DIR, f'{experiment_name}_sample_predictions.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Zapisano przykładowe predykcje: {filepath}")


def plot_experiments_comparison(results_df, metric='test_accuracy'):
    """
    Rysuje porównanie wyników eksperymentów.
    
    Args:
        results_df: DataFrame z wynikami eksperymentów
        metric: Metryka do porównania
    """
    ensure_results_dir()
    
    plt.figure(figsize=(12, 6))
    
    # Sortuj według metryki
    results_sorted = results_df.sort_values(metric, ascending=False)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results_sorted)))
    
    bars = plt.bar(
        results_sorted['experiment_name'], 
        results_sorted[metric],
        color=colors,
        edgecolor='black',
        linewidth=1
    )
    
    # Dodaj wartości na słupkach
    for bar, val in zip(bars, results_sorted[metric]):
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            bar.get_height() + 0.5,
            f'{val:.2f}%',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    plt.xlabel('Eksperyment', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Porównanie dokładności eksperymentów', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Zapisz wykres
    filepath = os.path.join(RESULTS_DIR, 'experiments_comparison.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Zapisano porównanie eksperymentów: {filepath}")


def plot_class_accuracy(y_true, y_pred, experiment_name='model'):
    """
    Rysuje dokładność dla każdej klasy.
    
    Args:
        y_true: Prawdziwe etykiety
        y_pred: Przewidziane etykiety
        experiment_name: Nazwa eksperymentu
    """
    ensure_results_dir()
    
    cm = confusion_matrix(y_true, y_pred)
    class_accuracy = cm.diagonal() / cm.sum(axis=1) * 100
    
    plt.figure(figsize=(12, 6))
    
    colors = plt.cm.RdYlGn(class_accuracy / 100)
    
    bars = plt.bar(CLASS_NAMES, class_accuracy, color=colors, edgecolor='black', linewidth=1)
    
    # Dodaj wartości na słupkach
    for bar, val in zip(bars, class_accuracy):
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            bar.get_height() + 1,
            f'{val:.1f}%',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )
    
    plt.xlabel('Klasa', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Dokładność klasyfikacji dla każdej klasy', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Zapisz wykres
    filepath = os.path.join(RESULTS_DIR, f'{experiment_name}_class_accuracy.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Zapisano dokładność klas: {filepath}")


if __name__ == "__main__":
    # Test wizualizacji z przykładowymi danymi
    print("Test modułu wizualizacji...")
    
    # Przykładowa historia treningu
    history = {
        'train_loss': [2.3, 1.5, 1.0, 0.7, 0.5],
        'val_loss': [2.4, 1.6, 1.1, 0.8, 0.6],
        'train_acc': [20, 45, 60, 75, 85],
        'val_acc': [18, 42, 58, 72, 82]
    }
    
    plot_training_history(history, 'test')
    print("Test zakończony pomyślnie!")

