"""
Skrypt do ewaluacji wytrenowanych modeli CNN.
"""

import os
import argparse
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

from config import CLASS_NAMES, RESULTS_DIR, DEFAULT_CONFIG
from models.cnn import create_model
from utils.dataset import get_data_loaders
from utils.visualization import (
    plot_confusion_matrix, 
    plot_sample_predictions,
    plot_class_accuracy
)


def load_model(model_path, device):
    """
    Wczytuje wytrenowany model.
    
    Args:
        model_path: Ścieżka do pliku modelu
        device: Urządzenie (CPU/GPU)
        
    Returns:
        tuple: (model, config)
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    config = checkpoint.get('config', DEFAULT_CONFIG)
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Wczytano model z epoki {checkpoint.get('epoch', 'unknown')}")
    print(f"Dokładność walidacyjna: {checkpoint.get('val_acc', 'unknown'):.2f}%")
    
    return model, config


def evaluate_on_test(model, test_loader, device):
    """
    Ewaluuje model na zbiorze testowym.
    
    Args:
        model: Wytrenowany model
        test_loader: DataLoader z danymi testowymi
        device: Urządzenie
        
    Returns:
        tuple: (accuracy, loss, y_true, y_pred)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Ewaluacja'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    
    accuracy = accuracy_score(y_true, y_pred) * 100
    avg_loss = running_loss / len(test_loader.dataset)
    
    return accuracy, avg_loss, y_true, y_pred


def evaluate_model(model_path, experiment_name='model', generate_plots=True):
    """
    Pełna ewaluacja modelu z generowaniem raportów i wizualizacji.
    
    Args:
        model_path: Ścieżka do pliku modelu
        experiment_name: Nazwa eksperymentu
        generate_plots: Czy generować wykresy
        
    Returns:
        tuple: (accuracy, loss)
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUrządzenie: {device}")
    
    # Wczytaj model
    print(f"\nWczytywanie modelu: {model_path}")
    model, config = load_model(model_path, device)
    
    # Wczytaj dane testowe
    print("\nWczytywanie danych testowych...")
    _, _, test_loader = get_data_loaders(batch_size=config.get('batch_size', 64))
    
    # Ewaluacja
    print("\nEwaluacja na zbiorze testowym...")
    accuracy, loss, y_true, y_pred = evaluate_on_test(model, test_loader, device)
    
    print(f"\n{'='*60}")
    print(f"WYNIKI EWALUACJI: {experiment_name}")
    print(f"{'='*60}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test Loss: {loss:.4f}")
    print(f"{'='*60}")
    
    # Raport klasyfikacji
    print("\nRaport klasyfikacji:")
    print("-" * 60)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    print(report)
    
    # Zapisz raport do pliku
    report_path = os.path.join(RESULTS_DIR, f'{experiment_name}_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"WYNIKI EWALUACJI: {experiment_name}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Test Accuracy: {accuracy:.2f}%\n")
        f.write(f"Test Loss: {loss:.4f}\n")
        f.write(f"{'='*60}\n\n")
        f.write("Raport klasyfikacji:\n")
        f.write("-" * 60 + "\n")
        f.write(report)
    print(f"\nZapisano raport: {report_path}")
    
    # Generuj wizualizacje
    if generate_plots:
        print("\nGenerowanie wizualizacji...")
        
        # Macierz pomyłek
        plot_confusion_matrix(y_true, y_pred, experiment_name)
        
        # Dokładność dla każdej klasy
        plot_class_accuracy(y_true, y_pred, experiment_name)
        
        # Przykładowe predykcje
        plot_sample_predictions(model, test_loader, device, experiment_name)
    
    return accuracy, loss


def compare_models(model_paths, names=None):
    """
    Porównuje wiele modeli na zbiorze testowym.
    
    Args:
        model_paths: Lista ścieżek do modeli
        names: Lista nazw modeli (opcjonalnie)
        
    Returns:
        pd.DataFrame: Wyniki porównania
    """
    import pandas as pd
    
    if names is None:
        names = [f'model_{i}' for i in range(len(model_paths))]
    
    results = []
    
    for path, name in zip(model_paths, names):
        print(f"\n{'='*60}")
        print(f"Ewaluacja: {name}")
        print(f"{'='*60}")
        
        accuracy, loss = evaluate_model(
            model_path=path,
            experiment_name=name,
            generate_plots=True
        )
        
        results.append({
            'model_name': name,
            'model_path': path,
            'test_accuracy': accuracy,
            'test_loss': loss
        })
    
    results_df = pd.DataFrame(results)
    
    # Zapisz porównanie
    csv_path = os.path.join(RESULTS_DIR, 'models_comparison.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nZapisano porównanie modeli: {csv_path}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Ewaluacja modelu CNN Fashion MNIST')
    parser.add_argument('--model', type=str, required=True,
                        help='Ścieżka do pliku modelu (.pth)')
    parser.add_argument('--name', type=str, default='model',
                        help='Nazwa eksperymentu dla plików wyjściowych')
    parser.add_argument('--no-plots', action='store_true',
                        help='Nie generuj wizualizacji')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Błąd: Plik modelu nie istnieje: {args.model}")
        return
    
    evaluate_model(
        model_path=args.model,
        experiment_name=args.name,
        generate_plots=not args.no_plots
    )


if __name__ == "__main__":
    main()

