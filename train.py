"""
Główny skrypt treningowy dla modelu CNN Fashion MNIST.
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import EXPERIMENTS, DEFAULT_CONFIG, RESULTS_DIR
from models.cnn import create_model
from utils.dataset import get_data_loaders
from utils.visualization import plot_training_history


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Trenuje model przez jedną epokę.
    
    Returns:
        tuple: (średni loss, accuracy w %)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc='Trening', leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Waliduje model.
    
    Returns:
        tuple: (średni loss, accuracy w %)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Walidacja', leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def train_model(config, experiment_name='model', save_model=True):
    """
    Trenuje model CNN według podanej konfiguracji.
    
    Args:
        config: Słownik z konfiguracją
        experiment_name: Nazwa eksperymentu
        save_model: Czy zapisać model
        
    Returns:
        tuple: (model, history, best_val_acc)
    """
    # Upewnij się, że katalog results istnieje
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Ustawienia urządzenia
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUrządzenie: {device}")
    
    # Wczytaj dane
    print("\nWczytywanie danych...")
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=config['batch_size']
    )
    
    # Utwórz model
    print("\nTworzenie modelu...")
    model = create_model(config)
    model = model.to(device)
    print(f"Liczba parametrów: {model.get_num_params():,}")
    
    # Funkcja straty i optymalizator
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Historia treningu
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Early stopping
    best_val_acc = 0.0
    patience_counter = 0
    patience = config.get('early_stopping_patience', 5)
    
    print(f"\nRozpoczynanie treningu eksperymentu: {experiment_name}")
    print(f"Konfiguracja: {json.dumps(config, indent=2)}")
    print("-" * 60)
    
    for epoch in range(config['epochs']):
        # Trening
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Walidacja
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Zapisz historię
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoka [{epoch+1}/{config['epochs']}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            if save_model:
                model_path = os.path.join(RESULTS_DIR, f'{experiment_name}_best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': best_val_acc,
                    'config': config
                }, model_path)
                print(f"  -> Zapisano najlepszy model (Val Acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping po {epoch+1} epokach (brak poprawy przez {patience} epok)")
                break
    
    print("-" * 60)
    print(f"Trening zakończony. Najlepsza Val Acc: {best_val_acc:.2f}%")
    
    # Zapisz wykresy
    plot_training_history(history, experiment_name)
    
    # Zapisz historię do pliku JSON
    history_path = os.path.join(RESULTS_DIR, f'{experiment_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Zapisano historię treningu: {history_path}")
    
    return model, history, best_val_acc


def main():
    parser = argparse.ArgumentParser(description='Trening modelu CNN Fashion MNIST')
    parser.add_argument('--config', type=str, default='baseline',
                        choices=list(EXPERIMENTS.keys()) + ['default'],
                        help='Nazwa konfiguracji eksperymentu')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Nadpisz liczbę epok')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Nadpisz rozmiar batcha')
    parser.add_argument('--lr', type=float, default=None,
                        help='Nadpisz learning rate')
    parser.add_argument('--no-save', action='store_true',
                        help='Nie zapisuj modelu')
    
    args = parser.parse_args()
    
    # Wybierz konfigurację
    if args.config == 'default':
        config = DEFAULT_CONFIG.copy()
    else:
        config = EXPERIMENTS[args.config].copy()
    
    # Nadpisz parametry jeśli podane
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr
    
    # Trenuj model
    model, history, best_acc = train_model(
        config, 
        experiment_name=args.config,
        save_model=not args.no_save
    )
    
    print(f"\n{'='*60}")
    print(f"Eksperyment '{args.config}' zakończony!")
    print(f"Najlepsza dokładność walidacyjna: {best_acc:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

