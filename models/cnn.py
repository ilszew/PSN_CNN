"""
Architektura CNN dla klasyfikacji Fashion MNIST.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FashionCNN(nn.Module):
    """
    Sieć konwolucyjna do klasyfikacji Fashion MNIST.
    
    Architektura:
    - 3 warstwy konwolucyjne z BatchNorm i MaxPooling
    - 2 warstwy w pełni połączone z Dropout
    """
    
    def __init__(self, filters=[32, 64, 128], kernel_size=3, dropout=0.5, hidden_size=256, num_classes=10):
        """
        Args:
            filters: Lista liczby filtrów dla każdej warstwy konwolucyjnej
            kernel_size: Rozmiar kernela konwolucji
            dropout: Współczynnik dropout
            hidden_size: Rozmiar warstwy ukrytej FC
            num_classes: Liczba klas wyjściowych
        """
        super(FashionCNN, self).__init__()
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout
        
        # Padding dla zachowania rozmiaru
        padding = kernel_size // 2
        
        # Warstwa 1: Conv -> BatchNorm -> ReLU -> MaxPool
        # Input: 1x28x28 -> Output: filters[0]x14x14
        self.conv1 = nn.Conv2d(1, filters[0], kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Warstwa 2: Conv -> BatchNorm -> ReLU -> MaxPool
        # Input: filters[0]x14x14 -> Output: filters[1]x7x7
        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Warstwa 3: Conv -> BatchNorm -> ReLU
        # Input: filters[1]x7x7 -> Output: filters[2]x7x7 (bez pooling)
        self.conv3 = nn.Conv2d(filters[1], filters[2], kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(filters[2])
        
        # Oblicz rozmiar po spłaszczeniu
        # Po conv1+pool: 14x14, po conv2+pool: 7x7, po conv3: 7x7
        self.flatten_size = filters[2] * 7 * 7
        
        # Warstwy w pełni połączone
        self.fc1 = nn.Linear(self.flatten_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """
        Przepuszczenie danych przez sieć.
        
        Args:
            x: Tensor wejściowy o kształcie (batch, 1, 28, 28)
            
        Returns:
            Tensor wyjściowy o kształcie (batch, num_classes)
        """
        # Blok konwolucyjny 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Blok konwolucyjny 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Blok konwolucyjny 3
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Spłaszczenie
        x = x.view(-1, self.flatten_size)
        
        # Warstwy FC
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_num_params(self):
        """Zwraca liczbę parametrów modelu."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config):
    """
    Tworzy model CNN na podstawie konfiguracji.
    
    Args:
        config: Słownik z konfiguracją modelu
        
    Returns:
        FashionCNN: Zainicjalizowany model
    """
    model = FashionCNN(
        filters=config.get('filters', [32, 64, 128]),
        kernel_size=config.get('kernel_size', 3),
        dropout=config.get('dropout', 0.5),
        hidden_size=config.get('hidden_size', 256),
        num_classes=10
    )
    return model


if __name__ == "__main__":
    # Test modelu
    from config import DEFAULT_CONFIG
    
    model = create_model(DEFAULT_CONFIG)
    print(f"Architektura modelu:\n{model}")
    print(f"\nLiczba parametrów: {model.get_num_params():,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"\nKształt wejścia: {dummy_input.shape}")
    print(f"Kształt wyjścia: {output.shape}")

