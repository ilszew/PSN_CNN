"""
Moduł do ładowania i przygotowania danych Fashion MNIST.
"""

import os
import gzip
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from data.download_data import get_data_path


class FashionMNISTDataset(Dataset):
    """
    Dataset dla Fashion MNIST z plików .gz.
    """
    
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images: Numpy array z obrazami (N, 28, 28)
            labels: Numpy array z etykietami (N,)
            transform: Opcjonalne transformacje
        """
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Konwersja do tensora i dodanie kanału
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        
        # Normalizacja do zakresu [0, 1]
        image = image / 255.0
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_mnist_images(filepath):
    """
    Wczytuje obrazy z pliku w formacie MNIST (obsługuje .gz i surowe pliki).
    
    Args:
        filepath: Ścieżka do pliku
        
    Returns:
        numpy.ndarray: Tablica obrazów (N, 28, 28)
    """
    # Sprawdź czy plik jest skompresowany gzipem (magic number: 1f 8b)
    with open(filepath, 'rb') as f_test:
        magic_bytes = f_test.read(2)
        is_gzip = magic_bytes == b'\x1f\x8b'
    
    open_func = gzip.open if is_gzip else open
    
    with open_func(filepath, 'rb') as f:
        # Pomijamy nagłówek formatu MNIST: 
        # magic number (4), num_images (4), num_rows (4), num_cols (4) = 16 bajtów
        f.read(16)
        # Wczytujemy pozostałe dane jako obrazy
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(-1, 28, 28)


def load_mnist_labels(filepath):
    """
    Wczytuje etykiety z pliku w formacie MNIST (obsługuje .gz i surowe pliki).
    
    Args:
        filepath: Ścieżka do pliku
        
    Returns:
        numpy.ndarray: Tablica etykiet (N,)
    """
    # Sprawdź czy plik jest skompresowany gzipem
    with open(filepath, 'rb') as f_test:
        magic_bytes = f_test.read(2)
        is_gzip = magic_bytes == b'\x1f\x8b'
    
    open_func = gzip.open if is_gzip else open
    
    with open_func(filepath, 'rb') as f:
        # Pomijamy nagłówek formatu MNIST:
        # magic number (4) i liczbę elementów (4) = 8 bajtów
        f.read(8)
        # Wczytujemy etykiety
        return np.frombuffer(f.read(), dtype=np.uint8)


def get_data_loaders(batch_size=64, val_split=0.1667, num_workers=0):
    """
    Tworzy DataLoadery dla zbiorów train, validation i test.
    
    Args:
        batch_size: Rozmiar batcha
        val_split: Proporcja danych treningowych do walidacji (domyślnie ~10k z 60k)
        num_workers: Liczba workerów do ładowania danych
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Pobierz ścieżkę do danych
    data_path = get_data_path()
    
    # Znajdź pliki danych
    train_images_path = None
    train_labels_path = None
    test_images_path = None
    test_labels_path = None
    
    for root, dirs, files in os.walk(data_path):
        for file in files:
            filepath = os.path.join(root, file)
            if 'train-images' in file:
                train_images_path = filepath
            elif 'train-labels' in file:
                train_labels_path = filepath
            elif 't10k-images' in file:
                test_images_path = filepath
            elif 't10k-labels' in file:
                test_labels_path = filepath
    
    if not all([train_images_path, train_labels_path, test_images_path, test_labels_path]):
        raise FileNotFoundError("Nie znaleziono wszystkich plików Fashion MNIST")
    
    # Wczytaj dane
    print("Wczytywanie danych treningowych...")
    train_images = load_mnist_images(train_images_path)
    train_labels = load_mnist_labels(train_labels_path)
    
    print("Wczytywanie danych testowych...")
    test_images = load_mnist_images(test_images_path)
    test_labels = load_mnist_labels(test_labels_path)
    
    print(f"Rozmiar zbioru treningowego: {len(train_labels)}")
    print(f"Rozmiar zbioru testowego: {len(test_labels)}")
    
    # Transformacje - normalizacja
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    
    # Tworzenie datasetów
    full_train_dataset = FashionMNISTDataset(train_images, train_labels, transform=normalize)
    test_dataset = FashionMNISTDataset(test_images, test_labels, transform=normalize)
    
    # Podział train na train i validation
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Rozmiar zbioru treningowego po podziale: {len(train_dataset)}")
    print(f"Rozmiar zbioru walidacyjnego: {len(val_dataset)}")
    
    # Tworzenie DataLoaderów
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test ładowania danych
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=64)
    
    # Sprawdź kształt danych
    for images, labels in train_loader:
        print(f"Kształt batcha obrazów: {images.shape}")
        print(f"Kształt batcha etykiet: {labels.shape}")
        print(f"Zakres wartości: [{images.min():.2f}, {images.max():.2f}]")
        break

