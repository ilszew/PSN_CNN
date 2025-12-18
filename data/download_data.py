"""
Skrypt do pobierania zbioru danych Fashion MNIST przez kagglehub.
"""

import os
import kagglehub


def download_fashion_mnist():
    """
    Pobiera zbiór danych Fashion MNIST z Kaggle.
    
    Returns:
        str: Ścieżka do pobranych plików
    """
    print("Pobieranie zbioru danych Fashion MNIST...")
    path = kagglehub.dataset_download("zalando-research/fashionmnist")
    print(f"Ścieżka do plików: {path}")
    return path


def get_data_path():
    """
    Zwraca ścieżkę do danych Fashion MNIST.
    Jeśli dane nie istnieją, pobiera je.
    
    Returns:
        str: Ścieżka do danych
    """
    # Sprawdź czy dane już istnieją w cache kagglehub
    try:
        path = kagglehub.dataset_download("zalando-research/fashionmnist")
        return path
    except Exception as e:
        print(f"Błąd podczas pobierania danych: {e}")
        raise


if __name__ == "__main__":
    path = download_fashion_mnist()
    
    # Wyświetl zawartość katalogu
    print("\nZawartość katalogu:")
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")

