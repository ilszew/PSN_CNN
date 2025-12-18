"""
Konfiguracja hiperparametrów dla eksperymentów CNN Fashion MNIST.
"""

# Nazwy klas Fashion MNIST
CLASS_NAMES = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

# Domyślna konfiguracja
DEFAULT_CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 20,
    'filters': [32, 64, 128],
    'dropout': 0.5,
    'kernel_size': 3,
    'hidden_size': 256,
    'early_stopping_patience': 5,
}

# Konfiguracje eksperymentów
EXPERIMENTS = {
    'baseline': {
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 20,
        'filters': [32, 64, 128],
        'dropout': 0.5,
        'kernel_size': 3,
        'hidden_size': 256,
        'early_stopping_patience': 5,
    },
    'high_lr': {
        'learning_rate': 0.01,
        'batch_size': 64,
        'epochs': 20,
        'filters': [32, 64, 128],
        'dropout': 0.5,
        'kernel_size': 3,
        'hidden_size': 256,
        'early_stopping_patience': 5,
    },
    'large_batch': {
        'learning_rate': 0.001,
        'batch_size': 128,
        'epochs': 20,
        'filters': [32, 64, 128],
        'dropout': 0.5,
        'kernel_size': 3,
        'hidden_size': 256,
        'early_stopping_patience': 5,
    },
    'more_filters': {
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 20,
        'filters': [64, 128, 256],
        'dropout': 0.5,
        'kernel_size': 3,
        'hidden_size': 256,
        'early_stopping_patience': 5,
    },
    'less_dropout': {
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 20,
        'filters': [32, 64, 128],
        'dropout': 0.25,
        'kernel_size': 3,
        'hidden_size': 256,
        'early_stopping_patience': 5,
    },
}

# Ścieżki
RESULTS_DIR = 'results'
DATA_DIR = 'data'

