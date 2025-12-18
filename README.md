# PSN_CNN - Fashion MNIST Classification

Projekt sieci konwolucyjnej (CNN) do klasyfikacji obrazów z zestawu danych Fashion MNIST.

## Wymagania

- Python 3.10+
- PyTorch 2.0+
- CUDA (opcjonalnie, do obliczeń na GPU)

## Instalacja

1. Python instalacja

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
pip install -r requirements.txt
```

## 2. Uruchamianie serii eksperymentów

**Automatyczny wybór:**
```bash
python experiments.py
```

**Wymuszenie użycia CPU:**
```bash
python experiments.py --device cpu
```

**Wymuszenie użycia GPU (CUDA):**
```bash
python experiments.py --device cuda
```

## Struktura projektu

```
PSN_CNN/
├── config.py           # Konfiguracja hiperparametrów
├── train.py            # Skrypt treningowy
├── evaluate.py         # Skrypt ewaluacyjny
├── experiments.py      # Uruchamianie serii eksperymentów
├── models/
│   └── cnn.py          # Architektura sieci CNN
├── utils/
│   ├── dataset.py      # Ładowanie danych
│   └── visualization.py # Wizualizacje
├── data/               # Dane Fashion MNIST
├── results/            # Wyniki eksperymentów
└── requirements.txt    # Zależności
```

## Wyniki

Wyniki eksperymentów są zapisywane w katalogu `results/`:
- `*_best_model.pth` - wagi najlepszego modelu
- `*_history.json` - historia treningu
- `*_results.json` - wyniki eksperymentu
- `*_confusion_matrix.png` - macierz pomyłek
- `*_training_curves.png` - krzywe uczenia
- `experiments_summary.csv` - podsumowanie wszystkich eksperymentów


## Błąd "BadGzipFile"

```powershell
Remove-Item -Recurse -Force data\fashionmnist
python train.py --config baseline
```

## Brak GPU / CUDA nie działa

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Błąd "pip not recognized"

```powershell
.\.venv\Scripts\Activate.ps1
```

