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

## Użycie

### Trening pojedynczego modelu

**Automatyczny wybór urządzenia (GPU jeśli dostępne, w przeciwnym razie CPU):**
```bash
python train.py --config baseline
```

**Wymuszenie użycia CPU:**
```bash
python train.py --config baseline --device cpu
```

**Wymuszenie użycia GPU (CUDA):**
```bash
python train.py --config baseline --device cuda
```
 0.25 |

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


**Windows:**
```powershell
Remove-Item -Recurse -Force data\fashionmnist
python train.py --config baseline
```

## Brak GPU / CUDA nie działa

1. Sprawdź czy CUDA jest zainstalowana:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

2. Jeśli zwraca `False`, użyj CPU:
```bash
python train.py --config baseline --device cpu
```

## Błąd "pip not recognized"

Upewnij się, że środowisko wirtualne jest aktywowane:

**Windows:**
```powershell
.\.venv\Scripts\Activate.ps1
```

