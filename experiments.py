"""
Skrypt do uruchamiania serii eksperymentów z różnymi konfiguracjami.
"""

import os
import argparse
import json
from datetime import datetime
import pandas as pd
import torch

from config import EXPERIMENTS, RESULTS_DIR
from train import train_model
from evaluate import evaluate_model
from utils.visualization import plot_experiments_comparison


def run_all_experiments(experiments=None, skip_existing=False):
    """
    Uruchamia wszystkie zdefiniowane eksperymenty.
    
    Args:
        experiments: Lista nazw eksperymentów do uruchomienia (None = wszystkie)
        skip_existing: Czy pominąć eksperymenty z istniejącymi wynikami
        
    Returns:
        pd.DataFrame: Wyniki wszystkich eksperymentów
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if experiments is None:
        experiments = list(EXPERIMENTS.keys())
    
    results = []
    
    print("=" * 70)
    print("URUCHAMIANIE EKSPERYMENTÓW")
    print(f"Liczba eksperymentów: {len(experiments)}")
    print(f"Eksperymenty: {', '.join(experiments)}")
    print("=" * 70)
    
    for i, exp_name in enumerate(experiments, 1):
        print(f"\n{'#'*70}")
        print(f"# EKSPERYMENT {i}/{len(experiments)}: {exp_name}")
        print(f"{'#'*70}")
        
        # Sprawdź czy pominąć istniejący eksperyment
        model_path = os.path.join(RESULTS_DIR, f'{exp_name}_best_model.pth')
        if skip_existing and os.path.exists(model_path):
            print(f"Pomijanie eksperymentu '{exp_name}' - model już istnieje")
            
            # Wczytaj istniejące wyniki
            results_path = os.path.join(RESULTS_DIR, f'{exp_name}_results.json')
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    exp_results = json.load(f)
                    results.append(exp_results)
            continue
        
        config = EXPERIMENTS[exp_name]
        
        # Trenuj model
        model, history, best_val_acc = train_model(
            config,
            experiment_name=exp_name,
            save_model=True
        )
        
        # Ewaluuj na zbiorze testowym
        test_acc, test_loss = evaluate_model(
            model_path=model_path,
            experiment_name=exp_name,
            generate_plots=True
        )
        
        # Zapisz wyniki eksperymentu
        exp_results = {
            'experiment_name': exp_name,
            'config': config,
            'best_val_accuracy': best_val_acc,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'final_train_loss': history['train_loss'][-1],
            'final_train_acc': history['train_acc'][-1],
            'epochs_trained': len(history['train_loss']),
            'timestamp': datetime.now().isoformat()
        }
        
        results.append(exp_results)
        
        # Zapisz wyniki do pliku
        results_path = os.path.join(RESULTS_DIR, f'{exp_name}_results.json')
        with open(results_path, 'w') as f:
            json.dump(exp_results, f, indent=2)
        
        print(f"\nWyniki eksperymentu '{exp_name}':")
        print(f"  Val Accuracy: {best_val_acc:.2f}%")
        print(f"  Test Accuracy: {test_acc:.2f}%")
    
    # Utwórz DataFrame z wynikami
    results_df = pd.DataFrame(results)
    
    # Zapisz zbiorczy raport CSV
    csv_path = os.path.join(RESULTS_DIR, 'experiments_summary.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nZapisano podsumowanie eksperymentów: {csv_path}")
    
    # Wygeneruj wykres porównawczy
    if len(results) > 1:
        plot_experiments_comparison(results_df)
    
    # Wyświetl podsumowanie
    print("\n" + "=" * 70)
    print("PODSUMOWANIE EKSPERYMENTÓW")
    print("=" * 70)
    
    summary_cols = ['experiment_name', 'test_accuracy', 'best_val_accuracy', 'epochs_trained']
    print(results_df[summary_cols].to_string(index=False))
    
    # Znajdź najlepszy eksperyment
    best_idx = results_df['test_accuracy'].idxmax()
    best_exp = results_df.loc[best_idx]
    
    print(f"\n{'='*70}")
    print(f"NAJLEPSZY EKSPERYMENT: {best_exp['experiment_name']}")
    print(f"Test Accuracy: {best_exp['test_accuracy']:.2f}%")
    print(f"{'='*70}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Uruchom eksperymenty CNN Fashion MNIST')
    parser.add_argument('--experiments', nargs='+', default=None,
                        choices=list(EXPERIMENTS.keys()),
                        help='Lista eksperymentów do uruchomienia (domyślnie wszystkie)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Pomiń eksperymenty z istniejącymi wynikami')
    
    args = parser.parse_args()
    
    run_all_experiments(
        experiments=args.experiments,
        skip_existing=args.skip_existing
    )


if __name__ == "__main__":
    main()

