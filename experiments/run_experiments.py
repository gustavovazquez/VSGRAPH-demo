"""
Main Experiment Script

Run VS-Graph on all benchmark datasets from the paper.
Reproduces results from Tables II-III and Figure 1.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vsgraph import VSGraphEncoder, VSGraphEvaluator, load_tudataset


# Datasets from paper (Table I)
DATASETS = ['MUTAG', 'PTC_FM', 'PROTEINS', 'DD', 'NCI1']

# Expected results from paper (Figure 1)
PAPER_RESULTS = {
    'MUTAG': 88.47,
    'PTC_FM': 60.37,
    'PROTEINS': 73.29,
    'DD': 76.46,
    'NCI1': 63.19,
}


def run_single_dataset(
    dataset_name: str,
    dimension: int = 8192,
    diffusion_hops: int = 3,
    message_passing_layers: int = 2,
    blend_factor: float = 0.5,
    data_dir: str = './data',
    n_folds: int = 10,
    n_repeats: int = 3,
    seed: int = 42,
    verbose: bool = True
):
    """
    Run VS-Graph evaluation on a single dataset.
    
    Parameters
    ----------
    dataset_name : str
        Name of dataset
    dimension : int
        Hypervector dimensionality
    diffusion_hops : int
        Number of spike diffusion hops (K)
    message_passing_layers : int
        Number of message passing layers (L)
    blend_factor : float
        Residual blend factor (α)
    data_dir : str
        Data directory
    n_folds : int
        Number of CV folds
    n_repeats : int
        Number of CV repeats
    seed : int
        Random seed
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Evaluation results
    """
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*80}")
    
    # Load dataset
    print(f"\nLoading {dataset_name}...")
    graphs, labels, num_classes = load_tudataset(dataset_name, root_dir=data_dir)
    
    print(f"Number of graphs: {len(graphs)}")
    print(f"Number of classes: {num_classes}")
    
    # Create encoder
    encoder = VSGraphEncoder(
        dimension=dimension,
        diffusion_hops=diffusion_hops,
        message_passing_layers=message_passing_layers,
        blend_factor=blend_factor,
        seed=seed
    )
    
    print(f"\nEncoder configuration:")
    print(f"  Dimension (D): {dimension}")
    print(f"  Diffusion hops (K): {diffusion_hops}")
    print(f"  Message passing layers (L): {message_passing_layers}")
    print(f"  Blend factor (α): {blend_factor}")
    
    # Create evaluator
    evaluator = VSGraphEvaluator(
        encoder=encoder,
        n_folds=n_folds,
        n_repeats=n_repeats,
        seed=seed
    )
    
    # Run evaluation
    results = evaluator.evaluate(
        graphs=graphs,
        labels=labels,
        num_classes=num_classes,
        verbose=verbose
    )
    
    # Add dataset info to results
    results['dataset'] = dataset_name
    results['dimension'] = dimension
    results['diffusion_hops'] = diffusion_hops
    results['message_passing_layers'] = message_passing_layers
    results['blend_factor'] = blend_factor
    
    # Compare to paper results
    if dataset_name in PAPER_RESULTS:
        paper_acc = PAPER_RESULTS[dataset_name]
        our_acc = results['accuracy_mean']
        diff = our_acc - paper_acc
        
        print(f"\n{'='*80}")
        print(f"Comparison to Paper Results:")
        print(f"  Paper accuracy: {paper_acc:.2f}%")
        print(f"  Our accuracy: {our_acc:.2f}%")
        print(f"  Difference: {diff:+.2f}%")
        print(f"{'='*80}")
    
    return results


def run_all_datasets(
    datasets: list = None,
    dimension: int = 8192,
    diffusion_hops: int = 3,
    message_passing_layers: int = 2,
    blend_factor: float = 0.5,
    data_dir: str = './data',
    results_dir: str = './results',
    n_folds: int = 10,
    n_repeats: int = 3,
    seed: int = 42,
):
    """
    Run VS-Graph on all datasets and save results.
    """
    if datasets is None:
        datasets = DATASETS
    
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = []
    
    start_time = time.time()
    
    for dataset_name in datasets:
        try:
            results = run_single_dataset(
                dataset_name=dataset_name,
                dimension=dimension,
                diffusion_hops=diffusion_hops,
                message_passing_layers=message_passing_layers,
                blend_factor=blend_factor,
                data_dir=data_dir,
                n_folds=n_folds,
                n_repeats=n_repeats,
                seed=seed,
                verbose=True
            )
            all_results.append(results)
            
        except Exception as e:
            print(f"\nError running {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    total_time = time.time() - start_time
    
    # Create results DataFrame
    results_df = pd.DataFrame([{
        'Dataset': r['dataset'],
        'Accuracy (%)': f"{r['accuracy_mean']:.2f} ± {r['accuracy_std']:.2f}",
        'Train Time (ms)': f"{r['train_time_mean']:.3f} ± {r['train_time_std']:.3f}",
        'Inference Time (ms)': f"{r['inference_time_mean']:.3f} ± {r['inference_time_std']:.3f}",
        'Paper Accuracy (%)': f"{PAPER_RESULTS.get(r['dataset'], 'N/A')}",
    } for r in all_results])
    
    # Save results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_file = os.path.join(results_dir, f'results_{timestamp}.csv')
    results_df.to_csv(results_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETED")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"{'='*80}")
    print("\nResults Summary:")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {results_file}")
    
    return all_results, results_df


def main():
    parser = argparse.ArgumentParser(
        description='Run VS-Graph experiments on TUDataset benchmarks'
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=None,
        choices=DATASETS + ['all'],
        help='Datasets to run (default: all)'
    )
    parser.add_argument(
        '--dimension', '-d',
        type=int,
        default=8192,
        help='Hypervector dimension (default: 8192)'
    )
    parser.add_argument(
        '--diffusion-hops', '-k',
        type=int,
        default=3,
        help='Number of diffusion hops (default: 3)'
    )
    parser.add_argument(
        '--message-passing-layers', '-l',
        type=int,
        default=2,
        help='Number of message passing layers (default: 2)'
    )
    parser.add_argument(
        '--blend-factor', '-a',
        type=float,
        default=0.5,
        help='Blend factor alpha (default: 0.5)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Data directory (default: ./data)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='./results',
        help='Results directory (default: ./results)'
    )
    parser.add_argument(
        '--n-folds',
        type=int,
        default=10,
        help='Number of CV folds (default: 10)'
    )
    parser.add_argument(
        '--n-repeats',
        type=int,
        default=3,
        help='Number of CV repeats (default: 3)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Handle dataset selection
    if args.datasets is None or 'all' in args.datasets:
        datasets = DATASETS
    else:
        datasets = args.datasets
    
    # Run experiments
    run_all_datasets(
        datasets=datasets,
        dimension=args.dimension,
        diffusion_hops=args.diffusion_hops,
        message_passing_layers=args.message_passing_layers,
        blend_factor=args.blend_factor,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        n_folds=args.n_folds,
        n_repeats=args.n_repeats,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
