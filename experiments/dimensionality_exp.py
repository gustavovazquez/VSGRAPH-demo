"""
Dimensionality Reduction Experiment

Tests VS-Graph robustness across different hypervector dimensions.
Reproduces Figure 2 from the paper.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vsgraph import VSGraphEncoder, VSGraphEvaluator, load_tudataset


# Dimensionality values from paper (Figure 2)
DIMENSIONS = [128, 256, 512, 1024, 2048, 4096, 8192]

DATASETS = ['MUTAG', 'PTC_FM', 'PROTEINS', 'DD', 'NCI1']


def run_dimensionality_experiment(
    dataset_name: str,
    dimensions: list = None,
    diffusion_hops: int = 3,
    message_passing_layers: int = 2,
    blend_factor: float = 0.5,
    data_dir: str = './data',
    n_folds: int = 10,
    n_repeats: int = 3,
    seed: int = 42,
):
    """
    Run dimensionality reduction experiment on a single dataset.
    
    Parameters
    ----------
    dataset_name : str
        Name of dataset
    dimensions : list
        List of dimensions to test
    ... (other params same as run_experiments.py)
        
    Returns
    -------
    pd.DataFrame
        Results for each dimension
    """
    if dimensions is None:
        dimensions = DIMENSIONS
    
    print(f"\n{'='*80}")
    print(f"Dimensionality Experiment: {dataset_name}")
    print(f"Testing dimensions: {dimensions}")
    print(f"{'='*80}")
    
    # Load dataset once
    print(f"\nLoading {dataset_name}...")
    graphs, labels, num_classes = load_tudataset(dataset_name, root_dir=data_dir)
    
    results = []
    
    for dim in dimensions:
        print(f"\n{'-'*80}")
        print(f"Dimension: {dim}")
        print(f"{'-'*80}")
        
        # Create encoder with this dimension
        encoder = VSGraphEncoder(
            dimension=dim,
            diffusion_hops=diffusion_hops,
            message_passing_layers=message_passing_layers,
            blend_factor=blend_factor,
            seed=seed
        )
        
        # Evaluate
        evaluator = VSGraphEvaluator(
            encoder=encoder,
            n_folds=n_folds,
            n_repeats=n_repeats,
            seed=seed
        )
        
        result = evaluator.evaluate(
            graphs=graphs,
            labels=labels,
            num_classes=num_classes,
            verbose=False
        )
        
        # Store results
        results.append({
            'dimension': dim,
            'accuracy_mean': result['accuracy_mean'],
            'accuracy_std': result['accuracy_std'],
            'train_time_mean': result['train_time_mean'],
            'train_time_std': result['train_time_std'],
            'inference_time_mean': result['inference_time_mean'],
            'inference_time_std': result['inference_time_std'],
        })
        
        print(f"Accuracy: {result['accuracy_mean']:.2f} ± {result['accuracy_std']:.2f}%")
        print(f"Train time: {result['train_time_mean']:.3f} ± {result['train_time_std']:.3f} ms/graph")
        print(f"Inference time: {result['inference_time_mean']:.3f} ± {result['inference_time_std']:.3f} ms/graph")
    
    return pd.DataFrame(results)


def plot_dimensionality_results(
    results_dict: dict,
    output_dir: str = './results/plots'
):
    """
    Plot dimensionality experiment results (similar to Figure 2 in paper).
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping dataset names to results DataFrames
    output_dir : str
        Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subplots for each dataset
    n_datasets = len(results_dict)
    fig, axes = plt.subplots(1, n_datasets, figsize=(5*n_datasets, 4))
    
    if n_datasets == 1:
        axes = [axes]
    
    for idx, (dataset_name, df) in enumerate(results_dict.items()):
        ax = axes[idx]
        
        # Plot accuracy vs dimension
        ax.errorbar(
            df['dimension'],
            df['accuracy_mean'],
            yerr=df['accuracy_std'],
            marker='o',
            capsize=5,
            label='VS-Graph'
        )
        
        ax.set_xlabel('Hypervector Dimension (D)')
        ax.set_ylabel('Classification Accuracy (%)')
        ax.set_title(dataset_name)
        ax.set_xscale('log', base=2)
        ax.set_xticks(DIMENSIONS)
        ax.set_xticklabels([str(d) for d in DIMENSIONS], rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'dimensionality_accuracy.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nAccuracy plot saved to: {plot_file}")
    
    # Create timing plots
    fig, axes = plt.subplots(1, n_datasets, figsize=(5*n_datasets, 4))
    
    if n_datasets == 1:
        axes = [axes]
    
    for idx, (dataset_name, df) in enumerate(results_dict.items()):
        ax = axes[idx]
        
        # Plot train and inference time vs dimension
        ax.plot(df['dimension'], df['train_time_mean'], 
                marker='s', label='Train Time')
        ax.plot(df['dimension'], df['inference_time_mean'], 
                marker='^', label='Inference Time')
        
        ax.set_xlabel('Hypervector Dimension (D)')
        ax.set_ylabel('Time (ms/graph)')
        ax.set_title(dataset_name)
        ax.set_xscale('log', base=2)
        ax.set_xticks(DIMENSIONS)
        ax.set_xticklabels([str(d) for d in DIMENSIONS], rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'dimensionality_timing.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Timing plot saved to: {plot_file}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Run dimensionality reduction experiments'
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['MUTAG'],
        choices=DATASETS + ['all'],
        help='Datasets to test (default: MUTAG)'
    )
    parser.add_argument(
        '--dimensions',
        nargs='+',
        type=int,
        default=None,
        help=f'Dimensions to test (default: {DIMENSIONS})'
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
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate plots'
    )
    
    args = parser.parse_args()
    
    # Handle dataset selection
    if 'all' in args.datasets:
        datasets = DATASETS
    else:
        datasets = args.datasets
    
    # Handle dimensions
    dimensions = args.dimensions if args.dimensions else DIMENSIONS
    
    # Run experiments
    results_dict = {}
    
    for dataset_name in datasets:
        df = run_dimensionality_experiment(
            dataset_name=dataset_name,
            dimensions=dimensions,
            diffusion_hops=args.diffusion_hops,
            message_passing_layers=args.message_passing_layers,
            blend_factor=args.blend_factor,
            data_dir=args.data_dir,
            n_folds=args.n_folds,
            n_repeats=args.n_repeats,
            seed=args.seed,
        )
        
        results_dict[dataset_name] = df
        
        # Save individual results
        os.makedirs(args.results_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_file = os.path.join(
            args.results_dir,
            f'dimensionality_{dataset_name}_{timestamp}.csv'
        )
        df.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")
    
    # Generate plots if requested
    if args.plot:
        plot_dimensionality_results(
            results_dict,
            output_dir=os.path.join(args.results_dir, 'plots')
        )
    
    print(f"\n{'='*80}")
    print("DIMENSIONALITY EXPERIMENTS COMPLETED")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
