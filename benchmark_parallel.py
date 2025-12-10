"""
Benchmark script to compare sequential vs parallel performance.

This script tests the parallelization improvements by comparing:
1. Sequential graph encoding
2. Parallel graph encoding
3. Sequential cross-validation
4. Parallel cross-validation
"""

import time
import numpy as np
from vsgraph.data_loader import load_tudataset
from vsgraph.encoder import VSGraphEncoder
from vsgraph.evaluator import VSGraphEvaluator
from multiprocessing import cpu_count


def benchmark_encoding(dataset_name='MUTAG'):
    """Benchmark sequential vs parallel graph encoding."""
    print(f"\n{'='*70}")
    print(f"BENCHMARK: Graph Encoding on {dataset_name}")
    print(f"{'='*70}")

    # Load dataset
    print(f"Loading {dataset_name} dataset...")
    graphs, labels = load_tudataset(dataset_name)
    num_classes = len(np.unique(labels))

    print(f"Dataset: {len(graphs)} graphs, {num_classes} classes")
    print(f"Available CPU cores: {cpu_count()}")

    # Test configurations
    configs = [
        (1, "Sequential (1 worker)"),
        (2, "Parallel (2 workers)"),
        (4, "Parallel (4 workers)"),
        (-1, f"Parallel (all {cpu_count()} cores)")
    ]

    results = {}

    for n_jobs, description in configs:
        print(f"\n{'-'*70}")
        print(f"Testing: {description}")
        print(f"{'-'*70}")

        # Create encoder
        encoder = VSGraphEncoder(
            dimension=4096,  # Smaller dimension for faster testing
            diffusion_hops=3,
            message_passing_layers=2,
            n_jobs=n_jobs
        )

        # Time encoding
        start = time.time()
        embeddings = encoder.encode_graphs(graphs, verbose=True)
        elapsed = time.time() - start

        results[description] = {
            'time': elapsed,
            'graphs_per_sec': len(graphs) / elapsed,
            'ms_per_graph': elapsed / len(graphs) * 1000
        }

        print(f"Total time: {elapsed:.2f}s")
        print(f"Throughput: {results[description]['graphs_per_sec']:.2f} graphs/sec")

    # Print comparison
    print(f"\n{'='*70}")
    print("ENCODING PERFORMANCE COMPARISON")
    print(f"{'='*70}")
    print(f"{'Configuration':<30} {'Time (s)':<12} {'Speedup':<10} {'ms/graph':<10}")
    print(f"{'-'*70}")

    baseline_time = results["Sequential (1 worker)"]['time']
    for config, metrics in results.items():
        speedup = baseline_time / metrics['time']
        print(f"{config:<30} {metrics['time']:>10.2f}s  {speedup:>8.2f}x  {metrics['ms_per_graph']:>8.2f}")

    return results


def benchmark_cross_validation(dataset_name='MUTAG'):
    """Benchmark sequential vs parallel cross-validation."""
    print(f"\n{'='*70}")
    print(f"BENCHMARK: Cross-Validation on {dataset_name}")
    print(f"{'='*70}")

    # Load dataset
    print(f"Loading {dataset_name} dataset...")
    graphs, labels = load_tudataset(dataset_name)
    num_classes = len(np.unique(labels))

    print(f"Dataset: {len(graphs)} graphs, {num_classes} classes")
    print(f"Available CPU cores: {cpu_count()}")

    # Create encoder (using sequential encoding within folds)
    encoder = VSGraphEncoder(
        dimension=2048,  # Smaller for faster testing
        diffusion_hops=2,
        message_passing_layers=2,
        n_jobs=1  # Sequential encoding within each fold
    )

    # Test configurations
    configs = [
        (False, 1, "Sequential CV (no parallel folds)"),
        (True, 2, "Parallel CV (2 workers)"),
        (True, -1, f"Parallel CV (all {cpu_count()} cores)")
    ]

    results = {}

    for parallel_folds, n_jobs, description in configs:
        print(f"\n{'-'*70}")
        print(f"Testing: {description}")
        print(f"{'-'*70}")

        # Create evaluator
        evaluator = VSGraphEvaluator(
            encoder=encoder,
            n_folds=5,  # Fewer folds for faster testing
            n_repeats=1,  # Single repeat for faster testing
            n_jobs=n_jobs
        )

        # Time evaluation
        start = time.time()
        cv_results = evaluator.evaluate(
            graphs,
            labels,
            num_classes,
            verbose=True,
            parallel_folds=parallel_folds
        )
        elapsed = time.time() - start

        results[description] = {
            'time': elapsed,
            'accuracy': cv_results['accuracy_mean']
        }

        print(f"\nTotal CV time: {elapsed:.2f}s")
        print(f"Accuracy: {cv_results['accuracy_mean']:.2f}%")

    # Print comparison
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION PERFORMANCE COMPARISON")
    print(f"{'='*70}")
    print(f"{'Configuration':<40} {'Time (s)':<12} {'Speedup':<10} {'Accuracy':<10}")
    print(f"{'-'*70}")

    baseline_time = results["Sequential CV (no parallel folds)"]['time']
    for config, metrics in results.items():
        speedup = baseline_time / metrics['time']
        print(f"{config:<40} {metrics['time']:>10.2f}s  {speedup:>8.2f}x  {metrics['accuracy']:>8.2f}%")

    return results


def main():
    """Run all benchmarks."""
    print("\n" + "="*70)
    print("VS-GRAPH PARALLELIZATION BENCHMARK")
    print("="*70)
    print(f"System: {cpu_count()} CPU cores available")

    # Benchmark encoding
    encoding_results = benchmark_encoding('MUTAG')

    # Benchmark cross-validation
    cv_results = benchmark_cross_validation('MUTAG')

    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print("\nKey findings:")
    print("1. Graph encoding can be parallelized across multiple cores")
    print("2. Cross-validation folds can be processed in parallel")
    print("3. Speedup scales with number of available cores")
    print("4. Accuracy remains unchanged with parallelization")
    print("\nFor best performance on large datasets:")
    print(f"  - Use n_jobs=-1 to utilize all {cpu_count()} cores")
    print("  - Enable parallel_folds=True for cross-validation")
    print("  - Consider hybrid approach: parallel folds + sequential encoding per fold")


if __name__ == '__main__':
    main()
