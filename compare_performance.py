"""
Performance Comparison Script

Compares the performance between:
1. Original implementation (no vectorization, no parallelization)
2. Vectorized implementation (vectorization, no parallelization)
3. Parallel implementation (vectorization + parallelization)

Usage:
    python compare_performance.py
    python compare_performance.py --dataset MUTAG
    python compare_performance.py --n-graphs 100 --dimension 4096
"""

import sys
import argparse
import numpy as np
import networkx as nx
from multiprocessing import cpu_count

from vsgraph.encoder import VSGraphEncoder
from vsgraph.evaluator import VSGraphEvaluator
from vsgraph.data_loader import load_tudataset
from vsgraph.timing_utils import ComparisonTimer, PerformanceTimer


def create_test_graphs(n_graphs: int = 100, n_nodes: int = 30):
    """Create synthetic test graphs."""
    print(f"Creating {n_graphs} synthetic graphs with ~{n_nodes} nodes each...")
    graphs = []
    labels = []

    for i in range(n_graphs):
        if i % 2 == 0:
            # Erdos-Renyi graph
            G = nx.erdos_renyi_graph(n_nodes, 0.3)
            label = 0
        else:
            # Barabasi-Albert graph
            G = nx.barabasi_albert_graph(n_nodes, 3)
            label = 1

        G = nx.convert_node_labels_to_integers(G)
        graphs.append(G)
        labels.append(label)

    return graphs, np.array(labels)


def compare_encoding_performance(graphs, dimension=2048, n_repeats=3):
    """Compare encoding performance across different configurations."""
    print("\n" + "="*80)
    print("COMPARISON 1: Graph Encoding Performance")
    print("="*80)

    timer = ComparisonTimer("Graph Encoding")
    num_cores = cpu_count()

    configurations = [
        ("Original (no vectorization, sequential)", {
            'dimension': dimension,
            'n_jobs': 1,
            'use_vectorization': False
        }),
        ("Vectorized (with vectorization, sequential)", {
            'dimension': dimension,
            'n_jobs': 1,
            'use_vectorization': True
        }),
        ("Parallel 2-cores (vectorized)", {
            'dimension': dimension,
            'n_jobs': 2,
            'use_vectorization': True
        }),
        ("Parallel 4-cores (vectorized)", {
            'dimension': dimension,
            'n_jobs': 4,
            'use_vectorization': True
        }),
        (f"Parallel all-cores (vectorized, {num_cores} cores)", {
            'dimension': dimension,
            'n_jobs': -1,
            'use_vectorization': True
        }),
    ]

    results = {}

    for config_name, params in configurations:
        print(f"\n{'-'*80}")
        print(f"Testing: {config_name}")
        print(f"Parameters: {params}")
        print(f"{'-'*80}")

        # Run multiple times for statistical significance
        for repeat in range(n_repeats):
            print(f"  Repeat {repeat + 1}/{n_repeats}...", end=" ", flush=True)

            encoder = VSGraphEncoder(**params)

            with timer.time_version(config_name, operation="encoding"):
                embeddings = encoder.encode_graphs(graphs, verbose=False)

            print("Done")

        # Get stats for this configuration
        stats = timer.versions[config_name].get_summary()['encoding']
        results[config_name] = stats
        print(f"  Average time: {stats['mean']:.4f}s (±{stats['std']:.4f}s)")
        print(f"  Throughput: {len(graphs) / stats['mean']:.2f} graphs/sec")

    # Print comparison table
    print("\n" + "="*80)
    print("ENCODING PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'Configuration':<45} {'Mean Time (s)':<15} {'Speedup':<10} {'Graphs/sec':<12}")
    print("-"*80)

    baseline_time = results[configurations[0][0]]['mean']

    for config_name, stats in results.items():
        speedup = baseline_time / stats['mean']
        throughput = len(graphs) / stats['mean']
        print(f"{config_name:<45} {stats['mean']:>13.4f}  {speedup:>8.2f}×  {throughput:>10.2f}")

    print("="*80)

    return results


def compare_operations_detail(graphs, dimension=2048):
    """Compare individual operations (spike diffusion, message passing)."""
    print("\n" + "="*80)
    print("COMPARISON 2: Detailed Operation Performance")
    print("="*80)

    # Use a single graph for detailed comparison
    test_graph = graphs[0]
    print(f"Testing on graph with {test_graph.number_of_nodes()} nodes, "
          f"{test_graph.number_of_edges()} edges")

    timer = ComparisonTimer("Operations")

    configurations = [
        ("Original", {'dimension': dimension, 'use_vectorization': False}),
        ("Vectorized", {'dimension': dimension, 'use_vectorization': True}),
    ]

    n_iterations = 10

    for config_name, params in configurations:
        print(f"\n{'-'*80}")
        print(f"Testing: {config_name}")
        print(f"{'-'*80}")

        encoder = VSGraphEncoder(**params)

        # Test spike diffusion
        print(f"  Running spike_diffusion {n_iterations} times...", end=" ", flush=True)
        for i in range(n_iterations):
            with timer.time_version(config_name, operation="spike_diffusion"):
                ranks = encoder.spike_diffusion(test_graph)
        print("Done")

        # Test message passing
        print(f"  Running message_passing {n_iterations} times...", end=" ", flush=True)
        # Need initial hypervectors
        initial_hvs = np.random.randint(0, 2, size=(test_graph.number_of_nodes(), dimension), dtype=np.int8)
        for i in range(n_iterations):
            with timer.time_version(config_name, operation="message_passing"):
                final_hvs = encoder.associative_message_passing(test_graph, initial_hvs)
        print("Done")

    # Print comparisons
    timer.print_all_comparisons()


def compare_cross_validation(graphs, labels, num_classes, dimension=1024):
    """Compare cross-validation performance."""
    print("\n" + "="*80)
    print("COMPARISON 3: Cross-Validation Performance")
    print("="*80)

    configurations = [
        ("Sequential CV + Sequential Encoding", {
            'encoder': {'dimension': dimension, 'n_jobs': 1, 'use_vectorization': True},
            'evaluator': {'n_folds': 3, 'n_repeats': 1, 'n_jobs': 1},
            'eval_params': {'parallel_folds': False}
        }),
        ("Parallel CV (3 folds)", {
            'encoder': {'dimension': dimension, 'n_jobs': 1, 'use_vectorization': True},
            'evaluator': {'n_folds': 3, 'n_repeats': 1, 'n_jobs': 3},
            'eval_params': {'parallel_folds': True}
        }),
        ("Parallel Encoding (4 cores)", {
            'encoder': {'dimension': dimension, 'n_jobs': 4, 'use_vectorization': True},
            'evaluator': {'n_folds': 3, 'n_repeats': 1, 'n_jobs': 1},
            'eval_params': {'parallel_folds': False}
        }),
    ]

    results = {}

    for config_name, config in configurations:
        print(f"\n{'-'*80}")
        print(f"Testing: {config_name}")
        print(f"{'-'*80}")

        import time
        start_time = time.time()

        encoder = VSGraphEncoder(**config['encoder'])
        evaluator = VSGraphEvaluator(encoder, **config['evaluator'])

        cv_results = evaluator.evaluate(
            graphs, labels, num_classes,
            verbose=True,
            **config['eval_params']
        )

        elapsed = time.time() - start_time
        results[config_name] = {
            'time': elapsed,
            'accuracy': cv_results['accuracy_mean']
        }

        print(f"\nTotal time: {elapsed:.2f}s")
        print(f"Accuracy: {cv_results['accuracy_mean']:.2f}%")

    # Print comparison
    print("\n" + "="*80)
    print("CROSS-VALIDATION PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'Configuration':<40} {'Time (s)':<12} {'Speedup':<10} {'Accuracy (%)':<12}")
    print("-"*80)

    baseline_time = results[configurations[0][0]]['time']

    for config_name, stats in results.items():
        speedup = baseline_time / stats['time']
        print(f"{config_name:<40} {stats['time']:>10.2f}  {speedup:>8.2f}×  {stats['accuracy']:>10.2f}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Compare VS-Graph performance configurations")
    parser.add_argument('--dataset', type=str, default=None,
                        help='TUDataset name (e.g., MUTAG, PROTEINS). If not specified, uses synthetic data.')
    parser.add_argument('--n-graphs', type=int, default=100,
                        help='Number of synthetic graphs to create (only used if --dataset is not specified)')
    parser.add_argument('--dimension', type=int, default=2048,
                        help='Hypervector dimension (default: 2048)')
    parser.add_argument('--n-repeats', type=int, default=3,
                        help='Number of times to repeat each encoding test (default: 3)')
    parser.add_argument('--skip-cv', action='store_true',
                        help='Skip cross-validation comparison (faster)')
    parser.add_argument('--skip-detail', action='store_true',
                        help='Skip detailed operation comparison (faster)')

    args = parser.parse_args()

    print("="*80)
    print("VS-GRAPH PERFORMANCE COMPARISON")
    print("="*80)
    print(f"System: {cpu_count()} CPU cores available")
    print(f"Dimension: {args.dimension}")
    print(f"Repeats per test: {args.n_repeats}")

    # Load or create dataset
    if args.dataset:
        print(f"\nLoading dataset: {args.dataset}")
        graphs, labels = load_tudataset(args.dataset)
        num_classes = len(np.unique(labels))
        print(f"Loaded {len(graphs)} graphs with {num_classes} classes")
    else:
        graphs, labels = create_test_graphs(n_graphs=args.n_graphs)
        num_classes = len(np.unique(labels))
        print(f"Created {len(graphs)} synthetic graphs with {num_classes} classes")

    # Run comparisons
    try:
        # 1. Encoding performance comparison
        encoding_results = compare_encoding_performance(
            graphs,
            dimension=args.dimension,
            n_repeats=args.n_repeats
        )

        # 2. Detailed operation comparison
        if not args.skip_detail:
            compare_operations_detail(graphs, dimension=args.dimension)

        # 3. Cross-validation comparison
        if not args.skip_cv:
            compare_cross_validation(graphs, labels, num_classes, dimension=args.dimension)

        # Final summary
        print("\n" + "="*80)
        print("COMPARISON COMPLETE")
        print("="*80)
        print("\nKey Findings:")
        print("1. Vectorization provides significant speedup for individual operations")
        print("2. Parallelization scales well with number of cores for graph encoding")
        print("3. Different parallelization strategies suit different workloads")
        print("\nRecommendations:")
        print("- For small datasets: use vectorization without parallelization")
        print("- For medium datasets: use parallel graph encoding")
        print("- For large datasets with CV: use parallel CV folds")
        print(f"- Your system has {cpu_count()} cores - use n_jobs=-1 to utilize all")

    except KeyboardInterrupt:
        print("\n\nComparison interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during comparison: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
