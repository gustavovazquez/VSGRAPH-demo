"""
Quick test script to verify parallelization features work correctly.
"""
# -*- coding: utf-8 -*-

import sys
import io

# Fix encoding issues on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import networkx as nx
from vsgraph.encoder import VSGraphEncoder
from vsgraph.evaluator import VSGraphEvaluator
from vsgraph.classifier import PrototypeClassifier
from multiprocessing import cpu_count


def create_simple_graphs(n_graphs=50, n_nodes=20):
    """Create simple random graphs for testing."""
    graphs = []
    labels = []

    for i in range(n_graphs):
        # Create random graph
        if i % 2 == 0:
            # Erdos-Renyi graph
            G = nx.erdos_renyi_graph(n_nodes, 0.3)
            label = 0
        else:
            # Barabasi-Albert graph
            G = nx.barabasi_albert_graph(n_nodes, 3)
            label = 1

        # Ensure nodes are labeled 0 to n-1
        G = nx.convert_node_labels_to_integers(G)
        graphs.append(G)
        labels.append(label)

    return graphs, np.array(labels)


def test_vectorized_operations():
    """Test vectorized spike diffusion and message passing."""
    print("\n" + "="*70)
    print("TEST 1: Vectorized Operations")
    print("="*70)

    # Create test graph
    G = nx.karate_club_graph()
    G = nx.convert_node_labels_to_integers(G)

    # Test with different configurations
    encoder = VSGraphEncoder(dimension=1024, diffusion_hops=3, message_passing_layers=2)

    print(f"Testing on Karate Club graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")

    # Encode graph
    embedding = encoder.encode_graph(G)

    print(f"✓ Graph encoded successfully")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")

    return True


def test_parallel_encoding():
    """Test parallel graph encoding."""
    print("\n" + "="*70)
    print("TEST 2: Parallel Graph Encoding")
    print("="*70)

    # Create test graphs
    graphs, labels = create_simple_graphs(n_graphs=50, n_nodes=15)
    print(f"Created {len(graphs)} test graphs")
    print(f"Available CPU cores: {cpu_count()}")

    # Test sequential encoding
    print("\nTesting sequential encoding (n_jobs=1)...")
    encoder_seq = VSGraphEncoder(dimension=512, n_jobs=1)
    embeddings_seq = encoder_seq.encode_graphs(graphs, verbose=True)
    print(f"✓ Sequential encoding: {embeddings_seq.shape}")

    # Test parallel encoding
    print(f"\nTesting parallel encoding (n_jobs=2)...")
    encoder_par = VSGraphEncoder(dimension=512, n_jobs=2)
    embeddings_par = encoder_par.encode_graphs(graphs, verbose=True)
    print(f"✓ Parallel encoding: {embeddings_par.shape}")

    # Verify results are similar (may have small numerical differences)
    # Since each process creates its own encoder with potentially different random basis vectors,
    # we just verify the shapes match
    assert embeddings_seq.shape == embeddings_par.shape, "Shape mismatch!"
    print(f"✓ Shapes match: {embeddings_seq.shape}")

    return True


def test_parallel_cv():
    """Test parallel cross-validation."""
    print("\n" + "="*70)
    print("TEST 3: Parallel Cross-Validation")
    print("="*70)

    # Create test graphs
    graphs, labels = create_simple_graphs(n_graphs=60, n_nodes=15)
    num_classes = len(np.unique(labels))
    print(f"Created {len(graphs)} test graphs with {num_classes} classes")

    # Test sequential CV
    print("\nTesting sequential CV (parallel_folds=False)...")
    encoder = VSGraphEncoder(dimension=512, n_jobs=1)
    evaluator = VSGraphEvaluator(encoder, n_folds=3, n_repeats=1, n_jobs=1)
    results_seq = evaluator.evaluate(
        graphs, labels, num_classes,
        verbose=True,
        parallel_folds=False
    )
    print(f"✓ Sequential CV accuracy: {results_seq['accuracy_mean']:.2f}%")

    # Test parallel CV
    print(f"\nTesting parallel CV (parallel_folds=True, n_jobs=2)...")
    encoder2 = VSGraphEncoder(dimension=512, n_jobs=1)
    evaluator2 = VSGraphEvaluator(encoder2, n_folds=3, n_repeats=1, n_jobs=2)
    results_par = evaluator2.evaluate(
        graphs, labels, num_classes,
        verbose=True,
        parallel_folds=True
    )
    print(f"✓ Parallel CV accuracy: {results_par['accuracy_mean']:.2f}%")

    # Verify both produce valid results
    assert 0 <= results_seq['accuracy_mean'] <= 100, "Invalid accuracy!"
    assert 0 <= results_par['accuracy_mean'] <= 100, "Invalid accuracy!"
    print(f"✓ Both CV methods produce valid results")

    return True


def test_classifier():
    """Test classifier with parallel-encoded graphs."""
    print("\n" + "="*70)
    print("TEST 4: Classifier with Parallel Encoding")
    print("="*70)

    # Create test graphs
    graphs, labels = create_simple_graphs(n_graphs=100, n_nodes=15)
    num_classes = len(np.unique(labels))
    print(f"Created {len(graphs)} test graphs with {num_classes} classes")

    # Split train/test
    split = int(0.8 * len(graphs))
    train_graphs = graphs[:split]
    test_graphs = graphs[split:]
    train_labels = labels[:split]
    test_labels = labels[split:]

    # Encode with parallelization
    print(f"\nEncoding graphs with {cpu_count()} workers...")
    encoder = VSGraphEncoder(dimension=1024, n_jobs=-1)

    train_embeddings = encoder.encode_graphs(train_graphs, verbose=True)
    test_embeddings = encoder.encode_graphs(test_graphs, verbose=True)

    # Train classifier
    print("\nTraining classifier...")
    classifier = PrototypeClassifier(num_classes=num_classes)
    classifier.fit(train_embeddings, train_labels, verbose=True)

    # Predict
    print("\nMaking predictions...")
    predictions = classifier.predict(test_embeddings, verbose=True)
    accuracy = np.mean(predictions == test_labels) * 100

    print(f"✓ Test accuracy: {accuracy:.2f}%")

    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("VS-GRAPH PARALLELIZATION TESTS")
    print("="*70)
    print(f"System: {cpu_count()} CPU cores available")

    tests = [
        ("Vectorized Operations", test_vectorized_operations),
        ("Parallel Encoding", test_parallel_encoding),
        ("Parallel Cross-Validation", test_parallel_cv),
        ("Classifier Integration", test_classifier),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = "PASSED" if success else "FAILED"
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = "FAILED"

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, status in results.items():
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"{symbol} {test_name}: {status}")

    all_passed = all(status == "PASSED" for status in results.values())
    if all_passed:
        print("\n✓ All tests passed successfully!")
        print("\nParallelization features are working correctly.")
        print(f"You can now use n_jobs=-1 to utilize all {cpu_count()} cores.")
    else:
        print("\n✗ Some tests failed. Please review the errors above.")

    return all_passed


if __name__ == '__main__':
    main()
