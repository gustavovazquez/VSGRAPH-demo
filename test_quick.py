"""
Quick Test Script

Simple test to verify the installation and basic functionality.
"""

import sys
import numpy as np
import networkx as nx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vsgraph import VSGraphEncoder, PrototypeClassifier


def create_simple_graphs():
    """Create simple test graphs."""
    # Create a few simple graphs for testing
    graphs = []
    labels = []
    
    # Class 0: Small graphs with few edges
    for _ in range(10):
        G = nx.erdos_renyi_graph(5, 0.3, seed=42)
        G = nx.convert_node_labels_to_integers(G)
        graphs.append(G)
        labels.append(0)
    
    # Class 1: Larger graphs with more edges
    for _ in range(10):
        G = nx.erdos_renyi_graph(10, 0.5, seed=42)
        G = nx.convert_node_labels_to_integers(G)
        graphs.append(G)
        labels.append(1)
    
    return graphs, np.array(labels)


def test_encoder():
    """Test VS-Graph encoder."""
    print("\n" + "="*60)
    print("Test 1: VS-Graph Encoder")
    print("="*60)
    
    # Create simple graph
    G = nx.karate_club_graph()
    G = nx.convert_node_labels_to_integers(G)
    
    print(f"Test graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Create encoder
    encoder = VSGraphEncoder(
        dimension=1024,
        diffusion_hops=3,
        message_passing_layers=2,
        blend_factor=0.5,
        seed=42
    )
    
    print("\nEncoding graph...")
    embedding = encoder.encode_graph(G)
    
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding dtype: {embedding.dtype}")
    print(f"Embedding range: [{embedding.min():.3f}, {embedding.max():.3f}]")
    
    assert embedding.shape == (1024,), "Incorrect embedding shape"
    
    print("\n✓ Encoder test passed!")
    
    return encoder


def test_classifier():
    """Test Prototype Classifier."""
    print("\n" + "="*60)
    print("Test 2: Prototype Classifier")
    print("="*60)
    
    # Create dummy embeddings
    np.random.seed(42)
    
    # 20 embeddings, 2 classes
    embeddings = np.random.randn(20, 512).astype(np.float32)
    labels = np.array([0]*10 + [1]*10)
    
    print(f"Training data: {embeddings.shape}, {len(np.unique(labels))} classes")
    
    # Train classifier
    classifier = PrototypeClassifier(num_classes=2)
    classifier.fit(embeddings, labels, verbose=True)
    
    # Test prediction
    predictions = classifier.predict(embeddings)
    accuracy = np.mean(predictions == labels) * 100
    
    print(f"\nTrain accuracy: {accuracy:.2f}%")
    
    assert accuracy > 50, "Classifier performs worse than random"
    
    print("\n✓ Classifier test passed!")
    
    return classifier


def test_pipeline():
    """Test complete pipeline."""
    print("\n" + "="*60)
    print("Test 3: Complete Pipeline")
    print("="*60)
    
    # Create simple graphs
    graphs, labels = create_simple_graphs()
    
    print(f"Dataset: {len(graphs)} graphs, {len(np.unique(labels))} classes")
    
    # Split train/test
    split = 15
    train_graphs = graphs[:split]
    test_graphs = graphs[split:]
    train_labels = labels[:split]
    test_labels = labels[split:]
    
    print(f"Train: {len(train_graphs)} graphs")
    print(f"Test: {len(test_graphs)} graphs")
    
    # Create encoder
    encoder = VSGraphEncoder(
        dimension=512,
        diffusion_hops=2,
        message_passing_layers=1,
        blend_factor=0.5,
        seed=42
    )
    
    # Encode
    print("\nEncoding training graphs...")
    train_embeddings = encoder.encode_graphs(train_graphs, verbose=False)
    
    print("Encoding test graphs...")
    test_embeddings = encoder.encode_graphs(test_graphs, verbose=False)
    
    # Train classifier
    print("\nTraining classifier...")
    classifier = PrototypeClassifier(num_classes=2)
    classifier.fit(train_embeddings, train_labels, verbose=False)
    
    # Predict
    print("Testing...")
    predictions = classifier.predict(test_embeddings)
    accuracy = np.mean(predictions == test_labels) * 100
    
    print(f"\nTest accuracy: {accuracy:.2f}%")
    
    print("\n✓ Pipeline test passed!")


def main():
    print("\n" + "="*60)
    print("VS-Graph Quick Test")
    print("="*60)
    
    try:
        # Run tests
        test_encoder()
        test_classifier()
        test_pipeline()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nVS-Graph installation is working correctly!")
        print("You can now run experiments on real datasets.")
        print("\nNext steps:")
        print("  1. Run experiments: python experiments/run_experiments.py --datasets MUTAG")
        print("  2. Test dimensionality: python experiments/dimensionality_exp.py --datasets MUTAG --plot")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
