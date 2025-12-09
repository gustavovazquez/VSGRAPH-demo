"""
Evaluation Framework for VS-Graph

Implements 10-fold cross-validation with performance metrics.
Based on evaluation protocol from Section IV-B of the paper.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import StratifiedKFold
import time

from vsgraph.encoder import VSGraphEncoder
from vsgraph.classifier import PrototypeClassifier


class VSGraphEvaluator:
    """
    Evaluator for VS-Graph with k-fold cross-validation.
    
    Parameters
    ----------
    encoder : VSGraphEncoder
        VS-Graph encoder instance
    n_folds : int
        Number of cross-validation folds (default: 10)
    n_repeats : int
        Number of times to repeat CV (default: 3)
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        encoder: VSGraphEncoder,
        n_folds: int = 10,
        n_repeats: int = 3,
        seed: Optional[int] = 42
    ):
        self.encoder = encoder
        self.n_folds = n_folds
        self.n_repeats = n_repeats
        self.seed = seed
    
    def evaluate(
        self,
        graphs: List[nx.Graph],
        labels: np.ndarray,
        num_classes: int,
        verbose: bool = True
    ) -> Dict:
        """
        Perform k-fold cross-validation evaluation.
        
        Parameters
        ----------
        graphs : list of nx.Graph
            Input graphs
        labels : np.ndarray
            Graph labels
        num_classes : int
            Number of classes
        verbose : bool
            If True, print progress
            
        Returns
        -------
        dict
            Evaluation results with accuracy, training time, inference time
        """
        all_accuracies = []
        all_train_times = []
        all_inference_times = []
        
        # Convert graphs to numpy array for easier indexing
        graphs = np.array(graphs, dtype=object)
        
        for repeat in range(self.n_repeats):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Repeat {repeat + 1}/{self.n_repeats}")
                print(f"{'='*60}")
            
            # Create stratified k-fold splits
            seed = self.seed + repeat if self.seed is not None else None
            skf = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=seed
            )
            
            fold_accuracies = []
            fold_train_times = []
            fold_inference_times = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(graphs, labels)):
                if verbose:
                    print(f"\nFold {fold_idx + 1}/{self.n_folds}")
                
                # Split data
                train_graphs = graphs[train_idx].tolist()
                test_graphs = graphs[test_idx].tolist()
                train_labels = labels[train_idx]
                test_labels = labels[test_idx]
                
                # === TRAINING ===
                train_start = time.time()
                
                # Encode training graphs
                train_embeddings = self.encoder.encode_graphs(
                    train_graphs,
                    verbose=False
                )
                
                # Train classifier
                classifier = PrototypeClassifier(
                    num_classes=num_classes,
                    normalize=True
                )
                classifier.fit(train_embeddings, train_labels, verbose=False)
                
                train_time = time.time() - train_start
                train_time_per_graph = train_time / len(train_graphs) * 1000  # ms
                fold_train_times.append(train_time_per_graph)
                
                # === INFERENCE ===
                inference_start = time.time()
                
                # Encode test graphs
                test_embeddings = self.encoder.encode_graphs(
                    test_graphs,
                    verbose=False
                )
                
                # Predict
                predictions = classifier.predict(test_embeddings, verbose=False)
                
                inference_time = time.time() - inference_start
                inference_time_per_graph = inference_time / len(test_graphs) * 1000  # ms
                fold_inference_times.append(inference_time_per_graph)
                
                # Compute accuracy
                accuracy = np.mean(predictions == test_labels) * 100
                fold_accuracies.append(accuracy)
                
                if verbose:
                    print(f"  Accuracy: {accuracy:.2f}%")
                    print(f"  Train time: {train_time_per_graph:.3f} ms/graph")
                    print(f"  Inference time: {inference_time_per_graph:.3f} ms/graph")
            
            # Aggregate fold results for this repeat
            repeat_accuracy = np.mean(fold_accuracies)
            repeat_train_time = np.mean(fold_train_times)
            repeat_inference_time = np.mean(fold_inference_times)
            
            all_accuracies.append(repeat_accuracy)
            all_train_times.append(repeat_train_time)
            all_inference_times.append(repeat_inference_time)
            
            if verbose:
                print(f"\nRepeat {repeat + 1} Results:")
                print(f"  Mean Accuracy: {repeat_accuracy:.2f}%")
                print(f"  Mean Train Time: {repeat_train_time:.3f} ms/graph")
                print(f"  Mean Inference Time: {repeat_inference_time:.3f} ms/graph")
        
        # Final aggregated results
        results = {
            'accuracy_mean': np.mean(all_accuracies),
            'accuracy_std': np.std(all_accuracies),
            'accuracies': all_accuracies,
            'train_time_mean': np.mean(all_train_times),
            'train_time_std': np.std(all_train_times),
            'train_times': all_train_times,
            'inference_time_mean': np.mean(all_inference_times),
            'inference_time_std': np.std(all_inference_times),
            'inference_times': all_inference_times,
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"FINAL RESULTS (averaged over {self.n_repeats} repeats)")
            print(f"{'='*60}")
            print(f"Accuracy: {results['accuracy_mean']:.2f} ± {results['accuracy_std']:.2f}%")
            print(f"Train Time: {results['train_time_mean']:.3f} ± {results['train_time_std']:.3f} ms/graph")
            print(f"Inference Time: {results['inference_time_mean']:.3f} ± {results['inference_time_std']:.3f} ms/graph")
        
        return results


def quick_test(
    graphs: List[nx.Graph],
    labels: np.ndarray,
    num_classes: int,
    encoder_params: Optional[Dict] = None,
    test_size: float = 0.2,
    seed: int = 42
) -> Dict:
    """
    Quick single train/test split for debugging.
    
    Parameters
    ----------
    graphs : list of nx.Graph
        Input graphs
    labels : np.ndarray
        Graph labels
    num_classes : int
        Number of classes
    encoder_params : dict, optional
        Parameters for VSGraphEncoder
    test_size : float
        Fraction of data to use for testing
    seed : int
        Random seed
        
    Returns
    -------
    dict
        Test results
    """
    from sklearn.model_selection import train_test_split
    
    # Split data
    train_idx, test_idx = train_test_split(
        np.arange(len(graphs)),
        test_size=test_size,
        stratify=labels,
        random_state=seed
    )
    
    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    
    # Create encoder
    if encoder_params is None:
        encoder_params = {}
    encoder = VSGraphEncoder(**encoder_params)
    
    # Train
    print("Encoding training graphs...")
    train_embeddings = encoder.encode_graphs(train_graphs, verbose=True)
    
    print("\nTraining classifier...")
    classifier = PrototypeClassifier(num_classes=num_classes)
    classifier.fit(train_embeddings, train_labels, verbose=True)
    
    # Test
    print("\nEncoding test graphs...")
    test_embeddings = encoder.encode_graphs(test_graphs, verbose=True)
    
    print("\nPredicting...")
    predictions = classifier.predict(test_embeddings, verbose=True)
    
    accuracy = np.mean(predictions == test_labels) * 100
    
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'true_labels': test_labels
    }
