"""
Prototype Classifier Implementation

This module implements the prototype-based classification method from VS-Graph.

Based on Algorithm 1, lines 22-29 from paper: "VS-Graph: Scalable and Efficient
Graph Classification Using Hyperdimensional Computing" (arXiv:2512.03394v1)
"""

import numpy as np
from typing import List, Optional
import time


class PrototypeClassifier:
    """
    Prototype-based classifier using cosine similarity.
    
    Training creates one prototype per class by averaging embeddings.
    Inference assigns the class with maximum cosine similarity.
    
    Parameters
    ----------
    num_classes : int
        Number of classes
    normalize : bool
        Whether to L2-normalize embeddings and prototypes (default: True)
    epsilon : float
        Small constant for numerical stability (default: 1e-8)
    """
    
    def __init__(
        self,
        num_classes: int,
        normalize: bool = True,
        epsilon: float = 1e-8
    ):
        self.num_classes = num_classes
        self.normalize = normalize
        self.epsilon = epsilon
        
        self.prototypes: Optional[np.ndarray] = None
        self.class_counts: Optional[np.ndarray] = None
        self.is_fitted = False
    
    def _l2_normalize(self, vectors: np.ndarray) -> np.ndarray:
        """
        L2 normalize vectors.
        
        Parameters
        ----------
        vectors : np.ndarray
            Vectors to normalize, shape (n, D) or (D,)
            
        Returns
        -------
        np.ndarray
            Normalized vectors
        """
        if vectors.ndim == 1:
            norm = np.linalg.norm(vectors)
            return vectors / (norm + self.epsilon)
        else:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors / (norms + self.epsilon)
    
    def fit(self, embeddings: np.ndarray, labels: np.ndarray, verbose: bool = False):
        """
        Train the classifier by constructing class prototypes.
        
        Algorithm 1, lines 22-26:
        For each class c:
          - p'_c = (1/N_c) Σ_{G_k ∈ D_c} z_{G_k}
          - p_c = p'_c / (||p'_c||_2 + ε)
        
        Parameters
        ----------
        embeddings : np.ndarray
            Graph embeddings, shape (num_graphs, D)
        labels : np.ndarray
            Class labels, shape (num_graphs,)
        verbose : bool
            If True, print training info
        """
        start_time = time.time()
        
        num_graphs, D = embeddings.shape
        
        # Initialize prototypes
        self.prototypes = np.zeros((self.num_classes, D), dtype=np.float32)
        self.class_counts = np.zeros(self.num_classes, dtype=np.int32)
        
        # Compute prototype for each class (lines 22-26)
        for c in range(self.num_classes):
            # Get graphs belonging to class c
            class_mask = (labels == c)
            class_embeddings = embeddings[class_mask]
            
            N_c = class_embeddings.shape[0]
            self.class_counts[c] = N_c
            
            if N_c > 0:
                # Average embeddings: p'_c = (1/N_c) Σ z_G
                p_prime = np.mean(class_embeddings, axis=0)
                
                # Normalize: p_c = p'_c / (||p'_c||_2 + ε)
                if self.normalize:
                    self.prototypes[c] = self._l2_normalize(p_prime)
                else:
                    self.prototypes[c] = p_prime
            else:
                if verbose:
                    print(f"Warning: Class {c} has no training samples")
        
        self.is_fitted = True
        
        if verbose:
            train_time = time.time() - start_time
            avg_time = train_time / num_graphs * 1000  # ms per graph
            print(f"Training completed in {train_time:.3f}s")
            print(f"Average training time: {avg_time:.3f} ms/graph")
            print(f"Class distribution: {self.class_counts}")
    
    def predict(self, embeddings: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Predict class labels for graph embeddings.
        
        Algorithm 1, lines 27-29:
        - z_G = z_G / (||z_G||_2 + ε)
        - ŷ = argmax_c (z_G^T · p_c)
        
        Parameters
        ----------
        embeddings : np.ndarray
            Graph embeddings to classify, shape (num_graphs, D)
        verbose : bool
            If True, print inference info
            
        Returns
        -------
        np.ndarray
            Predicted labels, shape (num_graphs,)
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction")
        
        start_time = time.time()
        
        num_graphs = embeddings.shape[0]
        
        # Normalize test embeddings (line 28)
        if self.normalize:
            normalized_embeddings = self._l2_normalize(embeddings)
        else:
            normalized_embeddings = embeddings
        
        # Compute cosine similarity with all prototypes (line 29)
        # Shape: (num_graphs, num_classes)
        similarities = normalized_embeddings @ self.prototypes.T
        
        # Predict class with maximum similarity
        predictions = np.argmax(similarities, axis=1)
        
        if verbose:
            inference_time = time.time() - start_time
            avg_time = inference_time / num_graphs * 1000  # ms per graph
            print(f"Inference completed in {inference_time:.3f}s")
            print(f"Average inference time: {avg_time:.3f} ms/graph")
        
        return predictions
    
    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using softmax of cosine similarities.
        
        Parameters
        ----------
        embeddings : np.ndarray
            Graph embeddings, shape (num_graphs, D)
            
        Returns
        -------
        np.ndarray
            Class probabilities, shape (num_graphs, num_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction")
        
        # Normalize test embeddings
        if self.normalize:
            normalized_embeddings = self._l2_normalize(embeddings)
        else:
            normalized_embeddings = embeddings
        
        # Compute cosine similarity with all prototypes
        similarities = normalized_embeddings @ self.prototypes.T
        
        # Apply softmax
        exp_sim = np.exp(similarities - np.max(similarities, axis=1, keepdims=True))
        probabilities = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)
        
        return probabilities
    
    def score(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute classification accuracy.
        
        Parameters
        ----------
        embeddings : np.ndarray
            Graph embeddings, shape (num_graphs, D)
        labels : np.ndarray
            True labels, shape (num_graphs,)
            
        Returns
        -------
        float
            Accuracy score
        """
        predictions = self.predict(embeddings)
        accuracy = np.mean(predictions == labels)
        return accuracy
