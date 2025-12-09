"""
VS-Graph: Scalable and Efficient Graph Classification Using Hyperdimensional Computing

Implementation of the VS-Graph algorithm from:
"VS-Graph: Scalable and Efficient Graph Classification Using Hyperdimensional Computing"
arXiv:2512.03394v1

Main components:
- VSGraphEncoder: Spike diffusion + associative message passing
- PrototypeClassifier: Prototype-based classification
- TUDatasetLoader: Dataset loading utilities
- VSGraphEvaluator: Evaluation framework
"""

from vsgraph.encoder import VSGraphEncoder
from vsgraph.classifier import PrototypeClassifier
from vsgraph.data_loader import TUDatasetLoader, load_tudataset
from vsgraph.evaluator import VSGraphEvaluator, quick_test

__version__ = '1.0.0'
__author__ = 'VS-Graph Implementation'

__all__ = [
    'VSGraphEncoder',
    'PrototypeClassifier',
    'TUDatasetLoader',
    'load_tudataset',
    'VSGraphEvaluator',
    'quick_test',
]
