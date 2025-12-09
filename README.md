# VS-Graph: Scalable and Efficient Graph Classification Using Hyperdimensional Computing

Implementation of the **VS-Graph** algorithm from the paper:

> **VS-Graph: Scalable and Efficient Graph Classification Using Hyperdimensional Computing**  
> *Hamed Poursiami, Shay Snyder, Guojing Cong, Thomas Potok, Maryam Parsa*  
> arXiv:2512.03394v1 [cs.LG]  
> December 2025

## Overview

VS-Graph is a novel vector-symbolic graph learning framework that combines the computational efficiency of Hyperdimensional Computing (HDC) with the expressive power of message passing. Unlike traditional Graph Neural Networks (GNNs) that require gradient-based optimization, VS-Graph achieves competitive accuracy while being up to **450× faster** in training.

### Key Features

- **Spike Diffusion**: Topology-driven node identification mechanism
- **Associative Message Passing**: Multi-hop neighborhood aggregation in hyperdimensional space
- **Prototype Classification**: Non-parametric classification using class prototypes
- **No Backpropagation**: Weight-free architecture, single encoding pass
- **Dimension Robustness**: Maintains high accuracy from D=8192 down to D=128

### Performance Highlights

| Dataset | Accuracy | Training Speedup vs GIN |
|---------|----------|------------------------|
| MUTAG   | 88.47%   | 336×                   |
| PTC_FM  | 60.37%   | 244×                   |
| PROTEINS| 73.29%   | 214×                   |
| DD      | 76.46%   | 34×                    |
| NCI1    | 63.19%   | 452×                   |

## Installation

### Requirements

- Python 3.7+
- NumPy >= 1.21.0
- NetworkX >= 2.6.0
- scikit-learn >= 0.24.0
- matplotlib >= 3.4.0 (for plotting)
- pandas >= 1.3.0 (for results)

### Install from source

```bash
git clone https://github.com/yourusername/vsgraph.git
cd vsgraph
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from vsgraph import VSGraphEncoder, PrototypeClassifier, load_tudataset

# Load dataset
graphs, labels, num_classes = load_tudataset('MUTAG')

# Create encoder
encoder = VSGraphEncoder(
    dimension=8192,           # Hypervector dimensionality
    diffusion_hops=3,         # Spike diffusion iterations (K)
    message_passing_layers=2, # Message passing layers (L)
    blend_factor=0.5,         # Residual blend factor (α)
)

# Encode graphs
embeddings = encoder.encode_graphs(graphs, verbose=True)

# Train classifier
classifier = PrototypeClassifier(num_classes=num_classes)
classifier.fit(embeddings, labels)

# Predict
predictions = classifier.predict(embeddings)
```

### Run Full Evaluation

```python
from vsgraph import VSGraphEncoder, VSGraphEvaluator, load_tudataset

# Load data
graphs, labels, num_classes = load_tudataset('MUTAG')

# Create encoder and evaluator
encoder = VSGraphEncoder(dimension=8192)
evaluator = VSGraphEvaluator(
    encoder=encoder,
    n_folds=10,    # 10-fold cross-validation
    n_repeats=3    # Repeat 3 times
)

# Evaluate
results = evaluator.evaluate(graphs, labels, num_classes, verbose=True)

print(f"Accuracy: {results['accuracy_mean']:.2f} ± {results['accuracy_std']:.2f}%")
print(f"Train time: {results['train_time_mean']:.3f} ms/graph")
print(f"Inference time: {results['inference_time_mean']:.3f} ms/graph")
```

## Running Experiments

### Reproduce Paper Results

Run VS-Graph on all benchmark datasets:

```bash
cd experiments
python run_experiments.py --datasets all
```

Run on specific datasets:

```bash
python run_experiments.py --datasets MUTAG PROTEINS NCI1
```

### Dimensionality Reduction Experiments

Test robustness across different hypervector dimensions (reproduces Figure 2):

```bash
python dimensionality_exp.py --datasets MUTAG --plot
```

Test all dimensions from 128 to 8192:

```bash
python dimensionality_exp.py --datasets all --dimensions 128 256 512 1024 2048 4096 8192 --plot
```

### Hyperparameter Tuning

Experiment with different configurations:

```bash
# Try different spike diffusion hops
python run_experiments.py --datasets MUTAG --diffusion-hops 5

# Try different message passing layers
python run_experiments.py --datasets MUTAG --message-passing-layers 3

# Try different blend factors
python run_experiments.py --datasets MUTAG --blend-factor 0.7

# Use smaller dimension for faster experiments
python run_experiments.py --datasets MUTAG --dimension 1024
```

## Algorithm Details

### Spike Diffusion (Algorithm 1, Lines 2-9)

Computes topology-based node identifiers:

1. Initialize unit spikes: `s_i ← 1` for all nodes
2. For K diffusion hops:
   - Aggregate neighbor spikes: `s_i ← Σ_{j∈N(i)} s_j`
3. Rank nodes by spike values
4. Assign basis hypervectors based on ranks

### Associative Message Passing (Algorithm 1, Lines 13-19)

Multi-hop aggregation in hyperdimensional space:

1. For L message passing layers:
   - Aggregate neighbors with logical OR: `m_i^(l) = ∨_{j∈N(i)} h_j^(l)`
   - Residual blend update: `h_i^(l+1) = α·h_i^(l) + (1-α)·m_i^(l)`

### Graph-Level Readout (Algorithm 1, Line 20)

Average pooling over node hypervectors:

```
z_G = (1/|V|) Σ_{i∈V} h_i^(L)
```

### Prototype Classification (Algorithm 1, Lines 22-29)

**Training:**
- For each class c: `p_c = normalize(mean(z_G for G in class c))`

**Inference:**
- Predict: `ŷ = argmax_c (z_G^T · p_c)` (cosine similarity)

## Project Structure

```
vsgraph/
├── vsgraph/
│   ├── __init__.py           # Package initialization
│   ├── encoder.py            # VS-Graph encoder (Spike Diffusion + Message Passing)
│   ├── classifier.py         # Prototype classifier
│   ├── data_loader.py        # TUDataset loader
│   ├── evaluator.py          # Cross-validation framework
│   └── utils.py              # Utility functions
├── baselines/
│   └── graphhd.py            # GraphHD baseline implementation
├── experiments/
│   ├── run_experiments.py    # Main benchmark experiments
│   ├── dimensionality_exp.py # Dimensionality reduction study
│   └── timing_analysis.py    # Performance profiling
├── results/                  # Experiment results and plots
├── tests/                    # Unit tests
├── requirements.txt
├── README.md
└── LICENSE
```

## Datasets

The implementation supports all TUDataset benchmarks used in the paper:

| Dataset  | Graphs | Classes | Avg. Nodes | Avg. Edges | Domain       |
|----------|--------|---------|------------|------------|--------------|
| MUTAG    | 188    | 2       | 17.93      | 19.79      | Chemistry    |
| PTC_FM   | 349    | 2       | 14.11      | 14.48      | Chemistry    |
| PROTEINS | 1113   | 2       | 39.06      | 72.82      | Biology      |
| DD       | 1178   | 2       | 284.32     | 715.66     | Biology      |
| NCI1     | 4110   | 2       | 29.87      | 32.30      | Chemistry    |

Datasets are automatically downloaded from the TUDataset collection.

## Hyperparameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Dimension | D      | 8192    | Hypervector dimensionality |
| Diffusion Hops | K | 3    | Number of spike diffusion iterations |
| Message Passing Layers | L | 2 | Number of message passing layers |
| Blend Factor | α | 0.5   | Residual connection weight (0-1) |

**Note**: The paper doesn't explicitly specify K, L, α values. Default values are based on common HDC practices and preliminary experiments.

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{poursiami2025vsgraph,
  title={VS-Graph: Scalable and Efficient Graph Classification Using Hyperdimensional Computing},
  author={Poursiami, Hamed and Snyder, Shay and Cong, Guojing and Potok, Thomas and Parsa, Maryam},
  journal={arXiv preprint arXiv:2512.03394},
  year={2025}
}
```

And the GraphHD baseline:

```bibtex
@inproceedings{nunes2022graphhd,
  title={Graphhd: Efficient graph classification using hyperdimensional computing},
  author={Nunes, Igor and Heddes, Mike and Givargis, Tony and Nicolau, Alexandru and Veidenbaum, Alex},
  booktitle={2022 Design, Automation \& Test in Europe Conference \& Exhibition (DATE)},
  pages={1485--1490},
  year={2022},
  organization={IEEE}
}
```

## Comparison to Baselines

### VS-Graph vs GraphHD

- **Accuracy**: VS-Graph outperforms GraphHD by 4-5% on MUTAG and DD
- **Node Identification**: Spike Diffusion (VS-Graph) vs PageRank (GraphHD)
- **Aggregation**: Multi-hop message passing vs single edge binding
- **Robustness**: VS-Graph maintains accuracy at D=128; GraphHD degrades significantly

### VS-Graph vs GNNs

- **Training Speed**: 250-450× faster on average
- **Accuracy**: Competitive or better on MUTAG, PROTEINS, DD
- **Simplicity**: No gradient descent, no learnable parameters
- **Hardware**: Suitable for edge/neuromorphic devices

## Future Work

Potential extensions and improvements:

1. **Node/Edge Attributes**: Incorporate feature information when available
2. **Hyperparameter Optimization**: Automatic tuning of K, L, α
3. **GPU Acceleration**: Parallel encoding for large-scale datasets
4. **Neuromorphic Hardware**: Deploy on spiking neural network accelerators
5. **Other Graph Tasks**: Extend to node classification, link prediction

## License

MIT License - see LICENSE file for details

## Acknowledgments

This implementation is based on the paper "VS-Graph: Scalable and Efficient Graph Classification Using Hyperdimensional Computing" by Poursiami et al. (2025).

The research was funded in part by National Science Foundation (CCF2319619) and Department of Energy (DE-SC0025349, DE-AC05-00OR22725).

## Contact

For questions or issues, please open a GitHub issue or contact the repository maintainer.
