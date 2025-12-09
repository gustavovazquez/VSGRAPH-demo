"""
TUDataset Data Loader

This module handles loading and preprocessing of graph classification datasets
from the TUDataset collection.

Datasets used in paper: MUTAG, PTC_FM, PROTEINS, DD, NCI1
"""

import os
import urllib.request
import zipfile
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional
import pickle


class TUDatasetLoader:
    """
    Loader for TUDataset graph classification benchmarks.
    
    Parameters
    ----------
    root_dir : str
        Root directory for dataset storage
    dataset_name : str
        Name of dataset (MUTAG, PTC_FM, PROTEINS, DD, NCI1, ENZYMES)
    """
    
    DATASET_URLS = {
        'MUTAG': 'https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip',
        'PTC_FM': 'https://www.chrsmrrs.com/graphkerneldatasets/PTC_FM.zip',
        'PROTEINS': 'https://www.chrsmrrs.com/graphkerneldatasets/PROTEINS.zip',
        'DD': 'https://www.chrsmrrs.com/graphkerneldatasets/DD.zip',
        'NCI1': 'https://www.chrsmrrs.com/graphkerneldatasets/NCI1.zip',
        'ENZYMES': 'https://www.chrsmrrs.com/graphkerneldatasets/ENZYMES.zip',
    }
    
    def __init__(self, root_dir: str = './data', dataset_name: str = 'MUTAG'):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(root_dir, dataset_name)
        
        # Create directories if they don't exist
        os.makedirs(self.root_dir, exist_ok=True)
        
        self.graphs: List[nx.Graph] = []
        self.labels: np.ndarray = np.array([])
        self.num_classes: int = 0
    
    def download(self, force: bool = False):
        """
        Download dataset from TUDataset if not already present.
        
        Parameters
        ----------
        force : bool
            If True, re-download even if dataset exists
        """
        if self.dataset_name not in self.DATASET_URLS:
            raise ValueError(f"Unknown dataset: {self.dataset_name}. "
                           f"Available: {list(self.DATASET_URLS.keys())}")
        
        zip_path = os.path.join(self.root_dir, f"{self.dataset_name}.zip")
        
        # Check if already downloaded
        if os.path.exists(self.dataset_dir) and not force:
            print(f"Dataset {self.dataset_name} already exists at {self.dataset_dir}")
            return
        
        # Download
        print(f"Downloading {self.dataset_name}...")
        url = self.DATASET_URLS[self.dataset_name]
        
        try:
            urllib.request.urlretrieve(url, zip_path)
            print(f"Downloaded to {zip_path}")
            
            # Extract
            print(f"Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.root_dir)
            
            print(f"Extracted to {self.dataset_dir}")
            
            # Clean up zip file
            os.remove(zip_path)
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise
    
    def load(self, use_node_attrs: bool = False) -> Tuple[List[nx.Graph], np.ndarray]:
        """
        Load dataset from disk.
        
        Parameters
        ----------
        use_node_attrs : bool
            If True, include node attributes (default: False for topology-only)
            
        Returns
        -------
        graphs : list of nx.Graph
            List of graphs
        labels : np.ndarray
            Graph labels
        """
        # File paths
        edges_file = os.path.join(self.dataset_dir, f"{self.dataset_name}_A.txt")
        graph_indicator_file = os.path.join(self.dataset_dir, 
                                           f"{self.dataset_name}_graph_indicator.txt")
        labels_file = os.path.join(self.dataset_dir, 
                                  f"{self.dataset_name}_graph_labels.txt")
        
        if not os.path.exists(edges_file):
            raise FileNotFoundError(f"Dataset files not found. Please run download() first.")
        
        # Load edges
        edges = np.loadtxt(edges_file, delimiter=',', dtype=np.int32)
        
        # Load graph indicator (which graph each node belongs to)
        graph_indicator = np.loadtxt(graph_indicator_file, dtype=np.int32)
        
        # Load labels
        labels = np.loadtxt(labels_file, dtype=np.int32)
        
        # Convert labels to 0-indexed
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[label] for label in labels])
        
        self.num_classes = len(unique_labels)
        
        # Build graphs
        num_graphs = len(labels)
        graphs = [nx.Graph() for _ in range(num_graphs)]
        
        # Map global node IDs to graph-local node IDs
        node_to_graph = {}
        graph_node_counters = [0] * num_graphs
        
        for global_node_id, graph_id in enumerate(graph_indicator, start=1):
            graph_idx = graph_id - 1  # Convert to 0-indexed
            local_node_id = graph_node_counters[graph_idx]
            node_to_graph[global_node_id] = (graph_idx, local_node_id)
            graphs[graph_idx].add_node(local_node_id)
            graph_node_counters[graph_idx] += 1
        
        # Add edges
        for edge in edges:
            src_global, dst_global = edge
            src_graph_idx, src_local = node_to_graph[src_global]
            dst_graph_idx, dst_local = node_to_graph[dst_global]
            
            # Sanity check: edges should be within same graph
            assert src_graph_idx == dst_graph_idx, "Edge spans multiple graphs"
            
            graphs[src_graph_idx].add_edge(src_local, dst_local)
        
        self.graphs = graphs
        self.labels = labels
        
        print(f"Loaded {len(graphs)} graphs from {self.dataset_name}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Average nodes: {np.mean([g.number_of_nodes() for g in graphs]):.2f}")
        print(f"Average edges: {np.mean([g.number_of_edges() for g in graphs]):.2f}")
        
        return graphs, labels
    
    def get_dataset_stats(self) -> Dict:
        """
        Get statistics about the loaded dataset.
        
        Returns
        -------
        dict
            Dataset statistics
        """
        if not self.graphs:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        
        num_nodes = [g.number_of_nodes() for g in self.graphs]
        num_edges = [g.number_of_edges() for g in self.graphs]
        
        stats = {
            'name': self.dataset_name,
            'num_graphs': len(self.graphs),
            'num_classes': self.num_classes,
            'avg_nodes': np.mean(num_nodes),
            'std_nodes': np.std(num_nodes),
            'min_nodes': np.min(num_nodes),
            'max_nodes': np.max(num_nodes),
            'avg_edges': np.mean(num_edges),
            'std_edges': np.std(num_edges),
            'min_edges': np.min(num_edges),
            'max_edges': np.max(num_edges),
            'class_distribution': {
                i: np.sum(self.labels == i) for i in range(self.num_classes)
            }
        }
        
        return stats


def load_tudataset(dataset_name: str, root_dir: str = './data') -> Tuple[List[nx.Graph], np.ndarray, int]:
    """
    Convenience function to download and load a TUDataset.
    
    Parameters
    ----------
    dataset_name : str
        Name of dataset
    root_dir : str
        Root directory for data storage
        
    Returns
    -------
    graphs : list of nx.Graph
        Loaded graphs
    labels : np.ndarray
        Graph labels
    num_classes : int
        Number of classes
    """
    loader = TUDatasetLoader(root_dir=root_dir, dataset_name=dataset_name)
    
    # Download if needed
    try:
        loader.download()
    except:
        pass  # Might already be downloaded
    
    # Load
    graphs, labels = loader.load()
    
    return graphs, labels, loader.num_classes
