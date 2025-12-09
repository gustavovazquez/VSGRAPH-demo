"""
VS-Graph Encoder Implementation

This module implements the core VS-Graph encoding pipeline:
1. Spike Diffusion: Topology-driven node identification
2. Associative Message Passing: Multi-hop neighborhood aggregation
3. Graph-Level Readout: Global pooling

Based on Algorithm 1 from paper: "VS-Graph: Scalable and Efficient Graph
Classification Using Hyperdimensional Computing" (arXiv:2512.03394v1)
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import time


class VSGraphEncoder:
    """
    VS-Graph encoder implementing spike diffusion and associative message passing.
    
    Parameters
    ----------
    dimension : int
        Dimensionality of hypervectors (default: 8192)
    diffusion_hops : int
        Number of spike diffusion iterations (K in paper)
    message_passing_layers : int
        Number of message passing layers (L in paper)
    blend_factor : float
        Residual blend factor α ∈ [0,1] for message passing
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        dimension: int = 8192,
        diffusion_hops: int = 3,
        message_passing_layers: int = 2,
        blend_factor: float = 0.5,
        seed: Optional[int] = None
    ):
        self.D = dimension
        self.K = diffusion_hops
        self.L = message_passing_layers
        self.alpha = blend_factor
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize basis hypervectors for rank encoding
        # We'll create basis vectors on-demand based on max rank seen
        self.basis_memory: Dict[int, np.ndarray] = {}
        
    def _get_basis_hypervector(self, rank: int) -> np.ndarray:
        """
        Get or create a binary basis hypervector for a given rank.
        
        Parameters
        ----------
        rank : int
            Rank index
            
        Returns
        -------
        np.ndarray
            Binary hypervector {0,1}^D
        """
        if rank not in self.basis_memory:
            # Generate random binary hypervector
            self.basis_memory[rank] = np.random.randint(0, 2, size=self.D, dtype=np.int8)
        return self.basis_memory[rank]
    
    def spike_diffusion(self, graph: nx.Graph) -> np.ndarray:
        """
        Perform spike diffusion to obtain topology-based node rankings.
        
        Algorithm 1, lines 2-9:
        - Initialize unit spikes s_i ← 1 for all nodes
        - For K hops: s_i ← Σ_{j∈N(i)} s_j
        - Rank nodes by final spike values
        
        Parameters
        ----------
        graph : nx.Graph
            Input graph
            
        Returns
        -------
        np.ndarray
            Rank array where ranks[i] is the rank of node i
        """
        n = graph.number_of_nodes()
        
        # Initialize unit spikes (line 2)
        spikes = np.ones(n, dtype=np.float64)
        
        # Create adjacency list for efficient neighbor access
        adj_list = {i: list(graph.neighbors(i)) for i in range(n)}
        
        # Diffusion iterations (lines 3-8)
        for hop in range(self.K):
            new_spikes = np.zeros(n, dtype=np.float64)
            
            # For each node, aggregate spikes from neighbors (lines 5-6)
            for i in range(n):
                for j in adj_list[i]:
                    new_spikes[i] += spikes[j]
            
            spikes = new_spikes
        
        # Assign ranks based on spike values (line 9)
        # Higher spike value → lower rank (0 is highest)
        ranks = np.argsort(-spikes)  # Descending order
        rank_array = np.empty(n, dtype=np.int32)
        rank_array[ranks] = np.arange(n)
        
        return rank_array
    
    def associative_message_passing(
        self,
        graph: nx.Graph,
        initial_hypervectors: np.ndarray
    ) -> np.ndarray:
        """
        Perform associative message passing with logical OR aggregation.
        
        Algorithm 1, lines 13-19:
        - For L layers:
          - m_i^(l) = ∨_{j∈N(i)} h_j^(l)  (logical OR)
          - h_i^(l+1) = α·h_i^(l) + (1-α)·m_i^(l)  (residual blend)
        
        Parameters
        ----------
        graph : nx.Graph
            Input graph
        initial_hypervectors : np.ndarray
            Initial node hypervectors from spike diffusion, shape (n, D)
            
        Returns
        -------
        np.ndarray
            Final node hypervectors after L layers, shape (n, D)
        """
        n = graph.number_of_nodes()
        
        # Initialize with rank-based hypervectors (line 10-12)
        h = initial_hypervectors.astype(np.float32)
        
        # Create adjacency list
        adj_list = {i: list(graph.neighbors(i)) for i in range(n)}
        
        # Message passing layers (lines 13-19)
        for layer in range(self.L):
            messages = np.zeros((n, self.D), dtype=np.float32)
            
            # Aggregate neighbors with logical OR (lines 15-16)
            for i in range(n):
                if len(adj_list[i]) > 0:
                    # Logical OR: take max across neighbors for each dimension
                    neighbor_hvs = h[adj_list[i]]
                    messages[i] = np.max(neighbor_hvs, axis=0)
            
            # Residual blend update (lines 17-18)
            h = self.alpha * h + (1 - self.alpha) * messages
        
        return h
    
    def graph_level_readout(self, node_hypervectors: np.ndarray) -> np.ndarray:
        """
        Perform graph-level readout via average pooling.
        
        Algorithm 1, line 20:
        z_G = (1/|V|) Σ_{i∈V} h_i^(L)
        
        Parameters
        ----------
        node_hypervectors : np.ndarray
            Final node hypervectors, shape (n, D)
            
        Returns
        -------
        np.ndarray
            Graph-level embedding, shape (D,)
        """
        return np.mean(node_hypervectors, axis=0)
    
    def encode_graph(self, graph: nx.Graph) -> np.ndarray:
        """
        Encode a graph into a hyperdimensional representation.
        
        Complete pipeline from Algorithm 1:
        1. Spike diffusion (lines 2-9)
        2. Rank-based hypervector assignment (lines 10-12)
        3. Associative message passing (lines 13-19)
        4. Graph-level readout (line 20)
        
        Parameters
        ----------
        graph : nx.Graph
            Input graph with nodes labeled 0 to n-1
            
        Returns
        -------
        np.ndarray
            Graph embedding z_G ∈ R^D
        """
        n = graph.number_of_nodes()
        
        # Step 1: Spike Diffusion (lines 2-9)
        ranks = self.spike_diffusion(graph)
        
        # Step 2: Assign basis hypervectors based on ranks (lines 10-12)
        initial_hvs = np.zeros((n, self.D), dtype=np.int8)
        for i in range(n):
            initial_hvs[i] = self._get_basis_hypervector(ranks[i])
        
        # Step 3: Associative Message Passing (lines 13-19)
        final_hvs = self.associative_message_passing(graph, initial_hvs)
        
        # Step 4: Graph-Level Readout (line 20)
        graph_embedding = self.graph_level_readout(final_hvs)
        
        return graph_embedding
    
    def encode_graphs(self, graphs: List[nx.Graph], verbose: bool = False) -> np.ndarray:
        """
        Encode multiple graphs.
        
        Parameters
        ----------
        graphs : list of nx.Graph
            List of graphs to encode
        verbose : bool
            If True, print progress
            
        Returns
        -------
        np.ndarray
            Graph embeddings, shape (num_graphs, D)
        """
        num_graphs = len(graphs)
        embeddings = np.zeros((num_graphs, self.D), dtype=np.float32)
        
        start_time = time.time()
        
        for idx, graph in enumerate(graphs):
            embeddings[idx] = self.encode_graph(graph)
            
            if verbose and (idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (idx + 1) * 1000  # ms
                print(f"Encoded {idx + 1}/{num_graphs} graphs "
                      f"(avg: {avg_time:.2f} ms/graph)")
        
        if verbose:
            total_time = time.time() - start_time
            avg_time = total_time / num_graphs * 1000
            print(f"Total encoding time: {total_time:.2f}s "
                  f"(avg: {avg_time:.3f} ms/graph)")
        
        return embeddings
