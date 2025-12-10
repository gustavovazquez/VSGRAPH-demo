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
from multiprocessing import Pool, cpu_count
from functools import partial


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
    n_jobs : int
        Number of parallel jobs for encoding graphs (default: -1, uses all CPU cores)
        Set to 1 for sequential processing
    use_vectorization : bool
        If True, use optimized vectorized operations (default: True)
        Set to False to use original non-vectorized implementation for comparison
    """
    
    def __init__(
        self,
        dimension: int = 8192,
        diffusion_hops: int = 3,
        message_passing_layers: int = 2,
        blend_factor: float = 0.5,
        seed: Optional[int] = None,
        n_jobs: int = -1,
        use_vectorization: bool = True
    ):
        self.D = dimension
        self.K = diffusion_hops
        self.L = message_passing_layers
        self.alpha = blend_factor
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        self.use_vectorization = use_vectorization  # Feature flag for vectorization

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
    
    def _spike_diffusion_original(self, graph: nx.Graph) -> np.ndarray:
        """
        Original (non-vectorized) spike diffusion implementation.
        Used when use_vectorization=False for performance comparison.
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
        ranks = np.argsort(-spikes)  # Descending order
        rank_array = np.empty(n, dtype=np.int32)
        rank_array[ranks] = np.arange(n)

        return rank_array

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
        # Use original implementation if vectorization is disabled
        if not self.use_vectorization:
            return self._spike_diffusion_original(graph)

        # Vectorized implementation
        n = graph.number_of_nodes()

        # Initialize unit spikes (line 2)
        spikes = np.ones(n, dtype=np.float64)

        # Get adjacency matrix as sparse matrix for vectorized operations
        adj_matrix = nx.adjacency_matrix(graph, nodelist=range(n))

        # Diffusion iterations using matrix multiplication (vectorized)
        # new_spikes = A @ spikes (adjacency matrix times spike vector)
        for hop in range(self.K):
            spikes = adj_matrix @ spikes

        # Assign ranks based on spike values (line 9)
        # Higher spike value → lower rank (0 is highest)
        ranks = np.argsort(-spikes)  # Descending order
        rank_array = np.empty(n, dtype=np.int32)
        rank_array[ranks] = np.arange(n)

        return rank_array
    
    def _associative_message_passing_original(
        self,
        graph: nx.Graph,
        initial_hypervectors: np.ndarray
    ) -> np.ndarray:
        """
        Original (non-optimized) message passing implementation.
        Used when use_vectorization=False for performance comparison.
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
        # Use original implementation if vectorization is disabled
        if not self.use_vectorization:
            return self._associative_message_passing_original(graph, initial_hypervectors)

        # Optimized implementation
        n = graph.number_of_nodes()

        # Initialize with rank-based hypervectors (line 10-12)
        h = initial_hypervectors.astype(np.float32)

        # Get adjacency matrix for vectorized operations
        adj_matrix = nx.adjacency_matrix(graph, nodelist=range(n))

        # Message passing layers (lines 13-19)
        for layer in range(self.L):
            # For each node, compute max of neighbors' hypervectors
            messages = np.zeros((n, self.D), dtype=np.float32)

            # Vectorized neighbor aggregation using sparse matrix
            for i in range(n):
                # Get neighbors using adjacency matrix
                # Convert sparse row to array and find non-zero indices
                row = adj_matrix[i].toarray().flatten()
                neighbors = np.nonzero(row)[0]
                if len(neighbors) > 0:
                    # Logical OR: take max across neighbors for each dimension
                    messages[i] = np.max(h[neighbors], axis=0)

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
    
    def encode_graphs(self, graphs: List[nx.Graph], verbose: bool = False, n_jobs: Optional[int] = None) -> np.ndarray:
        """
        Encode multiple graphs with parallel processing support.

        Parameters
        ----------
        graphs : list of nx.Graph
            List of graphs to encode
        verbose : bool
            If True, print progress
        n_jobs : int, optional
            Number of parallel jobs. If None, uses self.n_jobs.
            Set to 1 for sequential processing.

        Returns
        -------
        np.ndarray
            Graph embeddings, shape (num_graphs, D)
        """
        num_graphs = len(graphs)
        embeddings = np.zeros((num_graphs, self.D), dtype=np.float32)

        start_time = time.time()

        # Determine number of workers
        workers = n_jobs if n_jobs is not None else self.n_jobs

        if workers == 1 or num_graphs < 10:
            # Sequential processing for small datasets or when explicitly requested
            for idx, graph in enumerate(graphs):
                embeddings[idx] = self.encode_graph(graph)

                if verbose and (idx + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (idx + 1) * 1000  # ms
                    print(f"Encoded {idx + 1}/{num_graphs} graphs "
                          f"(avg: {avg_time:.2f} ms/graph)")
        else:
            # Parallel processing for larger datasets
            if verbose:
                print(f"Using {workers} parallel workers for encoding {num_graphs} graphs")

            with Pool(processes=workers) as pool:
                # Use imap for better progress tracking
                results = pool.imap(self._encode_graph_wrapper, graphs)

                for idx, embedding in enumerate(results):
                    embeddings[idx] = embedding

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

    def _encode_graph_wrapper(self, graph: nx.Graph) -> np.ndarray:
        """
        Wrapper method for parallel encoding.
        Needed because multiprocessing requires picklable functions.

        Parameters
        ----------
        graph : nx.Graph
            Graph to encode

        Returns
        -------
        np.ndarray
            Graph embedding
        """
        return self.encode_graph(graph)
