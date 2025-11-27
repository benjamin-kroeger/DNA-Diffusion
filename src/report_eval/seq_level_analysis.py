#!/usr/bin/env python3
"""
Script 1: Comprehensive DNA Sequence Metrics Evaluation

Computes sequence-level metrics comparing ground truth vs synthetic sequences:
- Sequence novelty (Novelty@r for various radii)
- Sequence diversity (mean pairwise Hamming distance)
- K-mer frequency similarities (3, 5, 7, 9-mers)
- GC content similarity

Supports:
- Multiple synthetic datasets (for comparing different models)
- Per-cluster analysis (when cluster assignments are provided)
- Publication-quality visualizations

Author: DNA Diffusion Analysis Pipeline
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Union
from scipy.stats import entropy, ttest_ind, mannwhitneyu
from scipy.spatial.distance import jensenshannon
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from tqdm import tqdm
import argparse

warnings.filterwarnings('ignore')

# Set publication-quality plot defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (10, 8),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters."""
    kmer_sizes: List[int] = field(default_factory=lambda: [3, 5, 7, 9])
    novelty_radii: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 15, 20])
    sequence_length: int = 200
    n_diversity_samples: int = 2000  # Sample for pairwise diversity (full is O(n^2))
    random_seed: int = 42


class DNASequenceMetrics:
    """
    Comprehensive DNA sequence metrics calculator.

    Computes novelty, diversity, k-mer frequencies, and GC content
    with support for per-cluster analysis.
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        np.random.seed(self.config.random_seed)

        # Nucleotide encoding for vectorized operations
        self._nuc_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4,
                            'a': 0, 'c': 1, 'g': 2, 't': 3, 'n': 4}

    # ==================== Vectorized Sequence Operations ====================

    def _encode_sequences(self, sequences: List[str]) -> np.ndarray:
        """
        Encode sequences as integer matrix for vectorized operations.

        Args:
            sequences: List of DNA sequences (must be same length)

        Returns:
            np.ndarray of shape (n_sequences, seq_length) with dtype uint8
        """
        n_seqs = len(sequences)
        seq_len = len(sequences[0])

        # Pre-allocate array
        encoded = np.zeros((n_seqs, seq_len), dtype=np.uint8)

        for i, seq in enumerate(sequences):
            for j, nuc in enumerate(seq):
                encoded[i, j] = self._nuc_to_int.get(nuc, 4)

        return encoded

    def _encode_sequences_fast(self, sequences: List[str]) -> np.ndarray:
        """
        Fast encoding using numpy's vectorized string operations.
        """
        # Convert to numpy array of characters
        n_seqs = len(sequences)
        seq_len = len(sequences[0])

        # Create byte array from sequences
        seq_bytes = np.array([list(s.upper()) for s in sequences], dtype='U1')

        # Map to integers
        encoded = np.zeros((n_seqs, seq_len), dtype=np.uint8)
        encoded[seq_bytes == 'A'] = 0
        encoded[seq_bytes == 'C'] = 1
        encoded[seq_bytes == 'G'] = 2
        encoded[seq_bytes == 'T'] = 3
        encoded[seq_bytes == 'N'] = 4

        return encoded

    def _hamming_distance_matrix(self,
                                 seqs1_encoded: np.ndarray,
                                 seqs2_encoded: np.ndarray,
                                 batch_size: int = 1000) -> np.ndarray:
        """
        Compute pairwise Hamming distances between two sets of encoded sequences.

        Uses batched computation to manage memory for large datasets.

        Args:
            seqs1_encoded: (n1, L) encoded sequences
            seqs2_encoded: (n2, L) encoded sequences
            batch_size: Process this many query sequences at a time

        Returns:
            (n1, n2) matrix of Hamming distances
        """
        n1, seq_len = seqs1_encoded.shape
        n2 = seqs2_encoded.shape[0]

        # For small datasets, compute directly
        if n1 * n2 < 10_000_000:  # ~10M comparisons threshold
            # Broadcasting: (n1, 1, L) != (1, n2, L) -> (n1, n2, L)
            # Sum over L to get (n1, n2) distances
            distances = np.sum(seqs1_encoded[:, np.newaxis, :] != seqs2_encoded[np.newaxis, :, :], axis=2)
            return distances

        # For large datasets, batch to avoid memory issues
        distances = np.zeros((n1, n2), dtype=np.int32)

        for i in range(0, n1, batch_size):
            end_i = min(i + batch_size, n1)
            batch1 = seqs1_encoded[i:end_i]

            # Compute distances for this batch
            batch_dist = np.sum(batch1[:, np.newaxis, :] != seqs2_encoded[np.newaxis, :, :], axis=2)
            distances[i:end_i] = batch_dist

        return distances

    def _min_hamming_distances_vectorized(self,
                                          query_encoded: np.ndarray,
                                          reference_encoded: np.ndarray,
                                          batch_size: int = 500,
                                          show_progress: bool = True) -> np.ndarray:
        """
        Compute minimum Hamming distance from each query to any reference.

        Optimized for the novelty computation use case.

        Args:
            query_encoded: (n_query, L) encoded query sequences
            reference_encoded: (n_ref, L) encoded reference sequences
            batch_size: Process this many queries at a time
            show_progress: Show progress bar

        Returns:
            (n_query,) array of minimum distances
        """
        n_query = query_encoded.shape[0]
        n_ref = reference_encoded.shape[0]

        min_distances = np.zeros(n_query, dtype=np.int32)

        # Process in batches
        n_batches = (n_query + batch_size - 1) // batch_size

        if show_progress:
            from tqdm import tqdm
            batch_iter = tqdm(range(n_batches), desc="Computing novelty (vectorized)")
        else:
            batch_iter = range(n_batches)

        for batch_idx in batch_iter:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_query)

            query_batch = query_encoded[start_idx:end_idx]

            # Compute all distances for this batch: (batch_size, n_ref)
            # Using broadcasting: (batch, 1, L) != (1, n_ref, L) -> (batch, n_ref, L)
            # Sum over L -> (batch, n_ref)
            batch_distances = np.sum(
                query_batch[:, np.newaxis, :] != reference_encoded[np.newaxis, :, :],
                axis=2
            )

            # Get minimum for each query in batch
            min_distances[start_idx:end_idx] = np.min(batch_distances, axis=1)

        return min_distances

    # ==================== Core Metric Functions ====================

    @staticmethod
    def compute_gc_content(seq: str) -> float:
        """Compute GC content of a sequence."""
        seq = seq.upper()
        gc_count = seq.count('G') + seq.count('C')
        return gc_count / len(seq) if len(seq) > 0 else 0

    @staticmethod
    def hamming_distance(seq1: str, seq2: str) -> int:
        """Compute Hamming distance between two sequences."""
        return sum(c1 != c2 for c1, c2 in zip(seq1.upper(), seq2.upper()))

    @staticmethod
    def get_kmers(seq: str, k: int) -> List[str]:
        """Extract all k-mers from a sequence."""
        seq = seq.upper()
        return [seq[i:i + k] for i in range(len(seq) - k + 1)]

    def get_kmer_distribution(self, sequences: List[str], k: int) -> Dict[str, float]:
        """Get normalized k-mer frequency distribution."""
        kmer_counts = Counter()
        for seq in sequences:
            kmer_counts.update(self.get_kmers(seq, k))
        total = sum(kmer_counts.values())
        return {kmer: count / total for kmer, count in kmer_counts.items()}

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2 + 1e-10)

    # ==================== Novelty Metrics ====================

    def compute_novelty(self,
                        reference_seqs: List[str],
                        query_seqs: List[str],
                        show_progress: bool = True) -> Dict:
        """
        Compute Novelty@r metrics using vectorized operations.

        Novelty@r = fraction of query sequences with min Hamming distance > r from reference.

        This implementation uses numpy broadcasting for ~100x speedup over naive loops.
        """
        # Validate sequence lengths
        ref_len = len(reference_seqs[0])
        query_len = len(query_seqs[0])

        if ref_len != query_len:
            raise ValueError(f"Reference and query sequences must have same length. "
                             f"Got {ref_len} and {query_len}")

        # Encode sequences for vectorized computation
        if show_progress:
            print("  Encoding sequences...")

        reference_encoded = self._encode_sequences_fast(reference_seqs)
        query_encoded = self._encode_sequences_fast(query_seqs)

        # Compute minimum distances using vectorized method
        min_distances = self._min_hamming_distances_vectorized(
            query_encoded, reference_encoded,
            batch_size=500, show_progress=show_progress
        )

        results = {
            'min_distance_mean': float(np.mean(min_distances)),
            'min_distance_std': float(np.std(min_distances)),
            'min_distance_median': float(np.median(min_distances)),
            'min_distance_distribution': min_distances.tolist(),
        }

        # Compute novelty@r for each radius
        for r in self.config.novelty_radii:
            results[f'novelty@{r}'] = float(np.mean(min_distances > r))

        return results

    # ==================== Diversity Metrics ====================

    def compute_diversity(self,
                          sequences: List[str],
                          show_progress: bool = True) -> Dict:
        """
        Compute diversity metrics using pairwise Hamming distances.

        Uses vectorized operations for speed. For large datasets, samples
        sequences to avoid O(n^2) memory issues.
        """
        n = len(sequences)
        seq_len = len(sequences[0]) if sequences else self.config.sequence_length

        if n < 2:
            return {
                'mean_pairwise_hamming': 0.0,
                'std_pairwise_hamming': 0.0,
                'unique_ratio': 1.0,
                'shannon_entropy': 0.0,
            }

        # Determine if we need to sample
        max_seqs_for_full = self.config.n_diversity_samples

        if n > max_seqs_for_full:
            # Sample sequences for diversity computation
            if show_progress:
                print(f"  Sampling {max_seqs_for_full} sequences for diversity...")
            sample_idx = np.random.choice(n, max_seqs_for_full, replace=False)
            sample_seqs = [sequences[i] for i in sample_idx]
        else:
            sample_seqs = sequences

        # Encode sampled sequences
        if show_progress:
            print("  Encoding sequences for diversity...")
        encoded = self._encode_sequences_fast(sample_seqs)
        n_sample = len(sample_seqs)

        # Compute pairwise distances using vectorized method
        # For diversity we need upper triangle of distance matrix
        if show_progress:
            print("  Computing pairwise distances (vectorized)...")

        # Batch computation to manage memory
        batch_size = min(500, n_sample)
        pairwise_distances = []

        n_batches = (n_sample + batch_size - 1) // batch_size

        if show_progress:
            from tqdm import tqdm
            batch_iter = tqdm(range(n_batches), desc="Computing diversity")
        else:
            batch_iter = range(n_batches)

        for batch_idx in batch_iter:
            start_i = batch_idx * batch_size
            end_i = min(start_i + batch_size, n_sample)

            # Compute distances from batch to all sequences after it
            batch = encoded[start_i:end_i]

            # Compare with sequences from start_i onwards to get upper triangle
            for i_local, i_global in enumerate(range(start_i, end_i)):
                # Only compare with sequences that come after this one
                if i_global + 1 < n_sample:
                    remaining = encoded[i_global + 1:]
                    # Compute distances: (1, L) vs (n_remaining, L)
                    dists = np.sum(batch[i_local:i_local + 1] != remaining, axis=1) / seq_len
                    pairwise_distances.extend(dists.tolist())

        pairwise_distances = np.array(pairwise_distances)

        # Unique sequences ratio (on full dataset)
        unique_ratio = len(set(sequences)) / len(sequences)

        # Shannon entropy of sequence distribution
        seq_counter = Counter(sequences)
        seq_probs = np.array(list(seq_counter.values())) / len(sequences)
        shannon_ent = entropy(seq_probs)

        return {
            'mean_pairwise_hamming': float(np.mean(pairwise_distances)),
            'std_pairwise_hamming': float(np.std(pairwise_distances)),
            'unique_ratio': float(unique_ratio),
            'shannon_entropy': float(shannon_ent),
            'pairwise_distribution': pairwise_distances.tolist(),
        }

    # ==================== K-mer Analysis ====================

    def compute_kmer_metrics(self,
                             reference_seqs: List[str],
                             query_seqs: List[str]) -> Dict:
        """Compute k-mer frequency comparison metrics."""
        results = {}

        for k in self.config.kmer_sizes:
            ref_dist = self.get_kmer_distribution(reference_seqs, k)
            query_dist = self.get_kmer_distribution(query_seqs, k)

            # Align distributions
            all_kmers = sorted(set(ref_dist.keys()) | set(query_dist.keys()))
            ref_vec = np.array([ref_dist.get(km, 0) for km in all_kmers])
            query_vec = np.array([query_dist.get(km, 0) for km in all_kmers])

            # Normalize
            ref_vec = ref_vec / (ref_vec.sum() + 1e-10)
            query_vec = query_vec / (query_vec.sum() + 1e-10)

            # Compute metrics
            jsd = jensenshannon(ref_vec, query_vec)
            cos_sim = self.cosine_similarity(ref_vec, query_vec)

            # Overlap ratio
            shared = len(set(ref_dist.keys()) & set(query_dist.keys()))
            overlap = shared / len(all_kmers) if all_kmers else 0

            results[f'{k}-mer'] = {
                'jensen_shannon_divergence': float(jsd),
                'cosine_similarity': float(cos_sim),
                'overlap_ratio': float(overlap),
                'n_unique_reference': len(ref_dist),
                'n_unique_query': len(query_dist),
                'n_shared': shared,
            }

        return results

    # ==================== GC Content Analysis ====================

    def compute_gc_metrics(self,
                           reference_seqs: List[str],
                           query_seqs: List[str]) -> Dict:
        """Compute GC content comparison metrics."""
        ref_gc = np.array([self.compute_gc_content(s) for s in reference_seqs])
        query_gc = np.array([self.compute_gc_content(s) for s in query_seqs])

        # Statistical tests
        ttest = ttest_ind(ref_gc, query_gc)
        mwu = mannwhitneyu(ref_gc, query_gc, alternative='two-sided')

        return {
            'reference_mean': float(np.mean(ref_gc)),
            'reference_std': float(np.std(ref_gc)),
            'query_mean': float(np.mean(query_gc)),
            'query_std': float(np.std(query_gc)),
            'mean_difference': float(np.mean(query_gc) - np.mean(ref_gc)),
            'ttest_statistic': float(ttest.statistic),
            'ttest_pvalue': float(ttest.pvalue),
            'mannwhitney_pvalue': float(mwu.pvalue),
            'reference_distribution': ref_gc.tolist(),
            'query_distribution': query_gc.tolist(),
        }

    # ==================== Full Evaluation ====================

    def evaluate(self,
                 reference_seqs: List[str],
                 query_seqs: List[str],
                 query_name: str = "synthetic",
                 cluster_assignments: Optional[np.ndarray] = None,
                 show_progress: bool = True) -> Dict:
        """
        Run full evaluation suite.

        Args:
            reference_seqs: Ground truth sequences
            query_seqs: Synthetic/query sequences to evaluate
            query_name: Name identifier for the query dataset
            cluster_assignments: Optional cluster labels for query_seqs
            show_progress: Show progress bars

        Returns:
            Dictionary with all metrics
        """
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {query_name}")
        print(f"Reference: {len(reference_seqs)} sequences")
        print(f"Query: {len(query_seqs)} sequences")
        print(f"{'=' * 60}")

        results = {
            'name': query_name,
            'n_reference': len(reference_seqs),
            'n_query': len(query_seqs),
        }

        # Global metrics
        print("\n[1/4] Computing novelty metrics...")
        results['novelty'] = self.compute_novelty(reference_seqs, query_seqs, show_progress)

        print("\n[2/4] Computing diversity metrics...")
        results['diversity'] = self.compute_diversity(query_seqs, show_progress)

        print("\n[3/4] Computing k-mer metrics...")
        results['kmer'] = self.compute_kmer_metrics(reference_seqs, query_seqs)

        print("\n[4/4] Computing GC content metrics...")
        results['gc_content'] = self.compute_gc_metrics(reference_seqs, query_seqs)

        # Per-cluster analysis if provided
        if cluster_assignments is not None:
            print("\n[Bonus] Computing per-cluster metrics...")
            results['per_cluster'] = self._compute_per_cluster_metrics(
                reference_seqs, query_seqs, cluster_assignments, show_progress
            )

        return results

    def _compute_per_cluster_metrics(self,
                                     reference_seqs: List[str],
                                     query_seqs: List[str],
                                     cluster_assignments: np.ndarray,
                                     show_progress: bool) -> Dict:
        """Compute metrics for each cluster."""
        unique_clusters = np.unique(cluster_assignments)
        cluster_results = {}

        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise
                continue

            mask = cluster_assignments == cluster_id
            cluster_seqs = [query_seqs[i] for i in range(len(query_seqs)) if mask[i]]

            print(f"\n  Cluster {cluster_id}: {len(cluster_seqs)} sequences")

            cluster_results[int(cluster_id)] = {
                'n_sequences': len(cluster_seqs),
                'novelty': self.compute_novelty(reference_seqs, cluster_seqs, False),
                'diversity': self.compute_diversity(cluster_seqs, False),
                'kmer': self.compute_kmer_metrics(reference_seqs, cluster_seqs),
                'gc_content': self.compute_gc_metrics(reference_seqs, cluster_seqs),
            }

        return cluster_results


class MetricsVisualizer:
    """Generate publication-quality visualizations for DNA sequence metrics."""

    def __init__(self, output_dir: str = "figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color palette for multiple models
        self.model_colors = {
            'ground_truth': '#2ecc71',  # Green
            'original': '#e74c3c',  # Red
            'improved': '#3498db',  # Blue
            'model_3': '#9b59b6',  # Purple
            'model_4': '#f39c12',  # Orange
            'model_5': '#1abc9c',  # Teal
            'model_6': '#e67e22',  # Dark orange
            'model_7': '#34495e',  # Dark gray
        }

    def plot_novelty_curves(self,
                            results_dict: Dict[str, Dict],
                            output_name: str = "novelty_curves") -> str:
        """Plot Novelty@r curves for multiple models."""
        fig, ax = plt.subplots(figsize=(10, 7))

        for i, (name, results) in enumerate(results_dict.items()):
            novelty = results['novelty']
            radii = sorted([int(k.split('@')[1]) for k in novelty.keys() if k.startswith('novelty@')])
            values = [novelty[f'novelty@{r}'] for r in radii]

            color = list(self.model_colors.values())[i % len(self.model_colors)]
            ax.plot(radii, values, 'o-', label=name, color=color, linewidth=2, markersize=8)

        ax.set_xlabel('Radius (r) - Hamming Distance Threshold')
        ax.set_ylabel('Novelty@r (fraction of novel sequences)')
        ax.set_title('Sequence Novelty: Fraction of Sequences Beyond Distance r from Training Set')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def plot_gc_distributions(self,
                              results_dict: Dict[str, Dict],
                              reference_gc: List[float],
                              output_name: str = "gc_distributions") -> str:
        """Plot GC content distributions."""
        n_models = len(results_dict)
        fig, axes = plt.subplots(1, n_models + 1, figsize=(4 * (n_models + 1), 5))

        if n_models == 0:
            axes = [axes]

        # Plot reference
        axes[0].hist(reference_gc, bins=50, alpha=0.7, color=self.model_colors['ground_truth'],
                     edgecolor='black', linewidth=0.5)
        axes[0].axvline(np.mean(reference_gc), color='black', linestyle='--',
                        label=f'Mean: {np.mean(reference_gc):.3f}')
        axes[0].set_xlabel('GC Content')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Ground Truth')
        axes[0].legend()

        # Plot each model
        for i, (name, results) in enumerate(results_dict.items()):
            gc_dist = results['gc_content']['query_distribution']
            color = list(self.model_colors.values())[(i + 1) % len(self.model_colors)]

            axes[i + 1].hist(gc_dist, bins=50, alpha=0.7, color=color,
                             edgecolor='black', linewidth=0.5)
            axes[i + 1].axvline(np.mean(gc_dist), color='black', linestyle='--',
                                label=f'Mean: {np.mean(gc_dist):.3f}')
            axes[i + 1].set_xlabel('GC Content')
            axes[i + 1].set_title(f'{name}\np={results["gc_content"]["ttest_pvalue"]:.2e}')
            axes[i + 1].legend()

        plt.suptitle('GC Content Distributions', fontsize=16, y=1.02)
        plt.tight_layout()

        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def plot_kmer_heatmap(self,
                          results_dict: Dict[str, Dict],
                          output_name: str = "kmer_heatmap") -> str:
        """Plot k-mer similarity metrics as heatmap."""
        # Prepare data
        models = list(results_dict.keys())
        k_sizes = sorted(results_dict[models[0]]['kmer'].keys())

        # Create matrices for JSD and cosine similarity
        jsd_matrix = np.zeros((len(models), len(k_sizes)))
        cos_matrix = np.zeros((len(models), len(k_sizes)))

        for i, model in enumerate(models):
            for j, k in enumerate(k_sizes):
                jsd_matrix[i, j] = results_dict[model]['kmer'][k]['jensen_shannon_divergence']
                cos_matrix[i, j] = results_dict[model]['kmer'][k]['cosine_similarity']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # JSD heatmap (lower is better)
        sns.heatmap(jsd_matrix, ax=axes[0], annot=True, fmt='.4f',
                    xticklabels=[k.replace('-mer', '') for k in k_sizes],
                    yticklabels=models, cmap='Reds', vmin=0, vmax=0.5)
        axes[0].set_xlabel('K-mer Size')
        axes[0].set_ylabel('Model')
        axes[0].set_title('Jensen-Shannon Divergence\n(lower = more similar to GT)')

        # Cosine similarity heatmap (higher is better)
        sns.heatmap(cos_matrix, ax=axes[1], annot=True, fmt='.4f',
                    xticklabels=[k.replace('-mer', '') for k in k_sizes],
                    yticklabels=models, cmap='Greens', vmin=0.8, vmax=1.0)
        axes[1].set_xlabel('K-mer Size')
        axes[1].set_ylabel('Model')
        axes[1].set_title('Cosine Similarity\n(higher = more similar to GT)')

        plt.tight_layout()

        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def plot_diversity_comparison(self,
                                  results_dict: Dict[str, Dict],
                                  output_name: str = "diversity_comparison") -> str:
        """Plot diversity metrics comparison."""
        models = list(results_dict.keys())

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Mean pairwise Hamming distance
        hamming_vals = [results_dict[m]['diversity']['mean_pairwise_hamming'] for m in models]
        hamming_stds = [results_dict[m]['diversity']['std_pairwise_hamming'] for m in models]

        colors = [list(self.model_colors.values())[i % len(self.model_colors)]
                  for i in range(len(models))]

        axes[0].bar(models, hamming_vals, yerr=hamming_stds, color=colors,
                    edgecolor='black', capsize=5)
        axes[0].set_ylabel('Mean Pairwise Hamming Distance')
        axes[0].set_title('Sequence Diversity\n(higher = more diverse)')
        axes[0].tick_params(axis='x', rotation=45)

        # Unique ratio
        unique_vals = [results_dict[m]['diversity']['unique_ratio'] for m in models]
        axes[1].bar(models, unique_vals, color=colors, edgecolor='black')
        axes[1].set_ylabel('Unique Sequences Ratio')
        axes[1].set_title('Uniqueness\n(1.0 = all unique)')
        axes[1].set_ylim(0, 1.1)
        axes[1].tick_params(axis='x', rotation=45)

        # Shannon entropy
        entropy_vals = [results_dict[m]['diversity']['shannon_entropy'] for m in models]
        axes[2].bar(models, entropy_vals, color=colors, edgecolor='black')
        axes[2].set_ylabel('Shannon Entropy (bits)')
        axes[2].set_title('Entropy\n(higher = more varied)')
        axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def plot_cluster_comparison(self,
                                results_with_clusters: Dict,
                                output_name: str = "cluster_metrics") -> str:
        """Plot per-cluster metrics comparison."""
        if 'per_cluster' not in results_with_clusters:
            print("No cluster data available")
            return None

        cluster_data = results_with_clusters['per_cluster']
        clusters = sorted(cluster_data.keys())

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))

        # Cluster sizes
        sizes = [cluster_data[c]['n_sequences'] for c in clusters]
        axes[0, 0].bar([f'Cluster {c}' for c in clusters], sizes, color=colors, edgecolor='black')
        axes[0, 0].set_ylabel('Number of Sequences')
        axes[0, 0].set_title('Cluster Sizes')

        # Diversity per cluster
        div_vals = [cluster_data[c]['diversity']['mean_pairwise_hamming'] for c in clusters]
        axes[0, 1].bar([f'Cluster {c}' for c in clusters], div_vals, color=colors, edgecolor='black')
        axes[0, 1].set_ylabel('Mean Pairwise Hamming')
        axes[0, 1].set_title('Diversity per Cluster')

        # GC content per cluster
        gc_means = [cluster_data[c]['gc_content']['query_mean'] for c in clusters]
        gc_stds = [cluster_data[c]['gc_content']['query_std'] for c in clusters]
        axes[1, 0].bar([f'Cluster {c}' for c in clusters], gc_means, yerr=gc_stds,
                       color=colors, edgecolor='black', capsize=5)
        axes[1, 0].set_ylabel('GC Content')
        axes[1, 0].set_title('GC Content per Cluster')

        # Novelty@5 per cluster
        nov_vals = [cluster_data[c]['novelty'].get('novelty@5', 0) for c in clusters]
        axes[1, 1].bar([f'Cluster {c}' for c in clusters], nov_vals, color=colors, edgecolor='black')
        axes[1, 1].set_ylabel('Novelty@5')
        axes[1, 1].set_title('Novelty per Cluster')
        axes[1, 1].set_ylim(0, 1.1)

        plt.tight_layout()

        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def plot_summary_dashboard(self,
                               results_dict: Dict[str, Dict],
                               reference_gc: List[float],
                               output_name: str = "metrics_dashboard") -> str:
        """Create comprehensive dashboard with all key metrics."""
        fig = plt.figure(figsize=(20, 16))

        # Layout: 3 rows, 3 columns
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        models = list(results_dict.keys())
        colors = [list(self.model_colors.values())[i % len(self.model_colors)]
                  for i in range(len(models))]

        # 1. Novelty curves (top left, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        for i, (name, results) in enumerate(results_dict.items()):
            novelty = results['novelty']
            radii = sorted([int(k.split('@')[1]) for k in novelty.keys() if k.startswith('novelty@')])
            values = [novelty[f'novelty@{r}'] for r in radii]
            ax1.plot(radii, values, 'o-', label=name, color=colors[i], linewidth=2, markersize=6)
        ax1.set_xlabel('Radius (r)')
        ax1.set_ylabel('Novelty@r')
        ax1.set_title('A. Sequence Novelty')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Min distance distribution (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        for i, (name, results) in enumerate(results_dict.items()):
            ax2.hist(results['novelty']['min_distance_distribution'], bins=30,
                     alpha=0.5, label=name, color=colors[i], edgecolor='black')
        ax2.set_xlabel('Min Hamming Distance to Training')
        ax2.set_ylabel('Count')
        ax2.set_title('B. Distance to Nearest Training Sequence')
        ax2.legend()

        # 3. GC content (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(reference_gc, bins=40, alpha=0.6, label='Ground Truth',
                 color=self.model_colors['ground_truth'], edgecolor='black')
        for i, (name, results) in enumerate(results_dict.items()):
            ax3.hist(results['gc_content']['query_distribution'], bins=40,
                     alpha=0.4, label=name, color=colors[i], edgecolor='black')
        ax3.set_xlabel('GC Content')
        ax3.set_ylabel('Count')
        ax3.set_title('C. GC Content Distribution')
        ax3.legend()

        # 4. K-mer JSD (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        x = np.arange(len(self.config.kmer_sizes) if hasattr(self, 'config') else 4)
        width = 0.8 / len(models)

        for i, (name, results) in enumerate(results_dict.items()):
            k_sizes = sorted(results['kmer'].keys())
            jsd_vals = [results['kmer'][k]['jensen_shannon_divergence'] for k in k_sizes]
            ax4.bar(x + i * width, jsd_vals, width, label=name, color=colors[i], edgecolor='black')

        ax4.set_xlabel('K-mer Size')
        ax4.set_ylabel('JS Divergence')
        ax4.set_title('D. K-mer Frequency Similarity')
        ax4.set_xticks(x + width * (len(models) - 1) / 2)
        ax4.set_xticklabels([k.replace('-mer', '') for k in k_sizes])
        ax4.legend()

        # 5. Diversity metrics (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        diversity_vals = [results_dict[m]['diversity']['mean_pairwise_hamming'] for m in models]
        ax5.bar(models, diversity_vals, color=colors, edgecolor='black')
        ax5.set_ylabel('Mean Pairwise Hamming')
        ax5.set_title('E. Sequence Diversity')
        ax5.tick_params(axis='x', rotation=45)

        # 6. Summary table (bottom, spans all columns)
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')

        # Create summary table
        headers = ['Model', 'Novelty@5', 'Diversity', 'GC Diff', '5-mer JSD', 'Unique%']
        table_data = []
        for name, results in results_dict.items():
            row = [
                name,
                f"{results['novelty'].get('novelty@5', 0):.3f}",
                f"{results['diversity']['mean_pairwise_hamming']:.4f}",
                f"{results['gc_content']['mean_difference']:.4f}",
                f"{results['kmer']['5-mer']['jensen_shannon_divergence']:.4f}",
                f"{results['diversity']['unique_ratio'] * 100:.1f}%"
            ]
            table_data.append(row)

        table = ax6.table(cellText=table_data, colLabels=headers,
                          loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.8)
        ax6.set_title('F. Summary Metrics', fontsize=14, pad=20)

        plt.suptitle('DNA Sequence Generation Evaluation Dashboard', fontsize=18, y=0.98)

        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)


def load_sequences(filepath: str) -> List[str]:
    """Load sequences from various file formats."""
    filepath = Path(filepath)

    if filepath.suffix == '.csv':
        df = pd.read_csv(filepath)
        # Try common column names
        for col in ['sequence', 'seq', 'Sequence', 'SEQ']:
            if col in df.columns:
                return df[col].tolist()
        return df.iloc[:, 0].tolist()

    elif filepath.suffix == '.tsv':
        df = pd.read_csv(filepath, sep='\t')
        for col in ['sequence', 'seq', 'Sequence', 'SEQ']:
            if col in df.columns:
                return df[col].tolist()
        return df.iloc[:, 0].tolist()

    elif filepath.suffix == '.txt':
        with open(filepath, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    elif filepath.suffix in ['.fa', '.fasta']:
        sequences = []
        current_seq = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(''.join(current_seq))
                        current_seq = []
                else:
                    current_seq.append(line.strip())
        if current_seq:
            sequences.append(''.join(current_seq))
        return sequences

    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def main():
    parser = argparse.ArgumentParser(
        description='DNA Sequence Metrics Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with one synthetic dataset
  python 01_sequence_metrics_evaluation.py \\
      --reference data/ground_truth.txt \\
      --synthetic original:data/synthetic_original.txt

  # Multiple synthetic datasets
  python 01_sequence_metrics_evaluation.py \\
      --reference data/ground_truth.txt \\
      --synthetic original:data/synth_original.txt improved:data/synth_improved.txt

  # With cluster assignments
  python 01_sequence_metrics_evaluation.py \\
      --reference data/ground_truth.txt \\
      --synthetic original:data/synth.txt \\
      --clusters data/cluster_assignments.csv \\
      --output results/
        """
    )

    parser.add_argument('--reference', '-r', required=True,
                        help='Path to reference/ground truth sequences')
    parser.add_argument('--synthetic', '-s', nargs='+', required=True,
                        help='Synthetic datasets as name:path pairs (e.g., original:path/to/file.txt)')
    parser.add_argument('--clusters', '-c', default=None,
                        help='Path to CSV with cluster assignments (columns: id, cluster)')
    parser.add_argument('--output', '-o', default='evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--kmer-sizes', nargs='+', type=int, default=[3, 5, 7, 9],
                        help='K-mer sizes to analyze')
    parser.add_argument('--novelty-radii', nargs='+', type=int, default=[1, 2, 3, 5, 10, 15, 20],
                        help='Radii for novelty@r calculation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = EvaluationConfig(
        kmer_sizes=args.kmer_sizes,
        novelty_radii=args.novelty_radii,
        random_seed=args.seed
    )

    evaluator = DNASequenceMetrics(config)
    visualizer = MetricsVisualizer(output_dir=str(output_dir / 'figures'))

    # Load reference sequences
    print(f"Loading reference sequences from {args.reference}...")
    reference_seqs = load_sequences(args.reference)
    print(f"Loaded {len(reference_seqs)} reference sequences")

    # Load cluster assignments if provided
    cluster_assignments = None
    if args.clusters:
        cluster_df = pd.read_csv(args.clusters)
        cluster_assignments = cluster_df['cluster'].values
        print(f"Loaded cluster assignments: {len(np.unique(cluster_assignments))} clusters")

    # Parse synthetic datasets
    synthetic_datasets = {}
    for spec in args.synthetic:
        if ':' in spec:
            name, path = spec.split(':', 1)
        else:
            name = Path(spec).stem
            path = spec

        print(f"\nLoading synthetic dataset '{name}' from {path}...")
        synthetic_datasets[name] = load_sequences(path)
        print(f"Loaded {len(synthetic_datasets[name])} sequences")

    # Run evaluation for each dataset
    all_results = {}
    for name, synth_seqs in synthetic_datasets.items():
        results = evaluator.evaluate(
            reference_seqs,
            synth_seqs,
            query_name=name,
            cluster_assignments=cluster_assignments
        )
        all_results[name] = results

    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)

    ref_gc = [evaluator.compute_gc_content(s) for s in reference_seqs]

    figures = {
        'novelty_curves': visualizer.plot_novelty_curves(all_results),
        'gc_distributions': visualizer.plot_gc_distributions(all_results, ref_gc),
        'kmer_heatmap': visualizer.plot_kmer_heatmap(all_results),
        'diversity_comparison': visualizer.plot_diversity_comparison(all_results),
        'dashboard': visualizer.plot_summary_dashboard(all_results, ref_gc),
    }

    # Per-cluster plots if available
    for name, results in all_results.items():
        if 'per_cluster' in results:
            fig_path = visualizer.plot_cluster_comparison(
                results, output_name=f"cluster_metrics_{name}"
            )
            figures[f'cluster_metrics_{name}'] = fig_path

    # Save results to JSON
    results_file = output_dir / 'evaluation_results.json'

    # Remove distribution arrays for JSON (too large)
    json_results = {}
    for name, results in all_results.items():
        json_results[name] = {k: v for k, v in results.items()}
        # Remove large arrays
        if 'novelty' in json_results[name]:
            json_results[name]['novelty'] = {
                k: v for k, v in json_results[name]['novelty'].items()
                if not k.endswith('_distribution')
            }
        if 'diversity' in json_results[name]:
            json_results[name]['diversity'] = {
                k: v for k, v in json_results[name]['diversity'].items()
                if not k.endswith('_distribution')
            }
        if 'gc_content' in json_results[name]:
            json_results[name]['gc_content'] = {
                k: v for k, v in json_results[name]['gc_content'].items()
                if not k.endswith('_distribution')
            }

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Create summary CSV
    summary_rows = []
    for name, results in all_results.items():
        row = {
            'model': name,
            'n_sequences': results['n_query'],
            'novelty@1': results['novelty'].get('novelty@1', None),
            'novelty@5': results['novelty'].get('novelty@5', None),
            'novelty@10': results['novelty'].get('novelty@10', None),
            'min_dist_mean': results['novelty']['min_distance_mean'],
            'diversity': results['diversity']['mean_pairwise_hamming'],
            'unique_ratio': results['diversity']['unique_ratio'],
            'gc_difference': results['gc_content']['mean_difference'],
            'gc_pvalue': results['gc_content']['ttest_pvalue'],
        }
        # Add k-mer metrics
        for k in args.kmer_sizes:
            key = f'{k}-mer'
            if key in results['kmer']:
                row[f'{k}mer_jsd'] = results['kmer'][key]['jensen_shannon_divergence']
                row[f'{k}mer_cosine'] = results['kmer'][key]['cosine_similarity']

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_file = output_dir / 'evaluation_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to {summary_file}")

    # Print final summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated figures:")
    for name, path in figures.items():
        if path:
            print(f"  - {name}: {path}")

    print("\n" + summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
