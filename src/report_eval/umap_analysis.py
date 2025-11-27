#!/usr/bin/env python3
"""
Script 2: UMAP Embedding Visualization with HDBSCAN Clustering

Creates UMAP visualizations of DNA sequence embeddings with:
- Support for multiple synthetic datasets (up to 7+)
- HDBSCAN clustering to identify structure in embedding space
- Ground truth vs synthetic comparison
- Cluster assignment export for downstream analysis

Key output: Shows that synthetic data clusters into specific UMAP regions,
potentially indicating mode collapse.

Author: DNA Diffusion Analysis Pipeline
"""

import numpy as np
import pandas as pd
import h5py
import umap
import hdbscan
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import argparse
import json
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Publication-quality plot settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.figsize': (12, 10),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


@dataclass
class UMAPConfig:
    """Configuration for UMAP and clustering."""
    n_neighbors: int = 25
    min_dist: float = 0.1
    metric: str = 'euclidean'
    n_components: int = 2
    random_state: int = 42

    # HDBSCAN parameters - see docstring for tuning guidance
    min_cluster_size: int = 200  # Minimum points to form a cluster
    min_samples: int = 50  # Core point density threshold
    cluster_selection_epsilon: float = 0.0  # Merge clusters within this distance
    cluster_selection_method: str = 'eom'  # 'eom' (default) or 'leaf' for finer clusters
    alpha: float = 1.0  # Distance scaling (lower = more clusters)


# HDBSCAN Tuning Guide:
# ---------------------
# To find MORE/SMALLER clusters (more restrictive):
#   1. DECREASE min_cluster_size (e.g., 50, 20, 10)
#   2. DECREASE min_samples (e.g., 10, 5, 3)
#   3. Use cluster_selection_method='leaf' (finds finest granularity)
#   4. SET cluster_selection_epsilon=0.0 (don't merge nearby clusters)
#   5. DECREASE alpha (e.g., 0.5) - makes distance differences more pronounced
#
# To find FEWER/LARGER clusters (less restrictive):
#   1. INCREASE min_cluster_size (e.g., 500, 1000)
#   2. INCREASE min_samples (e.g., 100, 200)
#   3. Use cluster_selection_method='eom' (default, finds most stable)
#   4. INCREASE cluster_selection_epsilon (e.g., 0.5, 1.0) - merges nearby clusters
#
# Example for finding minute clusters:
#   --min-cluster-size 20 --min-samples 5 --cluster-method leaf --epsilon 0.0


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    filepath: str
    n_samples: Optional[int] = None
    color: str = '#888888'
    is_reference: bool = False


class EmbeddingLoader:
    """Load embeddings from various file formats."""

    @staticmethod
    def load_h5_embeddings(filepath: str,
                           n_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load embeddings from HDF5 file.

        Supports two formats:
        1. ID-based organization: Each sample is a group with 'embedding' dataset and metadata attrs
        2. Array-based organization: Single 'embeddings' dataset with separate metadata arrays

        Returns:
            embeddings: (N, L, D) or (N, D) array
            labels: (N,) array of labels/cell types (TAG field)
            ids: (N,) array of sequence IDs
        """
        with h5py.File(filepath, 'r') as f:
            # Print available keys for debugging
            top_keys = list(f.keys())
            print(f"  Available top-level keys: {top_keys[:10]}{'...' if len(top_keys) > 10 else ''}")

            # Check if this is ID-based organization (groups) or array-based
            first_key = top_keys[0] if top_keys else None

            if first_key and isinstance(f[first_key], h5py.Group) and 'embedding' in f[first_key]:
                # ID-based organization - each key is a sample ID with embedding inside
                print("  Detected ID-based H5 organization")
                return EmbeddingLoader._load_id_based_h5(f, n_samples)
            else:
                # Array-based organization
                print("  Detected array-based H5 organization")
                return EmbeddingLoader._load_array_based_h5(f, n_samples)

    @staticmethod
    def _load_id_based_h5(f: h5py.File,
                          n_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load from ID-based H5 format where each sample is a group.

        Expected structure:
            /<sample_id>/embedding: (L, D) array
            /<sample_id>.attrs['TAG']: string label
            /<sample_id>.attrs['data_label']: string label
        """
        ids = list(f.keys())

        # Sample if requested
        if n_samples is not None and n_samples < len(ids):
            # Use random sampling for better representation
            idx = np.random.choice(len(ids), n_samples, replace=False)
            ids = [ids[i] for i in sorted(idx)]

        num_samples = len(ids)

        # Get embedding shape from first sample
        first_embedding = f[ids[0]]['embedding'][:]
        emb_shape = first_embedding.shape

        print(f"  Loading {num_samples} samples with embedding shape {emb_shape}")

        # Pre-allocate arrays
        embeddings = np.zeros((num_samples,) + emb_shape, dtype='float32')
        labels = []
        tags = []

        # Load each embedding
        for idx, sample_id in enumerate(tqdm(ids, desc="Loading embeddings")):
            grp = f[sample_id]
            embeddings[idx] = grp['embedding'][:]

            # Get labels - try multiple attribute names
            label = None
            for attr_name in ['TAG', 'tag', 'data_label', 'datalabel', 'label']:
                if attr_name in grp.attrs:
                    label = grp.attrs[attr_name]
                    if isinstance(label, bytes):
                        label = label.decode()
                    break
            labels.append(label if label is not None else 'unknown')

            # Get TAG specifically for cell type
            tag = None
            for attr_name in ['TAG', 'tag', 'cell_type']:
                if attr_name in grp.attrs:
                    tag = grp.attrs[attr_name]
                    if isinstance(tag, bytes):
                        tag = tag.decode()
                    break
            tags.append(tag if tag is not None else 'unknown')

        return embeddings, np.array(tags), np.array(ids)

    @staticmethod
    def _load_array_based_h5(f: h5py.File,
                             n_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load from array-based H5 format with single embeddings array.

        Expected structure:
            /embeddings or /X: (N, L, D) or (N, D) array
            /labels or /tags: (N,) array
            /ids: (N,) array
        """
        # Find embeddings
        embeddings = None
        for key in ['embeddings', 'X', 'embedding', 'data']:
            if key in f and isinstance(f[key], h5py.Dataset):
                embeddings = f[key][:]
                print(f"  Found embeddings in key '{key}'")
                break

        if embeddings is None:
            # Try first dataset
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset) and len(f[key].shape) >= 2:
                    embeddings = f[key][:]
                    print(f"  Using first dataset '{key}' as embeddings")
                    break

        if embeddings is None:
            raise ValueError("Could not find embeddings in H5 file")

        # Find labels
        labels = None
        for key in ['tags', 'TAG', 'labels', 'datalabels', 'cell_types', 'y']:
            if key in f:
                labels = f[key][:]
                if len(labels) > 0 and isinstance(labels[0], bytes):
                    labels = np.array([l.decode() for l in labels])
                break

        if labels is None:
            labels = np.array(['unknown'] * len(embeddings))

        # Find IDs
        ids = None
        for key in ['ids', 'id', 'names', 'sequence_ids', 'dhs_id']:
            if key in f:
                ids = f[key][:]
                if len(ids) > 0 and isinstance(ids[0], bytes):
                    ids = np.array([i.decode() for i in ids])
                break

        if ids is None:
            ids = np.array([f'seq_{i}' for i in range(len(embeddings))])

        # Sample if requested
        if n_samples is not None and n_samples < len(embeddings):
            idx = np.random.choice(len(embeddings), n_samples, replace=False)
            embeddings = embeddings[idx]
            labels = labels[idx]
            ids = ids[idx]

        print(f"  Loaded embeddings shape: {embeddings.shape}")
        return embeddings, labels, ids

    @staticmethod
    def load_npy_embeddings(filepath: str,
                            n_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load embeddings from numpy file."""
        embeddings = np.load(filepath)

        if n_samples is not None and n_samples < len(embeddings):
            idx = np.random.choice(len(embeddings), n_samples, replace=False)
            embeddings = embeddings[idx]

        labels = np.array(['unknown'] * len(embeddings))
        ids = np.array([f'seq_{i}' for i in range(len(embeddings))])

        return embeddings, labels, ids


class UMAPVisualizer:
    """
    Create UMAP visualizations and perform clustering on DNA embeddings.
    """

    def __init__(self,
                 config: Optional[UMAPConfig] = None,
                 output_dir: str = 'umap_results'):
        self.config = config or UMAPConfig()
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

        self.reducer = None
        self.clusterer = None

    @staticmethod
    def extract_chromosome(sample_id: str) -> str:
        """
        Extract chromosome from sample ID.

        Expected format: chrY_2953025_2953206_2953130
        Returns: 'chrY' or 'unknown' if parsing fails
        """
        try:
            # Split by underscore and take first part
            parts = sample_id.split('_')
            if parts and parts[0].startswith('chr'):
                return parts[0]
            return 'unknown'
        except:
            return 'unknown'

    def plot_by_chromosome(self, results: Dict, output_name: str = 'umap_chromosomes') -> str:
        """Plot UMAP colored by chromosome."""
        fig, ax = plt.subplots(figsize=(16, 12))

        Z = results['umap_coords']
        sequence_ids = results['sequence_ids']

        # Extract chromosomes from IDs
        chromosomes = np.array([self.extract_chromosome(str(sid)) for sid in sequence_ids])
        unique_chroms = sorted(set(chromosomes), key=lambda x: (
            # Sort numerically for chr1-22, then X, Y, M, unknown
            int(x[3:]) if x[3:].isdigit() else
            {'X': 23, 'Y': 24, 'M': 25, 'MT': 25}.get(x[3:], 99)
        ) if x.startswith('chr') else 99)

        # Color palette - use tab20 for many chromosomes
        n_chroms = len(unique_chroms)
        if n_chroms <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, n_chroms))
        elif n_chroms <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, n_chroms))
        else:
            colors = plt.cm.nipy_spectral(np.linspace(0.1, 0.9, n_chroms))

        chrom_to_color = {chrom: colors[i] for i, chrom in enumerate(unique_chroms)}

        for chrom in unique_chroms:
            mask = chromosomes == chrom
            n_points = np.sum(mask)
            ax.scatter(Z[mask, 0], Z[mask, 1],
                       s=8, alpha=0.5, c=[chrom_to_color[chrom]],
                       label=f'{chrom} (n={n_points})', rasterized=True)

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('Embeddings by Chromosome')

        # Adjust legend for many chromosomes
        if n_chroms > 15:
            ax.legend(markerscale=3, loc='upper right', bbox_to_anchor=(1.15, 1),
                      fontsize=8, ncol=2)
        else:
            ax.legend(markerscale=3, loc='upper right', bbox_to_anchor=(1.15, 1))

        ax.grid(True, alpha=0.3)

        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def plot_chromosome_by_cluster(self, results: Dict, output_name: str = 'chromosome_cluster_composition') -> str:
        """
        Analyze chromosome distribution within each cluster.
        Shows if certain chromosomes are enriched in specific clusters.
        """
        clusters = results['cluster_labels']
        sequence_ids = results['sequence_ids']

        # Extract chromosomes
        chromosomes = np.array([self.extract_chromosome(str(sid)) for sid in sequence_ids])

        unique_clusters = sorted([c for c in np.unique(clusters) if c != -1])
        unique_chroms = sorted(set(chromosomes), key=lambda x: (
            int(x[3:]) if x.startswith('chr') and x[3:].isdigit() else
            {'chrX': 23, 'chrY': 24, 'chrM': 25, 'chrMT': 25}.get(x, 99)
        ))

        # Create composition matrix
        composition = np.zeros((len(unique_clusters), len(unique_chroms)))

        for i, cluster in enumerate(unique_clusters):
            cluster_mask = clusters == cluster
            cluster_chroms = chromosomes[cluster_mask]
            for j, chrom in enumerate(unique_chroms):
                composition[i, j] = np.sum(cluster_chroms == chrom)

        # Normalize to percentages per cluster
        composition_pct = composition / composition.sum(axis=1, keepdims=True) * 100

        # Plot heatmap
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Heatmap of percentages
        ax = axes[0]
        im = ax.imshow(composition_pct, aspect='auto', cmap='YlOrRd')
        ax.set_xticks(range(len(unique_chroms)))
        ax.set_xticklabels(unique_chroms, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(unique_clusters)))
        ax.set_yticklabels([f'Cluster {c}' for c in unique_clusters])
        ax.set_xlabel('Chromosome')
        ax.set_ylabel('Cluster')
        ax.set_title('Chromosome Distribution by Cluster (%)')
        plt.colorbar(im, ax=ax, label='Percentage')

        # Bar plot of total counts
        ax = axes[1]
        chrom_totals = [np.sum(chromosomes == c) for c in unique_chroms]
        bars = ax.bar(range(len(unique_chroms)), chrom_totals, color='steelblue')
        ax.set_xticks(range(len(unique_chroms)))
        ax.set_xticklabels(unique_chroms, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Chromosome')
        ax.set_ylabel('Count')
        ax.set_title('Total Sequences per Chromosome')

        plt.tight_layout()

        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def average_pool(self, embeddings: np.ndarray) -> np.ndarray:
        """Average pool 3D embeddings to 2D."""
        if len(embeddings.shape) == 3:
            return embeddings.mean(axis=1)
        return embeddings

    def fit_umap(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit UMAP on embeddings."""
        print(f"\nFitting UMAP on {len(embeddings)} samples...")

        self.reducer = umap.UMAP(
            n_neighbors=self.config.n_neighbors,
            min_dist=self.config.min_dist,
            metric=self.config.metric,
            n_components=self.config.n_components,
            random_state=self.config.random_state,
            low_memory=True,
            verbose=True
        )

        Z = self.reducer.fit_transform(embeddings)
        return Z

    def fit_hdbscan(self, umap_coords: np.ndarray) -> np.ndarray:
        """Fit HDBSCAN clustering on UMAP coordinates."""
        print("\nFitting HDBSCAN clustering...")
        print(f"  Parameters: min_cluster_size={self.config.min_cluster_size}, "
              f"min_samples={self.config.min_samples}, "
              f"epsilon={self.config.cluster_selection_epsilon}, "
              f"method={self.config.cluster_selection_method}, "
              f"alpha={self.config.alpha}")

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.min_cluster_size,
            min_samples=self.config.min_samples,
            metric='euclidean',
            cluster_selection_epsilon=self.config.cluster_selection_epsilon,
            cluster_selection_method=self.config.cluster_selection_method,
            alpha=self.config.alpha,
        )

        cluster_labels = self.clusterer.fit_predict(umap_coords)

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = np.sum(cluster_labels == -1)
        print(f"Found {n_clusters} clusters, {n_noise} noise points")

        return cluster_labels

    def load_previous_results(self, results_dir: str) -> Optional[Dict]:
        """
        Load previous UMAP results from a directory.

        Expects:
            - umap_coordinates.npy: (N, 2) array of UMAP coordinates
            - cluster_assignments.csv: DataFrame with metadata

        Returns:
            Dictionary with loaded results, or None if files not found
        """
        results_path = Path(results_dir)
        umap_file = results_path / 'umap_coordinates.npy'
        assignments_file = results_path / 'cluster_assignments.csv'

        if not umap_file.exists():
            print(f"  UMAP coordinates not found at {umap_file}")
            return None

        if not assignments_file.exists():
            print(f"  Cluster assignments not found at {assignments_file}")
            return None

        print(f"\nLoading previous UMAP results from {results_dir}...")

        # Load UMAP coordinates
        Z = np.load(umap_file)
        print(f"  Loaded UMAP coordinates: {Z.shape}")

        # Load metadata
        df = pd.read_csv(assignments_file)
        print(f"  Loaded metadata for {len(df)} samples")

        results = {
            'umap_coords': Z,
            'cluster_labels': df['cluster'].values,
            'source_labels': df['source'].values,
            'dataset_names': df['dataset'].values,
            'cell_type_labels': df['cell_type'].values,
            'sequence_ids': df['sequence_id'].values,
            'boundaries': {},  # Not needed for re-clustering
            'n_clusters': len(set(df['cluster'])) - (1 if -1 in df['cluster'].values else 0),
        }

        return results

    def recluster_from_previous(self, previous_results: Dict) -> Dict:
        """
        Re-run HDBSCAN on previously computed UMAP coordinates.

        Args:
            previous_results: Dictionary from load_previous_results()

        Returns:
            Updated results dictionary with new cluster assignments
        """
        print("\nRe-clustering with new HDBSCAN parameters...")

        Z = previous_results['umap_coords']
        cluster_labels = self.fit_hdbscan(Z)

        # Update results
        results = previous_results.copy()
        results['cluster_labels'] = cluster_labels
        results['n_clusters'] = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

        return results

    def analyze_multiple_datasets(self,
                                  reference_config: DatasetConfig,
                                  synthetic_configs: List[DatasetConfig],
                                  load_previous_umap: Optional[str] = None) -> Dict:
        """
        Analyze reference and multiple synthetic datasets.

        Args:
            reference_config: Configuration for reference dataset
            synthetic_configs: List of configurations for synthetic datasets
            load_previous_umap: Path to directory with previous UMAP results (optional)

        Returns comprehensive results including UMAP coordinates and cluster assignments.
        """
        np.random.seed(self.config.random_state)

        # Try to load previous UMAP results if requested
        if load_previous_umap:
            previous = self.load_previous_results(load_previous_umap)
            if previous is not None:
                # Re-cluster with current HDBSCAN parameters
                return self.recluster_from_previous(previous)
            else:
                print("  Could not load previous results, running full analysis...")

        # Load reference embeddings
        print(f"\nLoading reference: {reference_config.name}")
        ref_emb, ref_labels, ref_ids = self._load_dataset(reference_config)
        ref_emb_pooled = self.average_pool(ref_emb)

        # Load all synthetic embeddings
        synth_data = []
        for sc in synthetic_configs:
            print(f"\nLoading synthetic: {sc.name}")
            emb, labels, ids = self._load_dataset(sc)
            emb_pooled = self.average_pool(emb)
            synth_data.append({
                'config': sc,
                'embeddings': emb_pooled,
                'labels': labels,
                'ids': ids,
            })

        # Combine all embeddings for UMAP
        all_embeddings = [ref_emb_pooled]
        source_labels = ['reference'] * len(ref_emb_pooled)
        dataset_names = [reference_config.name] * len(ref_emb_pooled)
        all_labels = ref_labels.tolist()
        all_ids = ref_ids.tolist()

        boundaries = {'reference': (0, len(ref_emb_pooled))}
        current_idx = len(ref_emb_pooled)

        for sd in synth_data:
            all_embeddings.append(sd['embeddings'])
            source_labels.extend(['synthetic'] * len(sd['embeddings']))
            dataset_names.extend([sd['config'].name] * len(sd['embeddings']))
            all_labels.extend(sd['labels'].tolist())
            all_ids.extend(sd['ids'].tolist())

            boundaries[sd['config'].name] = (current_idx, current_idx + len(sd['embeddings']))
            current_idx += len(sd['embeddings'])

        all_embeddings = np.vstack(all_embeddings)
        source_labels = np.array(source_labels)
        dataset_names = np.array(dataset_names)
        all_labels = np.array(all_labels)
        all_ids = np.array(all_ids)

        print(f"\nTotal samples: {len(all_embeddings)}")

        # Fit UMAP
        Z = self.fit_umap(all_embeddings)

        # Fit HDBSCAN
        cluster_labels = self.fit_hdbscan(Z)

        # Prepare results
        results = {
            'umap_coords': Z,
            'cluster_labels': cluster_labels,
            'source_labels': source_labels,
            'dataset_names': dataset_names,
            'cell_type_labels': all_labels,
            'sequence_ids': all_ids,
            'boundaries': boundaries,
            'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            'reference_config': reference_config,
            'synthetic_configs': synthetic_configs,
        }

        return results

    def _load_dataset(self, config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load dataset based on file extension."""
        filepath = Path(config.filepath)

        if filepath.suffix == '.h5':
            return EmbeddingLoader.load_h5_embeddings(config.filepath, config.n_samples)
        elif filepath.suffix == '.npy':
            return EmbeddingLoader.load_npy_embeddings(config.filepath, config.n_samples)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

    # ==================== Visualization Methods ====================

    def plot_reference_vs_synthetic(self, results: Dict, output_name: str = 'umap_ref_vs_synth') -> str:
        """Plot reference vs all synthetic data."""
        fig, ax = plt.subplots(figsize=(14, 12))

        Z = results['umap_coords']
        source = results['source_labels']

        # Plot reference
        ref_mask = source == 'reference'
        ax.scatter(Z[ref_mask, 0], Z[ref_mask, 1],
                   s=8, alpha=0.5, c=self.model_colors['ground_truth'],
                   label='Ground Truth', rasterized=True)

        # Plot synthetic
        synth_mask = source == 'synthetic'
        ax.scatter(Z[synth_mask, 0], Z[synth_mask, 1],
                   s=8, alpha=0.5, c=self.model_colors['original'],
                   label='Synthetic', rasterized=True)

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('Ground Truth vs Synthetic Embeddings')
        ax.legend(markerscale=3, loc='upper right')
        ax.grid(True, alpha=0.3)

        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def plot_by_dataset(self, results: Dict, output_name: str = 'umap_by_dataset') -> str:
        """Plot all datasets with different colors."""
        fig, ax = plt.subplots(figsize=(14, 12))

        Z = results['umap_coords']
        dataset_names = results['dataset_names']
        unique_datasets = np.unique(dataset_names)

        colors = list(self.model_colors.values())

        for i, dataset in enumerate(unique_datasets):
            mask = dataset_names == dataset
            color = colors[i % len(colors)]

            ax.scatter(Z[mask, 0], Z[mask, 1],
                       s=8, alpha=0.5, c=color,
                       label=f'{dataset} (n={np.sum(mask)})', rasterized=True)

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('Embeddings by Dataset')
        ax.legend(markerscale=3, loc='upper right', bbox_to_anchor=(1.15, 1))
        ax.grid(True, alpha=0.3)

        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def plot_hdbscan_clusters(self, results: Dict, output_name: str = 'umap_clusters') -> str:
        """Plot HDBSCAN clusters."""
        fig, ax = plt.subplots(figsize=(14, 12))

        Z = results['umap_coords']
        clusters = results['cluster_labels']
        n_clusters = results['n_clusters']

        # Color palette for clusters
        palette = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))

        for k in np.unique(clusters):
            mask = clusters == k

            if k == -1:
                color = 'lightgray'
                label = f'Noise (n={np.sum(mask)})'
                alpha = 0.3
            else:
                color = palette[k % len(palette)]
                label = f'Cluster {k} (n={np.sum(mask)})'
                alpha = 0.6

            ax.scatter(Z[mask, 0], Z[mask, 1],
                       s=10, alpha=alpha, c=[color],
                       label=label, rasterized=True)

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(f'HDBSCAN Clusters ({n_clusters} clusters found)')
        ax.legend(markerscale=3, loc='upper right', bbox_to_anchor=(1.2, 1))
        ax.grid(True, alpha=0.3)

        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def plot_cluster_composition(self, results: Dict, output_name: str = 'cluster_composition') -> str:
        """Analyze and plot composition of each cluster by source."""
        clusters = results['cluster_labels']
        dataset_names = results['dataset_names']
        cell_type_labels = results['cell_type_labels']
        unique_clusters = sorted([c for c in np.unique(clusters) if c != -1])

        unique_datasets = np.unique(dataset_names)
        unique_cell_types = np.unique(cell_type_labels)

        # Create composition matrix
        composition = np.zeros((len(unique_clusters), len(unique_datasets)))
        cell_type_composition = np.zeros((len(unique_clusters), len(unique_cell_types)))

        for i, cluster in enumerate(unique_clusters):
            mask = clusters == cluster
            for j, dataset in enumerate(unique_datasets):
                composition[i, j] = np.sum((clusters == cluster) & (dataset_names == dataset))

        for i, cluster in enumerate(unique_clusters):
            mask = clusters == cluster
            for j, cell_type in enumerate(unique_cell_types):
                cell_type_composition[i, j] = np.sum((clusters == cluster) & (cell_type_labels == cell_type))

        # Normalize to percentages
        composition_pct = composition / composition.sum(axis=1, keepdims=True) * 100
        cell_type_pct = cell_type_composition / cell_type_composition.sum(axis=1, keepdims=True) * 100

        # Plot synth vs gt
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Absolute counts
        ax = axes[0]
        x = np.arange(len(unique_clusters))
        width = 0.8 / len(unique_datasets)
        colors = list(self.model_colors.values())

        for j, dataset in enumerate(unique_datasets):
            ax.bar(x + j * width, composition[:, j], width,
                   label=dataset, color=colors[j % len(colors)])

        ax.set_xlabel('Cluster')
        ax.set_ylabel('Count')
        ax.set_title('Cluster Composition (Absolute)')
        ax.set_xticks(x + width * (len(unique_datasets) - 1) / 2)
        ax.set_xticklabels([f'C{c}' for c in unique_clusters])
        ax.legend()

        # Percentage stacked bar
        ax = axes[1]
        bottom = np.zeros(len(unique_clusters))

        for j, dataset in enumerate(unique_datasets):
            ax.bar(x, composition_pct[:, j], width=0.6, bottom=bottom,
                   label=dataset, color=colors[j % len(colors)])
            bottom += composition_pct[:, j]

        ax.set_xlabel('Cluster')
        ax.set_ylabel('Percentage')
        ax.set_title('Cluster Composition (Percentage)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{c}' for c in unique_clusters])
        ax.legend(loc='upper right')
        ax.set_ylim(0, 105)

        plt.tight_layout()

        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def plot_cell_type_distribution(self, results: Dict, output_name: str = 'umap_cell_types') -> str:
        """Plot UMAP colored by cell type."""
        fig, ax = plt.subplots(figsize=(14, 12))

        Z = results['umap_coords']
        cell_types = results['cell_type_labels']
        unique_types = np.unique(cell_types)

        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_types)))

        for i, ct in enumerate(unique_types):
            mask = cell_types == ct
            ax.scatter(Z[mask, 0], Z[mask, 1],
                       s=8, alpha=0.5, c=[colors[i]],
                       label=f'{ct} (n={np.sum(mask)})', rasterized=True)

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('Embeddings by Cell Type')
        ax.legend(markerscale=3, loc='upper right', bbox_to_anchor=(1.25, 1), fontsize=8)
        ax.grid(True, alpha=0.3)

        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def plot_comprehensive_dashboard(self, results: Dict, output_name: str = 'umap_dashboard') -> str:
        """Create comprehensive 4-panel dashboard."""
        fig = plt.figure(figsize=(20, 18))

        Z = results['umap_coords']
        source = results['source_labels']
        dataset_names = results['dataset_names']
        clusters = results['cluster_labels']
        cell_types = results['cell_type_labels']

        # Panel 1: Reference vs Synthetic
        ax1 = fig.add_subplot(2, 2, 1)
        ref_mask = source == 'reference'
        synth_mask = source == 'synthetic'

        ax1.scatter(Z[ref_mask, 0], Z[ref_mask, 1], s=5, alpha=0.5,
                    c=self.model_colors['ground_truth'], label='Ground Truth', rasterized=True)
        ax1.scatter(Z[synth_mask, 0], Z[synth_mask, 1], s=5, alpha=0.5,
                    c=self.model_colors['original'], label='Synthetic', rasterized=True)
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        ax1.set_title('A. Ground Truth vs Synthetic')
        ax1.legend(markerscale=3)
        ax1.grid(True, alpha=0.3)

        # Panel 2: By Dataset
        ax2 = fig.add_subplot(2, 2, 2)
        unique_datasets = np.unique(dataset_names)
        colors = list(self.model_colors.values())

        for i, dataset in enumerate(unique_datasets):
            mask = dataset_names == dataset
            ax2.scatter(Z[mask, 0], Z[mask, 1], s=5, alpha=0.5,
                        c=colors[i % len(colors)], label=dataset, rasterized=True)

        ax2.set_xlabel('UMAP 1')
        ax2.set_ylabel('UMAP 2')
        ax2.set_title('B. By Dataset')
        ax2.legend(markerscale=3)
        ax2.grid(True, alpha=0.3)

        # Panel 3: HDBSCAN Clusters
        ax3 = fig.add_subplot(2, 2, 3)
        n_clusters = results['n_clusters']
        palette = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 1)))

        for k in np.unique(clusters):
            mask = clusters == k
            if k == -1:
                color = 'lightgray'
                alpha = 0.2
            else:
                color = palette[k % len(palette)]
                alpha = 0.6
            ax3.scatter(Z[mask, 0], Z[mask, 1], s=5, alpha=alpha,
                        c=[color], label=f'C{k}' if k != -1 else 'Noise', rasterized=True)

        ax3.set_xlabel('UMAP 1')
        ax3.set_ylabel('UMAP 2')
        ax3.set_title(f'C. HDBSCAN Clusters ({n_clusters} found)')
        ax3.legend(markerscale=3, ncol=2)
        ax3.grid(True, alpha=0.3)

        # Panel 4: By Cell Type
        ax4 = fig.add_subplot(2, 2, 4)
        unique_types = np.unique(cell_types)
        type_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_types)))

        for i, ct in enumerate(unique_types):
            mask = cell_types == ct
            ax4.scatter(Z[mask, 0], Z[mask, 1], s=5, alpha=0.5,
                        c=[type_colors[i]], label=ct, rasterized=True)

        ax4.set_xlabel('UMAP 1')
        ax4.set_ylabel('UMAP 2')
        ax4.set_title('D. By Cell Type')
        ax4.legend(markerscale=3, fontsize=8)
        ax4.grid(True, alpha=0.3)

        plt.suptitle('UMAP Embedding Analysis Dashboard', fontsize=18, y=0.98)
        plt.tight_layout()

        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def plot_synthetic_concentration(self, results: Dict, output_name: str = 'synthetic_concentration') -> str:
        """
        Show how synthetic data concentrates in specific regions.
        Key visualization for demonstrating mode collapse.
        """
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))

        Z = results['umap_coords']
        source = results['source_labels']
        clusters = results['cluster_labels']

        ref_mask = source == 'reference'
        synth_mask = source == 'synthetic'

        # Panel 1: Density plot for reference
        ax = axes[0]
        ax.hexbin(Z[ref_mask, 0], Z[ref_mask, 1], gridsize=50,
                  cmap='Greens', mincnt=1)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('Ground Truth Density')
        plt.colorbar(ax.collections[0], ax=ax, label='Count')

        # Panel 2: Density plot for synthetic
        ax = axes[1]
        ax.hexbin(Z[synth_mask, 0], Z[synth_mask, 1], gridsize=50,
                  cmap='Reds', mincnt=1)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('Synthetic Density')
        plt.colorbar(ax.collections[0], ax=ax, label='Count')

        # Panel 3: Cluster composition
        ax = axes[2]
        unique_clusters = sorted([c for c in np.unique(clusters) if c != -1])

        ref_counts = []
        synth_counts = []

        for c in unique_clusters:
            cluster_mask = clusters == c
            ref_counts.append(np.sum(cluster_mask & ref_mask))
            synth_counts.append(np.sum(cluster_mask & synth_mask))

        x = np.arange(len(unique_clusters))
        width = 0.35

        ax.bar(x - width / 2, ref_counts, width, label='Ground Truth',
               color=self.model_colors['ground_truth'])
        ax.bar(x + width / 2, synth_counts, width, label='Synthetic',
               color=self.model_colors['original'])

        ax.set_xlabel('Cluster')
        ax.set_ylabel('Count')
        ax.set_title('Samples per Cluster')
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{c}' for c in unique_clusters])
        ax.legend()

        plt.tight_layout()

        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def export_cluster_assignments(self, results: Dict, output_name: str = 'cluster_assignments') -> str:
        """Export cluster assignments to CSV for downstream analysis."""
        df = pd.DataFrame({
            'sequence_id': results['sequence_ids'],
            'source': results['source_labels'],
            'dataset': results['dataset_names'],
            'cell_type': results['cell_type_labels'],
            'cluster': results['cluster_labels'],
            'umap_1': results['umap_coords'][:, 0],
            'umap_2': results['umap_coords'][:, 1],
        })

        filepath = self.output_dir / f"{output_name}.csv"
        df.to_csv(filepath, index=False)

        # Also export UMAP coordinates as numpy array
        np.save(self.output_dir / 'umap_coordinates.npy', results['umap_coords'])

        return str(filepath)

    def export_cluster_statistics(self, results: Dict, output_name: str = 'cluster_statistics') -> str:
        """Export detailed statistics for each cluster."""
        clusters = results['cluster_labels']
        source = results['source_labels']
        dataset_names = results['dataset_names']
        cell_types = results['cell_type_labels']

        unique_clusters = sorted([c for c in np.unique(clusters) if c != -1])

        stats = []
        for c in unique_clusters:
            mask = clusters == c
            cluster_sources = source[mask]
            cluster_datasets = dataset_names[mask]
            cluster_types = cell_types[mask]

            stat = {
                'cluster': c,
                'total': int(np.sum(mask)),
                'n_reference': int(np.sum(cluster_sources == 'reference')),
                'n_synthetic': int(np.sum(cluster_sources == 'synthetic')),
                'pct_reference': float(np.mean(cluster_sources == 'reference') * 100),
                'pct_synthetic': float(np.mean(cluster_sources == 'synthetic') * 100),
            }

            # Dataset breakdown
            for dataset in np.unique(dataset_names):
                stat[f'n_{dataset}'] = int(np.sum(cluster_datasets == dataset))

            # Cell type breakdown
            for ct in np.unique(cell_types):
                stat[f'n_{ct}'] = int(np.sum(cluster_types == ct))

            stats.append(stat)

        # Add noise statistics
        noise_mask = clusters == -1
        if np.any(noise_mask):
            stats.append({
                'cluster': -1,
                'total': int(np.sum(noise_mask)),
                'n_reference': int(np.sum(source[noise_mask] == 'reference')),
                'n_synthetic': int(np.sum(source[noise_mask] == 'synthetic')),
                'pct_reference': float(np.mean(source[noise_mask] == 'reference') * 100),
                'pct_synthetic': float(np.mean(source[noise_mask] == 'synthetic') * 100),
            })

        df = pd.DataFrame(stats)
        filepath = self.output_dir / f"{output_name}.csv"
        df.to_csv(filepath, index=False)

        return str(filepath)


def main():
    parser = argparse.ArgumentParser(
        description='UMAP Embedding Visualization with HDBSCAN Clustering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with one synthetic dataset
  python 02_umap_clustering.py \\
      --reference ground_truth:data/gt_embeddings.h5 \\
      --synthetic original:data/synth_embeddings.h5

  # Multiple synthetic datasets with sampling
  python 02_umap_clustering.py \\
      --reference ground_truth:data/gt.h5:10000 \\
      --synthetic original:data/synth1.h5:5000 improved:data/synth2.h5:5000

  # Custom UMAP and HDBSCAN parameters
  python 02_umap_clustering.py \\
      --reference gt:data/gt.h5 \\
      --synthetic model1:data/s1.h5 \\
      --n-neighbors 30 --min-dist 0.05 \\
      --min-cluster-size 100

  # Re-cluster using previous UMAP (fast iteration on HDBSCAN params)
  python 02_umap_clustering.py \\
      --reference gt:data/gt.h5 \\
      --synthetic model1:data/s1.h5 \\
      --load-umap previous_results/ \\
      --min-cluster-size 20 --min-samples 5 --cluster-method leaf
        """
    )

    parser.add_argument('--reference', '-r', required=True,
                        help='Reference dataset as name:path[:n_samples]')
    parser.add_argument('--synthetic', '-s', nargs='+', required=True,
                        help='Synthetic datasets as name:path[:n_samples]')
    parser.add_argument('--output', '-o', default='umap_results',
                        help='Output directory')

    # UMAP parameters
    parser.add_argument('--n-neighbors', type=int, default=25,
                        help='UMAP n_neighbors parameter')
    parser.add_argument('--min-dist', type=float, default=0.1,
                        help='UMAP min_dist parameter')
    parser.add_argument('--metric', default='euclidean',
                        help='UMAP distance metric')

    # HDBSCAN parameters
    parser.add_argument('--min-cluster-size', type=int, default=200,
                        help='HDBSCAN min_cluster_size (smaller = more clusters)')
    parser.add_argument('--min-samples', type=int, default=50,
                        help='HDBSCAN min_samples (smaller = more clusters)')
    parser.add_argument('--cluster-epsilon', type=float, default=0.0,
                        help='HDBSCAN cluster_selection_epsilon (0 = no merging)')
    parser.add_argument('--cluster-method', choices=['eom', 'leaf'], default='eom',
                        help='HDBSCAN cluster_selection_method (leaf = finer clusters)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='HDBSCAN alpha (lower = more clusters)')

    # Load previous UMAP
    parser.add_argument('--load-umap', type=str, default=None,
                        help='Path to directory with previous UMAP results (skips UMAP, re-runs HDBSCAN only)')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Parse reference config
    ref_parts = args.reference.split(':')
    ref_config = DatasetConfig(
        name=ref_parts[0],
        filepath=ref_parts[1],
        n_samples=int(ref_parts[2]) if len(ref_parts) > 2 else None,
        is_reference=True
    )

    # Parse synthetic configs
    synth_configs = []
    for spec in args.synthetic:
        parts = spec.split(':')
        synth_configs.append(DatasetConfig(
            name=parts[0],
            filepath=parts[1],
            n_samples=int(parts[2]) if len(parts) > 2 else None,
            is_reference=False
        ))

    # Setup UMAP config
    umap_config = UMAPConfig(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.cluster_epsilon,
        cluster_selection_method=args.cluster_method,
        alpha=args.alpha
    )

    # Run analysis
    visualizer = UMAPVisualizer(config=umap_config, output_dir=args.output)
    results = visualizer.analyze_multiple_datasets(
        ref_config, synth_configs,
        load_previous_umap=args.load_umap
    )

    # Generate all visualizations
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)

    figures = {
        'ref_vs_synth': visualizer.plot_reference_vs_synthetic(results),
        'by_dataset': visualizer.plot_by_dataset(results),
        'clusters': visualizer.plot_hdbscan_clusters(results),
        'composition': visualizer.plot_cluster_composition(results),
        'cell_types': visualizer.plot_cell_type_distribution(results),
        'chromosomes': visualizer.plot_by_chromosome(results),
        'chrom_cluster': visualizer.plot_chromosome_by_cluster(results),
        'dashboard': visualizer.plot_comprehensive_dashboard(results),
        'concentration': visualizer.plot_synthetic_concentration(results),
    }

    # Export data
    cluster_file = visualizer.export_cluster_assignments(results)
    stats_file = visualizer.export_cluster_statistics(results)

    # Save config
    config_dict = {
        'reference': {
            'name': ref_config.name,
            'filepath': ref_config.filepath,
            'n_samples': ref_config.n_samples,
        },
        'synthetic': [
            {'name': sc.name, 'filepath': sc.filepath, 'n_samples': sc.n_samples}
            for sc in synth_configs
        ],
        'umap_config': {
            'n_neighbors': umap_config.n_neighbors,
            'min_dist': umap_config.min_dist,
            'metric': umap_config.metric,
        },
        'hdbscan_config': {
            'min_cluster_size': umap_config.min_cluster_size,
            'min_samples': umap_config.min_samples,
            'cluster_selection_epsilon': umap_config.cluster_selection_epsilon,
        },
        'results': {
            'n_clusters': results['n_clusters'],
            'n_total_samples': len(results['umap_coords']),
        }
    }

    with open(Path(args.output) / 'analysis_config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {args.output}")
    print(f"\nFound {results['n_clusters']} clusters")
    print(f"\nGenerated figures:")
    for name, path in figures.items():
        print(f"  - {name}: {path}")

    print(f"\nExported data:")
    print(f"  - Cluster assignments: {cluster_file}")
    print(f"  - Cluster statistics: {stats_file}")

    # Print cluster summary
    print("\nCluster Summary:")
    stats_df = pd.read_csv(stats_file)
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
