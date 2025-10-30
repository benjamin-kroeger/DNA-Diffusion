"""
Compare ground truth and synthetic DNA embeddings using UMAP visualization.
"""

import h5py
import umap
import matplotlib.pyplot as plt
import numpy as np
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore', message='n_jobs value')


def load_gt_embeddings_from_h5(filepath, n_samples=None):
    """
    Load ground truth embeddings from h5 file.

    Returns:
        X: (n_samples, L, D) embeddings
        datalabels: (n_samples,) numeric labels
        tags: (n_samples,) string tags
        ids: (n_samples,) sequence IDs
    """
    with h5py.File(filepath, 'r') as f:
        # Assuming your original load function works - use it
        from dnadiffusion.utils.caduceus_tokenization import load_embeddings_from_h5

    return load_embeddings_from_h5(filepath, n_samples=n_samples)


def load_synthetic_embeddings_from_h5(filepath, n_samples=None):
    """
    Load synthetic embeddings from h5 file with structure:

    seq_0/
        attrs: cell_type
        embedding: dataset with shape from sampled_images list
    seq_1/
        ...

    Returns:
        X: (n_samples, L, D) embeddings
        cell_types: (n_samples,) string cell type labels
    """
    embeddings = []
    cell_types = []

    with h5py.File(filepath, 'r') as f:
        # Get all sequence groups
        seq_keys = sorted([k for k in f.keys() if k.startswith('seq_')],
                          key=lambda x: int(x.split('_')[1]))

        if n_samples is not None:
            seq_keys = seq_keys[:n_samples]

        for seq_key in seq_keys:
            grp = f[seq_key]

            # Get embedding - it's stored as the last timestep from sampled_images
            # sampled_images is a list, so embedding dataset contains all timesteps
            emb_data = grp['embedding'][:]  # Shape depends on how it was saved

            # If it's the full list of timesteps, take the last one
            if emb_data.ndim == 5:  # (timesteps, B, C, D, L)
                emb_data = emb_data[-1]  # Take last timestep

            # Expected shape after taking last timestep: (B, C, D, L) or (C, D, L)
            # where B=1 (single sample), C=1 (channel)
            if emb_data.ndim == 4:  # (1, 1, D, L)
                emb_data = emb_data[0, 0]  # Remove batch and channel dims -> (D, L)
            elif emb_data.ndim == 3:  # (1, D, L)
                emb_data = emb_data[0]  # Remove channel dim -> (D, L)

            # Transpose to match GT format: (D, L) -> (L, D)
            emb_data = emb_data.T

            embeddings.append(emb_data)
            cell_types.append(grp.attrs['cell_type'])

    X = np.stack(embeddings, axis=0)  # (n_samples, L, D)
    cell_types = np.array(cell_types)

    return X, cell_types


def load_synthetic_embeddings_simple_format(filepath, n_samples=None):
    """
    Alternative loader if embeddings are in simpler format:

    embeddings: (N, D, L) dataset
    classes: (N,) dataset

    Returns:
        X: (n_samples, L, D) embeddings
        classes: (n_samples,) numeric labels
    """
    with h5py.File(filepath, 'r') as f:
        embeddings = f['embeddings'][:]  # (N, D, L)
        classes = f['classes'][:]  # (N,)

        if n_samples is not None:
            embeddings = embeddings[:n_samples]
            classes = classes[:n_samples]

        # Transpose to (N, L, D)
        X = embeddings.transpose(0, 2, 1)

        # Get cell type names from metadata if available
        if 'cell_type_name' in f.attrs:
            cell_type_name = f.attrs['cell_type_name']
            cell_types = np.array([cell_type_name] * len(classes))
            return X, cell_types, classes
        else:
            return X, classes


def average_pool_embeddings(X):
    """
    Average pool embeddings over length dimension.

    Args:
        X: (n_samples, L, D) embeddings

    Returns:
        X_pooled: (n_samples, D)
    """
    return X.mean(axis=1)


def create_comparison_umap(
    gt_filepath,
    synth_filepath,
    n_gt_samples=10000,
    n_synth_samples=10000,
    output_path='umap_comparison.png',
    use_simple_format=False,
):
    """
    Create UMAP comparison between ground truth and synthetic embeddings.

    Args:
        gt_filepath: Path to ground truth h5 file
        synth_filepath: Path to synthetic h5 file
        n_gt_samples: Number of GT samples to load
        n_synth_samples: Number of synthetic samples to load
        output_path: Where to save the plot
        use_simple_format: If True, use simple format loader for synthetic data
    """

    # Load ground truth embeddings
    print("Loading ground truth embeddings...")
    X_gt, datalabels_gt, tags_gt, ids_gt = load_gt_embeddings_from_h5(
        gt_filepath, n_samples=n_gt_samples
    )
    print(f"GT shape: {X_gt.shape}")
    print(f"Unique GT tags: {np.unique(tags_gt)}")

    # Load synthetic embeddings
    print("\nLoading synthetic embeddings...")
    if use_simple_format:
        result = load_synthetic_embeddings_simple_format(synth_filepath, n_samples=n_synth_samples)
        if len(result) == 3:
            X_synth, cell_types_synth, classes_synth = result
        else:
            X_synth, cell_types_synth = result
            classes_synth = None
    else:
        X_synth, cell_types_synth = load_synthetic_embeddings_from_h5(
            synth_filepath, n_samples=n_synth_samples
        )

    print(f"Synthetic shape: {X_synth.shape}")
    print(f"Unique synthetic cell types: {np.unique(cell_types_synth)}")

    # Average pool both datasets
    print("\nAverage pooling embeddings...")
    X_gt_pooled = average_pool_embeddings(X_gt)
    X_synth_pooled = average_pool_embeddings(X_synth)
    print(f"GT pooled shape: {X_gt_pooled.shape}")
    print(f"Synthetic pooled shape: {X_synth_pooled.shape}")

    # Combine datasets
    X_combined = np.vstack([X_gt_pooled, X_synth_pooled])
    n_gt = len(X_gt_pooled)
    n_synth = len(X_synth_pooled)

    # Create labels for source (GT vs Synthetic)
    source_labels = np.array(['GT'] * n_gt + ['Synthetic'] * n_synth)

    # Create combined cell type labels
    combined_labels = np.concatenate([tags_gt, cell_types_synth])

    # Run UMAP
    print(f"\nRunning UMAP on {len(X_combined):,} total samples...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=25,
        min_dist=0.1,
        metric='euclidean',
        random_state=42,
        low_memory=True,
        verbose=True
    )
    Z = reducer.fit_transform(X_combined)

    # Split back into GT and synthetic
    Z_gt = Z[:n_gt]
    Z_synth = Z[n_gt:]

    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(36, 10))

    # Plot 1: GT vs Synthetic (by source)
    axes[0].scatter(
        Z_gt[:, 0], Z_gt[:, 1],
        s=5, alpha=0.5, c='blue', label='Ground Truth'
    )
    axes[0].scatter(
        Z_synth[:, 0], Z_synth[:, 1],
        s=5, alpha=0.5, c='red', label='Synthetic'
    )
    axes[0].set_xlabel('UMAP 1', fontsize=12)
    axes[0].set_ylabel('UMAP 2', fontsize=12)
    axes[0].set_title(f'GT vs Synthetic\n(GT: {n_gt:,}, Synth: {n_synth:,})', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(markerscale=3, fontsize=10)

    # Plot 2: Colored by cell type (all data)
    unique_cell_types = np.unique(combined_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_cell_types)))
    cell_type_to_color = {ct: colors[i] for i, ct in enumerate(unique_cell_types)}

    for cell_type in unique_cell_types:
        mask = combined_labels == cell_type
        axes[1].scatter(
            Z[mask, 0], Z[mask, 1],
            s=5, alpha=0.6,
            label=cell_type,
            c=[cell_type_to_color[cell_type]]
        )

    axes[1].set_xlabel('UMAP 1', fontsize=12)
    axes[1].set_ylabel('UMAP 2', fontsize=12)
    axes[1].set_title(f'Colored by Cell Type\n({len(Z):,} sequences)', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=3, fontsize=8)

    # Plot 3: Side-by-side comparison for each cell type
    # Show GT in one style, synthetic in another for the same cell type
    for cell_type in unique_cell_types:
        # GT samples of this cell type
        mask_gt = (tags_gt == cell_type)
        if np.any(mask_gt):
            axes[2].scatter(
                Z_gt[mask_gt, 0], Z_gt[mask_gt, 1],
                s=8, alpha=0.6,
                c=[cell_type_to_color[cell_type]],
                marker='o',
                edgecolors='none',
                label=f'{cell_type} (GT)' if cell_type in cell_types_synth else cell_type
            )

        # Synthetic samples of this cell type
        mask_synth = (cell_types_synth == cell_type)
        if np.any(mask_synth):
            axes[2].scatter(
                Z_synth[mask_synth, 0], Z_synth[mask_synth, 1],
                s=8, alpha=0.6,
                c=[cell_type_to_color[cell_type]],
                marker='x',
                linewidths=1.5,
                label=f'{cell_type} (Synth)'
            )

    axes[2].set_xlabel('UMAP 1', fontsize=12)
    axes[2].set_ylabel('UMAP 2', fontsize=12)
    axes[2].set_title('GT (circles) vs Synthetic (x)\nby Cell Type', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {output_path}")
    plt.show()

    # Print statistics
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print(f"\nGround Truth:")
    print(f"  Total samples: {n_gt:,}")
    print(f"  Cell types: {np.unique(tags_gt)}")
    for ct in np.unique(tags_gt):
        count = np.sum(tags_gt == ct)
        print(f"    {ct}: {count:,}")

    print(f"\nSynthetic:")
    print(f"  Total samples: {n_synth:,}")
    print(f"  Cell types: {np.unique(cell_types_synth)}")
    for ct in np.unique(cell_types_synth):
        count = np.sum(cell_types_synth == ct)
        print(f"    {ct}: {count:,}")

    return Z, source_labels, combined_labels


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Paths to your data
    gt_filepath = "/home/benjaminkroeger/Documents/Master/UBC/Synthetic_data/DNA-Diffusion/data/embeddings/caduceus/train_embeddings.h5"
    synth_filepath = "/home/benjaminkroeger/Documents/Master/UBC/Synthetic_data/DNA-Diffusion/sampled_embedd.h5"  # UPDATE THIS

    # Run comparison
    Z, sources, labels = create_comparison_umap(
        gt_filepath=gt_filepath,
        synth_filepath=synth_filepath,
        n_gt_samples=10000,
        n_synth_samples=10000,
        output_path='umap_gt_vs_synthetic.png',
        use_simple_format=False,  # Set True if using the simple format loader
    )

    print("\nDone!")
