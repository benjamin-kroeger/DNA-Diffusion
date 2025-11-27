"""
Compare ground truth and synthetic DNA embeddings using UMAP visualization
and cluster the UMAP space into 3 clusters to inspect what they are.
"""

import h5py
import umap
import matplotlib.pyplot as plt
import numpy as np
import warnings
from pathlib import Path
from sklearn.cluster import KMeans

# Suppress warnings
warnings.filterwarnings('ignore', message='n_jobs value')


def load_embeddings_from_h5(filepath, n_samples=None):
    """
    Load ground truth embeddings from h5 file.

    Returns:
        X: (n_samples, L, D) embeddings
        datalabels: (n_samples,) numeric/string labels (your cell types)
        tags: (n_samples,) string tags
        ids: (n_samples,) sequence IDs
    """
    with h5py.File(filepath, 'r') as f:
        # your existing util does the real work
        from dnadiffusion.utils.caduceus_tokenization import load_embeddings_from_h5

        return load_embeddings_from_h5(filepath, n_samples=n_samples)


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
):
    # ------------------------------------------------------------------
    # 1) LOAD DATA
    # ------------------------------------------------------------------
    print("Loading ground truth embeddings...")
    X_gt, datalabels_gt, tags_gt, ids_gt = load_gt_embeddings_from_h5(
        gt_filepath, n_samples=n_gt_samples
    )
    print(f"GT shape: {X_gt.shape}")

    print("\nLoading synthetic embeddings...")
    X_synth, datalabels_synth, tags_synth, ids_synth = load_gt_embeddings_from_h5(
        synth_filepath, n_samples=n_synth_samples
    )
    print(f"Synth shape: {X_synth.shape}")

    # ------------------------------------------------------------------
    # 2) POOL
    # ------------------------------------------------------------------
    print("\nAverage pooling ...")
    X_gt_pooled = average_pool_embeddings(X_gt)      # (N_gt, D)
    X_synth_pooled = average_pool_embeddings(X_synth)  # (N_synth, D)

    X_combined = np.vstack([X_gt_pooled, X_synth_pooled])
    n_gt = len(X_gt_pooled)
    n_synth = len(X_synth_pooled)

    # labels for plotting
    source_labels = np.array(["GT"] * n_gt + ["Synthetic"] * n_synth)
    combined_cell_types = np.concatenate([tags_gt, datalabels_synth])

    # ------------------------------------------------------------------
    # 3) UMAP
    # ------------------------------------------------------------------
    print(f"\nRunning UMAP on {len(X_combined):,} points ...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=25,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
        low_memory=True,
        verbose=True,
    )
    Z = reducer.fit_transform(X_combined)  # (N_gt + N_synth, 2)
    Z_gt = Z[:n_gt]
    Z_synth = Z[n_gt:]

    # ------------------------------------------------------------------
    # 4) PLOT (2 subplots only)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    # 4.1 GT vs Synth
    axes[0].scatter(Z_gt[:, 0], Z_gt[:, 1], s=5, alpha=0.5, c="blue", label="Ground Truth")
    axes[0].scatter(Z_synth[:, 0], Z_synth[:, 1], s=5, alpha=0.5, c="red", label="Synthetic")
    axes[0].set_title(f"GT vs Synthetic\n(GT: {n_gt:,}, Synth: {n_synth:,})")
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(markerscale=3)

    # 4.2 all by cell type
    unique_ct = np.unique(combined_cell_types)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_ct)))
    ct2color = {ct: colors[i] for i, ct in enumerate(unique_ct)}
    for ct in unique_ct:
        m = (combined_cell_types == ct)
        axes[1].scatter(Z[m, 0], Z[m, 1], s=5, alpha=0.6, c=[ct2color[ct]], label=ct)
    axes[1].set_title(f"Colored by Cell Type\n({len(Z):,} sequences)")
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to {output_path}")

    # ------------------------------------------------------------------
    # 5) HDBSCAN ON UMAP SPACE
    # ------------------------------------------------------------------
    import hdbscan

    print("\nRunning HDBSCAN on UMAP embeddings ...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=200,      # tweak depending on dataset size
        min_samples=50,            # controls how conservative clustering is
        metric='euclidean',
        cluster_selection_epsilon=0.1
    )
    cluster_ids = clusterer.fit_predict(Z)
    n_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
    print(f"Found {n_clusters} clusters (plus noise).")

    # Plot clusters with noise
    fig2, ax = plt.subplots(figsize=(12, 10))
    palette = plt.cm.get_cmap("tab10", n_clusters)
    for k in np.unique(cluster_ids):
        mask = cluster_ids == k
        if k == -1:
            color = "lightgray"
            label = "noise"
        else:
            color = palette(k)
            label = f"cluster {k}"
        ax.scatter(Z[mask, 0], Z[mask, 1], s=6, alpha=0.6, color=color, label=label)

    ax.set_title(f"HDBSCAN clusters on UMAP space (found {n_clusters})")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig("umap_hdbscan_clusters.png", dpi=300, bbox_inches="tight")
    print("Saved HDBSCAN plot to umap_hdbscan_clusters.png")

    # ------------------------------------------------------------------
    # 5c) EXPORT CLUSTER MAPPING TO CSV
    # ------------------------------------------------------------------
    import pandas as pd
    import os

    # Build combined arrays
    ids_all = np.concatenate([ids_gt, ids_synth])
    source_all = np.array(["GT"] * n_gt + ["Synthetic"] * len(ids_synth))
    cluster_all = cluster_ids.astype(int)

    # optional: cell types (if you have them)
    try:
        celltypes_all = np.concatenate([tags_gt, datalabels_synth])
    except Exception:
        celltypes_all = np.array(["NA"] * len(ids_all))

    # make DataFrame
    df_clusters = pd.DataFrame({
        "id": ids_all,
        "source": source_all,
        "cluster": cluster_all,
        "cell_type": celltypes_all
    })

    # sort for readability
    df_clusters = df_clusters.sort_values(["cluster", "source"]).reset_index(drop=True)

    # save to csv
    out_path = "cluster_id_mapping.csv"
    df_clusters.to_csv(out_path, index=False)
    print(f"✅ Saved cluster-to-ID mapping to {os.path.abspath(out_path)}")

    # show summary
    print(df_clusters["cluster"].value_counts(dropna=False))
    print(df_clusters.head())

    # ------------------------------------------------------------------
    # 6) PLOT CLUSTERS (and centers if available)
    # ------------------------------------------------------------------
    fig2, ax = plt.subplots(figsize=(12, 10))

    unique_clusters = np.unique(cluster_ids)
    # if you used HDBSCAN, -1 means noise
    has_noise = -1 in unique_clusters
    n_clusters = len(unique_clusters) - (1 if has_noise else 0)

    # make a colormap big enough
    cmap = plt.cm.get_cmap("tab20", max(n_clusters, 1))

    for k in unique_clusters:
        mask = (cluster_ids == k)

        if k == -1:
            # noise points
            ax.scatter(
                Z[mask, 0], Z[mask, 1],
                s=5, alpha=0.25,
                color="lightgray",
                label="noise" if "noise" not in ax.get_legend_handles_labels()[1] else None,
            )
        else:
            ax.scatter(
                Z[mask, 0], Z[mask, 1],
                s=6, alpha=0.6,
                color=cmap(k),
                label=f"cluster {k}",
            )

    # optional: overlay GT/Synth faintly for orientation
    ax.scatter(Z_gt[:, 0], Z_gt[:, 1], s=2, alpha=0.12, color="blue", label="_gt_bg")
    ax.scatter(Z_synth[:, 0], Z_synth[:, 1], s=2, alpha=0.12, color="red", label="_synth_bg")

    # if the clustering algo has centers (e.g. KMeans), plot them
    if hasattr(clusterer, "cluster_centers_"):
        centers = clusterer.cluster_centers_
        for k, (cx, cy) in enumerate(centers):
            ax.scatter(
                cx, cy,
                s=320,
                color="black",
                marker="X",
                edgecolors="white",
                linewidths=1.5,
                zorder=10,
            )
            ax.text(
                cx + 0.15,
                cy + 0.15,
                f"C{k}",
                fontsize=12,
                weight="bold",
                color="black",
                zorder=11,
            )

    ax.set_title(f"UMAP with clustered regions ({n_clusters} clusters{', + noise' if has_noise else ''})")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, markerscale=2)
    plt.tight_layout()
    plt.savefig("umap_clusters.png", dpi=300, bbox_inches="tight")
    print("Saved plot to umap_clusters.png")

    return Z, source_labels, combined_cell_types, cluster_ids


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    gt_filepath = "/home/benjaminkroeger/Documents/Master/UBC/Synthetic_data/DNA-Diffusion/data/embeddings/caduceus/train_embeddings.h5"
    synth_filepath = "/home/benjaminkroeger/Documents/Master/UBC/Synthetic_data/DNA-Diffusion/data/outputs/original_model_colab/synth_emebeds.h5"

    Z, sources, labels, clusters = create_comparison_umap(
        gt_filepath=gt_filepath,
        synth_filepath=synth_filepath,
        n_gt_samples=10000,
        n_synth_samples=4000,
        output_path="umap_gt_vs_synthetic.png",
    )
    print("\nDone.")
