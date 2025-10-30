import h5py
import umap
import matplotlib.pyplot as plt
import numpy as np
import warnings
from dnadiffusion.utils.caduceus_tokenization import load_embeddings_from_h5

# Suppress the UMAP n_jobs warning
warnings.filterwarnings('ignore', message='n_jobs value')

# Parameters
n_samples = 40000

# Load embeddings and metadata
print("Loading embeddings...")
X_train, datalabels, tags, ids = load_embeddings_from_h5(
    "/home/benjaminkroeger/Documents/Master/UBC/Synthetic_data/DNA-Diffusion/data/embeddings/caduceus/train_embeddings.h5",
    n_samples=n_samples
)

B, L, D = X_train.shape
print(f"Original shape: {X_train.shape}")

# Average pool over length dimension to get (n_samples, D)
X_flat = X_train.mean(axis=1)  # Shape: (n_samples, D)
print(f"Averaged embedding shape: {X_flat.shape}")

# Apply UMAP with optimized parameters for large datasets
print("Running UMAP... (this may take a few minutes)")
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=25,
    min_dist=0.1,
    metric='euclidean',
    random_state=42,
    low_memory=True,
    verbose=True
)
Z = reducer.fit_transform(X_flat)

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(24, 10))

# Plot 1: Colored by TAG
unique_tags = np.unique(tags)
colors_tag = plt.cm.tab20(np.linspace(0, 1, len(unique_tags)))
tag_to_color = {tag: colors_tag[i] for i, tag in enumerate(unique_tags)}

for tag in unique_tags:
    mask = tags == tag
    axes[0].scatter(
        Z[mask, 0], Z[mask, 1],
        s=5, alpha=0.6,
        label=tag,
        c=[tag_to_color[tag]]
    )

axes[0].set_xlabel('UMAP 1', fontsize=12)
axes[0].set_ylabel('UMAP 2', fontsize=12)
axes[0].set_title(f'UMAP colored by TAG ({len(Z):,} sequences)', fontsize=14)
axes[0].grid(True, alpha=0.3)
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=3)

# Plot 2: Colored by datalabel
unique_labels = np.unique(datalabels)
colors_label = plt.cm.tab20b(np.linspace(0, 1, len(unique_labels)))
label_to_color = {label: colors_label[i] for i, label in enumerate(unique_labels)}

for label in unique_labels:
    mask = datalabels == label
    axes[1].scatter(
        Z[mask, 0], Z[mask, 1],
        s=5, alpha=0.6,
        label=label,
        c=[label_to_color[label]]
    )

axes[1].set_xlabel('UMAP 1', fontsize=12)
axes[1].set_ylabel('UMAP 2', fontsize=12)
axes[1].set_title(f'UMAP colored by datalabel ({len(Z):,} sequences)', fontsize=14)
axes[1].grid(True, alpha=0.3)
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=3)

plt.tight_layout()
plt.savefig('umap_40k_samples_colored.png', dpi=300, bbox_inches='tight')
plt.show()

print("Done!")
print(f"Unique TAGs: {unique_tags}")
print(f"Unique datalabels: {unique_labels}")
