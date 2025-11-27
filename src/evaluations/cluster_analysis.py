"""
Compute per-cluster summary statistics with k-mer analysis.
Focus on finding what makes clusters 0 and 1 special compared to cluster 2.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def load_and_merge_data(cluster_file, data_file):
    """Load cluster assignments and data table, then merge them properly."""
    print("Loading cluster assignments...")
    df_clusters = pd.read_csv(cluster_file)
    print(f"Cluster file shape: {df_clusters.shape}")

    print("\nLoading data table...")
    df_data = pd.read_csv(data_file, sep='\t')
    print(f"Data file shape: {df_data.shape}")

    print("\nMerging datasets...")
    df_merged = df_clusters.merge(
        df_data,
        left_on='id',
        right_on='dhs_id',
        how='left'
    )

    print(f"Merged shape: {df_merged.shape}")
    print(f"Merge success rate: {(~df_merged['dhs_id'].isna()).sum() / len(df_merged) * 100:.2f}%")

    return df_merged


def count_kmers(sequence, k):
    """Count k-mers in a sequence."""
    if pd.isna(sequence) or len(sequence) < k:
        return Counter()

    sequence = sequence.upper()
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    # Filter out k-mers with N's
    kmers = [kmer for kmer in kmers if 'N' not in kmer]
    return Counter(kmers)


def compute_kmer_frequencies(df, k, cluster_col='cluster'):
    """
    Compute k-mer frequencies for each cluster.

    Returns:
        DataFrame with k-mers as rows and clusters as columns (frequencies)
    """
    print(f"\nComputing {k}-mer frequencies...")

    # Get all possible k-mers
    all_kmers = [''.join(p) for p in product('ACGT', repeat=k)]

    cluster_kmer_counts = {}

    for cluster in sorted(df[cluster_col].unique()):
        cluster_seqs = df[df[cluster_col] == cluster]['sequence'].dropna()

        # Aggregate all k-mer counts for this cluster
        total_counts = Counter()
        for seq in cluster_seqs:
            total_counts.update(count_kmers(seq, k))

        # Convert to frequencies
        total = sum(total_counts.values())
        if total > 0:
            cluster_kmer_counts[cluster] = {kmer: count/total for kmer, count in total_counts.items()}
        else:
            cluster_kmer_counts[cluster] = {kmer: 0 for kmer in all_kmers}

    # Create DataFrame
    df_kmer = pd.DataFrame(cluster_kmer_counts).fillna(0)
    df_kmer = df_kmer.reindex(all_kmers, fill_value=0)

    return df_kmer


def find_discriminative_kmers(df_kmer, top_n=20):
    """
    Find k-mers that are most discriminative between clusters.
    Uses coefficient of variation and absolute differences.
    """
    # Calculate statistics
    df_stats = pd.DataFrame()
    df_stats['mean'] = df_kmer.mean(axis=1)
    df_stats['std'] = df_kmer.std(axis=1)
    df_stats['cv'] = df_stats['std'] / (df_stats['mean'] + 1e-10)  # Coefficient of variation
    df_stats['max'] = df_kmer.max(axis=1)
    df_stats['min'] = df_kmer.min(axis=1)
    df_stats['range'] = df_stats['max'] - df_stats['min']

    # For each cluster pair, compute absolute difference
    clusters = df_kmer.columns.tolist()
    for i, c1 in enumerate(clusters):
        for c2 in clusters[i+1:]:
            df_stats[f'diff_{c1}_vs_{c2}'] = abs(df_kmer[c1] - df_kmer[c2])

    # Sort by range (most discriminative)
    df_stats = df_stats.sort_values('range', ascending=False)

    return df_stats.head(top_n)


def plot_kmer_analysis(df_kmer, k, output_dir='cluster_plots', top_n=30):
    """Create visualizations for k-mer analysis."""

    # Find discriminative k-mers
    discriminative = find_discriminative_kmers(df_kmer, top_n=top_n)
    top_kmers = discriminative.index.tolist()

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Heatmap of top discriminative k-mers
    ax1 = fig.add_subplot(gs[0, :])
    data_to_plot = df_kmer.loc[top_kmers[:20]]
    sns.heatmap(data_to_plot.T, cmap='YlOrRd', annot=False, fmt='.4f',
                cbar_kws={'label': 'Frequency'}, ax=ax1)
    ax1.set_title(f'Top 20 Most Discriminative {k}-mers Across Clusters', fontsize=14, fontweight='bold')
    ax1.set_xlabel(f'{k}-mer', fontsize=12)
    ax1.set_ylabel('Cluster', fontsize=12)

    # 2. Enrichment in each cluster vs others
    ax2 = fig.add_subplot(gs[1, 0])
    clusters = df_kmer.columns.tolist()

    # For each cluster, find k-mers enriched in that cluster vs mean of others
    enrichment_data = []
    for cluster in clusters:
        other_clusters = [c for c in clusters if c != cluster]
        enrichment = df_kmer[cluster] - df_kmer[other_clusters].mean(axis=1)
        top_enriched = enrichment.nlargest(10)

        for kmer, enrich_val in top_enriched.items():
            enrichment_data.append({
                'cluster': cluster,
                'kmer': kmer,
                'enrichment': enrich_val,
                'freq_in_cluster': df_kmer.loc[kmer, cluster]
            })

    df_enrich = pd.DataFrame(enrichment_data)

    # Plot enrichment heatmap
    pivot_enrich = df_enrich.pivot_table(values='enrichment', index='kmer', columns='cluster', fill_value=0)
    sns.heatmap(pivot_enrich, cmap='RdBu_r', center=0, annot=False,
                cbar_kws={'label': 'Enrichment vs Others'}, ax=ax2)
    ax2.set_title(f'Top 10 Enriched {k}-mers per Cluster', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Cluster', fontsize=10)
    ax2.set_ylabel(f'{k}-mer', fontsize=10)

    # 3. Cluster-specific k-mer bar plots
    ax3 = fig.add_subplot(gs[1, 1])

    # Show mean frequency per cluster for top discriminative k-mers
    top_10_kmers = discriminative.index[:10].tolist()
    df_top10 = df_kmer.loc[top_10_kmers]

    x = np.arange(len(clusters))
    width = 0.8 / len(top_10_kmers)

    for i, kmer in enumerate(top_10_kmers):
        offset = (i - len(top_10_kmers)/2) * width + width/2
        ax3.bar(x + offset, df_top10.loc[kmer], width, label=kmer, alpha=0.8)

    ax3.set_xlabel('Cluster', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title(f'Top 10 Discriminative {k}-mer Frequencies', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(clusters)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. PCA of k-mer frequencies
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    ax4 = fig.add_subplot(gs[2, 0])

    # Standardize and run PCA
    scaler = StandardScaler()
    kmer_scaled = scaler.fit_transform(df_kmer.T)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(kmer_scaled)

    scatter = ax4.scatter(pca_result[:, 0], pca_result[:, 1],
                         c=range(len(clusters)), s=200, cmap='viridis', alpha=0.7)

    for i, cluster in enumerate(clusters):
        ax4.annotate(f'Cluster {cluster}',
                    (pca_result[i, 0], pca_result[i, 1]),
                    fontsize=12, fontweight='bold',
                    xytext=(5, 5), textcoords='offset points')

    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=10)
    ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=10)
    ax4.set_title(f'PCA of {k}-mer Frequencies', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. Coefficient of variation plot
    ax5 = fig.add_subplot(gs[2, 1])

    cv_data = discriminative[['mean', 'cv', 'range']].head(15)
    cv_data = cv_data.sort_values('cv', ascending=True)

    y_pos = np.arange(len(cv_data))
    ax5.barh(y_pos, cv_data['cv'], alpha=0.7, color='steelblue')
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(cv_data.index, fontsize=9)
    ax5.set_xlabel('Coefficient of Variation', fontsize=10)
    ax5.set_title(f'Variability of Top {k}-mers Across Clusters', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')

    plt.savefig(f'{output_dir}/{k}mer_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_dir}/{k}mer_analysis.png")
    plt.close()

    return discriminative, df_enrich


def plot_cluster_comparison(df_kmer_3, df_kmer_6, df_kmer_9, output_dir='cluster_plots'):
    """
    Create a focused comparison plot showing what's different between clusters.
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 18))

    for idx, (df_kmer, k) in enumerate([(df_kmer_3, 3), (df_kmer_6, 6), (df_kmer_9, 9)]):
        ax = axes[idx]

        # Find k-mers that distinguish cluster 0 and 1 from cluster 2
        clusters = df_kmer.columns.tolist()

        if 2 in clusters and (0 in clusters or 1 in clusters):
            # Calculate difference from cluster 2
            diff_scores = pd.DataFrame()

            for c in [0, 1, -1]:  # Check clusters 0, 1, and noise
                if c in clusters:
                    diff_scores[f'cluster_{c}_vs_2'] = df_kmer[c] - df_kmer[2]

            # Find top k-mers with largest absolute differences
            if not diff_scores.empty:
                diff_scores['max_abs_diff'] = diff_scores.abs().max(axis=1)
                top_diff = diff_scores.nlargest(20, 'max_abs_diff')

                # Plot heatmap
                plot_data = df_kmer.loc[top_diff.index]
                sns.heatmap(plot_data.T, cmap='RdYlBu_r', center=plot_data.values.mean(),
                           annot=True, fmt='.4f', cbar_kws={'label': 'Frequency'}, ax=ax)
                ax.set_title(f'{k}-mers Most Different Between Clusters 0,1 vs 2',
                           fontsize=14, fontweight='bold')
                ax.set_xlabel(f'{k}-mer', fontsize=12)
                ax.set_ylabel('Cluster', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/cluster_comparison_kmers.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_dir}/cluster_comparison_kmers.png")
    plt.close()


def analyze_gc_content(df_merged, output_dir='cluster_plots'):
    """Analyze GC content differences between clusters."""
    print("\nAnalyzing GC content...")

    def calc_gc(seq):
        if pd.isna(seq):
            return np.nan
        seq = seq.upper()
        gc_count = seq.count('G') + seq.count('C')
        total = len([b for b in seq if b in 'ACGT'])
        return gc_count / total if total > 0 else np.nan

    df_merged['gc_content'] = df_merged['sequence'].apply(calc_gc)

    # Plot GC content distribution
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Box plot
    clusters = sorted([c for c in df_merged['cluster'].unique() if c != -1])
    if -1 in df_merged['cluster'].unique():
        clusters.append(-1)

    df_merged.boxplot(column='gc_content', by='cluster', ax=axes[0])
    axes[0].set_xlabel('Cluster', fontsize=12)
    axes[0].set_ylabel('GC Content', fontsize=12)
    axes[0].set_title('GC Content Distribution by Cluster', fontsize=14, fontweight='bold')
    plt.sca(axes[0])
    plt.xticks(rotation=0)

    # Violin plot with statistics
    for cluster in clusters:
        data = df_merged[df_merged['cluster'] == cluster]['gc_content'].dropna()
        parts = axes[1].violinplot([data], positions=[cluster], showmeans=True, showmedians=True)

    axes[1].set_xlabel('Cluster', fontsize=12)
    axes[1].set_ylabel('GC Content', fontsize=12)
    axes[1].set_title('GC Content Distribution (Violin Plot)', fontsize=14, fontweight='bold')
    axes[1].set_xticks(clusters)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add mean values as text
    for cluster in clusters:
        mean_gc = df_merged[df_merged['cluster'] == cluster]['gc_content'].mean()
        axes[1].text(cluster, mean_gc, f'{mean_gc:.3f}',
                    ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/gc_content_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_dir}/gc_content_analysis.png")
    plt.close()

    # Print statistics
    print("\nGC Content Statistics by Cluster:")
    gc_stats = df_merged.groupby('cluster')['gc_content'].agg(['count', 'mean', 'median', 'std'])
    print(gc_stats)

    return gc_stats


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    # File paths
    cluster_file = "/home/benjaminkroeger/Documents/Master/UBC/Synthetic_data/DNA-Diffusion/src/evaluations/cluster_id_mapping.csv"
    data_file = "/home/benjaminkroeger/Documents/Master/UBC/Synthetic_data/DNA-Diffusion/data/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt"  # Update this path

    output_dir = 'cluster_plots'
    Path(output_dir).mkdir(exist_ok=True)

    # Load and merge data
    df_merged = load_and_merge_data(cluster_file, data_file)

    # Basic statistics
    print("\n" + "="*80)
    print("BASIC CLUSTER COMPOSITION")
    print("="*80)
    composition = df_merged.groupby('cluster').agg({
        'id': 'count',
        'source': lambda x: (x == 'GT').sum(),
    }).rename(columns={'id': 'total_count', 'source': 'gt_count'})
    composition['synthetic_count'] = composition['total_count'] - composition['gt_count']
    print(composition)

    # GC content analysis
    gc_stats = analyze_gc_content(df_merged, output_dir)

    # K-mer analysis
    print("\n" + "="*80)
    print("K-MER FREQUENCY ANALYSIS")
    print("="*80)

    all_kmer_stats = {}

    # 3-mers
    print("\n--- 3-mer Analysis ---")
    df_kmer_3 = compute_kmer_frequencies(df_merged, k=3)
    disc_3, enrich_3 = plot_kmer_analysis(df_kmer_3, k=3, output_dir=output_dir)
    all_kmer_stats['3mer_discriminative'] = disc_3
    all_kmer_stats['3mer_enrichment'] = enrich_3
    print("\nTop 10 discriminative 3-mers:")
    print(disc_3[['mean', 'std', 'range']].head(10))

    # 6-mers
    print("\n--- 6-mer Analysis ---")
    df_kmer_6 = compute_kmer_frequencies(df_merged, k=6)
    disc_6, enrich_6 = plot_kmer_analysis(df_kmer_6, k=6, output_dir=output_dir)
    all_kmer_stats['6mer_discriminative'] = disc_6
    all_kmer_stats['6mer_enrichment'] = enrich_6
    print("\nTop 10 discriminative 6-mers:")
    print(disc_6[['mean', 'std', 'range']].head(10))

    # 9-mers
    print("\n--- 9-mer Analysis ---")
    df_kmer_9 = compute_kmer_frequencies(df_merged, k=9)
    disc_9, enrich_9 = plot_kmer_analysis(df_kmer_9, k=9, output_dir=output_dir)
    all_kmer_stats['9mer_discriminative'] = disc_9
    all_kmer_stats['9mer_enrichment'] = enrich_9
    print("\nTop 10 discriminative 9-mers:")
    print(disc_9[['mean', 'std', 'range']].head(10))

    # Cluster comparison plot
    plot_cluster_comparison(df_kmer_3, df_kmer_6, df_kmer_9, output_dir)

    # Save k-mer statistics to Excel
    print("\nSaving k-mer statistics to Excel...")
    with pd.ExcelWriter('kmer_cluster_statistics.xlsx', engine='openpyxl') as writer:
        gc_stats.to_excel(writer, sheet_name='GC_Content')
        for key, df in all_kmer_stats.items():
            # Truncate sheet name if too long
            sheet_name = key[:31]
            df.to_excel(writer, sheet_name=sheet_name)

        # Also save full frequency tables
        df_kmer_3.to_excel(writer, sheet_name='3mer_frequencies')
        df_kmer_6.to_excel(writer, sheet_name='6mer_frequencies')
        df_kmer_9.to_excel(writer, sheet_name='9mer_frequencies')

    print(f"✓ Saved kmer_cluster_statistics.xlsx")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"K-mer analysis plots saved to: {output_dir}/")
    print(f"Statistics saved to: kmer_cluster_statistics.xlsx")
