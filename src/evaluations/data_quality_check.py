"""
Validate whether low-complexity regions (homopolymers) in clusters 0,1 are
artifacts or functional sequences using the DHS Index dataset.

Based on Meuleman et al. 2020 Nature paper data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def load_data_with_metadata(cluster_file, data_file):
    """Load data with all DHS Index metadata."""
    print("Loading data with metadata...")
    df_clusters = pd.read_csv(cluster_file)
    df_data = pd.read_csv(data_file, sep='\t')
    df_merged = df_clusters.merge(df_data, left_on='id', right_on='dhs_id', how='left')
    return df_merged


def check_genomic_annotations(df_merged, output_dir='validation_analysis'):
    """
    Check if sequences overlap with known functional regions.

    Key validations from the paper:
    1. Distance to TSS (promoters)
    2. DNase signal strength
    3. Number of biosamples with signal
    4. Chromosomal distribution
    """
    Path(output_dir).mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print("GENOMIC ANNOTATION VALIDATION")
    print("=" * 80)

    # 1. CHECK CHROMOSOMAL DISTRIBUTION
    print("\n1. CHROMOSOMAL DISTRIBUTION:")
    print("   Artifacts often cluster on specific chromosomes or telomeric regions")

    chrom_dist = df_merged.groupby(['cluster', 'chr']).size().unstack(fill_value=0)

    # Calculate chi-square test for uniform distribution
    for cluster in [0, 1, 2, -1]:
        if cluster in df_merged['cluster'].unique():
            obs_freq = chrom_dist.loc[cluster]
            expected = np.full(len(obs_freq), obs_freq.sum() / len(obs_freq))
            chi2, pval = stats.chisquare(obs_freq, expected)

            print(f"\n   Cluster {cluster}:")
            print(f"     Chi-square test for uniform distribution: p={pval:.2e}")
            if pval < 0.001:
                print(f"     ⚠️  NON-UNIFORM distribution (potential artifact clustering)")
            else:
                print(f"     ✓ Uniform distribution (looks normal)")

            # Check for chromosome-specific enrichment
            top_chroms = obs_freq.nlargest(3)
            print(f"     Top 3 chromosomes: {dict(top_chroms)}")

    # Plot chromosomal distribution
    fig, ax = plt.subplots(figsize=(16, 6))
    chrom_pct = chrom_dist.div(chrom_dist.sum(axis=1), axis=0) * 100
    chrom_pct.T.plot(kind='bar', ax=ax, alpha=0.7)
    ax.set_xlabel('Chromosome', fontsize=12)
    ax.set_ylabel('Percentage of Sequences', fontsize=12)
    ax.set_title('Chromosomal Distribution by Cluster', fontsize=14, fontweight='bold')
    ax.legend(title='Cluster')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/chromosomal_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n   ✓ Saved {output_dir}/chromosomal_distribution.png")
    plt.close()

    # 2. CHECK DNASE SIGNAL STRENGTH
    print("\n2. DNASE SIGNAL STRENGTH:")
    print("   Functional DHSs have strong signals; artifacts have weak/variable signals")

    if 'total_signal' in df_merged.columns:
        signal_stats = df_merged.groupby('cluster')['total_signal'].agg(['mean', 'median', 'std'])
        print("\n   Signal statistics by cluster:")
        print(signal_stats)

        # Compare clusters 0,1 vs 2
        cluster_01_signal = df_merged[df_merged['cluster'].isin([0, 1])]['total_signal'].dropna()
        cluster_2_signal = df_merged[df_merged['cluster'] == 2]['total_signal'].dropna()

        stat, pval = stats.mannwhitneyu(cluster_01_signal, cluster_2_signal)
        print(f"\n   Mann-Whitney U test (clusters 0,1 vs 2): p={pval:.2e}")

        if cluster_01_signal.median() < cluster_2_signal.median() * 0.5:
            print(f"   ⚠️  Clusters 0,1 have MUCH WEAKER signal (potential low-quality)")
        else:
            print(f"   ✓ Signal strength comparable (looks functional)")

        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Boxplot
        df_merged.boxplot(column='total_signal', by='cluster', ax=axes[0])
        axes[0].set_xlabel('Cluster', fontsize=11)
        axes[0].set_ylabel('Total DNase Signal', fontsize=11)
        axes[0].set_title('DNase Signal Strength by Cluster', fontsize=12, fontweight='bold')

        # Histogram
        for cluster in sorted(df_merged['cluster'].unique()):
            data = df_merged[df_merged['cluster'] == cluster]['total_signal'].dropna()
            axes[1].hist(data, bins=50, alpha=0.5, label=f'Cluster {cluster}', density=True)
        axes[1].set_xlabel('Total DNase Signal', fontsize=11)
        axes[1].set_ylabel('Density', fontsize=11)
        axes[1].set_title('DNase Signal Distribution', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].set_xlim(0, df_merged['total_signal'].quantile(0.99))

        plt.tight_layout()
        plt.savefig(f'{output_dir}/dnase_signal_comparison.png', dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved {output_dir}/dnase_signal_comparison.png")
        plt.close()

    # 3. CHECK NUMBER OF BIOSAMPLES (numsamples)
    print("\n3. BIOSAMPLE BREADTH:")
    print("   Functional DHSs active in multiple cell types; artifacts often cell-specific")

    if 'numsamples' in df_merged.columns:
        sample_stats = df_merged.groupby('cluster')['numsamples'].agg(['mean', 'median', 'std'])
        print("\n   Number of biosamples by cluster:")
        print(sample_stats)

        # Test
        cluster_01_samples = df_merged[df_merged['cluster'].isin([0, 1])]['numsamples'].dropna()
        cluster_2_samples = df_merged[df_merged['cluster'] == 2]['numsamples'].dropna()

        stat, pval = stats.mannwhitneyu(cluster_01_samples, cluster_2_samples)
        print(f"\n   Mann-Whitney U test: p={pval:.2e}")

        if cluster_01_samples.median() <= 2:
            print(f"   ⚠️  Clusters 0,1 active in FEW cell types (potential artifacts/cell-specific noise)")
        else:
            print(f"   ✓ Active across multiple cell types (functional)")

        # Visualize
        fig, ax = plt.subplots(figsize=(10, 6))
        df_merged.boxplot(column='numsamples', by='cluster', ax=ax)
        ax.set_xlabel('Cluster', fontsize=11)
        ax.set_ylabel('Number of Biosamples', fontsize=11)
        ax.set_title('Biosample Breadth by Cluster', fontsize=12, fontweight='bold')
        plt.suptitle('')  # Remove default title
        plt.tight_layout()
        plt.savefig(f'{output_dir}/biosample_breadth.png', dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved {output_dir}/biosample_breadth.png")
        plt.close()

    # 4. CHECK DHS WIDTH
    print("\n4. DHS WIDTH DISTRIBUTION:")
    print("   Artifacts often have unusual widths (very narrow or very wide)")

    if 'DHS_width' in df_merged.columns:
        width_stats = df_merged.groupby('cluster')['DHS_width'].agg(['mean', 'median', 'std'])
        print("\n   DHS width statistics:")
        print(width_stats)

        # Check for outliers
        for cluster in [0, 1, 2]:
            if cluster in df_merged['cluster'].unique():
                widths = df_merged[df_merged['cluster'] == cluster]['DHS_width'].dropna()
                q1, q3 = widths.quantile([0.25, 0.75])
                iqr = q3 - q1
                outliers = widths[(widths < q1 - 1.5 * iqr) | (widths > q3 + 1.5 * iqr)]
                pct_outliers = len(outliers) / len(widths) * 100

                print(f"\n   Cluster {cluster}:")
                print(f"     Outliers: {pct_outliers:.1f}%")
                if pct_outliers > 20:
                    print(f"     ⚠️  HIGH outlier rate (potential quality issues)")

    # 5. SUMMIT POSITION ANALYSIS
    print("\n5. SUMMIT POSITION WITHIN DHS:")
    print("   Artifacts may have off-center summits")

    if 'DHS_width' in df_merged.columns and 'summit' in df_merged.columns and 'start' in df_merged.columns:
        # Calculate relative summit position
        df_merged['summit_relative'] = (df_merged['summit'] - df_merged['start']) / df_merged['DHS_width']

        for cluster in [0, 1, 2]:
            if cluster in df_merged['cluster'].unique():
                rel_pos = df_merged[df_merged['cluster'] == cluster]['summit_relative'].dropna()
                mean_pos = rel_pos.mean()
                print(f"\n   Cluster {cluster}: mean summit position = {mean_pos:.3f} (0.5 = centered)")

                if abs(mean_pos - 0.5) > 0.15:
                    print(f"     ⚠️  Off-center summits (unusual)")
                else:
                    print(f"     ✓ Well-centered summits (normal)")


def analyze_sequence_quality(df_merged, output_dir='validation_analysis'):
    """
    Analyze sequence quality metrics.
    """
    print("\n" + "=" * 80)
    print("SEQUENCE QUALITY ANALYSIS")
    print("=" * 80)

    # Check for N content
    print("\n1. N-BASE CONTENT:")
    df_merged['n_count'] = df_merged['sequence'].apply(
        lambda x: x.upper().count('N') if pd.notna(x) else 0
    )

    for cluster in [0, 1, 2, -1]:
        if cluster in df_merged['cluster'].unique():
            n_counts = df_merged[df_merged['cluster'] == cluster]['n_count']
            seqs_with_n = (n_counts > 0).sum()
            pct_with_n = seqs_with_n / len(n_counts) * 100

            print(f"\n   Cluster {cluster}:")
            print(f"     Sequences with N bases: {seqs_with_n} ({pct_with_n:.2f}%)")

            if pct_with_n > 10:
                print(f"     ⚠️  HIGH N-content (potential assembly/sequencing issues)")
            else:
                print(f"     ✓ Low N-content")

    # Check for extreme GC content
    print("\n2. EXTREME GC CONTENT:")
    df_merged['gc_extreme'] = df_merged['sequence'].apply(
        lambda x: (lambda s: (s.count('G') + s.count('C')) / len(s) if len(s) > 0 else 0)(
            x.upper() if pd.notna(x) else ''
        )
    ).apply(lambda gc: gc < 0.25 or gc > 0.75)

    for cluster in [0, 1, 2]:
        if cluster in df_merged['cluster'].unique():
            extreme = df_merged[df_merged['cluster'] == cluster]['gc_extreme']
            pct_extreme = extreme.sum() / len(extreme) * 100

            print(f"\n   Cluster {cluster}:")
            print(f"     Sequences with extreme GC (<25% or >75%): {pct_extreme:.2f}%")

            if pct_extreme > 30:
                print(f"     ⚠️  HIGH rate of extreme GC (unusual)")


def create_validation_report(df_merged, output_dir='validation_analysis'):
    """
    Generate final validation report.
    """
    report_file = output_dir / Path('VALIDATION_REPORT.txt')

    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("VALIDATION REPORT: Are Clusters 0,1 Artifacts or Functional?\n")
        f.write("=" * 80 + "\n\n")

        f.write("## DECISION CRITERIA ##\n\n")
        f.write("Clusters 0,1 are likely ARTIFACTS if they show:\n")
        f.write("  ❌ Non-uniform chromosomal distribution (chr-specific clustering)\n")
        f.write("  ❌ Significantly weaker DNase signal than cluster 2\n")
        f.write("  ❌ Active in very few biosamples (≤2)\n")
        f.write("  ❌ Unusual DHS widths (many outliers)\n")
        f.write("  ❌ High N-base content (>10%)\n")
        f.write("  ❌ High rate of extreme GC content (>30%)\n\n")

        f.write("Clusters 0,1 are likely FUNCTIONAL if they show:\n")
        f.write("  ✓ Uniform chromosomal distribution\n")
        f.write("  ✓ Strong DNase signals (comparable to cluster 2)\n")
        f.write("  ✓ Active across multiple cell types (>3 biosamples)\n")
        f.write("  ✓ Normal DHS characteristics\n")
        f.write("  ✓ Contain known regulatory motifs (TATA boxes, poly-A signals)\n\n")

        f.write("## NEXT STEPS ##\n\n")
        f.write("1. If ARTIFACTS:\n")
        f.write("   - Filter out clusters 0,1 from training data\n")
        f.write("   - Retrain DNA-Diffusion model on cluster 2 only\n")
        f.write("   - Model correctly avoided generating these artifacts!\n\n")

        f.write("2. If FUNCTIONAL:\n")
        f.write("   - These are specialized regulatory regions (promoters, poly-A signals)\n")
        f.write("   - Model SHOULD learn to generate these patterns\n")
        f.write("   - Need to condition model on genomic context (distance to TSS, etc.)\n\n")

        f.write("3. Additional validation:\n")
        f.write("   - Overlap with known promoters (GENCODE TSS ±1kb)\n")
        f.write("   - Check overlap with CpG islands\n")
        f.write("   - Validate with ENCODE cCRE annotations\n")
        f.write("   - Run ChromHMM state enrichment analysis\n")

    print(f"\n✓ Saved validation report: {report_file}")

    with open(report_file, 'r') as f:
        print("\n" + f.read())


def main():
    """Main validation pipeline."""
    cluster_file = "/home/benjaminkroeger/Documents/Master/UBC/Synthetic_data/DNA-Diffusion/src/evaluations/cluster_id_mapping.csv"
    data_file = "/home/benjaminkroeger/Documents/Master/UBC/Synthetic_data/DNA-Diffusion/data/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt"  # UPDATE THIS

    output_dir = Path('validation_analysis')
    output_dir.mkdir(exist_ok=True)

    # Load data
    df_merged = load_data_with_metadata(cluster_file, data_file)

    # Run validations
    check_genomic_annotations(df_merged, output_dir)
    analyze_sequence_quality(df_merged, output_dir)

    # Generate report
    create_validation_report(df_merged, output_dir)

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\nCheck {output_dir}/VALIDATION_REPORT.txt for conclusions")


if __name__ == "__main__":
    main()
