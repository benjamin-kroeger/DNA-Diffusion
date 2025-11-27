#!/usr/bin/env python3
"""
Script 3: STREME Motif Discovery and Cluster Feature Analysis

Performs motif discovery and feature extraction to characterize differences
between UMAP clusters:
- STREME motif enrichment analysis
- Sequence complexity metrics (linguistic complexity, entropy)
- Low-complexity region detection
- Nucleotide composition analysis
- GC skew and other structural features

Key insight: Identifies that small clusters may represent low-complexity
or repetitive regions that the model fails to generate.

Author: DNA Diffusion Analysis Pipeline

Requirements:
    - MEME Suite (for STREME): https://meme-suite.org/
    - Install: apt-get install meme-suite OR conda install -c bioconda meme
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import subprocess
import tempfile
import os
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.stats import entropy, mannwhitneyu, kruskal
import argparse
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
    'legend.fontsize': 11,
    'figure.figsize': (12, 10),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


@dataclass
class STREMEConfig:
    """Configuration for STREME analysis."""
    min_motif_width: int = 6
    max_motif_width: int = 15
    n_motifs: int = 10
    pvalue_threshold: float = 0.05
    streme_path: str = 'streme'  # Path to STREME executable


class SequenceComplexityAnalyzer:
    """
    Analyze sequence complexity using various metrics.
    """

    @staticmethod
    def linguistic_complexity(seq: str, k: int = 3) -> float:
        """
        Compute linguistic complexity: ratio of observed k-mers to possible k-mers.

        Low complexity sequences have fewer unique k-mers.
        """
        seq = seq.upper()
        if len(seq) < k:
            return 0.0

        observed_kmers = set()
        for i in range(len(seq) - k + 1):
            observed_kmers.add(seq[i:i + k])

        # Maximum possible k-mers in sequence of this length
        max_possible = min(4 ** k, len(seq) - k + 1)

        return len(observed_kmers) / max_possible if max_possible > 0 else 0.0

    @staticmethod
    def shannon_entropy(seq: str) -> float:
        """Compute Shannon entropy of nucleotide frequencies."""
        seq = seq.upper()
        counts = Counter(seq)
        total = sum(counts.values())
        probs = np.array([counts.get(n, 0) / total for n in 'ACGT'])
        return entropy(probs, base=2)

    @staticmethod
    def gc_content(seq: str) -> float:
        """Compute GC content."""
        seq = seq.upper()
        gc = seq.count('G') + seq.count('C')
        return gc / len(seq) if len(seq) > 0 else 0.0

    @staticmethod
    def gc_skew(seq: str) -> float:
        """Compute GC skew: (G-C)/(G+C)."""
        seq = seq.upper()
        g, c = seq.count('G'), seq.count('C')
        return (g - c) / (g + c) if (g + c) > 0 else 0.0

    @staticmethod
    def at_skew(seq: str) -> float:
        """Compute AT skew: (A-T)/(A+T)."""
        seq = seq.upper()
        a, t = seq.count('A'), seq.count('T')
        return (a - t) / (a + t) if (a + t) > 0 else 0.0

    @staticmethod
    def dinucleotide_frequencies(seq: str) -> Dict[str, float]:
        """Compute dinucleotide frequencies."""
        seq = seq.upper()
        dinucs = [seq[i:i + 2] for i in range(len(seq) - 1)]
        counts = Counter(dinucs)
        total = len(dinucs)

        all_dinucs = [a + b for a in 'ACGT' for b in 'ACGT']
        return {d: counts.get(d, 0) / total for d in all_dinucs}

    @staticmethod
    def count_homopolymers(seq: str, min_length: int = 4) -> Dict[str, int]:
        """Count homopolymer runs (consecutive identical nucleotides)."""
        seq = seq.upper()
        counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0, 'total': 0}

        for nuc in 'ACGT':
            pattern = nuc * min_length
            count = 0
            i = 0
            while i < len(seq):
                if seq[i:].startswith(pattern):
                    count += 1
                    # Skip to end of homopolymer
                    while i < len(seq) and seq[i] == nuc:
                        i += 1
                else:
                    i += 1
            counts[nuc] = count

        counts['total'] = sum(counts[n] for n in 'ACGT')
        return counts

    @staticmethod
    def count_repeat_units(seq: str, unit_sizes: List[int] = [2, 3, 4]) -> Dict[str, int]:
        """Count tandem repeats of various unit sizes."""
        seq = seq.upper()
        results = {}

        for unit_size in unit_sizes:
            # Find consecutive repeats of any unit
            repeat_count = 0
            i = 0
            while i < len(seq) - unit_size * 2 + 1:
                unit = seq[i:i + unit_size]
                repeat_length = 1

                j = i + unit_size
                while j <= len(seq) - unit_size and seq[j:j + unit_size] == unit:
                    repeat_length += 1
                    j += unit_size

                if repeat_length >= 3:  # At least 3 consecutive repeats
                    repeat_count += 1
                    i = j
                else:
                    i += 1

            results[f'{unit_size}mer_repeats'] = repeat_count

        return results

    @staticmethod
    def low_complexity_fraction(seq: str, window_size: int = 20, threshold: float = 0.5) -> float:
        """
        Compute fraction of sequence that is low complexity.
        Uses sliding window linguistic complexity.
        """
        seq = seq.upper()
        if len(seq) < window_size:
            return 0.0

        low_complexity_positions = 0

        for i in range(len(seq) - window_size + 1):
            window = seq[i:i + window_size]
            lc = SequenceComplexityAnalyzer.linguistic_complexity(window, k=2)
            if lc < threshold:
                low_complexity_positions += 1

        return low_complexity_positions / (len(seq) - window_size + 1)


class STREMERunner:
    """
    Run STREME motif discovery and parse results.
    """

    def __init__(self, config: Optional[STREMEConfig] = None):
        self.config = config or STREMEConfig()

    def check_streme_available(self) -> bool:
        """Check if STREME is available."""
        try:
            result = subprocess.run(
                [self.config.streme_path, '--version'],
                capture_output=True, text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def write_fasta(self, sequences: List[str], filepath: str, prefix: str = 'seq'):
        """Write sequences to FASTA file."""
        with open(filepath, 'w') as f:
            for i, seq in enumerate(sequences):
                f.write(f'>{prefix}_{i}\n{seq}\n')

    def run_streme(self,
                   positive_seqs: List[str],
                   negative_seqs: Optional[List[str]] = None,
                   output_dir: str = 'streme_output') -> Dict:
        """
        Run STREME motif discovery.

        Args:
            positive_seqs: Sequences to find enriched motifs in
            negative_seqs: Control sequences (optional, uses shuffled if not provided)
            output_dir: Output directory for STREME results

        Returns:
            Dictionary with motif results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write positive sequences
            pos_fasta = os.path.join(tmpdir, 'positive.fa')
            self.write_fasta(positive_seqs, pos_fasta, prefix='pos')

            # Build command
            cmd = [
                self.config.streme_path,
                '--p', pos_fasta,
                '--oc', str(output_path),
                '--minw', str(self.config.min_motif_width),
                '--maxw', str(self.config.max_motif_width),
                '--nmotifs', str(self.config.n_motifs),
                '--thresh', str(self.config.pvalue_threshold),
                '--dna',
            ]

            # Add negative sequences if provided
            if negative_seqs is not None:
                neg_fasta = os.path.join(tmpdir, 'negative.fa')
                self.write_fasta(negative_seqs, neg_fasta, prefix='neg')
                cmd.extend(['--n', neg_fasta])

            print(f"Running STREME: {' '.join(cmd)}")

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

                if result.returncode != 0:
                    print(f"STREME error: {result.stderr}")
                    return {'success': False, 'error': result.stderr}

                # Parse results
                return self.parse_streme_output(output_path)

            except subprocess.TimeoutExpired:
                return {'success': False, 'error': 'STREME timed out'}
            except Exception as e:
                return {'success': False, 'error': str(e)}

    def parse_streme_output(self, output_dir: Path) -> Dict:
        """Parse STREME output files."""
        results = {'success': True, 'motifs': []}

        # Parse streme.txt for motif information
        streme_txt = output_dir / 'streme.txt'
        if streme_txt.exists():
            with open(streme_txt, 'r') as f:
                content = f.read()

            # Extract motif information using regex
            motif_pattern = r'MOTIF\s+(\S+)\s+STREME-(\d+)'
            pvalue_pattern = r'letter-probability matrix:.*?E=\s*([\d.e+-]+)'

            motif_matches = re.findall(motif_pattern, content)

            for consensus, motif_num in motif_matches:
                results['motifs'].append({
                    'consensus': consensus,
                    'motif_id': f'STREME-{motif_num}',
                })

        # Check for HTML output with more details
        streme_html = output_dir / 'streme.html'
        if streme_html.exists():
            results['html_report'] = str(streme_html)

        return results


class ClusterFeatureAnalyzer:
    """
    Analyze features that distinguish different clusters.
    """

    def __init__(self, output_dir: str = 'cluster_analysis'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.complexity_analyzer = SequenceComplexityAnalyzer()
        self.streme_runner = STREMERunner()

    def compute_sequence_features(self,
                                  sequences: List[str],
                                  show_progress: bool = True) -> pd.DataFrame:
        """
        Compute comprehensive features for a list of sequences.
        """
        features = []

        iterator = tqdm(sequences, desc="Computing features") if show_progress else sequences

        for seq in iterator:
            feat = {
                'gc_content': self.complexity_analyzer.gc_content(seq),
                'gc_skew': self.complexity_analyzer.gc_skew(seq),
                'at_skew': self.complexity_analyzer.at_skew(seq),
                'shannon_entropy': self.complexity_analyzer.shannon_entropy(seq),
                'linguistic_complexity_k2': self.complexity_analyzer.linguistic_complexity(seq, k=2),
                'linguistic_complexity_k3': self.complexity_analyzer.linguistic_complexity(seq, k=3),
                'linguistic_complexity_k4': self.complexity_analyzer.linguistic_complexity(seq, k=4),
                'low_complexity_fraction': self.complexity_analyzer.low_complexity_fraction(seq),
            }

            # Homopolymer counts
            homopolymers = self.complexity_analyzer.count_homopolymers(seq)
            feat['homopolymer_total'] = homopolymers['total']
            for nuc in 'ACGT':
                feat[f'homopolymer_{nuc}'] = homopolymers[nuc]

            # Repeat units
            repeats = self.complexity_analyzer.count_repeat_units(seq)
            feat.update(repeats)

            # Dinucleotide frequencies
            dinucs = self.complexity_analyzer.dinucleotide_frequencies(seq)
            for dinuc, freq in dinucs.items():
                feat[f'dinuc_{dinuc}'] = freq

            features.append(feat)

        return pd.DataFrame(features)

    def analyze_clusters(self,
                         sequences: List[str],
                         cluster_assignments: np.ndarray,
                         sequence_ids: Optional[np.ndarray] = None,
                         run_streme: bool = True) -> Dict:
        """
        Comprehensive analysis of cluster differences.
        """
        unique_clusters = sorted([c for c in np.unique(cluster_assignments) if c != -1])

        print(f"\nAnalyzing {len(unique_clusters)} clusters...")

        # Compute features for all sequences
        print("\nComputing sequence features...")
        features_df = self.compute_sequence_features(sequences)
        features_df['cluster'] = cluster_assignments
        if sequence_ids is not None:
            features_df['sequence_id'] = sequence_ids

        # Compute cluster statistics
        print("\nComputing cluster statistics...")
        cluster_stats = self._compute_cluster_statistics(features_df, unique_clusters)

        # Statistical tests between clusters
        print("\nRunning statistical tests...")
        pairwise_tests = self._run_pairwise_tests(features_df, unique_clusters)

        # Run STREME if available
        streme_results = {}
        if run_streme and self.streme_runner.check_streme_available():
            print("\nRunning STREME motif discovery...")
            streme_results = self._run_cluster_streme(sequences, cluster_assignments, unique_clusters)
        elif run_streme:
            print("\nSTREME not available, skipping motif discovery")

        # Identify distinguishing features
        distinguishing_features = self._identify_distinguishing_features(
            features_df, unique_clusters, pairwise_tests
        )

        results = {
            'features_df': features_df,
            'cluster_stats': cluster_stats,
            'pairwise_tests': pairwise_tests,
            'streme_results': streme_results,
            'distinguishing_features': distinguishing_features,
        }

        return results

    def _compute_cluster_statistics(self,
                                    features_df: pd.DataFrame,
                                    clusters: List[int]) -> Dict:
        """Compute summary statistics for each cluster."""
        stats = {}

        feature_cols = [c for c in features_df.columns
                        if c not in ['cluster', 'sequence_id']]

        for cluster in clusters:
            cluster_df = features_df[features_df['cluster'] == cluster]

            stats[cluster] = {
                'n_sequences': len(cluster_df),
                'features': {}
            }

            for col in feature_cols:
                stats[cluster]['features'][col] = {
                    'mean': float(cluster_df[col].mean()),
                    'std': float(cluster_df[col].std()),
                    'median': float(cluster_df[col].median()),
                    'min': float(cluster_df[col].min()),
                    'max': float(cluster_df[col].max()),
                }

        return stats

    def _run_pairwise_tests(self,
                            features_df: pd.DataFrame,
                            clusters: List[int]) -> Dict:
        """Run statistical tests between cluster pairs."""
        feature_cols = [c for c in features_df.columns
                        if c not in ['cluster', 'sequence_id']]

        tests = {}

        for i, c1 in enumerate(clusters):
            for c2 in clusters[i + 1:]:
                key = f'{c1}_vs_{c2}'
                tests[key] = {}

                df1 = features_df[features_df['cluster'] == c1]
                df2 = features_df[features_df['cluster'] == c2]

                for col in feature_cols:
                    try:
                        stat, pval = mannwhitneyu(df1[col], df2[col], alternative='two-sided')
                        effect_size = abs(df1[col].mean() - df2[col].mean()) / \
                                      (df1[col].std() + df2[col].std() + 1e-10)

                        tests[key][col] = {
                            'statistic': float(stat),
                            'pvalue': float(pval),
                            'effect_size': float(effect_size),
                            'significant': pval < 0.05,
                        }
                    except Exception:
                        pass

        return tests

    def _run_cluster_streme(self,
                            sequences: List[str],
                            cluster_assignments: np.ndarray,
                            clusters: List[int]) -> Dict:
        """Run STREME for each cluster vs others."""
        results = {}

        for cluster in clusters:
            print(f"\n  Running STREME for cluster {cluster}...")

            cluster_seqs = [sequences[i] for i in range(len(sequences))
                            if cluster_assignments[i] == cluster]
            other_seqs = [sequences[i] for i in range(len(sequences))
                          if cluster_assignments[i] != cluster and cluster_assignments[i] != -1]

            # Sample if too many sequences
            if len(cluster_seqs) > 1000:
                idx = np.random.choice(len(cluster_seqs), 1000, replace=False)
                cluster_seqs = [cluster_seqs[i] for i in idx]

            if len(other_seqs) > 2000:
                idx = np.random.choice(len(other_seqs), 2000, replace=False)
                other_seqs = [other_seqs[i] for i in idx]

            output_dir = self.output_dir / f'streme_cluster_{cluster}'
            streme_result = self.streme_runner.run_streme(
                cluster_seqs, other_seqs, str(output_dir)
            )

            results[cluster] = streme_result

        return results

    def _identify_distinguishing_features(self,
                                          features_df: pd.DataFrame,
                                          clusters: List[int],
                                          pairwise_tests: Dict) -> Dict:
        """Identify features that best distinguish clusters."""
        feature_cols = [c for c in features_df.columns
                        if c not in ['cluster', 'sequence_id']]

        # Kruskal-Wallis test across all clusters
        kruskal_results = {}
        for col in feature_cols:
            groups = [features_df[features_df['cluster'] == c][col].values
                      for c in clusters]
            try:
                stat, pval = kruskal(*groups)
                kruskal_results[col] = {
                    'statistic': float(stat),
                    'pvalue': float(pval),
                    'significant': pval < 0.05,
                }
            except Exception:
                pass

        # Rank features by significance
        ranked_features = sorted(
            kruskal_results.items(),
            key=lambda x: x[1]['pvalue']
        )

        return {
            'kruskal_wallis': kruskal_results,
            'ranked_features': [(f, r['pvalue']) for f, r in ranked_features[:20]],
        }

    # ==================== Visualization Methods ====================

    def plot_feature_distributions(self,
                                   features_df: pd.DataFrame,
                                   output_name: str = 'feature_distributions') -> str:
        """Plot distributions of key features by cluster."""
        key_features = [
            'gc_content', 'shannon_entropy', 'linguistic_complexity_k3',
            'low_complexity_fraction', 'homopolymer_total'
        ]

        # Filter to features that exist
        key_features = [f for f in key_features if f in features_df.columns]

        n_features = len(key_features)
        fig, axes = plt.subplots(2, (n_features + 1) // 2, figsize=(5 * ((n_features + 1) // 2), 10))
        axes = axes.flatten()

        clusters = sorted([c for c in features_df['cluster'].unique() if c != -1])
        colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))

        for i, feature in enumerate(key_features):
            ax = axes[i]

            for j, cluster in enumerate(clusters):
                data = features_df[features_df['cluster'] == cluster][feature]
                ax.hist(data, bins=30, alpha=0.5, label=f'C{cluster}', color=colors[j])

            ax.set_xlabel(feature.replace('_', ' ').title())
            ax.set_ylabel('Count')
            ax.legend()

        # Hide unused axes
        for i in range(len(key_features), len(axes)):
            axes[i].axis('off')

        plt.suptitle('Feature Distributions by Cluster', fontsize=16)
        plt.tight_layout()

        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def plot_cluster_comparison_heatmap(self,
                                        cluster_stats: Dict,
                                        output_name: str = 'cluster_heatmap') -> str:
        """Create heatmap comparing cluster feature means."""
        clusters = sorted(cluster_stats.keys())

        # Select key features
        key_features = [
            'gc_content', 'shannon_entropy', 'linguistic_complexity_k3',
            'low_complexity_fraction', 'homopolymer_total',
            'gc_skew', 'at_skew'
        ]

        # Filter to available features
        available = list(cluster_stats[clusters[0]]['features'].keys())
        key_features = [f for f in key_features if f in available]

        # Build matrix
        data = np.zeros((len(clusters), len(key_features)))
        for i, cluster in enumerate(clusters):
            for j, feat in enumerate(key_features):
                data[i, j] = cluster_stats[cluster]['features'][feat]['mean']

        # Normalize columns for visualization
        data_norm = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-10)

        fig, ax = plt.subplots(figsize=(12, 6))

        sns.heatmap(data_norm, ax=ax, annot=True, fmt='.2f',
                    xticklabels=[f.replace('_', '\n') for f in key_features],
                    yticklabels=[f'Cluster {c}' for c in clusters],
                    cmap='RdBu_r', center=0)

        ax.set_title('Cluster Feature Comparison\n(Z-scored)')

        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def plot_complexity_scatter(self,
                                features_df: pd.DataFrame,
                                output_name: str = 'complexity_scatter') -> str:
        """Scatter plot of complexity metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        clusters = sorted([c for c in features_df['cluster'].unique() if c != -1])
        colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))

        # Plot 1: GC content vs entropy
        ax = axes[0]
        for i, cluster in enumerate(clusters):
            mask = features_df['cluster'] == cluster
            ax.scatter(features_df.loc[mask, 'gc_content'],
                       features_df.loc[mask, 'shannon_entropy'],
                       s=10, alpha=0.5, c=[colors[i]], label=f'C{cluster}')
        ax.set_xlabel('GC Content')
        ax.set_ylabel('Shannon Entropy')
        ax.set_title('GC Content vs Entropy')
        ax.legend()

        # Plot 2: Linguistic complexity vs low complexity fraction
        ax = axes[1]
        for i, cluster in enumerate(clusters):
            mask = features_df['cluster'] == cluster
            ax.scatter(features_df.loc[mask, 'linguistic_complexity_k3'],
                       features_df.loc[mask, 'low_complexity_fraction'],
                       s=10, alpha=0.5, c=[colors[i]], label=f'C{cluster}')
        ax.set_xlabel('Linguistic Complexity (k=3)')
        ax.set_ylabel('Low Complexity Fraction')
        ax.set_title('Linguistic Complexity vs Low Complexity')
        ax.legend()

        # Plot 3: GC content vs linguistic complexity
        ax = axes[2]
        for i, cluster in enumerate(clusters):
            mask = features_df['cluster'] == cluster
            ax.scatter(features_df.loc[mask, 'gc_content'],
                       features_df.loc[mask, 'linguistic_complexity_k3'],
                       s=10, alpha=0.5, c=[colors[i]], label=f'C{cluster}')
        ax.set_xlabel('GC Content')
        ax.set_ylabel('Linguistic Complexity (k=3)')
        ax.set_title('GC Content vs Complexity')
        ax.legend()

        plt.tight_layout()

        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def plot_distinguishing_features(self,
                                     distinguishing_features: Dict,
                                     output_name: str = 'distinguishing_features') -> str:
        """Bar plot of most distinguishing features."""
        ranked = distinguishing_features['ranked_features'][:15]

        fig, ax = plt.subplots(figsize=(12, 8))

        features = [r[0].replace('_', '\n') for r in ranked]
        pvalues = [-np.log10(r[1] + 1e-100) for r in ranked]

        bars = ax.barh(features, pvalues, color='steelblue', edgecolor='black')

        ax.axvline(-np.log10(0.05), color='red', linestyle='--',
                   label='p=0.05 threshold')
        ax.axvline(-np.log10(0.01), color='orange', linestyle='--',
                   label='p=0.01 threshold')

        ax.set_xlabel('-log10(p-value)')
        ax.set_title('Most Distinguishing Features Between Clusters\n(Kruskal-Wallis test)')
        ax.legend()
        ax.invert_yaxis()

        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def create_summary_report(self, results: Dict, output_name: str = 'cluster_analysis_report') -> str:
        """Generate comprehensive summary report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CLUSTER FEATURE ANALYSIS REPORT")
        report_lines.append("=" * 80)

        # Cluster statistics
        report_lines.append("\n\n## CLUSTER STATISTICS\n")
        for cluster, stats in results['cluster_stats'].items():
            report_lines.append(f"\n### Cluster {cluster}")
            report_lines.append(f"Number of sequences: {stats['n_sequences']}")
            report_lines.append("\nKey features:")

            for feat in ['gc_content', 'shannon_entropy', 'linguistic_complexity_k3',
                         'low_complexity_fraction']:
                if feat in stats['features']:
                    f = stats['features'][feat]
                    report_lines.append(f"  {feat}: {f['mean']:.4f} ± {f['std']:.4f}")

        # Distinguishing features
        report_lines.append("\n\n## MOST DISTINGUISHING FEATURES\n")
        for feat, pval in results['distinguishing_features']['ranked_features'][:10]:
            report_lines.append(f"  {feat}: p = {pval:.2e}")

        # STREME results
        if results['streme_results']:
            report_lines.append("\n\n## STREME MOTIF DISCOVERY\n")
            for cluster, streme in results['streme_results'].items():
                report_lines.append(f"\n### Cluster {cluster}")
                if streme.get('success'):
                    for motif in streme.get('motifs', [])[:5]:
                        report_lines.append(f"  Motif: {motif['consensus']}")
                else:
                    report_lines.append(f"  Error: {streme.get('error', 'Unknown')}")

        report_text = '\n'.join(report_lines)

        filepath = self.output_dir / f"{output_name}.txt"
        with open(filepath, 'w') as f:
            f.write(report_text)

        return str(filepath)


def load_sequences_and_clusters(sequences_file: str,
                                clusters_file: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Load sequences and their cluster assignments."""
    # Load cluster assignments
    clusters_df = pd.read_csv(clusters_file)
    clusters_df = clusters_df[~clusters_df["sequence_id"].str.startswith("synth")]

    # Load sequences
    seq_path = Path(sequences_file)
    if seq_path.suffix == '.txt':
        with open(sequences_file, 'r') as f:
            all_sequences = [line.strip() for line in f if line.strip()]
    elif seq_path.suffix == '.csv':
        seq_df = pd.read_csv(sequences_file)
        for col in ['sequence', 'seq', 'Sequence']:
            if col in seq_df.columns:
                all_sequences = seq_df[col].tolist()
                break
    elif seq_path.suffix == '.tsv':
        seq_df = pd.read_csv(sequences_file, sep='\t')
        for col in ['sequence', 'seq', 'Sequence']:
            if col in seq_df.columns:
                all_sequences = seq_df[col].tolist()
                break
    else:
        raise ValueError(f"Unsupported sequence file format: {seq_path.suffix}")

    # Match sequences to cluster assignments
    # This assumes the clusters_df has sequence_id that matches the order
    cluster_assignments = clusters_df['cluster'].values
    sequence_ids = clusters_df['sequence_id'].values if 'sequence_id' in clusters_df.columns else None

    return all_sequences, cluster_assignments, sequence_ids


def main():
    parser = argparse.ArgumentParser(
        description='STREME Motif Discovery and Cluster Feature Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze clusters with STREME
  python 03_cluster_feature_analysis.py \\
      --sequences data/all_sequences.txt \\
      --clusters umap_results/cluster_assignments.csv \\
      --output cluster_analysis/

  # Skip STREME (faster)
  python 03_cluster_feature_analysis.py \\
      --sequences data/sequences.txt \\
      --clusters clusters.csv \\
      --no-streme
        """
    )

    parser.add_argument('--sequences', '-s', required=True,
                        help='Path to sequences file')
    parser.add_argument('--clusters', '-c', required=True,
                        help='Path to cluster assignments CSV')
    parser.add_argument('--output', '-o', default='cluster_analysis',
                        help='Output directory')
    parser.add_argument('--no-streme', action='store_true',
                        help='Skip STREME motif discovery')

    args = parser.parse_args()

    # Load data
    print("Loading sequences and cluster assignments...")
    sequences, cluster_assignments, sequence_ids = load_sequences_and_clusters(
        args.sequences, args.clusters
    )
    print(f"Loaded {len(sequences)} sequences")

    # Run analysis
    analyzer = ClusterFeatureAnalyzer(output_dir=args.output)
    results = analyzer.analyze_clusters(
        sequences,
        cluster_assignments,
        sequence_ids,
        run_streme=not args.no_streme
    )

    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)

    figures = {
        'feature_distributions': analyzer.plot_feature_distributions(results['features_df']),
        'cluster_heatmap': analyzer.plot_cluster_comparison_heatmap(results['cluster_stats']),
        'complexity_scatter': analyzer.plot_complexity_scatter(results['features_df']),
        'distinguishing_features': analyzer.plot_distinguishing_features(results['distinguishing_features']),
    }

    # Save report
    report_file = analyzer.create_summary_report(results)

    # Save features dataframe
    features_file = Path(args.output) / 'sequence_features.csv'
    results['features_df'].to_csv(features_file, index=False)

    # Save cluster statistics as JSON
    stats_file = Path(args.output) / 'cluster_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(results['cluster_stats'], f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {args.output}")
    print("\nGenerated figures:")
    for name, path in figures.items():
        print(f"  - {name}: {path}")
    print(f"\nReport: {report_file}")
    print(f"Features CSV: {features_file}")

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    print("\nTop distinguishing features:")
    for feat, pval in results['distinguishing_features']['ranked_features'][:5]:
        print(f"  {feat}: p = {pval:.2e}")


if __name__ == "__main__":
    main()
