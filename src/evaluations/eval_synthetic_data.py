"""
DNA Sequence Generation Evaluation Framework with Bootstrapping

Computes comprehensive metrics for evaluating synthetic DNA sequences
against test/reference sequences with confidence intervals.
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
from scipy.stats import entropy, ks_2samp, wasserstein_distance, ttest_ind
from scipy.spatial.distance import jensenshannon
import warnings
from dataclasses import dataclass

from tqdm import tqdm

warnings.filterwarnings('ignore')


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap resampling."""
    n_bootstrap: int = 1000
    confidence_level: float = 0.95
    random_seed: Optional[int] = 42


class DNASequenceEvaluator:
    """
    Comprehensive evaluation framework for DNA sequence generation with bootstrapping.

    Usage:
        evaluator = DNASequenceEvaluator()
        results_df, detailed_results = evaluator.evaluate(test_sequences, synthetic_sequences)
    """

    def __init__(self,
                 kmer_sizes: List[int] = [3, 4, 5],
                 novelty_radii: List[int] = [1, 2, 3, 5],
                 sequence_length: int = 200,
                 bootstrap_config: Optional[BootstrapConfig] = None,
                 verbose: bool = True):
        """
        Initialize the evaluator.

        Args:
            kmer_sizes: List of k-mer sizes to analyze (default: [3, 4, 5])
            novelty_radii: List of radius values for novelty@r calculation (default: [1, 2, 3, 5])
            sequence_length: Expected sequence length for normalization (default: 200)
            bootstrap_config: Configuration for bootstrapping (default: 1000 iterations, 95% CI)
            verbose: Whether to print progress messages
        """
        self.kmer_sizes = kmer_sizes
        self.novelty_radii = novelty_radii
        self.sequence_length = sequence_length
        self.bootstrap_config = bootstrap_config or BootstrapConfig()
        self.verbose = verbose

        if self.bootstrap_config.random_seed is not None:
            np.random.seed(self.bootstrap_config.random_seed)

    def evaluate(self,
                 test_sequences: List[str],
                 synthetic_sequences: List[str],
                 enable_bootstrap: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Main evaluation method - computes all metrics with bootstrapped confidence intervals.

        Args:
            test_sequences: List of real/test DNA sequences
            synthetic_sequences: List of generated/synthetic DNA sequences
            enable_bootstrap: Whether to compute bootstrap confidence intervals

        Returns:
            Tuple of:
                - DataFrame with all metrics, standard errors, and confidence intervals
                - Dictionary with detailed results including bootstrap distributions
        """
        if self.verbose:
            print(f"Evaluating {len(synthetic_sequences)} synthetic sequences against {len(test_sequences)} test sequences...")
            if enable_bootstrap:
                print(f"Bootstrap iterations: {self.bootstrap_config.n_bootstrap}")

        # Compute point estimates
        results = self._compute_all_metrics(test_sequences, synthetic_sequences)

        # Compute bootstrap confidence intervals
        bootstrap_results = {}
        if enable_bootstrap:
            if self.verbose:
                print("\nPerforming bootstrap resampling...")
            bootstrap_results = self._bootstrap_metrics(test_sequences, synthetic_sequences)

        # Convert to DataFrame
        results_df = self._results_to_dataframe(results, bootstrap_results, enable_bootstrap)

        # Create detailed results dictionary
        detailed_results = {
            'point_estimates': results,
            'bootstrap': bootstrap_results if enable_bootstrap else {},
            'config': {
                'n_test': len(test_sequences),
                'n_synthetic': len(synthetic_sequences),
                'kmer_sizes': self.kmer_sizes,
                'novelty_radii': self.novelty_radii,
                'sequence_length': self.sequence_length,
                'bootstrap_iterations': self.bootstrap_config.n_bootstrap if enable_bootstrap else 0,
                'confidence_level': self.bootstrap_config.confidence_level
            }
        }

        if self.verbose:
            print("\nEvaluation complete!")
            self._print_summary(results_df)

        return results_df, detailed_results

    def _compute_all_metrics(self, test_seqs: List[str], synth_seqs: List[str]) -> Dict:
        """Compute all metrics (point estimates)."""
        results = {}

        # 1. GC Content Comparison (difference metrics only)
        results['gc_content'] = self._evaluate_gc_content(test_seqs, synth_seqs)

        # 2. K-mer Frequency Comparison
        results['kmer_analysis'] = self._evaluate_kmer_frequencies(test_seqs, synth_seqs)

        # 3. Novelty Metrics (Novelty@r)
        results['novelty'] = self._evaluate_novelty(test_seqs, synth_seqs)

        # 4. Diversity Metrics (Mean pairwise normalized Hamming distance)
        results['diversity'] = self._evaluate_diversity(synth_seqs)

        # 5. Nucleotide Distribution
        results['nucleotide_dist'] = self._evaluate_nucleotide_distribution(test_seqs, synth_seqs)

        return results

    def _bootstrap_metrics(self, test_seqs: List[str], synth_seqs: List[str]) -> Dict:
        """
        Perform bootstrap resampling to estimate confidence intervals.

        Returns dictionary with bootstrap distributions and confidence intervals for each metric.
        """
        n_bootstrap = self.bootstrap_config.n_bootstrap
        alpha = 1 - self.bootstrap_config.confidence_level

        # Store bootstrap samples for each metric
        bootstrap_samples = defaultdict(list)

        for i in tqdm(range(n_bootstrap)):
            if self.verbose and (i + 1) % 100 == 0:
                print(f"  Bootstrap iteration {i + 1}/{n_bootstrap}")

            # Resample with replacement
            test_sample = self._resample(test_seqs)
            synth_sample = self._resample(synth_seqs)

            # Compute metrics on resampled data
            results = self._compute_all_metrics(test_sample, synth_sample)

            # Extract key metrics for bootstrapping
            self._extract_bootstrap_metrics(results, bootstrap_samples)

        # Compute confidence intervals
        bootstrap_results = {}
        for metric_name, samples in bootstrap_samples.items():
            samples = np.array(samples)
            bootstrap_results[metric_name] = {
                'mean': np.mean(samples),
                'std_error': np.std(samples),
                'ci_lower': np.percentile(samples, 100 * alpha / 2),
                'ci_upper': np.percentile(samples, 100 * (1 - alpha / 2)),
                'distribution': samples
            }

        return bootstrap_results

    def _extract_bootstrap_metrics(self, results: Dict, bootstrap_samples: Dict):
        """Extract key metrics from results for bootstrap sampling."""
        # GC Content differences
        bootstrap_samples['gc_mean_diff'].append(results['gc_content']['mean_difference'])
        bootstrap_samples['gc_wasserstein'].append(results['gc_content']['wasserstein_distance'])

        # K-mer metrics
        for k in self.kmer_sizes:
            key = f'{k}-mer'
            if key in results['kmer_analysis']:
                bootstrap_samples[f'kmer_{k}_jsd'].append(
                    results['kmer_analysis'][key]['jensen_shannon_divergence']
                )
                bootstrap_samples[f'kmer_{k}_cosine'].append(
                    results['kmer_analysis'][key]['cosine_similarity']
                )

        # Novelty@r
        for r in self.novelty_radii:
            key = f'novelty@{r}'
            if key in results['novelty']:
                bootstrap_samples[f'novelty_r{r}'].append(results['novelty'][key])

        # Diversity
        bootstrap_samples['diversity'].append(results['diversity']['mean_pairwise_hamming'])

        # Nucleotide distribution
        bootstrap_samples['nucleotide_jsd'].append(
            results['nucleotide_dist']['jensen_shannon_divergence']
        )

    def _results_to_dataframe(self, results: Dict, bootstrap_results: Dict,
                              enable_bootstrap: bool) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        rows = []

        # Helper function to add rows
        def add_row(category, metric, value, unit='', interpretation=''):
            row = {
                'category': category,
                'metric': metric,
                'value': value,
                'unit': unit,
                'interpretation': interpretation
            }

            if enable_bootstrap:
                metric_key = self._get_bootstrap_key(category, metric)
                if metric_key in bootstrap_results:
                    bs = bootstrap_results[metric_key]
                    row['std_error'] = bs['std_error']
                    row['ci_lower'] = bs['ci_lower']
                    row['ci_upper'] = bs['ci_upper']
                    row['trustworthiness'] = self._assess_trustworthiness(bs['std_error'], value)
                else:
                    row['std_error'] = np.nan
                    row['ci_lower'] = np.nan
                    row['ci_upper'] = np.nan
                    row['trustworthiness'] = 'N/A'

            rows.append(row)

        # GC Content (difference metrics only)
        gc = results['gc_content']
        add_row('GC Content', 'Mean Difference', gc['mean_difference'], '%',
                'Difference in GC% (synthetic - test); closer to 0 is better')
        add_row('GC Content', 'Wasserstein Distance', gc['wasserstein_distance'], '',
                'Distribution distance; lower is better')
        add_row('GC Content', 'T-test p-value', gc['ttest_pvalue'], '',
                'p>0.05 indicates no significant difference')

        # K-mer Analysis
        for k, metrics in results['kmer_analysis'].items():
            add_row('K-mer Analysis', f'{k} JS Divergence', metrics['jensen_shannon_divergence'], '',
                    'Lower is better (0=identical, 1=completely different)')
            add_row('K-mer Analysis', f'{k} Cosine Similarity', metrics['cosine_similarity'], '',
                    'Higher is better (1=identical, 0=orthogonal)')
            add_row('K-mer Analysis', f'{k} Overlap Ratio', metrics['overlap_ratio'], '',
                    'Higher is better (shared k-mers)')

        # Novelty@r
        nov = results['novelty']
        for r in self.novelty_radii:
            key = f'novelty@{r}'
            if key in nov:
                add_row('Novelty', f'Novelty@{r}', nov[key], '',
                        f'Fraction of synthetics with min Hamming distance > {r} from train set; higher is better')

        # Diversity
        div = results['diversity']
        add_row('Diversity', 'Mean Pairwise Hamming Distance', div['mean_pairwise_hamming'], '',
                'Higher is better (more diverse); normalized by sequence length')
        add_row('Diversity', 'Std Pairwise Hamming Distance', div['std_pairwise_hamming'], '',
                'Variability in diversity')

        # Additional diversity metrics
        add_row('Diversity', 'Unique Sequences Ratio', div['unique_sequences_ratio'], '',
                'Higher is better (no duplicates = 1.0)')
        add_row('Diversity', 'Shannon Entropy', div['sequence_entropy'], 'bits',
                'Higher is better (more varied sequences)')

        # Nucleotide Distribution
        nuc = results['nucleotide_dist']
        add_row('Nucleotide Distribution', 'JS Divergence', nuc['jensen_shannon_divergence'], '',
                'Lower is better (similar A/C/G/T ratios)')
        add_row('Nucleotide Distribution', 'Total Variation Distance', nuc['total_variation_distance'], '',
                'Lower is better (similar distribution)')

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Reorder columns
        base_cols = ['category', 'metric', 'value', 'unit', 'interpretation']
        if enable_bootstrap:
            base_cols.extend(['std_error', 'ci_lower', 'ci_upper', 'trustworthiness'])

        return df[base_cols]

    def _get_bootstrap_key(self, category: str, metric: str) -> str:
        """Map category and metric to bootstrap result key."""
        mapping = {
            ('GC Content', 'Mean Difference'): 'gc_mean_diff',
            ('GC Content', 'Wasserstein Distance'): 'gc_wasserstein',
            ('Diversity', 'Mean Pairwise Hamming Distance'): 'diversity',
            ('Nucleotide Distribution', 'JS Divergence'): 'nucleotide_jsd',
        }

        # Handle k-mer metrics
        for k in self.kmer_sizes:
            mapping[(f'K-mer Analysis', f'{k}-mer JS Divergence')] = f'kmer_{k}_jsd'
            mapping[(f'K-mer Analysis', f'{k}-mer Cosine Similarity')] = f'kmer_{k}_cosine'

        # Handle novelty@r
        for r in self.novelty_radii:
            mapping[(f'Novelty', f'Novelty@{r}')] = f'novelty_r{r}'

        return mapping.get((category, metric), '')

    @staticmethod
    def _assess_trustworthiness(std_error: float, value: float) -> str:
        """
        Assess the trustworthiness of a metric based on its standard error.

        Returns: 'High', 'Medium', 'Low', or 'Very Low'
        """
        if np.isnan(std_error) or value == 0:
            return 'N/A'

        # Coefficient of variation (relative std error)
        cv = abs(std_error / value) if value != 0 else float('inf')

        if cv < 0.05:
            return 'High'
        elif cv < 0.15:
            return 'Medium'
        elif cv < 0.30:
            return 'Low'
        else:
            return 'Very Low'

    @staticmethod
    def _resample(sequences: List[str]) -> List[str]:
        """Bootstrap resample sequences with replacement."""
        indices = np.random.choice(len(sequences), size=len(sequences), replace=True)
        return [sequences[i] for i in indices]

    # ==================== Metric Computation Methods ====================

    def _evaluate_gc_content(self, test_seqs: List[str], synth_seqs: List[str]) -> Dict:
        """Evaluate GC content - return difference metrics only."""
        test_gc = np.array([self._compute_gc_content(seq) for seq in test_seqs])
        synth_gc = np.array([self._compute_gc_content(seq) for seq in synth_seqs])

        # T-test for statistical significance
        ttest_result = ttest_ind(test_gc, synth_gc)

        return {
            'mean_difference': np.mean(synth_gc) - np.mean(test_gc),
            'std_difference': np.sqrt(np.var(test_gc) + np.var(synth_gc)),
            'wasserstein_distance': wasserstein_distance(test_gc, synth_gc),
            'ttest_statistic': ttest_result.statistic,
            'ttest_pvalue': ttest_result.pvalue
        }

    def _evaluate_kmer_frequencies(self, test_seqs: List[str], synth_seqs: List[str]) -> Dict:
        """Evaluate k-mer frequency distributions."""
        results = {}

        for k in self.kmer_sizes:
            test_kmers = self._get_kmer_distribution(test_seqs, k)
            synth_kmers = self._get_kmer_distribution(synth_seqs, k)

            # Align the two distributions (same keys)
            all_kmers = set(test_kmers.keys()) | set(synth_kmers.keys())
            test_freq = np.array([test_kmers.get(km, 0) for km in sorted(all_kmers)])
            synth_freq = np.array([synth_kmers.get(km, 0) for km in sorted(all_kmers)])

            # Normalize to probabilities
            test_freq = test_freq / (test_freq.sum() + 1e-10)
            synth_freq = synth_freq / (synth_freq.sum() + 1e-10)

            results[f'{k}-mer'] = {
                'jensen_shannon_divergence': jensenshannon(test_freq, synth_freq),
                'cosine_similarity': self._cosine_similarity(test_freq, synth_freq),
                'total_unique_test': len(test_kmers),
                'total_unique_synthetic': len(synth_kmers),
                'overlap_ratio': len(set(test_kmers.keys()) & set(synth_kmers.keys())) / len(all_kmers)
            }

        return results

    def _evaluate_novelty(self, test_seqs: List[str], synth_seqs: List[str]) -> Dict:
        """
        Evaluate novelty using Novelty@r metric.

        Novelty@r = (1/|S|) * sum_{x in S} I[min_{y in T} d_H(x,y) > r]
        where d_H is Hamming distance, S is synthetic set, T is test set, r is radius
        """
        results = {}

        # Compute minimum Hamming distance for each synthetic sequence
        min_hamming_distances = []
        for synth_seq in tqdm(synth_seqs,desc='Computing min Hamming distances for novelty'):
            min_dist = min(self._hamming_distance(synth_seq, test_seq) for test_seq in test_seqs)
            min_hamming_distances.append(min_dist)

        min_hamming_distances = np.array(min_hamming_distances)

        # Compute Novelty@r for each radius
        for r in self.novelty_radii:
            novelty_at_r = np.mean(min_hamming_distances > r)
            results[f'novelty@{r}'] = novelty_at_r

        # Store min distances for additional analysis
        results['min_hamming_mean'] = np.mean(min_hamming_distances)
        results['min_hamming_std'] = np.std(min_hamming_distances)
        results['min_hamming_median'] = np.median(min_hamming_distances)

        return results

    def _evaluate_diversity(self, synth_seqs: List[str]) -> Dict:
        """
        Evaluate diversity using mean pairwise normalized Hamming distance.

        Div = (2 / (|S|(|S|-1))) * sum_{i<j} (d_H(x_i, x_j) / L)
        where L is sequence length
        """
        n = len(synth_seqs)

        if n < 2:
            return {
                'mean_pairwise_hamming': 0.0,
                'std_pairwise_hamming': 0.0,
                'unique_sequences_ratio': 1.0,
                'sequence_entropy': 0.0
            }

        # Compute all pairwise Hamming distances (normalized)
        pairwise_distances = []
        with tqdm(total=n * (n - 1) // 2, desc='Computing pairwise Hamming distances for diversity') as pbar:
            for i in range(n):
                for j in range(i + 1, n):
                    dist = self._hamming_distance(synth_seqs[i], synth_seqs[j])
                    normalized_dist = dist / self.sequence_length
                    pairwise_distances.append(normalized_dist)
                    pbar.update(1)

        pairwise_distances = np.array(pairwise_distances)

        # Unique sequences ratio
        unique_ratio = len(set(synth_seqs)) / len(synth_seqs)

        # Sequence entropy
        seq_counter = Counter(synth_seqs)
        seq_probs = np.array(list(seq_counter.values())) / len(synth_seqs)
        sequence_entropy = entropy(seq_probs)

        return {
            'mean_pairwise_hamming': np.mean(pairwise_distances),
            'std_pairwise_hamming': np.std(pairwise_distances),
            'unique_sequences_ratio': unique_ratio,
            'sequence_entropy': sequence_entropy
        }

    def _evaluate_nucleotide_distribution(self, test_seqs: List[str], synth_seqs: List[str]) -> Dict:
        """Evaluate overall nucleotide frequency distributions."""
        test_dist = self._get_nucleotide_distribution(test_seqs)
        synth_dist = self._get_nucleotide_distribution(synth_seqs)

        nucleotides = ['A', 'C', 'G', 'T']
        test_freq = np.array([test_dist.get(n, 0) for n in nucleotides])
        synth_freq = np.array([synth_dist.get(n, 0) for n in nucleotides])

        return {
            'test_distribution': {n: test_dist.get(n, 0) for n in nucleotides},
            'synthetic_distribution': {n: synth_dist.get(n, 0) for n in nucleotides},
            'jensen_shannon_divergence': jensenshannon(test_freq, synth_freq),
            'total_variation_distance': 0.5 * np.sum(np.abs(test_freq - synth_freq))
        }

    # ==================== Helper Methods ====================

    @staticmethod
    def _compute_gc_content(seq: str) -> float:
        """Compute GC content of a sequence."""
        seq = seq.upper()
        gc_count = seq.count('G') + seq.count('C')
        return gc_count / len(seq) if len(seq) > 0 else 0

    @staticmethod
    def _hamming_distance(seq1: str, seq2: str) -> int:
        """
        Compute Hamming distance between two sequences.
        Assumes sequences are of equal length.
        """
        if len(seq1) != len(seq2):
            raise ValueError(f"Sequences must be of equal length. Got {len(seq1)} and {len(seq2)}")
        return sum(c1 != c2 for c1, c2 in zip(seq1.upper(), seq2.upper()))

    @staticmethod
    def _get_kmers(seq: str, k: int) -> List[str]:
        """Extract all k-mers from a sequence."""
        return [seq[i:i + k] for i in range(len(seq) - k + 1)]

    def _get_kmer_distribution(self, sequences: List[str], k: int) -> Dict[str, float]:
        """Get normalized k-mer frequency distribution."""
        kmer_counts = Counter()
        for seq in sequences:
            kmer_counts.update(self._get_kmers(seq.upper(), k))

        total = sum(kmer_counts.values())
        return {kmer: count / total for kmer, count in kmer_counts.items()}

    @staticmethod
    def _get_nucleotide_distribution(sequences: List[str]) -> Dict[str, float]:
        """Get overall nucleotide frequency distribution."""
        nucleotide_counts = Counter()
        for seq in sequences:
            nucleotide_counts.update(seq.upper())

        total = sum(nucleotide_counts.values())
        return {nuc: count / total for nuc, count in nucleotide_counts.items()}

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2 + 1e-10)

    def _print_summary(self, df: pd.DataFrame):
        """Print a summary of evaluation results from DataFrame."""
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY WITH CONFIDENCE INTERVALS")
        print("=" * 80)

        # Print by category
        for category in df['category'].unique():
            print(f"\n[{category}]")
            cat_df = df[df['category'] == category]

            for _, row in cat_df.iterrows():
                metric_name = row['metric']
                value = row['value']
                unit = row['unit']

                # Format value display
                if isinstance(value, float):
                    if abs(value) < 0.001:
                        value_str = f"{value:.6f}"
                    else:
                        value_str = f"{value:.4f}"
                else:
                    value_str = str(value)

                # Add confidence interval if available
                if 'ci_lower' in row and not pd.isna(row['ci_lower']):
                    ci_str = f" (95% CI: [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}])"
                    trust_str = f" [{row['trustworthiness']} trust]"
                else:
                    ci_str = ""
                    trust_str = ""

                print(f"  {metric_name}: {value_str} {unit}{ci_str}{trust_str}")

        print("\n" + "=" * 80)
        print("\nTrustworthiness Guide:")
        print("  High: Standard error < 5% of value (very reliable)")
        print("  Medium: Standard error 5-15% of value (reasonably reliable)")
        print("  Low: Standard error 15-30% of value (interpret with caution)")
        print("  Very Low: Standard error > 30% of value (unreliable, need more data)")
        print("=" * 80 + "\n")


# Example usage
if __name__ == "__main__":
    # Example test sequences
    np.random.seed(42)

    # Initialize evaluator
    config = BootstrapConfig(n_bootstrap=8, confidence_level=0.95, random_seed=42)
    evaluator = DNASequenceEvaluator(
        kmer_sizes=[3, 4, 5],
        novelty_radii=[1, 2, 3, 5, 10],
        sequence_length=200,
        bootstrap_config=config,
        verbose=True
    )

    test_sequences = \
    pd.read_csv("/home/benjaminkroeger/Documents/Master/UBC/Synthetic_data/DNA-Diffusion/data/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt",sep="\t")["sequence"].sample(n=4000).tolist()
    with open(
        "/home/benjaminkroeger/Documents/Master/UBC/Synthetic_data/DNA-Diffusion/data/outputs/original_model_colab/combined_dna_diff_original_seqs.txt",
        "r") as f:
        synthetic_sequences = f.readlines()
        synthetic_sequences = [x.strip() for x in synthetic_sequences]

    # Run evaluation
    results_df, detailed_results = evaluator.evaluate(
        test_sequences,
        synthetic_sequences,
        enable_bootstrap=True
    )

    # Display DataFrame
    print("\n" + "=" * 80)
    print("RESULTS DATAFRAME")
    print("=" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    print(results_df.to_string(index=False))

    # Save to CSV
    results_df.to_csv('dna_evaluation_results.csv', index=False)
    print("\n✓ Results saved to 'dna_evaluation_results.csv'")

    print("\n" + "=" * 80)
    print("KEY INTERPRETATIONS")
    print("=" * 80)
    print("""
GC Content Mean Difference: How much GC% differs (synthetic - test)
    • Close to 0 = good match
    • Check p-value: >0.05 means no significant difference

Novelty@r: Fraction of sequences with min Hamming distance > r from training
    • Novelty@1 = completely unique sequences
    • Novelty@5 = sequences with at least 5 different positions
    • Higher = more novel (but may indicate mode collapse if too high)

Diversity (Mean Pairwise Hamming): Average normalized Hamming distance
    • Range: [0, 1]
    • 0 = all identical
    • 1 = maximally different
    • Higher = more diverse generation

K-mer JS Divergence: How different k-mer distributions are
    • 0 = identical distributions
    • Lower = better match to training distribution
    """)
