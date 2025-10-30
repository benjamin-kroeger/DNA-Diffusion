"""
DNA Sequence Generation Evaluation Framework with Bootstrapping

Computes comprehensive metrics for evaluating synthetic DNA sequences
against test/reference sequences with confidence intervals.
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
from scipy.stats import entropy, ks_2samp, wasserstein_distance
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
                 bootstrap_config: Optional[BootstrapConfig] = None,
                 verbose: bool = True):
        """
        Initialize the evaluator.

        Args:
            kmer_sizes: List of k-mer sizes to analyze (default: [3, 4, 5])
            bootstrap_config: Configuration for bootstrapping (default: 1000 iterations, 95% CI)
            verbose: Whether to print progress messages
        """
        self.kmer_sizes = kmer_sizes
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

        # 1. GC Content Comparison
        results['gc_content'] = self._evaluate_gc_content(test_seqs, synth_seqs)

        # 2. K-mer Frequency Comparison
        results['kmer_analysis'] = self._evaluate_kmer_frequencies(test_seqs, synth_seqs)

        # 3. Novelty Metrics
        results['novelty'] = self._evaluate_novelty(test_seqs, synth_seqs)

        # 4. Diversity Metrics
        results['diversity'] = self._evaluate_diversity(synth_seqs)

        # 5. Sequence Length Statistics
        results['length_stats'] = self._evaluate_length_distribution(test_seqs, synth_seqs)

        # 6. Nucleotide Distribution
        results['nucleotide_dist'] = self._evaluate_nucleotide_distribution(test_seqs, synth_seqs)

        # 7. Entropy Metrics
        results['entropy'] = self._evaluate_entropy(test_seqs, synth_seqs)

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
        # GC Content
        bootstrap_samples['gc_content_diff'].append(results['gc_content']['mean_absolute_difference'])
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

        # Novelty
        bootstrap_samples['exact_match_ratio'].append(results['novelty']['exact_match_ratio'])
        bootstrap_samples['min_edit_distance'].append(results['novelty']['min_edit_distance_mean'])

        # Diversity
        bootstrap_samples['unique_sequences_ratio'].append(results['diversity']['unique_sequences_ratio'])
        bootstrap_samples['self_bleu'].append(results['diversity']['self_bleu_mean'])
        bootstrap_samples['distinct_2'].append(results['diversity']['distinct-2'])
        bootstrap_samples['pairwise_edit_distance'].append(results['diversity']['pairwise_edit_distance_mean'])

        # Length
        bootstrap_samples['length_ks_statistic'].append(results['length_stats']['ks_statistic'])

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

        # GC Content
        gc = results['gc_content']
        add_row('GC Content', 'Test Mean', gc['test_mean'], '%', 'Reference baseline')
        add_row('GC Content', 'Synthetic Mean', gc['synthetic_mean'], '%', 'Generated sequences')
        add_row('GC Content', 'Absolute Difference', gc['mean_absolute_difference'], '%',
                'Lower is better (closer to test)')
        add_row('GC Content', 'Wasserstein Distance', gc['wasserstein_distance'], '',
                'Lower is better (similar distribution)')
        add_row('GC Content', 'KS Test p-value', gc['ks_pvalue'], '',
                'Higher is better (p>0.05 = similar)')

        # K-mer Analysis
        for k, metrics in results['kmer_analysis'].items():
            add_row('K-mer Analysis', f'{k} JS Divergence', metrics['jensen_shannon_divergence'], '',
                    'Lower is better (0=identical, 1=completely different)')
            add_row('K-mer Analysis', f'{k} Cosine Similarity', metrics['cosine_similarity'], '',
                    'Higher is better (1=identical, 0=orthogonal)')
            add_row('K-mer Analysis', f'{k} Overlap Ratio', metrics['overlap_ratio'], '',
                    'Higher is better (shared k-mers)')

        # Novelty
        nov = results['novelty']
        add_row('Novelty', 'Exact Match Ratio', nov['exact_match_ratio'], '',
                'Lower is better (less memorization)')
        add_row('Novelty', 'Min Edit Distance Mean', nov['min_edit_distance_mean'], 'bp',
                'Higher is better (more novel)')
        add_row('Novelty', 'Min Edit Distance Std', nov['min_edit_distance_std'], 'bp',
                'Variability in novelty')

        # Diversity
        div = results['diversity']
        add_row('Diversity', 'Unique Sequences Ratio', div['unique_sequences_ratio'], '',
                'Higher is better (more diverse)')
        add_row('Diversity', 'Self-BLEU Mean', div['self_bleu_mean'], '',
                'Lower is better (less self-similar)')
        add_row('Diversity', 'Distinct-1', div['distinct-1'], '',
                'Higher is better (unique 1-mers)')
        add_row('Diversity', 'Distinct-2', div['distinct-2'], '',
                'Higher is better (unique 2-mers)')
        add_row('Diversity', 'Distinct-3', div['distinct-3'], '',
                'Higher is better (unique 3-mers)')
        add_row('Diversity', 'Pairwise Edit Distance Mean', div['pairwise_edit_distance_mean'], 'bp',
                'Higher is better (more diverse)')
        add_row('Diversity', 'Sequence Entropy', div['sequence_entropy'], 'bits',
                'Higher is better (more varied)')

        # Length Statistics
        length = results['length_stats']
        add_row('Length', 'Test Mean', length['test_mean'], 'bp', 'Reference baseline')
        add_row('Length', 'Synthetic Mean', length['synthetic_mean'], 'bp', 'Generated sequences')
        add_row('Length', 'KS Test Statistic', length['ks_statistic'], '',
                'Lower is better (similar distribution)')
        add_row('Length', 'KS Test p-value', length['ks_pvalue'], '',
                'Higher is better (p>0.05 = similar)')

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
            ('GC Content', 'Absolute Difference'): 'gc_content_diff',
            ('GC Content', 'Wasserstein Distance'): 'gc_wasserstein',
            ('Novelty', 'Exact Match Ratio'): 'exact_match_ratio',
            ('Novelty', 'Min Edit Distance Mean'): 'min_edit_distance',
            ('Diversity', 'Unique Sequences Ratio'): 'unique_sequences_ratio',
            ('Diversity', 'Self-BLEU Mean'): 'self_bleu',
            ('Diversity', 'Distinct-2'): 'distinct_2',
            ('Diversity', 'Pairwise Edit Distance Mean'): 'pairwise_edit_distance',
            ('Length', 'KS Test Statistic'): 'length_ks_statistic',
            ('Nucleotide Distribution', 'JS Divergence'): 'nucleotide_jsd',
        }

        # Handle k-mer metrics
        for k in self.kmer_sizes:
            mapping[(f'K-mer Analysis', f'{k}-mer JS Divergence')] = f'kmer_{k}_jsd'
            mapping[(f'K-mer Analysis', f'{k}-mer Cosine Similarity')] = f'kmer_{k}_cosine'

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

    # ==================== Original Metric Methods ====================

    def _evaluate_gc_content(self, test_seqs: List[str], synth_seqs: List[str]) -> Dict:
        """Evaluate GC content comparison between test and synthetic sequences."""
        test_gc = [self._compute_gc_content(seq) for seq in test_seqs]
        synth_gc = [self._compute_gc_content(seq) for seq in synth_seqs]

        return {
            'test_mean': np.mean(test_gc),
            'test_std': np.std(test_gc),
            'synthetic_mean': np.mean(synth_gc),
            'synthetic_std': np.std(synth_gc),
            'mean_absolute_difference': abs(np.mean(test_gc) - np.mean(synth_gc)),
            'ks_statistic': ks_2samp(test_gc, synth_gc).statistic,
            'ks_pvalue': ks_2samp(test_gc, synth_gc).pvalue,
            'wasserstein_distance': wasserstein_distance(test_gc, synth_gc)
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
        """Evaluate novelty of synthetic sequences."""
        # 1. Exact match ratio
        exact_matches = sum(1 for seq in synth_seqs if seq in test_seqs)

        # 2. Minimum edit distance to training set
        min_edit_distances = []
        for synth_seq in synth_seqs:
            min_dist = min(self._edit_distance(synth_seq, test_seq) for test_seq in test_seqs)
            min_edit_distances.append(min_dist)

        # 3. K-mer novelty (percentage of k-mers not in test set)
        kmer_novelty = {}
        for k in self.kmer_sizes:
            test_kmers = set()
            for seq in test_seqs:
                test_kmers.update(self._get_kmers(seq, k))

            novel_kmer_counts = []
            for seq in synth_seqs:
                synth_kmers = set(self._get_kmers(seq, k))
                novel_count = len(synth_kmers - test_kmers)
                novel_kmer_counts.append(novel_count / (len(synth_kmers) + 1e-10))

            kmer_novelty[f'{k}-mer_novelty'] = np.mean(novel_kmer_counts)

        return {
            'exact_match_ratio': exact_matches / len(synth_seqs),
            'min_edit_distance_mean': np.mean(min_edit_distances),
            'min_edit_distance_median': np.median(min_edit_distances),
            'min_edit_distance_std': np.std(min_edit_distances),
            **kmer_novelty
        }

    def _evaluate_diversity(self, synth_seqs: List[str]) -> Dict:
        """Evaluate diversity within synthetic sequences."""
        # 1. Unique sequences ratio
        unique_ratio = len(set(synth_seqs)) / len(synth_seqs)

        # 2. Self-BLEU (lower is more diverse)
        self_bleu_scores = []
        for i, seq in enumerate(synth_seqs):
            other_seqs = synth_seqs[:i] + synth_seqs[i+1:]
            if other_seqs:
                bleu = self._compute_self_bleu(seq, other_seqs, n=4)
                self_bleu_scores.append(bleu)

        # 3. Distinct-n (higher is more diverse)
        distinct_n = {}
        for n in [1, 2, 3]:
            distinct_n[f'distinct-{n}'] = self._compute_distinct_n(synth_seqs, n)

        # 4. Pairwise edit distance (sample for efficiency)
        sample_size = min(100, len(synth_seqs))
        sample_indices = np.random.choice(len(synth_seqs), sample_size, replace=False)
        pairwise_dists = []
        for i in range(len(sample_indices)):
            for j in range(i+1, len(sample_indices)):
                dist = self._edit_distance(synth_seqs[sample_indices[i]],
                                          synth_seqs[sample_indices[j]])
                pairwise_dists.append(dist)

        # 5. Entropy of sequences
        seq_counter = Counter(synth_seqs)
        seq_probs = np.array(list(seq_counter.values())) / len(synth_seqs)
        sequence_entropy = entropy(seq_probs)

        return {
            'unique_sequences_ratio': unique_ratio,
            'self_bleu_mean': np.mean(self_bleu_scores) if self_bleu_scores else 0,
            'self_bleu_std': np.std(self_bleu_scores) if self_bleu_scores else 0,
            **distinct_n,
            'pairwise_edit_distance_mean': np.mean(pairwise_dists) if pairwise_dists else 0,
            'pairwise_edit_distance_std': np.std(pairwise_dists) if pairwise_dists else 0,
            'sequence_entropy': sequence_entropy
        }

    def _evaluate_length_distribution(self, test_seqs: List[str], synth_seqs: List[str]) -> Dict:
        """Evaluate sequence length distributions."""
        test_lengths = [len(seq) for seq in test_seqs]
        synth_lengths = [len(seq) for seq in synth_seqs]

        return {
            'test_mean': np.mean(test_lengths),
            'test_std': np.std(test_lengths),
            'synthetic_mean': np.mean(synth_lengths),
            'synthetic_std': np.std(synth_lengths),
            'ks_statistic': ks_2samp(test_lengths, synth_lengths).statistic,
            'ks_pvalue': ks_2samp(test_lengths, synth_lengths).pvalue
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

    def _evaluate_entropy(self, test_seqs: List[str], synth_seqs: List[str]) -> Dict:
        """Evaluate various entropy measures."""
        # Positional entropy (if sequences have same length)
        test_pos_entropy = self._compute_positional_entropy(test_seqs)
        synth_pos_entropy = self._compute_positional_entropy(synth_seqs)

        return {
            'test_positional_entropy_mean': np.mean(test_pos_entropy) if test_pos_entropy else None,
            'synthetic_positional_entropy_mean': np.mean(synth_pos_entropy) if synth_pos_entropy else None,
        }

    # ==================== Helper Methods ====================

    @staticmethod
    def _compute_gc_content(seq: str) -> float:
        """Compute GC content of a sequence."""
        seq = seq.upper()
        gc_count = seq.count('G') + seq.count('C')
        return gc_count / len(seq) if len(seq) > 0 else 0

    @staticmethod
    def _get_kmers(seq: str, k: int) -> List[str]:
        """Extract all k-mers from a sequence."""
        return [seq[i:i+k] for i in range(len(seq) - k + 1)]

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
    def _edit_distance(seq1: str, seq2: str) -> int:
        """Compute Levenshtein edit distance."""
        if len(seq1) < len(seq2):
            return DNASequenceEvaluator._edit_distance(seq2, seq1)

        if len(seq2) == 0:
            return len(seq1)

        previous_row = range(len(seq2) + 1)
        for i, c1 in enumerate(seq1):
            current_row = [i + 1]
            for j, c2 in enumerate(seq2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2 + 1e-10)

    def _compute_self_bleu(self, seq: str, other_seqs: List[str], n: int = 4) -> float:
        """Compute self-BLEU score for a sequence against others."""
        ref_ngrams = [Counter(self._get_kmers(s.upper(), n)) for s in other_seqs]
        hyp_ngrams = Counter(self._get_kmers(seq.upper(), n))

        if not hyp_ngrams or not ref_ngrams:
            return 0.0

        scores = []
        for ref in ref_ngrams:
            matches = sum(min(hyp_ngrams[ng], ref[ng]) for ng in hyp_ngrams)
            total = sum(hyp_ngrams.values())
            scores.append(matches / total if total > 0 else 0)

        return np.mean(scores)

    def _compute_distinct_n(self, sequences: List[str], n: int) -> float:
        """Compute distinct-n metric (ratio of unique n-grams)."""
        all_ngrams = []
        for seq in sequences:
            all_ngrams.extend(self._get_kmers(seq.upper(), n))

        if not all_ngrams:
            return 0.0

        return len(set(all_ngrams)) / len(all_ngrams)

    @staticmethod
    def _compute_positional_entropy(sequences: List[str]) -> List[float]:
        """Compute entropy at each position (for equal-length sequences)."""
        if not sequences:
            return []

        # Check if all sequences have same length
        lengths = [len(s) for s in sequences]
        if len(set(lengths)) > 1:
            return []  # Can't compute positional entropy for variable lengths

        seq_length = lengths[0]
        entropies = []

        for pos in range(seq_length):
            nucleotides = [seq[pos].upper() for seq in sequences]
            counts = Counter(nucleotides)
            probs = np.array(list(counts.values())) / len(nucleotides)
            entropies.append(entropy(probs))

        return entropies

    def _print_summary(self, df: pd.DataFrame):
        """Print a summary of evaluation results from DataFrame."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY WITH CONFIDENCE INTERVALS")
        print("="*80)

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
                if row['interpretation']:
                    print(f"    → {row['interpretation']}")

        print("\n" + "="*80)
        print("\nTrustworthiness Guide:")
        print("  High: Standard error < 5% of value (very reliable)")
        print("  Medium: Standard error 5-15% of value (reasonably reliable)")
        print("  Low: Standard error 15-30% of value (interpret with caution)")
        print("  Very Low: Standard error > 30% of value (unreliable, need more data)")
        print("="*80 + "\n")


# Example usage
if __name__ == "__main__":
    # Example test sequences
    np.random.seed(42)

    # Generate more realistic example data
    def generate_random_dna(length, gc_content=0.5):
        """Generate random DNA sequence with specified GC content."""
        seq = []
        for _ in range(length):
            if np.random.random() < gc_content:
                seq.append(np.random.choice(['G', 'C']))
            else:
                seq.append(np.random.choice(['A', 'T']))
        return ''.join(seq)

    # Create test dataset
    test_sequences = [generate_random_dna(50, gc_content=0.52) for _ in range(100)]

    # Create synthetic dataset (slightly different GC content, some duplicates)
    synthetic_sequences = [generate_random_dna(50, gc_content=0.48) for _ in range(80)]
    # Add some exact matches to test novelty
    synthetic_sequences.extend(test_sequences[:5])
    # Add some duplicates to test diversity
    synthetic_sequences.extend([synthetic_sequences[0]] * 3)

    print("="*80)
    print("DNA SEQUENCE EVALUATION EXAMPLE")
    print("="*80)
    print(f"\nTest sequences: {len(test_sequences)}")
    print(f"Synthetic sequences: {len(synthetic_sequences)}")
    print("\nNote: This example uses 100 bootstrap iterations for speed.")
    print("For publication-quality results, use 1000+ iterations.\n")

    # Initialize evaluator with fewer bootstrap iterations for demo
    config = BootstrapConfig(n_bootstrap=10, confidence_level=0.95, random_seed=42)
    evaluator = DNASequenceEvaluator(
        kmer_sizes=[3, 4, 5],
        bootstrap_config=config,
        verbose=True
    )

    # Run evaluation
    results_df, detailed_results = evaluator.evaluate(
        test_sequences,
        synthetic_sequences,
        enable_bootstrap=True
    )

    # Display DataFrame
    print("\n" + "="*80)
    print("RESULTS DATAFRAME (First 10 rows)")
    print("="*80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)
    print(results_df.head(10).to_string(index=False))

    # Save to CSV
    results_df.to_csv('dna_evaluation_results.csv', index=False)
    print("\n✓ Results saved to 'dna_evaluation_results.csv'")

    # Access detailed bootstrap distributions
    print("\n" + "="*80)
    print("BOOTSTRAP DISTRIBUTION EXAMPLE")
    print("="*80)
    gc_diff_dist = detailed_results['bootstrap']['gc_content_diff']['distribution']
    print(f"GC Content Difference Bootstrap Distribution:")
    print(f"  Mean: {np.mean(gc_diff_dist):.4f}")
    print(f"  Std Error: {np.std(gc_diff_dist):.4f}")
    print(f"  95% CI: [{np.percentile(gc_diff_dist, 2.5):.4f}, {np.percentile(gc_diff_dist, 97.5):.4f}]")

    # Example: Filter for high-trust metrics
    print("\n" + "="*80)
    print("HIGH-TRUST METRICS ONLY")
    print("="*80)
    high_trust = results_df[results_df['trustworthiness'] == 'High']
    if len(high_trust) > 0:
        print(high_trust[['category', 'metric', 'value', 'std_error', 'trustworthiness']].to_string(index=False))
    else:
        print("No metrics with 'High' trustworthiness in this example.")
        print("(Increase bootstrap iterations or sample size for better precision)")

    print("\n" + "="*80)
    print("USAGE TIPS")
    print("="*80)
    print("""
1. Access results as DataFrame:
   results_df['value']  # All metric values
   results_df[results_df['category'] == 'Novelty']  # Filter by category

2. Export results:
   results_df.to_csv('results.csv')
   results_df.to_latex('results.tex')

3. Access bootstrap distributions:
   detailed_results['bootstrap']['gc_content_diff']['distribution']

4. Interpret trustworthiness:
   - High: Report with confidence
   - Medium: Report with standard errors
   - Low/Very Low: Need more data or report cautiously

5. For publication:
   - Use n_bootstrap >= 1000
   - Report confidence intervals for key metrics
   - Check trustworthiness before making strong claims
    """)
