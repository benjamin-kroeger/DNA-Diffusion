#!/usr/bin/env python3
"""
Script 1: DNA Sequence Metrics Evaluation

Computes sequence-level metrics comparing ground truth vs synthetic sequences:
- K-mer frequency similarities (3, 5, 7, 9-mers) with Jensen-Shannon Divergence
- GC content similarity
- Discriminator accuracy (can a classifier distinguish real from synthetic?)

Metrics explained:
- Discriminator: Trains a classifier on k-mer features to distinguish GT from synthetic.
  If accuracy >> 50%, synthetic is detectably different from real.
- K-mer JSD: Jensen-Shannon Divergence between k-mer frequency distributions.
  Lower = more similar to reference.
- GC Content: Basic compositional similarity.

Author: DNA Diffusion Analysis Pipeline
"""

import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Tuple, Optional, Union
from scipy.stats import entropy, ttest_ind, mannwhitneyu, sem
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
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
    sequence_length: int = 200
    random_seed: int = 42


class DNASequenceMetrics:
    """
    DNA sequence metrics calculator for evaluating generative models.

    Computes k-mer frequencies, GC content, and discriminator accuracy.
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        np.random.seed(self.config.random_seed)

    # ==================== Core Metric Functions ====================

    @staticmethod
    def compute_gc_content(seq: str) -> float:
        """Compute GC content of a sequence."""
        seq = seq.upper()
        gc_count = seq.count('G') + seq.count('C')
        return gc_count / len(seq) if len(seq) > 0 else 0

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

    # ==================== Diversity Metrics ====================

    def compute_diversity(self, sequences: List[str]) -> Dict:
        """Compute diversity metrics for a set of sequences."""
        n = len(sequences)

        if n < 2:
            return {
                'unique_ratio': 1.0,
                'shannon_entropy': 0.0,
            }

        unique_ratio = len(set(sequences)) / len(sequences)
        seq_counter = Counter(sequences)
        seq_probs = np.array(list(seq_counter.values())) / len(sequences)
        shannon_ent = entropy(seq_probs)

        return {
            'unique_ratio': float(unique_ratio),
            'shannon_entropy': float(shannon_ent),
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

            results[f'{k}-mer'] = {
                'jensen_shannon_divergence': float(jsd),
                'cosine_similarity': float(cos_sim),
                'n_unique_reference': len(ref_dist),
                'n_unique_query': len(query_dist),
            }

        return results

    # ==================== GC Content Analysis ====================

    def compute_gc_metrics(self,
                           reference_seqs: List[str],
                           query_seqs: List[str]) -> Dict:
        """Compute GC content comparison metrics."""
        ref_gc = np.array([self.compute_gc_content(s) for s in reference_seqs])
        query_gc = np.array([self.compute_gc_content(s) for s in query_seqs])

        ttest = ttest_ind(ref_gc, query_gc)
        mwu = mannwhitneyu(ref_gc, query_gc, alternative='two-sided')

        return {
            'reference_mean': float(np.mean(ref_gc)),
            'reference_std': float(np.std(ref_gc)),
            'reference_se': float(sem(ref_gc)),
            'query_mean': float(np.mean(query_gc)),
            'query_std': float(np.std(query_gc)),
            'query_se': float(sem(query_gc)),
            'mean_difference': float(np.mean(query_gc) - np.mean(ref_gc)),
            'ttest_statistic': float(ttest.statistic),
            'ttest_pvalue': float(ttest.pvalue),
            'mannwhitney_pvalue': float(mwu.pvalue),
        }

    # ==================== K-mer Feature Extraction ====================

    def _get_kmer_feature_vectors(self,
                                  sequences: List[str],
                                  k: int = 4) -> np.ndarray:
        """Convert sequences to k-mer frequency vectors."""
        from itertools import product
        all_kmers = [''.join(p) for p in product('ACGT', repeat=k)]
        kmer_to_idx = {km: i for i, km in enumerate(all_kmers)}
        n_kmers = len(all_kmers)

        X = np.zeros((len(sequences), n_kmers), dtype=np.float32)

        for i, seq in enumerate(sequences):
            seq = seq.upper()
            kmer_counts = Counter()
            for j in range(len(seq) - k + 1):
                kmer = seq[j:j + k]
                if kmer in kmer_to_idx:
                    kmer_counts[kmer] += 1

            total = sum(kmer_counts.values())
            if total > 0:
                for kmer, count in kmer_counts.items():
                    X[i, kmer_to_idx[kmer]] = count / total

        return X

    # ==================== Discriminator Metrics ====================

    def compute_discriminator_metrics(self,
                                      reference_seqs: List[str],
                                      query_seqs: List[str],
                                      k: int = 4,
                                      n_samples: int = 5000,
                                      show_progress: bool = True) -> Dict:
        """
        Train a classifier to distinguish real from synthetic sequences.

        If accuracy >> 50%, the synthetic data is detectably different from real.
        Uses balanced sampling and balanced accuracy for fair evaluation.

        Returns:
            Dictionary with balanced accuracy, AUC, and top discriminating k-mers
        """
        if show_progress:
            print("  Building k-mer feature vectors...")

        # Balanced sampling
        n_per_class = min(n_samples, len(reference_seqs), len(query_seqs))

        if show_progress:
            print(f"  Using balanced sampling: {n_per_class} sequences per class")

        # Sample from reference
        if len(reference_seqs) > n_per_class:
            ref_idx = np.random.choice(len(reference_seqs), n_per_class, replace=False)
            ref_sample = [reference_seqs[i] for i in ref_idx]
        else:
            ref_sample = reference_seqs

        # Sample from query
        if len(query_seqs) > n_per_class:
            query_idx = np.random.choice(len(query_seqs), n_per_class, replace=False)
            query_sample = [query_seqs[i] for i in query_idx]
        else:
            query_sample = query_seqs

        # Get feature vectors
        X_ref = self._get_kmer_feature_vectors(ref_sample, k)
        X_query = self._get_kmer_feature_vectors(query_sample, k)

        # Combine and create labels
        X = np.vstack([X_ref, X_query])
        y = np.array([0] * len(X_ref) + [1] * len(X_query))

        # Shuffle
        shuffle_idx = np.random.permutation(len(y))
        X = X[shuffle_idx]
        y = y[shuffle_idx]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if show_progress:
            print("  Training classifiers with 5-fold cross-validation...")

        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                    class_weight='balanced', random_state=42, n_jobs=-1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        rf_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring='balanced_accuracy')
        rf_auc_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring='roc_auc')

        # Train Logistic Regression
        lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        lr_scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring='balanced_accuracy')
        lr_auc_scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring='roc_auc')

        # Get feature importances from final RF model
        rf.fit(X_scaled, y)
        from itertools import product
        all_kmers = [''.join(p) for p in product('ACGT', repeat=k)]
        importance_idx = np.argsort(rf.feature_importances_)[::-1][:20]
        top_kmers = [(all_kmers[i], float(rf.feature_importances_[i])) for i in importance_idx]

        rf_acc = float(np.mean(rf_scores))
        rf_acc_std = float(np.std(rf_scores))
        rf_auc = float(np.mean(rf_auc_scores))
        rf_auc_std = float(np.std(rf_auc_scores))

        # Interpretation
        if rf_acc < 0.55:
            interpretation = "indistinguishable (excellent generation)"
        elif rf_acc < 0.60:
            interpretation = "slightly distinguishable"
        elif rf_acc < 0.70:
            interpretation = "moderately distinguishable"
        elif rf_acc < 0.80:
            interpretation = "highly distinguishable"
        else:
            interpretation = "very easily distinguishable (poor generation)"

        return {
            'balanced_accuracy': rf_acc,
            'balanced_accuracy_std': rf_acc_std,
            'balanced_accuracy_ci95': (rf_acc - 1.96 * rf_acc_std, rf_acc + 1.96 * rf_acc_std),
            'roc_auc': rf_auc,
            'roc_auc_std': rf_auc_std,
            'roc_auc_ci95': (rf_auc - 1.96 * rf_auc_std, rf_auc + 1.96 * rf_auc_std),
            'logistic_regression_balanced_acc': float(np.mean(lr_scores)),
            'logistic_regression_auc': float(np.mean(lr_auc_scores)),
            'baseline_accuracy': 0.5,
            'interpretation': interpretation,
            'top_discriminating_kmers': top_kmers,
            'kmer_size': k,
            'n_per_class': n_per_class,
            'cv_folds': 5,
        }

    # ==================== Cell-Type Fidelity ====================

    def compute_celltype_fidelity(self,
                                  reference_seqs: List[str],
                                  reference_celltypes: List[str],
                                  query_seqs: List[str],
                                  query_celltypes: List[str],
                                  k: int = 4,
                                  test_size: float = 0.2,
                                  show_progress: bool = True) -> Dict:
        """Measure how well synthetic sequences preserve cell-type specificity."""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        if show_progress:
            print("  Building k-mer feature vectors for cell-type classification...")

        le = LabelEncoder()
        all_celltypes = list(set(reference_celltypes) | set(query_celltypes))
        le.fit(all_celltypes)

        ref_labels = le.transform(reference_celltypes)
        query_labels = le.transform(query_celltypes)

        n_classes = len(le.classes_)

        if show_progress:
            print(f"  Found {n_classes} cell types: {list(le.classes_)}")

        X_ref = self._get_kmer_feature_vectors(reference_seqs, k)
        X_query = self._get_kmer_feature_vectors(query_seqs, k)

        X_train, X_test_gt, y_train, y_test_gt = train_test_split(
            X_ref, ref_labels, test_size=test_size, stratify=ref_labels, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_gt_scaled = scaler.transform(X_test_gt)
        X_query_scaled = scaler.transform(X_query)

        if show_progress:
            print("  Training classifier on GT...")

        rf = RandomForestClassifier(n_estimators=100, max_depth=15,
                                    class_weight='balanced', random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)

        gt_accuracy = rf.score(X_test_gt_scaled, y_test_gt)
        synth_accuracy = rf.score(X_query_scaled, query_labels)

        fidelity_gap = gt_accuracy - synth_accuracy
        baseline = 1.0 / n_classes

        if fidelity_gap < 0.05:
            interpretation = "excellent - synthetic preserves cell-type specificity"
        elif fidelity_gap < 0.10:
            interpretation = "good - minor loss of cell-type specificity"
        elif fidelity_gap < 0.20:
            interpretation = "moderate - noticeable loss of cell-type specificity"
        else:
            interpretation = "poor - synthetic loses cell-type identity"

        return {
            'gt_accuracy': float(gt_accuracy),
            'synthetic_accuracy': float(synth_accuracy),
            'fidelity_gap': float(fidelity_gap),
            'baseline_accuracy': float(baseline),
            'interpretation': interpretation,
            'n_classes': n_classes,
            'cell_types': list(le.classes_),
        }

    # ==================== Full Evaluation ====================

    def evaluate(self,
                 reference_seqs: List[str],
                 query_seqs: List[str],
                 query_name: str = "synthetic",
                 reference_celltypes: Optional[List[str]] = None,
                 query_celltypes: Optional[List[str]] = None,
                 show_progress: bool = True) -> Dict:
        """Run full evaluation suite."""
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

        print("\n[1/4] Computing diversity metrics...")
        results['diversity'] = self.compute_diversity(query_seqs)

        print("\n[2/4] Computing k-mer metrics...")
        results['kmer'] = self.compute_kmer_metrics(reference_seqs, query_seqs)

        print("\n[3/4] Computing GC content metrics...")
        results['gc_content'] = self.compute_gc_metrics(reference_seqs, query_seqs)

        print("\n[4/4] Training discriminator (can we tell synthetic from real?)...")
        results['discriminator'] = self.compute_discriminator_metrics(
            reference_seqs, query_seqs, k=4, show_progress=show_progress
        )

        # Cell-type fidelity if provided
        if reference_celltypes is not None and query_celltypes is not None:
            print("\n[Bonus] Computing cell-type conditional fidelity...")
            results['celltype_fidelity'] = self.compute_celltype_fidelity(
                reference_seqs, reference_celltypes,
                query_seqs, query_celltypes,
                k=4, show_progress=show_progress
            )

        return results


class MetricsVisualizer:
    """Generate publication-quality visualizations for DNA sequence metrics."""

    def __init__(self, output_dir: str = "figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_colors = {
            'ground_truth': '#2ecc71',
            'reference': '#2ecc71',
        }

    def _get_color(self, name: str, idx: int = 0) -> str:
        """Get color for a model name."""
        if name.lower() in self.model_colors:
            return self.model_colors[name.lower()]
        colors = ['#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22']
        return colors[idx % len(colors)]

    def plot_gc_content_barplot(self,
                                results_dict: Dict[str, Dict],
                                reference_gc_mean: float,
                                reference_gc_se: float,
                                output_name: str = "gc_content_barplot") -> str:
        """Plot GC content as bar plot with standard error bars."""
        models = ['Reference'] + list(results_dict.keys())
        means = [reference_gc_mean] + [results_dict[m]['gc_content']['query_mean'] for m in results_dict]
        ses = [reference_gc_se] + [results_dict[m]['gc_content']['query_se'] for m in results_dict]

        fig, ax = plt.subplots(figsize=(8, 6))

        colors = [self._get_color('reference')] + [self._get_color(m, i) for i, m in enumerate(results_dict.keys())]

        bars = ax.bar(models, means, yerr=ses, capsize=5, color=colors,
                      edgecolor='black', linewidth=1.5, error_kw={'linewidth': 2})

        ax.set_ylabel('GC Content')
        ax.set_title('GC Content Comparison')

        # Set y-axis to show differences clearly
        y_min = min(means) - max(ses) * 3
        y_max = max(means) + max(ses) * 3
        ax.set_ylim(max(0, y_min), min(1, y_max))

        # Add significance markers
        for i, (name, results) in enumerate(results_dict.items()):
            pval = results['gc_content']['ttest_pvalue']
            if pval < 0.001:
                sig = '***'
            elif pval < 0.01:
                sig = '**'
            elif pval < 0.05:
                sig = '*'
            else:
                sig = 'ns'
            ax.text(i + 1, means[i + 1] + ses[i + 1] + 0.005, sig,
                    ha='center', va='bottom', fontsize=12)

        plt.tight_layout()
        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def plot_kmer_jsd_heatmap(self,
                              results_dict: Dict[str, Dict],
                              output_name: str = "kmer_jsd_heatmap") -> str:
        """Plot k-mer Jensen-Shannon Divergence as heatmap."""
        models = list(results_dict.keys())
        k_sizes = sorted(results_dict[models[0]]['kmer'].keys())

        jsd_matrix = np.zeros((len(models), len(k_sizes)))

        for i, model in enumerate(models):
            for j, k in enumerate(k_sizes):
                jsd_matrix[i, j] = results_dict[model]['kmer'][k]['jensen_shannon_divergence']

        fig, ax = plt.subplots(figsize=(10, max(4, len(models) * 0.8)))

        sns.heatmap(jsd_matrix, ax=ax, annot=True, fmt='.4f',
                    xticklabels=[k.replace('-mer', '') for k in k_sizes],
                    yticklabels=models, cmap='Reds', vmin=0,
                    cbar_kws={'label': 'Jensen-Shannon Divergence'})

        ax.set_xlabel('K-mer Size')
        ax.set_ylabel('Model')
        ax.set_title('K-mer Distribution Similarity')

        plt.tight_layout()
        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def plot_discriminator_accuracy(self,
                                    results_dict: Dict[str, Dict],
                                    output_name: str = "discriminator_accuracy") -> str:
        """Plot discriminator balanced accuracy with confidence intervals."""
        models = list(results_dict.keys())

        accs = []
        stds = []
        for m in models:
            if 'discriminator' in results_dict[m]:
                accs.append(results_dict[m]['discriminator']['balanced_accuracy'])
                stds.append(results_dict[m]['discriminator']['balanced_accuracy_std'])
            else:
                accs.append(0.5)
                stds.append(0)

        fig, ax = plt.subplots(figsize=(8, 6))

        colors = [self._get_color(m, i) for i, m in enumerate(models)]

        # Plot bars with error bars (95% CI)
        bars = ax.bar(models, accs, yerr=[1.96 * s for s in stds],
                      capsize=5, color=colors, edgecolor='black', linewidth=1.5)

        # Add baseline at 0.5
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, label='Random chance (0.5)')

        # Add interpretation zones
        ax.axhspan(0.5, 0.55, alpha=0.1, color='green', label='Indistinguishable')
        ax.axhspan(0.55, 0.70, alpha=0.1, color='yellow')
        ax.axhspan(0.70, 1.0, alpha=0.1, color='red')

        ax.set_ylabel('Balanced Accuracy')
        ax.set_title('Discriminator Accuracy\n(Can a classifier tell synthetic from real?)')
        ax.set_ylim(0.4, 1.0)
        ax.legend(loc='upper right')

        # Add value labels
        for bar, acc, std in zip(bars, accs, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.96 * std + 0.02,
                    f'{acc:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def plot_discriminator_auc(self,
                               results_dict: Dict[str, Dict],
                               output_name: str = "discriminator_auc") -> str:
        """Plot discriminator ROC-AUC with confidence intervals."""
        models = list(results_dict.keys())

        aucs = []
        stds = []
        for m in models:
            if 'discriminator' in results_dict[m]:
                aucs.append(results_dict[m]['discriminator']['roc_auc'])
                stds.append(results_dict[m]['discriminator']['roc_auc_std'])
            else:
                aucs.append(0.5)
                stds.append(0)

        fig, ax = plt.subplots(figsize=(8, 8))


        bars = ax.bar(models, aucs, yerr=[1.96 * s for s in stds],
                      capsize=5, color="gray", edgecolor='black', linewidth=1.5)

        # Add baseline at 0.5
        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Random chance (0.5)')

        ax.set_ylabel('ROC-AUC')
        ax.set_title('Discriminator ROC-AUC')
        ax.set_ylim(0.4, 1.05)
        ax.legend(loc='upper right')

        # Add value labels
        for bar, auc, std in zip(bars, aucs, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.96 * std + 0.02,
                    f'{auc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def plot_top_discriminating_kmers(self,
                                      results_dict: Dict[str, Dict],
                                      output_name: str = "top_discriminating_kmers") -> str:
        """Plot top discriminating k-mers for each model."""
        n_models = len(results_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))

        if n_models == 1:
            axes = [axes]

        for ax, (name, results) in zip(axes, results_dict.items()):
            if 'discriminator' not in results:
                continue

            top_kmers = results['discriminator']['top_discriminating_kmers'][:10]
            kmers = [k[0] for k in top_kmers]
            importances = [k[1] for k in top_kmers]

            color = self._get_color(name, list(results_dict.keys()).index(name))

            ax.barh(range(len(kmers)), importances, color=color, edgecolor='black')
            ax.set_yticks(range(len(kmers)))
            ax.set_yticklabels(kmers, fontfamily='monospace')
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'{name}\nTop Discriminating 4-mers')
            ax.invert_yaxis()

        plt.tight_layout()
        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def plot_summary_dashboard(self,
                               results_dict: Dict[str, Dict],
                               reference_gc_mean: float,
                               reference_gc_se: float,
                               output_name: str = "metrics_dashboard") -> str:
        """Create comprehensive dashboard with key metrics."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        models = list(results_dict.keys())
        colors = [self._get_color(m, i) for i, m in enumerate(models)]

        # 1. GC Content Bar Plot (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        all_models = ['Reference'] + models
        gc_means = [reference_gc_mean] + [results_dict[m]['gc_content']['query_mean'] for m in models]
        gc_ses = [reference_gc_se] + [results_dict[m]['gc_content']['query_se'] for m in models]
        all_colors = [self._get_color('reference')] + colors

        ax1.bar(all_models, gc_means, yerr=gc_ses, capsize=5, color=all_colors,
                edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('GC Content')
        ax1.set_title('A. GC Content')
        ax1.tick_params(axis='x', rotation=45)

        y_min = min(gc_means) - max(gc_ses) * 3
        y_max = max(gc_means) + max(gc_ses) * 3
        ax1.set_ylim(max(0, y_min), min(1, y_max))

        # 2. K-mer JSD Heatmap (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        k_sizes = sorted(results_dict[models[0]]['kmer'].keys())
        jsd_matrix = np.zeros((len(models), len(k_sizes)))
        for i, model in enumerate(models):
            for j, k in enumerate(k_sizes):
                jsd_matrix[i, j] = results_dict[model]['kmer'][k]['jensen_shannon_divergence']

        sns.heatmap(jsd_matrix, ax=ax2, annot=True, fmt='.4f',
                    xticklabels=[k.replace('-mer', '') for k in k_sizes],
                    yticklabels=models, cmap='Reds', vmin=0)
        ax2.set_xlabel('K-mer Size')
        ax2.set_title('B. K-mer Jensen-Shannon Divergence')

        # 3. Discriminator Accuracy (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])
        accs = [results_dict[m]['discriminator']['balanced_accuracy'] for m in models]
        stds = [results_dict[m]['discriminator']['balanced_accuracy_std'] for m in models]

        bars = ax3.bar(models, accs, yerr=[1.96 * s for s in stds],
                       capsize=5, color=colors, edgecolor='black', linewidth=1.5)
        ax3.axhline(y=0.5, color='gray', linestyle='--', linewidth=2)
        ax3.set_ylabel('Balanced Accuracy')
        ax3.set_title('C. Discriminator Accuracy')
        ax3.set_ylim(0.4, 1.0)
        ax3.tick_params(axis='x', rotation=45)

        for bar, acc in zip(bars, accs):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                     f'{acc:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 4. Summary Table (bottom right)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')

        headers = ['Model', 'GC Diff', '5-mer JSD', 'Disc. Acc.', 'Interpretation']
        table_data = []
        for name, results in results_dict.items():
            disc_acc = results['discriminator']['balanced_accuracy']
            interp = results['discriminator']['interpretation'].split(' ')[0]  # First word

            row = [
                name,
                f"{results['gc_content']['mean_difference']:.4f}",
                f"{results['kmer']['5-mer']['jensen_shannon_divergence']:.4f}",
                f"{disc_acc:.1%}",
                interp,
            ]
            table_data.append(row)

        table = ax4.table(cellText=table_data, colLabels=headers,
                          loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        ax4.set_title('D. Summary', fontsize=14, pad=20)

        plt.suptitle('DNA Sequence Generation Evaluation', fontsize=18, y=0.98)

        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)


def load_sequences(filepath: str, return_celltypes: bool = False) -> Union[List[str], Tuple[List[str], List[str]]]:
    """Load sequences from various file formats."""
    filepath = Path(filepath)
    celltypes = None

    if filepath.suffix == '.csv':
        df = pd.read_csv(filepath)
        sequences = None
        for col in ['sequence', 'seq', 'Sequence', 'SEQ']:
            if col in df.columns:
                sequences = df[col].tolist()
                break
        if sequences is None:
            sequences = df.iloc[:, 0].tolist()

        if return_celltypes:
            for col in ['TAG', 'tag', 'celltype', 'cell_type', 'CellType']:
                if col in df.columns:
                    celltypes = df[col].tolist()
                    break

    elif filepath.suffix == '.tsv':
        df = pd.read_csv(filepath, sep='\t')
        sequences = None
        for col in ['sequence', 'seq', 'Sequence', 'SEQ']:
            if col in df.columns:
                sequences = df[col].tolist()
                break
        if sequences is None:
            sequences = df.iloc[:, 0].tolist()

        if return_celltypes:
            for col in ['TAG', 'tag', 'celltype', 'cell_type', 'CellType']:
                if col in df.columns:
                    celltypes = df[col].tolist()
                    break

    elif filepath.suffix == '.txt':
        with open(filepath, 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]

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

    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    if return_celltypes:
        return sequences, celltypes
    return sequences


def main():
    parser = argparse.ArgumentParser(
        description='DNA Sequence Metrics Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 01_sequence_metrics_evaluation.py \\
      --reference data/ground_truth.csv \\
      --synthetic original:data/synthetic.csv

  python 01_sequence_metrics_evaluation.py \\
      --reference data/gt.csv \\
      --synthetic model1:data/synth1.csv model2:data/synth2.csv \\
      --celltype-fidelity
        """
    )

    parser.add_argument('--reference', '-r', required=True,
                        help='Path to reference/ground truth sequences')
    parser.add_argument('--synthetic', '-s', nargs='+', required=True,
                        help='Synthetic datasets as name:path pairs')
    parser.add_argument('--output', '-o', default='evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--kmer-sizes', nargs='+', type=int, default=[3, 5, 7, 9],
                        help='K-mer sizes to analyze')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--celltype-fidelity', action='store_true',
                        help='Compute cell-type conditional fidelity (requires TAG column)')

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = EvaluationConfig(
        kmer_sizes=args.kmer_sizes,
        random_seed=args.seed
    )

    evaluator = DNASequenceMetrics(config)
    visualizer = MetricsVisualizer(output_dir=str(output_dir / 'figures'))

    # Load reference sequences
    print(f"Loading reference sequences from {args.reference}...")
    if args.celltype_fidelity:
        reference_seqs, reference_celltypes = load_sequences(args.reference, return_celltypes=True)
        if reference_celltypes is None:
            print("WARNING: No cell type column found. Disabling cell-type fidelity.")
            args.celltype_fidelity = False
    else:
        reference_seqs = load_sequences(args.reference)
        reference_celltypes = None
    print(f"Loaded {len(reference_seqs)} reference sequences")

    # Compute reference GC stats
    ref_gc = np.array([evaluator.compute_gc_content(s) for s in reference_seqs])
    ref_gc_mean = float(np.mean(ref_gc))
    ref_gc_se = float(sem(ref_gc))

    # Parse synthetic datasets
    synthetic_datasets = {}
    synthetic_celltypes = {}
    for spec in args.synthetic:
        if ':' in spec:
            name, path = spec.split(':', 1)
        else:
            name = Path(spec).stem
            path = spec

        print(f"\nLoading synthetic dataset '{name}' from {path}...")
        if args.celltype_fidelity:
            seqs, celltypes = load_sequences(path, return_celltypes=True)
            synthetic_datasets[name] = seqs
            synthetic_celltypes[name] = celltypes
        else:
            synthetic_datasets[name] = load_sequences(path)
        print(f"Loaded {len(synthetic_datasets[name])} sequences")

    # Run evaluation
    all_results = {}
    for name, synth_seqs in synthetic_datasets.items():
        ref_ct = reference_celltypes if args.celltype_fidelity else None
        syn_ct = synthetic_celltypes.get(name) if args.celltype_fidelity else None

        results = evaluator.evaluate(
            reference_seqs,
            synth_seqs,
            query_name=name,
            reference_celltypes=ref_ct,
            query_celltypes=syn_ct,
        )
        all_results[name] = results

    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)

    figures = {
        'gc_content': visualizer.plot_gc_content_barplot(all_results, ref_gc_mean, ref_gc_se),
        'kmer_jsd': visualizer.plot_kmer_jsd_heatmap(all_results),
        'discriminator_accuracy': visualizer.plot_discriminator_accuracy(all_results),
        'discriminator_auc': visualizer.plot_discriminator_auc(all_results),
        'top_kmers': visualizer.plot_top_discriminating_kmers(all_results),
        'dashboard': visualizer.plot_summary_dashboard(all_results, ref_gc_mean, ref_gc_se),
    }

    # Save results to JSON
    results_file = output_dir / 'evaluation_results.json'
    json_results = {}
    for name, results in all_results.items():
        json_results[name] = results.copy()
        # Convert tuples to lists for JSON
        if 'discriminator' in json_results[name]:
            disc = json_results[name]['discriminator']
            if 'balanced_accuracy_ci95' in disc:
                disc['balanced_accuracy_ci95'] = list(disc['balanced_accuracy_ci95'])
            if 'roc_auc_ci95' in disc:
                disc['roc_auc_ci95'] = list(disc['roc_auc_ci95'])

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Create summary CSV
    summary_rows = []
    for name, results in all_results.items():
        row = {
            'model': name,
            'n_sequences': results['n_query'],
            'gc_mean': results['gc_content']['query_mean'],
            'gc_se': results['gc_content']['query_se'],
            'gc_difference': results['gc_content']['mean_difference'],
            'gc_pvalue': results['gc_content']['ttest_pvalue'],
        }

        # K-mer metrics
        for k in args.kmer_sizes:
            key = f'{k}-mer'
            if key in results['kmer']:
                row[f'{k}mer_jsd'] = results['kmer'][key]['jensen_shannon_divergence']

        # Discriminator
        row['discriminator_balanced_acc'] = results['discriminator']['balanced_accuracy']
        row['discriminator_balanced_acc_std'] = results['discriminator']['balanced_accuracy_std']
        row['discriminator_auc'] = results['discriminator']['roc_auc']
        row['discriminator_interpretation'] = results['discriminator']['interpretation']

        # Cell-type fidelity
        if 'celltype_fidelity' in results:
            row['celltype_gt_accuracy'] = results['celltype_fidelity']['gt_accuracy']
            row['celltype_synth_accuracy'] = results['celltype_fidelity']['synthetic_accuracy']
            row['celltype_fidelity_gap'] = results['celltype_fidelity']['fidelity_gap']

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_file = output_dir / 'evaluation_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to {summary_file}")

    # Print summary
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
