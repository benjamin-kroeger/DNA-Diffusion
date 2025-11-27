#!/usr/bin/env python3
"""
Script 4: Sequence Classification and UMAP Validation

Classifies sequences based on the distinguishing features discovered in Script 3,
then maps classifications onto UMAP to validate findings:
- Rule-based classification using complexity thresholds
- Random Forest/Gradient Boosting classifier training
- Classification overlay on UMAP
- Validation of mode collapse hypothesis

Key output: Confirms that synthetic data fails to generate low-complexity regions
by showing clear separation in UMAP space.

Author: DNA Diffusion Analysis Pipeline
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
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
class ClassificationCriteria:
    """
    Criteria for classifying sequences into complexity categories.
    These thresholds should be tuned based on the cluster analysis results.
    """
    # Low complexity criteria
    low_complexity_threshold: float = 0.3  # linguistic complexity k=3
    low_entropy_threshold: float = 1.5  # Shannon entropy
    high_lc_fraction_threshold: float = 0.4  # low_complexity_fraction

    # GC content extremes
    gc_low_threshold: float = 0.35
    gc_high_threshold: float = 0.65

    # Repetitive/homopolymer criteria
    homopolymer_threshold: int = 3  # number of homopolymer runs

    # Category names
    categories: Dict = None

    def __post_init__(self):
        if self.categories is None:
            self.categories = {
                0: 'High Complexity',
                1: 'Low Complexity',
                2: 'Extreme GC',
                3: 'Repetitive',
            }


class SequenceClassifier:
    """
    Classify DNA sequences based on complexity and structural features.
    """

    def __init__(self, criteria: Optional[ClassificationCriteria] = None):
        self.criteria = criteria or ClassificationCriteria()
        self.ml_classifier = None
        self.scaler = StandardScaler()

    # ==================== Feature Extraction ====================

    @staticmethod
    def compute_gc_content(seq: str) -> float:
        """Compute GC content."""
        seq = seq.upper()
        gc = seq.count('G') + seq.count('C')
        return gc / len(seq) if len(seq) > 0 else 0.0

    @staticmethod
    def linguistic_complexity(seq: str, k: int = 3) -> float:
        """Compute linguistic complexity."""
        seq = seq.upper()
        if len(seq) < k:
            return 0.0

        observed_kmers = set()
        for i in range(len(seq) - k + 1):
            observed_kmers.add(seq[i:i + k])

        max_possible = min(4 ** k, len(seq) - k + 1)
        return len(observed_kmers) / max_possible if max_possible > 0 else 0.0

    @staticmethod
    def shannon_entropy(seq: str) -> float:
        """Compute Shannon entropy."""
        from collections import Counter
        from scipy.stats import entropy

        seq = seq.upper()
        counts = Counter(seq)
        total = sum(counts.values())
        probs = np.array([counts.get(n, 0) / total for n in 'ACGT'])
        return entropy(probs, base=2)

    @staticmethod
    def low_complexity_fraction(seq: str, window_size: int = 20, threshold: float = 0.5) -> float:
        """Compute fraction of low complexity regions."""
        seq = seq.upper()
        if len(seq) < window_size:
            return 0.0

        low_complexity_positions = 0

        for i in range(len(seq) - window_size + 1):
            window = seq[i:i + window_size]
            # Simple k=2 linguistic complexity for speed
            kmers = set(window[j:j + 2] for j in range(len(window) - 1))
            lc = len(kmers) / min(16, len(window) - 1)
            if lc < threshold:
                low_complexity_positions += 1

        return low_complexity_positions / (len(seq) - window_size + 1)

    @staticmethod
    def count_homopolymers(seq: str, min_length: int = 4) -> int:
        """Count homopolymer runs."""
        seq = seq.upper()
        total = 0

        for nuc in 'ACGT':
            pattern = nuc * min_length
            i = 0
            while i < len(seq):
                if seq[i:].startswith(pattern):
                    total += 1
                    while i < len(seq) and seq[i] == nuc:
                        i += 1
                else:
                    i += 1

        return total

    def extract_features(self, seq: str) -> Dict:
        """Extract all features for a sequence."""
        return {
            'gc_content': self.compute_gc_content(seq),
            'linguistic_complexity_k3': self.linguistic_complexity(seq, k=3),
            'shannon_entropy': self.shannon_entropy(seq),
            'low_complexity_fraction': self.low_complexity_fraction(seq),
            'homopolymer_count': self.count_homopolymers(seq),
        }

    def extract_features_batch(self, sequences: List[str], show_progress: bool = True) -> pd.DataFrame:
        """Extract features for multiple sequences."""
        features = []

        iterator = tqdm(sequences, desc="Extracting features") if show_progress else sequences
        for seq in iterator:
            features.append(self.extract_features(seq))

        return pd.DataFrame(features)

    # ==================== Rule-Based Classification ====================

    def classify_rule_based(self, features: pd.DataFrame) -> np.ndarray:
        """
        Classify sequences using rule-based criteria.

        Categories:
        0: High Complexity (normal)
        1: Low Complexity
        2: Extreme GC
        3: Repetitive
        """
        n = len(features)
        labels = np.zeros(n, dtype=int)

        # Low complexity
        low_complexity_mask = (
            (features['linguistic_complexity_k3'] < self.criteria.low_complexity_threshold) |
            (features['shannon_entropy'] < self.criteria.low_entropy_threshold) |
            (features['low_complexity_fraction'] > self.criteria.high_lc_fraction_threshold)
        )
        labels[low_complexity_mask] = 1

        # Extreme GC (only if not already low complexity)
        extreme_gc_mask = (
            (labels == 0) &
            ((features['gc_content'] < self.criteria.gc_low_threshold) |
             (features['gc_content'] > self.criteria.gc_high_threshold))
        )
        labels[extreme_gc_mask] = 2

        # Repetitive (only if not already classified)
        repetitive_mask = (
            (labels == 0) &
            (features['homopolymer_count'] >= self.criteria.homopolymer_threshold)
        )
        labels[repetitive_mask] = 3

        return labels

    # ==================== ML-Based Classification ====================

    def train_classifier(self,
                         features: pd.DataFrame,
                         labels: np.ndarray,
                         classifier_type: str = 'rf') -> Dict:
        """
        Train ML classifier on labeled data.

        Args:
            features: Feature dataframe
            labels: True labels (e.g., cluster assignments)
            classifier_type: 'rf' for Random Forest, 'gb' for Gradient Boosting

        Returns:
            Training results including cross-validation scores
        """
        # Prepare features
        feature_cols = ['gc_content', 'linguistic_complexity_k3', 'shannon_entropy',
                        'low_complexity_fraction', 'homopolymer_count']
        X = features[feature_cols].values
        y = labels

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # Initialize classifier
        if classifier_type == 'rf':
            self.ml_classifier = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
        else:
            self.ml_classifier = GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42
            )

        # Train
        self.ml_classifier.fit(X_train, y_train)

        # Evaluate
        y_pred = self.ml_classifier.predict(X_test)

        # Cross-validation
        cv_scores = cross_val_score(self.ml_classifier, X_scaled, y, cv=5)

        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_importances': dict(zip(feature_cols,
                                            self.ml_classifier.feature_importances_.tolist())),
        }

        return results

    def predict_ml(self, features: pd.DataFrame) -> np.ndarray:
        """Predict using trained ML classifier."""
        if self.ml_classifier is None:
            raise ValueError("ML classifier not trained. Call train_classifier first.")

        feature_cols = ['gc_content', 'linguistic_complexity_k3', 'shannon_entropy',
                        'low_complexity_fraction', 'homopolymer_count']
        X = features[feature_cols].values
        X_scaled = self.scaler.transform(X)

        return self.ml_classifier.predict(X_scaled)


class UMAPValidator:
    """
    Validate classifications by overlaying on UMAP coordinates.
    """

    def __init__(self, output_dir: str = 'validation_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color schemes
        self.category_colors = {
            0: '#2ecc71',  # High Complexity - Green
            1: '#e74c3c',  # Low Complexity - Red
            2: '#3498db',  # Extreme GC - Blue
            3: '#9b59b6',  # Repetitive - Purple
        }

        self.category_names = {
            0: 'High Complexity',
            1: 'Low Complexity',
            2: 'Extreme GC',
            3: 'Repetitive',
        }

    def plot_classification_overlay(self,
                                    umap_coords: np.ndarray,
                                    classifications: np.ndarray,
                                    source_labels: np.ndarray,
                                    output_name: str = 'classification_overlay') -> str:
        """
        Create 4-panel figure showing classification overlay on UMAP.
        """
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))

        # Panel A: All classifications
        ax = axes[0, 0]
        for cat in sorted(np.unique(classifications)):
            mask = classifications == cat
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       s=8, alpha=0.5, c=self.category_colors.get(cat, '#888888'),
                       label=f'{self.category_names.get(cat, f"Class {cat}")} (n={np.sum(mask)})',
                       rasterized=True)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('A. Classification Overlay')
        ax.legend(markerscale=3)
        ax.grid(True, alpha=0.3)

        # Panel B: Reference only
        ax = axes[0, 1]
        ref_mask = source_labels == 'reference'
        for cat in sorted(np.unique(classifications)):
            mask = (classifications == cat) & ref_mask
            if np.sum(mask) > 0:
                ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                           s=8, alpha=0.5, c=self.category_colors.get(cat, '#888888'),
                           label=f'{self.category_names.get(cat, f"Class {cat}")} (n={np.sum(mask)})',
                           rasterized=True)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('B. Ground Truth Only')
        ax.legend(markerscale=3)
        ax.grid(True, alpha=0.3)

        # Panel C: Synthetic only
        ax = axes[1, 0]
        synth_mask = source_labels == 'synthetic'
        for cat in sorted(np.unique(classifications)):
            mask = (classifications == cat) & synth_mask
            if np.sum(mask) > 0:
                ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                           s=8, alpha=0.5, c=self.category_colors.get(cat, '#888888'),
                           label=f'{self.category_names.get(cat, f"Class {cat}")} (n={np.sum(mask)})',
                           rasterized=True)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('C. Synthetic Only')
        ax.legend(markerscale=3)
        ax.grid(True, alpha=0.3)

        # Panel D: Category distribution comparison
        ax = axes[1, 1]
        categories = sorted(np.unique(classifications))

        ref_counts = [np.sum((classifications == c) & ref_mask) for c in categories]
        synth_counts = [np.sum((classifications == c) & synth_mask) for c in categories]

        # Normalize to percentages
        ref_pct = np.array(ref_counts) / np.sum(ref_counts) * 100
        synth_pct = np.array(synth_counts) / np.sum(synth_counts) * 100 if np.sum(synth_counts) > 0 else np.zeros(len(categories))

        x = np.arange(len(categories))
        width = 0.35

        ax.bar(x - width / 2, ref_pct, width, label='Ground Truth', color='#2ecc71')
        ax.bar(x + width / 2, synth_pct, width, label='Synthetic', color='#e74c3c')

        ax.set_xlabel('Category')
        ax.set_ylabel('Percentage')
        ax.set_title('D. Category Distribution Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([self.category_names.get(c, f'Class {c}') for c in categories], rotation=45)
        ax.legend()

        plt.suptitle('Sequence Classification Validation on UMAP', fontsize=18, y=0.98)
        plt.tight_layout()

        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def plot_mode_collapse_evidence(self,
                                    umap_coords: np.ndarray,
                                    classifications: np.ndarray,
                                    source_labels: np.ndarray,
                                    output_name: str = 'mode_collapse_evidence') -> str:
        """
        Create visualization specifically highlighting mode collapse.
        """
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))

        ref_mask = source_labels == 'reference'
        synth_mask = source_labels == 'synthetic'

        # Low complexity regions
        low_complex_mask = classifications == 1

        # Panel 1: Reference density with low complexity highlighted
        ax = axes[0]
        ax.scatter(umap_coords[ref_mask & ~low_complex_mask, 0],
                   umap_coords[ref_mask & ~low_complex_mask, 1],
                   s=5, alpha=0.3, c='gray', label='Other GT', rasterized=True)
        ax.scatter(umap_coords[ref_mask & low_complex_mask, 0],
                   umap_coords[ref_mask & low_complex_mask, 1],
                   s=15, alpha=0.8, c='#e74c3c', label='Low Complexity GT', rasterized=True)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('Ground Truth: Low Complexity Regions')
        ax.legend(markerscale=2)
        ax.grid(True, alpha=0.3)

        # Panel 2: Synthetic coverage
        ax = axes[1]
        ax.scatter(umap_coords[ref_mask, 0], umap_coords[ref_mask, 1],
                   s=5, alpha=0.15, c='gray', label='GT (background)', rasterized=True)
        ax.scatter(umap_coords[synth_mask, 0], umap_coords[synth_mask, 1],
                   s=8, alpha=0.5, c='#3498db', label='Synthetic', rasterized=True)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('Synthetic Data Coverage')
        ax.legend(markerscale=2)
        ax.grid(True, alpha=0.3)

        # Panel 3: Gap analysis - where is synthetic missing?
        ax = axes[2]
        # Show GT low complexity in red
        ax.scatter(umap_coords[ref_mask & low_complex_mask, 0],
                   umap_coords[ref_mask & low_complex_mask, 1],
                   s=15, alpha=0.8, c='#e74c3c', label='GT Low Complexity', rasterized=True)
        # Show synthetic low complexity in blue (if any)
        ax.scatter(umap_coords[synth_mask & low_complex_mask, 0],
                   umap_coords[synth_mask & low_complex_mask, 1],
                   s=15, alpha=0.8, c='#3498db', label='Synth Low Complexity', rasterized=True)

        # Add statistics
        n_gt_low = np.sum(ref_mask & low_complex_mask)
        n_synth_low = np.sum(synth_mask & low_complex_mask)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(f'Mode Collapse: Low Complexity Gap\nGT: {n_gt_low}, Synth: {n_synth_low}')
        ax.legend(markerscale=2)
        ax.grid(True, alpha=0.3)

        plt.suptitle('Evidence of Mode Collapse in Synthetic Generation', fontsize=16)
        plt.tight_layout()

        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def plot_feature_importance_validation(self,
                                           features_df: pd.DataFrame,
                                           classifications: np.ndarray,
                                           output_name: str = 'feature_validation') -> str:
        """
        Plot feature distributions by classification to validate separation.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        feature_cols = ['gc_content', 'linguistic_complexity_k3', 'shannon_entropy',
                        'low_complexity_fraction', 'homopolymer_count']

        unique_classes = sorted(np.unique(classifications))
        colors = [self.category_colors.get(c, '#888888') for c in unique_classes]

        for i, col in enumerate(feature_cols):
            ax = axes[i]

            data = [features_df.loc[classifications == c, col].values
                    for c in unique_classes]

            bp = ax.boxplot(data, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_xticklabels([self.category_names.get(c, f'Class {c}')
                                for c in unique_classes], rotation=45)
            ax.set_ylabel(col.replace('_', ' ').title())
            ax.set_title(f'{col.replace("_", " ").title()} by Category')

        # Hide last subplot if unused
        axes[-1].axis('off')

        plt.suptitle('Feature Distributions by Classification Category', fontsize=16)
        plt.tight_layout()

        filepath = self.output_dir / f"{output_name}.png"
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def create_summary_statistics(self,
                                  classifications: np.ndarray,
                                  source_labels: np.ndarray,
                                  features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create summary statistics table.
        """
        ref_mask = source_labels == 'reference'
        synth_mask = source_labels == 'synthetic'

        rows = []
        for cat in sorted(np.unique(classifications)):
            cat_mask = classifications == cat

            row = {
                'Category': self.category_names.get(cat, f'Class {cat}'),
                'GT Count': int(np.sum(cat_mask & ref_mask)),
                'GT %': float(np.sum(cat_mask & ref_mask) / np.sum(ref_mask) * 100),
                'Synth Count': int(np.sum(cat_mask & synth_mask)),
                'Synth %': float(np.sum(cat_mask & synth_mask) / np.sum(synth_mask) * 100) if np.sum(synth_mask) > 0 else 0,
            }

            # Add mean features for this category
            for col in ['gc_content', 'linguistic_complexity_k3', 'shannon_entropy']:
                row[f'{col}_mean'] = float(features_df.loc[cat_mask, col].mean())

            rows.append(row)

        return pd.DataFrame(rows)


def load_data_for_validation(cluster_assignments_file: str,
                             umap_coords_file: Optional[str] = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load cluster assignments and UMAP coordinates.
    """
    # Load cluster assignments (which should have UMAP coords)
    df = pd.read_csv(cluster_assignments_file)

    # Check for UMAP columns
    if 'umap_1' in df.columns and 'umap_2' in df.columns:
        umap_coords = df[['umap_1', 'umap_2']].values
    elif umap_coords_file is not None:
        umap_coords = np.load(umap_coords_file)
    else:
        raise ValueError("UMAP coordinates not found. Provide umap_coords_file or ensure cluster_assignments has umap_1, umap_2 columns")

    return df, umap_coords


def main():
    parser = argparse.ArgumentParser(
        description='Sequence Classification and UMAP Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with cluster assignments from Script 2
  python 04_classification_validation.py \\
      --clusters umap_results/cluster_assignments.csv \\
      --output validation_results/

  # With custom complexity thresholds
  python 04_classification_validation.py \\
      --clusters clusters.csv \\
      --low-complexity-threshold 0.25 \\
      --gc-low 0.30 --gc-high 0.70

  # Train ML classifier using cluster labels
  python 04_classification_validation.py \\
      --clusters clusters.csv \\
      --train-ml
        """
    )

    parser.add_argument('--clusters', '-c', required=True,
                        help='Path to cluster assignments CSV (from Script 2)')
    parser.add_argument('--umap-coords', default=None,
                        help='Path to UMAP coordinates .npy file (if not in clusters CSV)')
    parser.add_argument('--sequences', '-s', default=None,
                        help='Path to sequences file (optional, for feature extraction)')
    parser.add_argument('--features', '-f', default=None,
                        help='Path to precomputed features CSV (from Script 3)')
    parser.add_argument('--output', '-o', default='validation_results',
                        help='Output directory')

    # Classification thresholds
    parser.add_argument('--low-complexity-threshold', type=float, default=0.3,
                        help='Linguistic complexity threshold for low complexity')
    parser.add_argument('--gc-low', type=float, default=0.35,
                        help='Low GC content threshold')
    parser.add_argument('--gc-high', type=float, default=0.65,
                        help='High GC content threshold')

    parser.add_argument('--train-ml', action='store_true',
                        help='Train ML classifier using cluster labels')

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    df, umap_coords = load_data_for_validation(args.clusters, args.umap_coords)

    source_labels = df['source'].values if 'source' in df.columns else np.array(['unknown'] * len(df))

    # Load or compute features
    if args.features:
        print(f"Loading precomputed features from {args.features}")
        features_df = pd.read_csv(args.features)
    elif args.sequences:
        print("Extracting features from sequences...")
        # Load sequences
        seq_path = Path(args.sequences)
        if seq_path.suffix == '.txt':
            with open(args.sequences, 'r') as f:
                sequences = [line.strip() for line in f if line.strip()]
        else:
            seq_df = pd.read_csv(args.sequences, sep='\t' if seq_path.suffix == '.tsv' else ',')
            for col in ['sequence', 'seq', 'Sequence']:
                if col in seq_df.columns:
                    sequences = seq_df[col].tolist()
                    break

        # Extract features
        classifier = SequenceClassifier()
        features_df = classifier.extract_features_batch(sequences)
    else:
        # Try to find features in the same directory as clusters
        features_path = Path(args.clusters).parent / 'sequence_features.csv'
        if features_path.exists():
            print(f"Loading features from {features_path}")
            features_df = pd.read_csv(features_path)
        else:
            raise ValueError("No features found. Provide --sequences or --features")

    # Initialize classifier with criteria
    criteria = ClassificationCriteria(
        low_complexity_threshold=args.low_complexity_threshold,
        gc_low_threshold=args.gc_low,
        gc_high_threshold=args.gc_high,
    )
    classifier = SequenceClassifier(criteria=criteria)

    # Run rule-based classification
    print("\nRunning rule-based classification...")
    classifications = classifier.classify_rule_based(features_df)

    # Print classification summary
    print("\nClassification Summary:")
    for cat in sorted(np.unique(classifications)):
        n = np.sum(classifications == cat)
        pct = n / len(classifications) * 100
        print(f"  {criteria.categories.get(cat, f'Class {cat}')}: {n} ({pct:.1f}%)")

    # Train ML classifier if requested
    ml_results = None
    if args.train_ml and 'cluster' in df.columns:
        print("\nTraining ML classifier...")
        cluster_labels = df['cluster'].values
        ml_results = classifier.train_classifier(features_df, cluster_labels)

        print(f"\nML Classifier Results:")
        print(f"  Accuracy: {ml_results['accuracy']:.3f}")
        print(f"  CV Mean: {ml_results['cv_mean']:.3f} ± {ml_results['cv_std']:.3f}")
        print("\nFeature Importances:")
        for feat, imp in sorted(ml_results['feature_importances'].items(),
                                key=lambda x: -x[1]):
            print(f"  {feat}: {imp:.3f}")

    # Initialize validator
    validator = UMAPValidator(output_dir=str(output_dir))

    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating validation visualizations...")
    print("=" * 60)

    figures = {
        'classification_overlay': validator.plot_classification_overlay(
            umap_coords, classifications, source_labels
        ),
        'mode_collapse_evidence': validator.plot_mode_collapse_evidence(
            umap_coords, classifications, source_labels
        ),
        'feature_validation': validator.plot_feature_importance_validation(
            features_df, classifications
        ),
    }

    # Create summary statistics
    summary_df = validator.create_summary_statistics(
        classifications, source_labels, features_df
    )

    # Save results
    # Add classifications to original dataframe
    df['complexity_class'] = classifications
    df['complexity_class_name'] = [criteria.categories.get(c, f'Class {c}')
                                   for c in classifications]

    results_file = output_dir / 'classification_results.csv'
    df.to_csv(results_file, index=False)

    summary_file = output_dir / 'classification_summary.csv'
    summary_df.to_csv(summary_file, index=False)

    # Save ML results if available
    if ml_results:
        ml_file = output_dir / 'ml_classifier_results.json'
        # Remove non-serializable parts
        ml_results_save = {k: v for k, v in ml_results.items()
                           if k not in ['classification_report']}
        with open(ml_file, 'w') as f:
            json.dump(ml_results_save, f, indent=2)

    # Print final summary
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated figures:")
    for name, path in figures.items():
        print(f"  - {name}: {path}")

    print(f"\nResults saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    ref_mask = source_labels == 'reference'
    synth_mask = source_labels == 'synthetic'

    for cat in sorted(np.unique(classifications)):
        cat_name = criteria.categories.get(cat, f'Class {cat}')
        gt_pct = np.sum((classifications == cat) & ref_mask) / np.sum(ref_mask) * 100
        synth_pct = np.sum((classifications == cat) & synth_mask) / np.sum(synth_mask) * 100 if np.sum(synth_mask) > 0 else 0

        if gt_pct > synth_pct * 2:
            print(f"\n⚠️  {cat_name}: Underrepresented in synthetic data")
            print(f"   GT: {gt_pct:.1f}% vs Synth: {synth_pct:.1f}%")
        elif synth_pct > gt_pct * 2:
            print(f"\n⚠️  {cat_name}: Overrepresented in synthetic data")
            print(f"   GT: {gt_pct:.1f}% vs Synth: {synth_pct:.1f}%")


if __name__ == "__main__":
    main()
