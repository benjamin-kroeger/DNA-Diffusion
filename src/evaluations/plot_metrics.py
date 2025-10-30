"""
DNA Evaluation Multi-Run Comparison Plotter

Visualizes and compares metrics across multiple evaluation runs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class DNAEvaluationPlotter:
    """
    Create comprehensive comparison plots for multiple DNA evaluation runs.

    Usage:
        plotter = DNAEvaluationPlotter()
        plotter.plot_comparison(
            dataframes=[df1, df2, df3],
            labels=['Model A', 'Model B', 'Model C']
        )
    """

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid', figsize: Tuple[int, int] = (16, 12)):
        """
        Initialize the plotter.

        Args:
            style: Matplotlib style (default: 'seaborn-v0_8-darkgrid')
            figsize: Default figure size for plots
        """
        self.style = style
        self.figsize = figsize

        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')

        # Color palette
        self.colors = sns.color_palette("husl", 10)

    def plot_comparison(self,
                        dataframes: List[pd.DataFrame],
                        labels: List[str],
                        output_dir: Optional[str] = None,
                        show_plots: bool = True,
                        include_ci: bool = True,
                        metrics_to_plot: Optional[List[str]] = None,
                        save_format: str = 'png',
                        dpi: int = 300) -> Dict[str, plt.Figure]:
        """
        Create comprehensive comparison plots across multiple runs.

        Args:
            dataframes: List of results DataFrames from different runs
            labels: List of labels for each run (e.g., model names)
            output_dir: Directory to save plots (optional)
            show_plots: Whether to display plots
            include_ci: Whether to include confidence intervals
            metrics_to_plot: Specific metrics to plot (None = all)
            save_format: Format for saved plots ('png', 'pdf', 'svg')
            dpi: Resolution for saved plots

        Returns:
            Dictionary mapping plot names to figure objects
        """
        if len(dataframes) != len(labels):
            raise ValueError("Number of dataframes must match number of labels")

        if len(dataframes) < 2:
            raise ValueError("Need at least 2 dataframes to compare")

        print(f"Creating comparison plots for {len(dataframes)} runs: {', '.join(labels)}")

        # Validate and align dataframes
        dataframes = self._validate_dataframes(dataframes)

        # Create output directory if specified
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        figures = {}

        # 1. Overview comparison (all metrics)
        print("\n1. Creating overview comparison plot...")
        fig = self._plot_overview_comparison(dataframes, labels, include_ci, metrics_to_plot)
        figures['overview'] = fig
        self._save_and_show(fig, 'overview_comparison', output_dir, show_plots, save_format, dpi)

        # 2. Category-wise comparison
        print("2. Creating category-wise comparison plots...")
        category_figs = self._plot_by_category(dataframes, labels, include_ci, metrics_to_plot)
        figures.update(category_figs)
        for cat_name, fig in category_figs.items():
            self._save_and_show(fig, f'category_{cat_name.lower().replace(" ", "_")}',
                                output_dir, show_plots, save_format, dpi)

        # 3. Radar/Spider plot for key metrics
        print("3. Creating radar plot...")
        fig = self._plot_radar_comparison(dataframes, labels)
        figures['radar'] = fig
        self._save_and_show(fig, 'radar_comparison', output_dir, show_plots, save_format, dpi)

        # 4. Trustworthiness heatmap
        if include_ci and 'trustworthiness' in dataframes[0].columns:
            print("4. Creating trustworthiness heatmap...")
            fig = self._plot_trustworthiness_heatmap(dataframes, labels)
            figures['trustworthiness'] = fig
            self._save_and_show(fig, 'trustworthiness_heatmap', output_dir, show_plots, save_format, dpi)

        # 5. Metric-specific detailed comparisons
        print("5. Creating detailed metric comparisons...")
        detail_figs = self._plot_detailed_metrics(dataframes, labels, include_ci, metrics_to_plot)
        figures.update(detail_figs)
        for metric_name, fig in detail_figs.items():
            self._save_and_show(fig, f'detail_{metric_name.lower().replace(" ", "_").replace("/", "_")}',
                                output_dir, show_plots, save_format, dpi)

        # 6. Statistical summary table
        print("6. Creating statistical summary...")
        fig = self._plot_summary_table(dataframes, labels)
        figures['summary_table'] = fig
        self._save_and_show(fig, 'summary_table', output_dir, show_plots, save_format, dpi)

        print(f"\n✓ Created {len(figures)} comparison plots")
        if output_dir:
            print(f"✓ Plots saved to: {output_dir}")

        return figures

    def _validate_dataframes(self, dataframes: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Validate that all dataframes have the same structure."""
        required_cols = ['category', 'metric', 'value']

        for i, df in enumerate(dataframes):
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"DataFrame {i} missing required columns: {required_cols}")

        # Check that all dataframes have the same metrics
        metrics_0 = set(zip(dataframes[0]['category'], dataframes[0]['metric']))
        for i, df in enumerate(dataframes[1:], 1):
            metrics_i = set(zip(df['category'], df['metric']))
            if metrics_0 != metrics_i:
                print(f"Warning: DataFrame {i} has different metrics than DataFrame 0")

        return dataframes

    def _plot_overview_comparison(self,
                                  dataframes: List[pd.DataFrame],
                                  labels: List[str],
                                  include_ci: bool,
                                  metrics_to_plot: Optional[List[str]]) -> plt.Figure:
        """Create overview comparison plot with all metrics."""
        # Prepare data
        df_combined = self._combine_dataframes(dataframes, labels)

        if metrics_to_plot:
            df_combined = df_combined[df_combined['metric'].isin(metrics_to_plot)]

        # Create figure
        n_metrics = len(df_combined['metric'].unique())
        height = max(12, n_metrics * 0.5)
        fig, ax = plt.subplots(figsize=(self.figsize[0], height))

        # Group by metric for plotting
        metrics = df_combined['metric'].unique()
        x_pos = np.arange(len(metrics))
        width = 0.8 / len(labels)

        for i, label in enumerate(labels):
            df_label = df_combined[df_combined['run'] == label]
            values = [df_label[df_label['metric'] == m]['value'].values[0] if len(df_label[df_label['metric'] == m]) > 0 else 0
                      for m in metrics]

            offset = (i - len(labels) / 2 + 0.5) * width
            bars = ax.barh(x_pos + offset, values, width, label=label, alpha=0.8, color=self.colors[i])

            # Add confidence intervals if available
            if include_ci and 'ci_lower' in df_label.columns:
                for j, m in enumerate(metrics):
                    metric_data = df_label[df_label['metric'] == m]
                    if len(metric_data) > 0 and not pd.isna(metric_data['ci_lower'].values[0]):
                        ci_lower = metric_data['ci_lower'].values[0]
                        ci_upper = metric_data['ci_upper'].values[0]
                        value = metric_data['value'].values[0]
                        ax.errorbar(value, x_pos[j] + offset,
                                    xerr=[[value - ci_lower], [ci_upper - value]],
                                    fmt='none', color='black', alpha=0.5, capsize=3, linewidth=1)

        ax.set_yticks(x_pos)
        ax.set_yticklabels(metrics, fontsize=9)
        ax.set_xlabel('Value', fontsize=12, fontweight='bold')
        ax.set_title('Overview: All Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        return fig

    def _plot_by_category(self,
                          dataframes: List[pd.DataFrame],
                          labels: List[str],
                          include_ci: bool,
                          metrics_to_plot: Optional[List[str]]) -> Dict[str, plt.Figure]:
        """Create separate plots for each category."""
        df_combined = self._combine_dataframes(dataframes, labels)

        if metrics_to_plot:
            df_combined = df_combined[df_combined['metric'].isin(metrics_to_plot)]

        figures = {}
        categories = df_combined['category'].unique()

        for category in categories:
            df_cat = df_combined[df_combined['category'] == category]

            # Create subplot
            n_metrics = len(df_cat['metric'].unique())
            height = max(6, n_metrics * 0.6)
            fig, ax = plt.subplots(figsize=(14, height))

            metrics = df_cat['metric'].unique()
            x_pos = np.arange(len(metrics))
            width = 0.8 / len(labels)

            for i, label in enumerate(labels):
                df_label = df_cat[df_cat['run'] == label]
                values = [df_label[df_label['metric'] == m]['value'].values[0] if len(df_label[df_label['metric'] == m]) > 0 else 0
                          for m in metrics]

                offset = (i - len(labels) / 2 + 0.5) * width
                bars = ax.barh(x_pos + offset, values, width, label=label, alpha=0.8, color=self.colors[i])

                # Add confidence intervals
                if include_ci and 'ci_lower' in df_label.columns:
                    for j, m in enumerate(metrics):
                        metric_data = df_label[df_label['metric'] == m]
                        if len(metric_data) > 0 and not pd.isna(metric_data['ci_lower'].values[0]):
                            ci_lower = metric_data['ci_lower'].values[0]
                            ci_upper = metric_data['ci_upper'].values[0]
                            value = metric_data['value'].values[0]
                            ax.errorbar(value, x_pos[j] + offset,
                                        xerr=[[value - ci_lower], [ci_upper - value]],
                                        fmt='none', color='black', alpha=0.5, capsize=3, linewidth=1)

            ax.set_yticks(x_pos)
            ax.set_yticklabels(metrics, fontsize=10)
            ax.set_xlabel('Value', fontsize=12, fontweight='bold')
            ax.set_title(f'{category} - Detailed Comparison', fontsize=14, fontweight='bold', pad=15)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3, axis='x')

            plt.tight_layout()
            figures[category] = fig

        return figures

    def _plot_radar_comparison(self, dataframes: List[pd.DataFrame], labels: List[str]) -> plt.Figure:
        """Create radar/spider plot for key metrics."""
        # Select key metrics for radar plot (normalized to 0-1 scale)
        key_metrics = {
            'GC Content': 'Wasserstein Distance',
            'K-mer Analysis': '3-mer JS Divergence',
            'Novelty': 'Min Edit Distance Mean',
            'Diversity': 'Unique Sequences Ratio',
            'Diversity': 'Distinct-2'
        }

        # Extract and normalize metrics
        df_combined = self._combine_dataframes(dataframes, labels)

        # Select metrics
        radar_metrics = []
        for cat, metric in key_metrics.items():
            mask = (df_combined['category'] == cat) & (df_combined['metric'] == metric)
            if mask.any():
                radar_metrics.append((cat, metric))

        if len(radar_metrics) < 3:
            # Not enough metrics for radar plot, use top metrics instead
            value_counts = df_combined.groupby(['category', 'metric']).size()
            radar_metrics = [(cat, metric) for cat, metric in value_counts.head(6).index]

        # Create figure
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='polar')

        # Number of variables
        num_vars = len(radar_metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        # Plot for each run
        for i, label in enumerate(labels):
            df_label = df_combined[df_combined['run'] == label]

            values = []
            for cat, metric in radar_metrics:
                mask = (df_label['category'] == cat) & (df_label['metric'] == metric)
                if mask.any():
                    value = df_label[mask]['value'].values[0]
                else:
                    value = 0
                values.append(value)

            # Normalize values to 0-1 scale per metric
            if i == 0:
                self.radar_max_values = {}
                for j, (cat, metric) in enumerate(radar_metrics):
                    all_values = []
                    for df in dataframes:
                        mask = (df['category'] == cat) & (df['metric'] == metric)
                        if mask.any():
                            all_values.append(df[mask]['value'].values[0])
                    self.radar_max_values[(cat, metric)] = max(all_values) if all_values else 1

            normalized_values = [values[j] / self.radar_max_values[radar_metrics[j]]
                                 for j in range(len(values))]
            normalized_values += normalized_values[:1]  # Complete the circle

            ax.plot(angles, normalized_values, 'o-', linewidth=2, label=label, color=self.colors[i])
            ax.fill(angles, normalized_values, alpha=0.15, color=self.colors[i])

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f"{cat}\n{metric}" for cat, metric in radar_metrics], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        ax.set_title('Key Metrics Radar Comparison (Normalized)',
                     fontsize=16, fontweight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_trustworthiness_heatmap(self, dataframes: List[pd.DataFrame], labels: List[str]) -> plt.Figure:
        """Create heatmap showing trustworthiness of metrics across runs."""
        # Prepare data
        trust_map = {'High': 4, 'Medium': 3, 'Low': 2, 'Very Low': 1, 'N/A': 0}

        metrics = dataframes[0]['metric'].unique()
        trust_matrix = np.zeros((len(metrics), len(labels)))

        for i, label in enumerate(labels):
            df = dataframes[i]
            for j, metric in enumerate(metrics):
                trust = df[df['metric'] == metric]['trustworthiness'].values
                if len(trust) > 0:
                    trust_matrix[j, i] = trust_map.get(trust[0], 0)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2), max(12, len(metrics) * 0.4)))

        im = ax.imshow(trust_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=4)

        # Set ticks
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(metrics)))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_yticklabels(metrics, fontsize=9)

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_ticks([0, 1, 2, 3, 4])
        cbar.set_ticklabels(['N/A', 'Very Low', 'Low', 'Medium', 'High'])
        cbar.ax.set_ylabel('Trustworthiness', rotation=-90, va="bottom", fontsize=12)

        # Add values to cells
        for i in range(len(metrics)):
            for j in range(len(labels)):
                value = trust_matrix[i, j]
                text_color = 'white' if value < 2 else 'black'
                ax.text(j, i, ['N/A', 'VL', 'L', 'M', 'H'][int(value)],
                        ha="center", va="center", color=text_color, fontsize=9, fontweight='bold')

        ax.set_title('Metric Trustworthiness Across Runs', fontsize=16, fontweight='bold', pad=20)
        fig.tight_layout()
        return fig

    def _plot_detailed_metrics(self,
                               dataframes: List[pd.DataFrame],
                               labels: List[str],
                               include_ci: bool,
                               metrics_to_plot: Optional[List[str]]) -> Dict[str, plt.Figure]:
        """Create detailed plots for important metrics with confidence intervals."""
        df_combined = self._combine_dataframes(dataframes, labels)

        # Select important metrics for detailed view
        important_metrics = [
            'Wasserstein Distance',
            '3-mer JS Divergence',
            'Min Edit Distance Mean',
            'Unique Sequences Ratio',
            'Self-BLEU Mean',
            'Distinct-2'
        ]

        if metrics_to_plot:
            important_metrics = [m for m in important_metrics if m in metrics_to_plot]

        figures = {}

        for metric in important_metrics:
            df_metric = df_combined[df_combined['metric'] == metric]

            if len(df_metric) == 0:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))

            x_pos = np.arange(len(labels))
            values = [df_metric[df_metric['run'] == label]['value'].values[0]
                      if len(df_metric[df_metric['run'] == label]) > 0 else 0
                      for label in labels]

            bars = ax.bar(x_pos, values, color=self.colors[:len(labels)], alpha=0.8, edgecolor='black')

            # Add confidence intervals
            if include_ci and 'ci_lower' in df_metric.columns:
                errors = []
                for label in labels:
                    metric_data = df_metric[df_metric['run'] == label]
                    if len(metric_data) > 0 and not pd.isna(metric_data['ci_lower'].values[0]):
                        ci_lower = metric_data['ci_lower'].values[0]
                        ci_upper = metric_data['ci_upper'].values[0]
                        value = metric_data['value'].values[0]
                        errors.append([[value - ci_lower], [ci_upper - value]])
                    else:
                        errors.append([[0], [0]])

                errors = np.array(errors).squeeze()
                ax.errorbar(x_pos, values, yerr=errors.T, fmt='none',
                            color='black', capsize=5, linewidth=2, alpha=0.7)

            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{value:.4f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, fontsize=12)
            ax.set_ylabel('Value', fontsize=12, fontweight='bold')
            ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            figures[metric] = fig

        return figures

    def _plot_summary_table(self, dataframes: List[pd.DataFrame], labels: List[str]) -> plt.Figure:
        """Create a summary table with key statistics."""
        # Calculate summary statistics
        summary_data = []

        for i, (df, label) in enumerate(zip(dataframes, labels)):
            # Count metrics by trustworthiness
            if 'trustworthiness' in df.columns:
                trust_counts = df['trustworthiness'].value_counts()
                high_trust = trust_counts.get('High', 0)
                medium_trust = trust_counts.get('Medium', 0)
                low_trust = trust_counts.get('Low', 0) + trust_counts.get('Very Low', 0)
            else:
                high_trust = medium_trust = low_trust = 0

            # Key metric values
            gc_diff = df[df['metric'] == 'Absolute Difference']['value'].values
            gc_diff = gc_diff[0] if len(gc_diff) > 0 else np.nan

            novelty = df[df['metric'] == 'Min Edit Distance Mean']['value'].values
            novelty = novelty[0] if len(novelty) > 0 else np.nan

            diversity = df[df['metric'] == 'Unique Sequences Ratio']['value'].values
            diversity = diversity[0] if len(diversity) > 0 else np.nan

            summary_data.append({
                'Model': label,
                'High Trust': high_trust,
                'Med Trust': medium_trust,
                'Low Trust': low_trust,
                'GC Diff': f'{gc_diff:.4f}' if not np.isnan(gc_diff) else 'N/A',
                'Novelty': f'{novelty:.2f}' if not np.isnan(novelty) else 'N/A',
                'Diversity': f'{diversity:.3f}' if not np.isnan(diversity) else 'N/A'
            })

        # Create table
        fig, ax = plt.subplots(figsize=(14, max(4, len(labels) * 0.8)))
        ax.axis('tight')
        ax.axis('off')

        df_summary = pd.DataFrame(summary_data)
        table = ax.table(cellText=df_summary.values, colLabels=df_summary.columns,
                         cellLoc='center', loc='center', colWidths=[0.2, 0.12, 0.12, 0.12, 0.15, 0.15, 0.14])

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)

        # Style header
        for i in range(len(df_summary.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Style rows
        for i in range(1, len(df_summary) + 1):
            for j in range(len(df_summary.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        ax.set_title('Summary Statistics Across Runs', fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()
        return fig

    def _combine_dataframes(self, dataframes: List[pd.DataFrame], labels: List[str]) -> pd.DataFrame:
        """Combine multiple dataframes with run labels."""
        combined = []
        for df, label in zip(dataframes, labels):
            df_copy = df.copy()
            df_copy['run'] = label
            combined.append(df_copy)
        return pd.concat(combined, ignore_index=True)

    def _save_and_show(self, fig: plt.Figure, name: str, output_dir: Optional[str],
                       show: bool, fmt: str, dpi: int):
        """Save and/or show a figure."""
        if output_dir:
            filepath = Path(output_dir) / f"{name}.{fmt}"
            fig.savefig(filepath, format=fmt, dpi=dpi, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("DNA EVALUATION COMPARISON PLOTTER - EXAMPLE")
    print("=" * 80)

    # Simulate example dataframes (normally you'd load these from CSV or evaluation runs)
    np.random.seed(42)


    def create_example_df(noise_level: float = 0.1) -> pd.DataFrame:
        """Create an example results dataframe with some noise."""
        data = {
            'category': ['GC Content', 'GC Content', 'K-mer Analysis', 'K-mer Analysis',
                         'Novelty', 'Novelty', 'Diversity', 'Diversity', 'Diversity'],
            'metric': ['Absolute Difference', 'Wasserstein Distance', '3-mer JS Divergence',
                       '3-mer Cosine Similarity', 'Exact Match Ratio', 'Min Edit Distance Mean',
                       'Unique Sequences Ratio', 'Self-BLEU Mean', 'Distinct-2'],
            'value': [0.04, 0.15, 0.12, 0.88, 0.05, 25.3, 0.92, 0.35, 0.78],
            'unit': ['%', '', '', '', '', 'bp', '', '', ''],
            'interpretation': ['Lower is better'] * 9,
            'std_error': [0.005, 0.02, 0.015, 0.03, 0.008, 1.2, 0.02, 0.04, 0.03],
            'ci_lower': [0.035, 0.13, 0.10, 0.85, 0.04, 23.5, 0.90, 0.30, 0.75],
            'ci_upper': [0.045, 0.17, 0.14, 0.91, 0.06, 27.1, 0.94, 0.40, 0.81],
            'trustworthiness': ['High', 'Medium', 'High', 'Medium', 'High', 'Medium', 'High', 'Low', 'High']
        }
        df = pd.DataFrame(data)

        # Add noise
        #df['value'] = df['value'] * (1 + np.random.randn(len(df)) * noise_level)
        #df['ci_lower'] = df['ci_lower'] * (1 + np.random.randn(len(df)) * noise_level)
        #df['ci_upper'] = df['ci_upper'] * (1 + np.random.randn(len(df)) * noise_level)

        return df


    # Create example dataframes for 3 different models
    df_model_a = create_example_df(noise_level=0.05)
    df_model_b = create_example_df(noise_level=0.15)
    df_model_c = create_example_df(noise_level=0.10)

    # Initialize plotter
    plotter = DNAEvaluationPlotter(figsize=(16, 12))

    # Create comparison plots
    print("\nGenerating comparison plots...")
    figures = plotter.plot_comparison(
        dataframes=[df_model_a, df_model_b, df_model_c],
        labels=['Model A (Baseline)', 'Model B (Fine-tuned)', 'Model C (Augmented)'],
        output_dir='comparison_plots',
        show_plots=False,  # Set to True to display plots
        include_ci=True,
        save_format='png',
        dpi=300
    )

    print("\n" + "=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)
    print("""
# Basic usage - Load results from CSV files
df1 = pd.read_csv('run1_results.csv')
df2 = pd.read_csv('run2_results.csv')
df3 = pd.read_csv('run3_results.csv')

# Create plotter
plotter = DNAEvaluationPlotter()

# Generate all comparison plots
figures = plotter.plot_comparison(
    dataframes=[df1, df2, df3],
    labels=['Baseline', 'Model A', 'Model B'],
    output_dir='comparison_plots',
    show_plots=True,
    include_ci=True
)

# Plot only specific metrics
figures = plotter.plot_comparison(
    dataframes=[df1, df2],
    labels=['Old Model', 'New Model'],
    metrics_to_plot=['Min Edit Distance Mean', 'Unique Sequences Ratio'],
    output_dir='key_metrics'
)

# Access individual figures
overview_fig = figures['overview']
radar_fig = figures['radar']

# Save in different formats
plotter.plot_comparison(
    dataframes=[df1, df2],
    labels=['Model 1', 'Model 2'],
    output_dir='publication_plots',
    save_format='pdf',  # or 'svg', 'png'
    dpi=600,  # High resolution for publication
    show_plots=False
)

# Compare arbitrary number of runs
dfs = [pd.read_csv(f'run_{i}_results.csv') for i in range(1, 11)]
labels = [f'Run {i}' for i in range(1, 11)]
figures = plotter.plot_comparison(dfs, labels, output_dir='all_runs')
    """)
