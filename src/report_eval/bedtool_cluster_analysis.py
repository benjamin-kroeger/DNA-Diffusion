#!/usr/bin/env python3
"""
Genomic Annotation Analysis for Cluster Characterization

Uses bedtools to intersect cluster sequences with:
- RepeatMasker (Alu, LINE, SINE, etc.)
- ENCODE cCREs (enhancers, promoters, insulators)
- Gene annotations (proximity to exons/splice sites)

Requirements:
- bedtools installed and in PATH
- Annotation files downloaded (see --help for details)

Usage:
    python genomic_annotation_analysis.py \
        --sequences sequences.csv \
        --clusters cluster_assignments.csv \
        --repeatmasker ~/annotations/repeatmasker_hg38.bed \
        --ccres ~/annotations/GRCh38-cCREs.bed \
        --exons ~/annotations/gencode_exons.bed \
        --output annotation_results/
"""

import numpy as np
import pandas as pd
import subprocess
import tempfile
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact
import argparse
import warnings
import os
os.environ['PATH'] = os.environ['PATH'] + ":" + '/home/benjaminkroeger/Downloads/bedtools-2.31.1/bedtools2/bin'

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'figure.figsize': (12, 8),
    'figure.dpi': 150,
    'savefig.dpi': 300,
})


class GenomicAnnotationAnalyzer:
    """Analyze genomic annotations for sequence clusters."""

    def __init__(self, output_dir: str = "annotation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = tempfile.mkdtemp()

    def create_cluster_bed_files(self,
                                 sequences_df: pd.DataFrame,
                                 clusters_df: pd.DataFrame) -> Dict[int, str]:
        """Create BED files for each cluster."""

        # Merge sequences with clusters
        # Handle different ID formats
        if 'dhs_id' in sequences_df.columns:
            seq_id_col = 'dhs_id'
        else:
            seq_id_col = sequences_df.columns[0]

        if 'sequence_id' in clusters_df.columns:
            clust_id_col = 'sequence_id'
        else:
            clust_id_col = clusters_df.columns[0]

        # Try to match IDs
        # The cluster sequence_id might be like "chr1_100075908_100076080_100075990"
        # The sequences dhs_id might be the same or similar

        # Extract coordinates from cluster sequence_id if needed
        def parse_sequence_id(seq_id):
            parts = seq_id.split('_')
            if len(parts) >= 3:
                chrom = parts[0]
                start = int(parts[1])
                end = int(parts[2])
                return chrom, start, end
            return None, None, None

        # Create merged dataframe
        if 'chr' in sequences_df.columns and 'start' in sequences_df.columns:
            # Sequences have coordinates directly
            merged = sequences_df.copy()

            # Add cluster info
            # Create a mapping key
            clusters_df['_merge_key'] = clusters_df[clust_id_col]
            sequences_df['_merge_key'] = sequences_df[seq_id_col]

            merged = sequences_df.merge(
                clusters_df[['_merge_key', 'cluster']],
                on='_merge_key',
                how='inner'
            )
        else:
            # Parse coordinates from sequence IDs
            coords = clusters_df[clust_id_col].apply(parse_sequence_id)
            clusters_df['chr'] = [c[0] for c in coords]
            clusters_df['start'] = [c[1] for c in coords]
            clusters_df['end'] = [c[2] for c in coords]
            merged = clusters_df

        print(f"Merged {len(merged)} sequences with cluster assignments")

        # Create BED files per cluster
        bed_files = {}
        unique_clusters = sorted(merged['cluster'].unique())

        for cluster in unique_clusters:
            cluster_data = merged[merged['cluster'] == cluster]
            bed_path = Path(self.temp_dir) / f"cluster_{cluster}.bed"

            # Write BED file: chr, start, end, name, score, strand
            with open(bed_path, 'w') as f:
                for _, row in cluster_data.iterrows():
                    chrom = row['chr']
                    start = int(row['start'])
                    end = int(row['end'])
                    name = row.get(seq_id_col, f"{chrom}_{start}_{end}")
                    f.write(f"{chrom}\t{start}\t{end}\t{name}\t0\t.\n")

            bed_files[cluster] = str(bed_path)
            print(f"  Cluster {cluster}: {len(cluster_data)} regions → {bed_path}")

        # Also create combined BED file
        all_bed_path = Path(self.temp_dir) / "all_clusters.bed"
        with open(all_bed_path, 'w') as f:
            for _, row in merged.iterrows():
                chrom = row['chr']
                start = int(row['start'])
                end = int(row['end'])
                cluster = row['cluster']
                name = f"{row.get(seq_id_col, '')}|cluster_{cluster}"
                f.write(f"{chrom}\t{start}\t{end}\t{name}\t0\t.\n")

        bed_files['all'] = str(all_bed_path)
        self.merged_df = merged
        self.bed_files = bed_files

        return bed_files

    def sort_bed_file(self, bed_file: str) -> str:
        """Sort a BED file for bedtools compatibility."""
        sorted_file = bed_file.replace('.bed', '_sorted.bed')
        if sorted_file == bed_file:
            sorted_file = bed_file + '.sorted'

        cmd = f"sort -k1,1 -k2,2n {bed_file} > {sorted_file}"
        try:
            subprocess.run(cmd, shell=True, check=True)
            return sorted_file
        except subprocess.CalledProcessError:
            return bed_file

    def run_bedtools_intersect(self,
                               query_bed: str,
                               annotation_bed: str,
                               options: str = "-wa -wb") -> pd.DataFrame:
        """Run bedtools intersect and return results."""
        # Sort both files first
        sorted_query = self.sort_bed_file(query_bed)
        sorted_annotation = self.sort_bed_file(annotation_bed)

        cmd = f"bedtools intersect -a {sorted_query} -b {sorted_annotation} {options}"

        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)

            if not result.stdout.strip():
                return pd.DataFrame()

            # Parse output
            lines = result.stdout.strip().split('\n')
            data = [line.split('\t') for line in lines]

            return pd.DataFrame(data)

        except subprocess.CalledProcessError as e:
            print(f"Error running bedtools: {e.stderr}")
            return pd.DataFrame()

    def analyze_repeatmasker(self, repeatmasker_bed: str) -> pd.DataFrame:
        """Analyze overlap with RepeatMasker annotations."""
        print("\nAnalyzing RepeatMasker overlaps...")

        results = []

        for cluster, bed_file in self.bed_files.items():
            if cluster == 'all':
                continue

            # Get overlaps
            overlaps = self.run_bedtools_intersect(bed_file, repeatmasker_bed)

            if overlaps.empty:
                results.append({
                    'cluster': cluster,
                    'total_regions': len(self.merged_df[self.merged_df['cluster'] == cluster]),
                    'regions_with_repeat': 0,
                    'repeat_fraction': 0,
                })
                continue

            # Count unique regions with overlaps
            regions_with_overlap = overlaps[3].nunique()  # Column 3 is query name
            total_regions = len(self.merged_df[self.merged_df['cluster'] == cluster])

            # Count repeat types (column 9 or 10 typically has repeat class)
            # RepeatMasker BED format varies, try to find repeat info
            n_cols = len(overlaps.columns)

            # Typically: query(0-5), annotation(6+)
            # annotation columns: chr, start, end, name, class, strand, family
            if n_cols >= 11:
                repeat_names = overlaps[9].value_counts()  # repeat name
                repeat_classes = overlaps[10].value_counts() if n_cols > 10 else None
            elif n_cols >= 10:
                repeat_names = overlaps[9].value_counts()
                repeat_classes = None
            else:
                repeat_names = pd.Series()
                repeat_classes = None

            # Count Alu specifically
            alu_overlaps = overlaps[overlaps.apply(lambda x: 'Alu' in str(x.values), axis=1)]
            line_overlaps = overlaps[overlaps.apply(lambda x: 'LINE' in str(x.values) or 'L1' in str(x.values), axis=1)]
            sine_overlaps = overlaps[overlaps.apply(lambda x: 'SINE' in str(x.values), axis=1)]

            result = {
                'cluster': cluster,
                'total_regions': total_regions,
                'regions_with_repeat': regions_with_overlap,
                'repeat_fraction': regions_with_overlap / total_regions if total_regions > 0 else 0,
                'alu_overlaps': len(alu_overlaps),
                'alu_fraction': alu_overlaps[3].nunique() / total_regions if total_regions > 0 else 0,
                'line_overlaps': len(line_overlaps),
                'line_fraction': line_overlaps[3].nunique() / total_regions if total_regions > 0 else 0,
                'sine_overlaps': len(sine_overlaps),
                'sine_fraction': sine_overlaps[3].nunique() / total_regions if total_regions > 0 else 0,
            }

            # Add top repeat types
            if len(repeat_names) > 0:
                for i, (name, count) in enumerate(repeat_names.head(5).items()):
                    result[f'top_repeat_{i + 1}'] = f"{name} ({count})"

            results.append(result)

        self.repeatmasker_results = pd.DataFrame(results)
        return self.repeatmasker_results

    def analyze_ccres(self, ccres_bed: str) -> pd.DataFrame:
        """Analyze overlap with ENCODE cCREs."""
        print("\nAnalyzing ENCODE cCRE overlaps...")

        results = []

        for cluster, bed_file in self.bed_files.items():
            if cluster == 'all':
                continue

            overlaps = self.run_bedtools_intersect(bed_file, ccres_bed)

            total_regions = len(self.merged_df[self.merged_df['cluster'] == cluster])

            if overlaps.empty:
                results.append({
                    'cluster': cluster,
                    'total_regions': total_regions,
                    'ccre_overlaps': 0,
                    'ccre_fraction': 0,
                })
                continue

            regions_with_ccre = overlaps[3].nunique()

            # cCRE types are usually in column 9 or name field
            # Types: PLS (promoter-like), pELS (proximal enhancer-like),
            #        dELS (distal enhancer-like), CTCF-only, DNase-H3K4me3

            n_cols = len(overlaps.columns)
            ccre_types = Counter()

            for _, row in overlaps.iterrows():
                row_str = '|'.join(str(x) for x in row.values)
                if 'PLS' in row_str:
                    ccre_types['promoter_like'] += 1
                elif 'pELS' in row_str:
                    ccre_types['proximal_enhancer'] += 1
                elif 'dELS' in row_str:
                    ccre_types['distal_enhancer'] += 1
                elif 'CTCF' in row_str:
                    ccre_types['CTCF_only'] += 1
                elif 'DNase' in row_str:
                    ccre_types['DNase_H3K4me3'] += 1
                else:
                    ccre_types['other'] += 1

            result = {
                'cluster': cluster,
                'total_regions': total_regions,
                'ccre_overlaps': regions_with_ccre,
                'ccre_fraction': regions_with_ccre / total_regions if total_regions > 0 else 0,
            }

            for ctype, count in ccre_types.items():
                result[f'{ctype}_count'] = count
                result[f'{ctype}_fraction'] = count / total_regions if total_regions > 0 else 0

            results.append(result)

        self.ccre_results = pd.DataFrame(results)
        return self.ccre_results

    def analyze_splice_proximity(self, exons_bed: str, max_distance: int = 100) -> pd.DataFrame:
        """Analyze proximity to exon boundaries (splice sites)."""
        print(f"\nAnalyzing proximity to exons (within {max_distance}bp)...")

        # Sort exons file once
        sorted_exons = self.sort_bed_file(exons_bed)

        results = []

        for cluster, bed_file in self.bed_files.items():
            if cluster == 'all':
                continue

            # Sort cluster BED file
            sorted_cluster = self.sort_bed_file(bed_file)

            # Use bedtools closest to find nearest exon
            cmd = f"bedtools closest -a {sorted_cluster} -b {sorted_exons} -d"

            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)

                if not result.stdout.strip():
                    continue

                lines = result.stdout.strip().split('\n')

                distances = []
                near_splice = 0
                overlaps_exon = 0

                for line in lines:
                    parts = line.split('\t')
                    if len(parts) > 0:
                        try:
                            dist = int(parts[-1])  # Last column is distance
                            distances.append(dist)
                            if dist == 0:
                                overlaps_exon += 1
                            elif dist <= max_distance:
                                near_splice += 1
                        except (ValueError, IndexError):
                            pass

                total_regions = len(self.merged_df[self.merged_df['cluster'] == cluster])

                results.append({
                    'cluster': cluster,
                    'total_regions': total_regions,
                    'overlaps_exon': overlaps_exon,
                    'overlaps_exon_fraction': overlaps_exon / total_regions if total_regions > 0 else 0,
                    'near_splice_site': near_splice,
                    'near_splice_fraction': near_splice / total_regions if total_regions > 0 else 0,
                    'mean_distance_to_exon': np.mean(distances) if distances else np.nan,
                    'median_distance_to_exon': np.median(distances) if distances else np.nan,
                })

            except subprocess.CalledProcessError as e:
                print(f"  Error for cluster {cluster}: {e.stderr}")

        self.splice_results = pd.DataFrame(results)
        return self.splice_results

    def statistical_comparison(self) -> Dict:
        """Run statistical tests comparing clusters."""
        print("\nRunning statistical comparisons...")

        stats_results = {}

        # RepeatMasker chi-square test
        if hasattr(self, 'repeatmasker_results') and len(self.repeatmasker_results) > 1:
            df = self.repeatmasker_results

            # Create contingency table: clusters x (has_repeat, no_repeat)
            contingency = []
            for _, row in df.iterrows():
                with_repeat = row['regions_with_repeat']
                without_repeat = row['total_regions'] - row['regions_with_repeat']
                contingency.append([with_repeat, without_repeat])

            contingency = np.array(contingency)

            if contingency.sum() > 0 and contingency.shape[0] > 1:
                chi2, pval, dof, expected = chi2_contingency(contingency)
                stats_results['repeatmasker_chi2'] = {
                    'chi2': chi2,
                    'pvalue': pval,
                    'dof': dof,
                }

            # Alu-specific test
            contingency_alu = []
            for _, row in df.iterrows():
                with_alu = int(row.get('alu_fraction', 0) * row['total_regions'])
                without_alu = row['total_regions'] - with_alu
                contingency_alu.append([with_alu, without_alu])

            contingency_alu = np.array(contingency_alu)

            if contingency_alu.sum() > 0 and contingency_alu.shape[0] > 1:
                chi2, pval, dof, expected = chi2_contingency(contingency_alu)
                stats_results['alu_chi2'] = {
                    'chi2': chi2,
                    'pvalue': pval,
                    'dof': dof,
                }

        # cCRE chi-square test
        if hasattr(self, 'ccre_results') and len(self.ccre_results) > 1:
            df = self.ccre_results

            contingency = []
            for _, row in df.iterrows():
                with_ccre = row['ccre_overlaps']
                without_ccre = row['total_regions'] - row['ccre_overlaps']
                contingency.append([with_ccre, without_ccre])

            contingency = np.array(contingency)

            if contingency.sum() > 0 and contingency.shape[0] > 1:
                chi2, pval, dof, expected = chi2_contingency(contingency)
                stats_results['ccre_chi2'] = {
                    'chi2': chi2,
                    'pvalue': pval,
                    'dof': dof,
                }

        self.stats_results = stats_results
        return stats_results

    def generate_visualizations(self) -> Dict[str, str]:
        """Generate visualization plots."""
        print("\nGenerating visualizations...")

        figures = {}
        figures_dir = self.output_dir / 'figures'
        figures_dir.mkdir(exist_ok=True)

        # Plot 1: RepeatMasker results
        if hasattr(self, 'repeatmasker_results') and len(self.repeatmasker_results) > 0:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            df = self.repeatmasker_results
            clusters = df['cluster'].astype(str).tolist()
            colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12'][:len(clusters)]

            # Overall repeat fraction
            axes[0].bar(clusters, df['repeat_fraction'] * 100, color=colors, edgecolor='black')
            axes[0].set_ylabel('% of Regions')
            axes[0].set_xlabel('Cluster')
            axes[0].set_title('A. Regions Overlapping Repeats')

            # Alu fraction
            if 'alu_fraction' in df.columns:
                axes[1].bar(clusters, df['alu_fraction'] * 100, color=colors, edgecolor='black')
                axes[1].set_ylabel('% of Regions')
                axes[1].set_xlabel('Cluster')
                axes[1].set_title('B. Regions Overlapping Alu Elements')

            # LINE fraction
            if 'line_fraction' in df.columns:
                axes[2].bar(clusters, df['line_fraction'] * 100, color=colors, edgecolor='black')
                axes[2].set_ylabel('% of Regions')
                axes[2].set_xlabel('Cluster')
                axes[2].set_title('C. Regions Overlapping LINE Elements')

            plt.tight_layout()
            filepath = figures_dir / 'repeatmasker_overlap.png'
            plt.savefig(filepath)
            plt.close()
            figures['repeatmasker'] = str(filepath)

        # Plot 2: cCRE results
        if hasattr(self, 'ccre_results') and len(self.ccre_results) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            df = self.ccre_results
            clusters = df['cluster'].astype(str).tolist()
            colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12'][:len(clusters)]

            # Overall cCRE fraction
            axes[0].bar(clusters, df['ccre_fraction'] * 100, color=colors, edgecolor='black')
            axes[0].set_ylabel('% of Regions')
            axes[0].set_xlabel('Cluster')
            axes[0].set_title('A. Regions Overlapping ENCODE cCREs')

            # cCRE type breakdown (stacked bar)
            ccre_types = ['promoter_like_fraction', 'proximal_enhancer_fraction',
                          'distal_enhancer_fraction', 'CTCF_only_fraction']
            ccre_labels = ['Promoter', 'Prox. Enhancer', 'Dist. Enhancer', 'CTCF']

            available_types = [t for t in ccre_types if t in df.columns]

            if available_types:
                bottom = np.zeros(len(clusters))
                for ctype, label in zip(available_types, ccre_labels[:len(available_types)]):
                    values = df[ctype].fillna(0) * 100
                    axes[1].bar(clusters, values, bottom=bottom, label=label, edgecolor='black')
                    bottom += values.values

                axes[1].set_ylabel('% of Regions')
                axes[1].set_xlabel('Cluster')
                axes[1].set_title('B. cCRE Types by Cluster')
                axes[1].legend(loc='upper right')

            plt.tight_layout()
            filepath = figures_dir / 'ccre_overlap.png'
            plt.savefig(filepath)
            plt.close()
            figures['ccre'] = str(filepath)

        # Plot 3: Splice site proximity
        if hasattr(self, 'splice_results') and len(self.splice_results) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            df = self.splice_results
            clusters = df['cluster'].astype(str).tolist()
            colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12'][:len(clusters)]

            # Near splice site fraction
            axes[0].bar(clusters, df['near_splice_fraction'] * 100, color=colors, edgecolor='black')
            axes[0].set_ylabel('% of Regions')
            axes[0].set_xlabel('Cluster')
            axes[0].set_title('A. Regions Near Splice Sites (<100bp)')

            # Median distance to exon
            axes[1].bar(clusters, df['median_distance_to_exon'], color=colors, edgecolor='black')
            axes[1].set_ylabel('Distance (bp)')
            axes[1].set_xlabel('Cluster')
            axes[1].set_title('B. Median Distance to Nearest Exon')

            plt.tight_layout()
            filepath = figures_dir / 'splice_proximity.png'
            plt.savefig(filepath)
            plt.close()
            figures['splice'] = str(filepath)

        # Plot 4: Summary heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        summary_data = []
        clusters = []

        for cluster in sorted(self.merged_df['cluster'].unique()):
            if cluster == 'all':
                continue
            clusters.append(f'Cluster {cluster}')

            row = {}

            if hasattr(self, 'repeatmasker_results'):
                rm = self.repeatmasker_results[self.repeatmasker_results['cluster'] == cluster]
                if len(rm) > 0:
                    row['Repeat Overlap'] = rm['repeat_fraction'].values[0] * 100
                    row['Alu Overlap'] = rm.get('alu_fraction', pd.Series([0])).values[0] * 100
                    row['LINE Overlap'] = rm.get('line_fraction', pd.Series([0])).values[0] * 100

            if hasattr(self, 'ccre_results'):
                cc = self.ccre_results[self.ccre_results['cluster'] == cluster]
                if len(cc) > 0:
                    row['cCRE Overlap'] = cc['ccre_fraction'].values[0] * 100

            if hasattr(self, 'splice_results'):
                sp = self.splice_results[self.splice_results['cluster'] == cluster]
                if len(sp) > 0:
                    row['Near Splice'] = sp['near_splice_fraction'].values[0] * 100

            summary_data.append(row)

        if summary_data and clusters:
            summary_df = pd.DataFrame(summary_data, index=clusters)

            sns.heatmap(summary_df, annot=True, fmt='.1f', cmap='YlOrRd',
                        ax=ax, cbar_kws={'label': '% of Regions'})
            ax.set_title('Genomic Annotation Summary by Cluster')

            plt.tight_layout()
            filepath = figures_dir / 'annotation_summary_heatmap.png'
            plt.savefig(filepath)
            plt.close()
            figures['summary'] = str(filepath)

        return figures

    def generate_report(self) -> str:
        """Generate text report."""
        report = []
        report.append("=" * 80)
        report.append("GENOMIC ANNOTATION ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")

        # RepeatMasker results
        if hasattr(self, 'repeatmasker_results'):
            report.append("## REPEATMASKER OVERLAP")
            report.append("")

            for _, row in self.repeatmasker_results.iterrows():
                report.append(f"### Cluster {row['cluster']} (n={row['total_regions']})")
                report.append(f"  - Regions with repeat elements: {row['regions_with_repeat']} ({row['repeat_fraction'] * 100:.1f}%)")

                if 'alu_fraction' in row:
                    report.append(f"  - Alu elements: {row['alu_fraction'] * 100:.1f}%")
                if 'line_fraction' in row:
                    report.append(f"  - LINE elements: {row['line_fraction'] * 100:.1f}%")
                if 'sine_fraction' in row:
                    report.append(f"  - SINE elements: {row['sine_fraction'] * 100:.1f}%")

                # Top repeats
                top_repeats = [row.get(f'top_repeat_{i}', '') for i in range(1, 6)]
                top_repeats = [r for r in top_repeats if r]
                if top_repeats:
                    report.append(f"  - Top repeat types: {', '.join(top_repeats[:3])}")

                report.append("")

            if 'repeatmasker_chi2' in self.stats_results:
                stats = self.stats_results['repeatmasker_chi2']
                report.append(f"**Chi-square test (repeat overlap):** χ²={stats['chi2']:.2f}, p={stats['pvalue']:.2e}")

            if 'alu_chi2' in self.stats_results:
                stats = self.stats_results['alu_chi2']
                report.append(f"**Chi-square test (Alu overlap):** χ²={stats['chi2']:.2f}, p={stats['pvalue']:.2e}")

            report.append("")

        # cCRE results
        if hasattr(self, 'ccre_results'):
            report.append("## ENCODE cCRE OVERLAP")
            report.append("")

            for _, row in self.ccre_results.iterrows():
                report.append(f"### Cluster {row['cluster']} (n={row['total_regions']})")
                report.append(f"  - Regions with cCRE: {row['ccre_overlaps']} ({row['ccre_fraction'] * 100:.1f}%)")

                for ctype in ['promoter_like', 'proximal_enhancer', 'distal_enhancer', 'CTCF_only']:
                    if f'{ctype}_count' in row:
                        report.append(f"  - {ctype.replace('_', ' ').title()}: {row[f'{ctype}_count']}")

                report.append("")

            if 'ccre_chi2' in self.stats_results:
                stats = self.stats_results['ccre_chi2']
                report.append(f"**Chi-square test (cCRE overlap):** χ²={stats['chi2']:.2f}, p={stats['pvalue']:.2e}")

            report.append("")

        # Splice proximity results
        if hasattr(self, 'splice_results'):
            report.append("## SPLICE SITE PROXIMITY")
            report.append("")

            for _, row in self.splice_results.iterrows():
                report.append(f"### Cluster {row['cluster']} (n={row['total_regions']})")
                report.append(f"  - Overlaps exon: {row['overlaps_exon']} ({row['overlaps_exon_fraction'] * 100:.1f}%)")
                report.append(f"  - Near splice site (<100bp): {row['near_splice_site']} ({row['near_splice_fraction'] * 100:.1f}%)")
                report.append(f"  - Median distance to exon: {row['median_distance_to_exon']:.0f} bp")
                report.append("")

        # Conclusions
        report.append("## CONCLUSIONS")
        report.append("")

        if hasattr(self, 'repeatmasker_results'):
            df = self.repeatmasker_results

            # Find cluster with highest Alu
            if 'alu_fraction' in df.columns:
                max_alu_cluster = df.loc[df['alu_fraction'].idxmax(), 'cluster']
                max_alu_frac = df['alu_fraction'].max()
                report.append(f"- **Highest Alu overlap:** Cluster {max_alu_cluster} ({max_alu_frac * 100:.1f}%)")

            # Find cluster with highest repeat overlap
            max_repeat_cluster = df.loc[df['repeat_fraction'].idxmax(), 'cluster']
            max_repeat_frac = df['repeat_fraction'].max()
            report.append(f"- **Highest repeat overlap:** Cluster {max_repeat_cluster} ({max_repeat_frac * 100:.1f}%)")

        if hasattr(self, 'splice_results'):
            df = self.splice_results

            # Find cluster closest to splice sites
            min_dist_cluster = df.loc[df['median_distance_to_exon'].idxmin(), 'cluster']
            min_dist = df['median_distance_to_exon'].min()
            report.append(f"- **Closest to exons:** Cluster {min_dist_cluster} (median {min_dist:.0f}bp)")

        report.append("")

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description='Genomic annotation analysis for cluster characterization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Required annotation files:
  RepeatMasker: Download from UCSC (rmsk.txt.gz), convert to BED
  ENCODE cCREs: Download from https://api.wenglab.org/screen_v13/fdownloads/
  GENCODE exons: Extract from GTF annotation

Example:
  python genomic_annotation_analysis.py \\
      --sequences sequences.csv \\
      --clusters cluster_assignments.csv \\
      --repeatmasker ~/annotations/repeatmasker_hg38.bed \\
      --ccres ~/annotations/GRCh38-cCREs.bed \\
      --exons ~/annotations/gencode_exons.bed \\
      --output annotation_results/
        """
    )

    parser.add_argument('--sequences', '-s', required=True,
                        help='CSV file with sequences (must have chr, start, end columns)')
    parser.add_argument('--clusters', '-c', required=True,
                        help='CSV file with cluster assignments')
    parser.add_argument('--repeatmasker', '-r',
                        help='RepeatMasker BED file (optional)')
    parser.add_argument('--ccres', '-e',
                        help='ENCODE cCREs BED file (optional)')
    parser.add_argument('--exons', '-x',
                        help='Exons BED file for splice site analysis (optional)')
    parser.add_argument('--output', '-o', default='annotation_results',
                        help='Output directory')

    args = parser.parse_args()

    # Check that at least one annotation file is provided
    if not any([args.repeatmasker, args.ccres, args.exons]):
        parser.error("At least one annotation file (--repeatmasker, --ccres, or --exons) is required")

    # Check bedtools is installed
    try:
        subprocess.run(['bedtools', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: bedtools not found. Please install bedtools:")
        print("  conda install -c bioconda bedtools")
        print("  # or")
        print("  sudo apt-get install bedtools")
        return

    # Load data
    print("Loading data...")
    sequences_df = pd.read_csv(args.sequences)
    clusters_df = pd.read_csv(args.clusters)

    print(f"  Sequences: {len(sequences_df)}")
    print(f"  Cluster assignments: {len(clusters_df)}")

    # Initialize analyzer
    analyzer = GenomicAnnotationAnalyzer(output_dir=args.output)

    # Create BED files
    print("\nCreating BED files for each cluster...")
    analyzer.create_cluster_bed_files(sequences_df, clusters_df)

    # Run analyses
    if args.repeatmasker:
        print(f"\nUsing RepeatMasker file: {args.repeatmasker}")
        analyzer.analyze_repeatmasker(args.repeatmasker)

    if args.ccres:
        print(f"\nUsing cCREs file: {args.ccres}")
        analyzer.analyze_ccres(args.ccres)

    if args.exons:
        print(f"\nUsing exons file: {args.exons}")
        analyzer.analyze_splice_proximity(args.exons)

    # Statistical tests
    analyzer.statistical_comparison()

    # Generate outputs
    figures = analyzer.generate_visualizations()
    report = analyzer.generate_report()

    # Save results
    output_dir = Path(args.output)

    if hasattr(analyzer, 'repeatmasker_results'):
        analyzer.repeatmasker_results.to_csv(output_dir / 'repeatmasker_results.csv', index=False)

    if hasattr(analyzer, 'ccre_results'):
        analyzer.ccre_results.to_csv(output_dir / 'ccre_results.csv', index=False)

    if hasattr(analyzer, 'splice_results'):
        analyzer.splice_results.to_csv(output_dir / 'splice_results.csv', index=False)

    with open(output_dir / 'annotation_report.txt', 'w') as f:
        f.write(report)

    # Print report
    print("\n" + report)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    for name, path in figures.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
