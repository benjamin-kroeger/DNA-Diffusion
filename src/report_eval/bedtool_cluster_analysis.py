#!/usr/bin/env python3
"""
Comprehensive Genomic Annotation Analysis Tool

Analyzes genomic regions against a broad range of annotations:
- Repeats (RepeatMasker: Alu, LINE, SINE, LTR, DNA transposons, etc.)
- Regulatory elements (ENCODE cCREs: enhancers, promoters, insulators)
- Gene features (exons, introns, UTRs, splice sites, TSS)
- Chromatin state (ChromHMM)
- Conservation (phastCons, phyloP)
- CpG islands
- Transcription factor binding sites
- DNase hypersensitivity
- Histone modifications

Automatically downloads and preprocesses annotation files if missing.

Usage:
    python comprehensive_genomic_analyzer.py \
        --sequences sequences.csv \
        --clusters cluster_assignments.csv \
        --genome hg38 \
        --output annotation_results/
"""

import numpy as np
import pandas as pd
import subprocess
import tempfile
import gzip
import shutil
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
import argparse
import warnings
import os
import urllib.request
import sys

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'figure.figsize': (12, 8),
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

# =============================================================================
# ANNOTATION SOURCES CONFIGURATION
# =============================================================================

ANNOTATION_SOURCES = {
    'hg38': {
        'repeatmasker': {
            'url': 'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/rmsk.txt.gz',
            'description': 'RepeatMasker annotations (Alu, LINE, SINE, LTR, etc.)',
            'format': 'ucsc_rmsk',
        },
        'cpg_islands': {
            'url': 'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cpgIslandExt.txt.gz',
            'description': 'CpG Islands',
            'format': 'ucsc_cpg',
        },
        'refseq_genes': {
            'url': 'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/ncbiRefSeq.txt.gz',
            'description': 'RefSeq gene annotations',
            'format': 'ucsc_refseq',
        },
        'gc_content': {
            'url': 'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/gc5BaseBw.txt.gz',
            'description': 'GC content (5bp windows)',
            'format': 'ucsc_gc',
        },
        'simple_repeats': {
            'url': 'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/simpleRepeat.txt.gz',
            'description': 'Simple tandem repeats (microsatellites)',
            'format': 'ucsc_simple_repeat',
        },
        'encode_ccres': {
            'url': 'https://downloads.wenglab.org/V3/GRCh38-cCREs.bed',
            'description': 'ENCODE cCREs (enhancers, promoters, etc.)',
            'format': 'bed',
        },
        'gencode_genes': {
            'url': 'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz',
            'description': 'GENCODE comprehensive gene annotation',
            'format': 'gtf',
        },
        'phastcons': {
            'url': 'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/hg38.phastCons100way.bw',
            'description': 'PhastCons 100-way conservation scores',
            'format': 'bigwig',
        },
        'chromhmm_k562': {
            'url': 'https://www.encodeproject.org/files/ENCFF481URC/@@download/ENCFF481URC.bed.gz',
            'description': 'ChromHMM segmentation for K562',
            'format': 'ucsc_chromhmm',
        },
        'dnase_clusters': {
            'url': 'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/wgEncodeRegDnaseClustered.txt.gz',
            'description': 'DNase I hypersensitivity clusters',
            'format': 'ucsc_dnase',
        },
        'tfbs_clusters': {
            'url': 'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/encRegTfbsClustered.txt.gz',
            'description': 'Transcription factor binding site clusters',
            'format': 'ucsc_tfbs',
        },
    },
    'hg19': {
        'repeatmasker': {
            'url': 'https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/rmsk.txt.gz',
            'description': 'RepeatMasker annotations',
            'format': 'ucsc_rmsk',
        },
        'cpg_islands': {
            'url': 'https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/cpgIslandExt.txt.gz',
            'description': 'CpG Islands',
            'format': 'ucsc_cpg',
        },
        'refseq_genes': {
            'url': 'https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/ncbiRefSeq.txt.gz',
            'description': 'RefSeq gene annotations',
            'format': 'ucsc_refseq',
        },
        'simple_repeats': {
            'url': 'https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/simpleRepeat.txt.gz',
            'description': 'Simple tandem repeats',
            'format': 'ucsc_simple_repeat',
        },
        'encode_ccres': {
            'url': 'https://api.wenglab.org/screen_v13/fdownloads/GRCh37-cCREs.bed',
            'description': 'ENCODE cCREs',
            'format': 'bed',
        },
    },
    'mm10': {
        'repeatmasker': {
            'url': 'https://hgdownload.soe.ucsc.edu/goldenPath/mm10/database/rmsk.txt.gz',
            'description': 'RepeatMasker annotations',
            'format': 'ucsc_rmsk',
        },
        'cpg_islands': {
            'url': 'https://hgdownload.soe.ucsc.edu/goldenPath/mm10/database/cpgIslandExt.txt.gz',
            'description': 'CpG Islands',
            'format': 'ucsc_cpg',
        },
        'refseq_genes': {
            'url': 'https://hgdownload.soe.ucsc.edu/goldenPath/mm10/database/ncbiRefSeq.txt.gz',
            'description': 'RefSeq gene annotations',
            'format': 'ucsc_refseq',
        },
    },
}


# =============================================================================
# ANNOTATION FILE MANAGER
# =============================================================================

class AnnotationFileManager:
    """Manages downloading, converting, and caching annotation files."""

    def __init__(self, cache_dir: str = "~/.genomic_annotations", genome: str = "hg38"):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.genome = genome
        self.genome_dir = self.cache_dir / genome
        self.genome_dir.mkdir(exist_ok=True)

        # Add bedtools to PATH if needed
        self._setup_bedtools_path()

    def _setup_bedtools_path(self):
        """Try to find bedtools and add to PATH."""
        common_paths = [
            '/usr/bin',
            '/usr/local/bin',
            os.path.expanduser('~/miniconda3/bin'),
            os.path.expanduser('~/anaconda3/bin'),
            '/home/benjaminkroeger/Downloads/bedtools-2.31.1/bedtools2/bin',
        ]
        for p in common_paths:
            if os.path.exists(os.path.join(p, 'bedtools')):
                os.environ['PATH'] = os.environ['PATH'] + ":" + p
                break

    def download_file(self, url: str, dest_path: Path, description: str = "") -> bool:
        """Download a file with progress indication."""
        print(f"  Downloading {description or url}...")
        try:
            def report_progress(block_num, block_size, total_size):
                if total_size > 0:
                    pct = min(100, block_num * block_size * 100 // total_size)
                    sys.stdout.write(f"\r    Progress: {pct}%")
                    sys.stdout.flush()

            urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
            print()
            return True
        except Exception as e:
            print(f"\n  ERROR downloading {url}: {e}")
            return False

    def decompress_gzip(self, gz_path: Path, out_path: Path) -> bool:
        """Decompress a gzip file."""
        try:
            with gzip.open(gz_path, 'rb') as f_in:
                with open(out_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return True
        except Exception as e:
            print(f"  ERROR decompressing {gz_path}: {e}")
            return False

    def sort_bed_file(self, bed_path: Path) -> Path:
        """Sort a BED file for bedtools compatibility."""
        sorted_path = bed_path.with_suffix('.sorted.bed')
        if sorted_path.exists():
            return sorted_path

        print(f"  Sorting {bed_path.name}...")
        cmd = f"sort -k1,1 -k2,2n {bed_path} > {sorted_path}"
        try:
            subprocess.run(cmd, shell=True, check=True)
            return sorted_path
        except subprocess.CalledProcessError as e:
            print(f"  ERROR sorting: {e}")
            return bed_path

    def convert_ucsc_rmsk_to_bed(self, ucsc_path: Path, bed_path: Path) -> bool:
        """Convert UCSC RepeatMasker format to BED."""
        print(f"  Converting RepeatMasker to BED...")
        try:
            with open(ucsc_path, 'r') as fin, open(bed_path, 'w') as fout:
                for line in fin:
                    parts = line.strip().split('\t')
                    if len(parts) >= 17:
                        # UCSC rmsk: bin, swScore, milliDiv, milliDel, milliIns,
                        # genoName, genoStart, genoEnd, genoLeft, strand,
                        # repName, repClass, repFamily, repStart, repEnd, repLeft, id
                        chrom = parts[5]
                        start = parts[6]
                        end = parts[7]
                        strand = parts[9]
                        rep_name = parts[10]
                        rep_class = parts[11]
                        rep_family = parts[12]
                        fout.write(f"{chrom}\t{start}\t{end}\t{rep_name}\t0\t{strand}\t{rep_class}\t{rep_family}\n")
            return True
        except Exception as e:
            print(f"  ERROR converting: {e}")
            return False

    def convert_ucsc_cpg_to_bed(self, ucsc_path: Path, bed_path: Path) -> bool:
        """Convert UCSC CpG island format to BED."""
        print(f"  Converting CpG islands to BED...")
        try:
            with open(ucsc_path, 'r') as fin, open(bed_path, 'w') as fout:
                for line in fin:
                    parts = line.strip().split('\t')
                    if len(parts) >= 11:
                        # bin, chrom, chromStart, chromEnd, name, length, cpgNum, gcNum, perCpg, perGc, obsExp
                        chrom = parts[1]
                        start = parts[2]
                        end = parts[3]
                        name = parts[4]
                        cpg_num = parts[6]
                        gc_pct = parts[9]
                        obs_exp = parts[10]
                        fout.write(f"{chrom}\t{start}\t{end}\t{name}\t{cpg_num}\t.\t{gc_pct}\t{obs_exp}\n")
            return True
        except Exception as e:
            print(f"  ERROR converting: {e}")
            return False

    def convert_ucsc_refseq_to_bed(self, ucsc_path: Path, bed_path: Path) -> bool:
        """Convert UCSC RefSeq format to multiple BED files (genes, exons, UTRs)."""
        print(f"  Converting RefSeq to BED files...")
        try:
            genes_path = bed_path.with_name(bed_path.stem + '_genes.bed')
            exons_path = bed_path.with_name(bed_path.stem + '_exons.bed')
            tss_path = bed_path.with_name(bed_path.stem + '_tss.bed')
            utr5_path = bed_path.with_name(bed_path.stem + '_5utr.bed')
            utr3_path = bed_path.with_name(bed_path.stem + '_3utr.bed')
            introns_path = bed_path.with_name(bed_path.stem + '_introns.bed')

            with open(ucsc_path, 'r') as fin, \
                open(genes_path, 'w') as f_genes, \
                open(exons_path, 'w') as f_exons, \
                open(tss_path, 'w') as f_tss, \
                open(utr5_path, 'w') as f_utr5, \
                open(utr3_path, 'w') as f_utr3, \
                open(introns_path, 'w') as f_introns:

                for line in fin:
                    parts = line.strip().split('\t')
                    if len(parts) < 16:
                        continue

                    # RefSeq format columns
                    chrom = parts[2]
                    strand = parts[3]
                    tx_start = int(parts[4])
                    tx_end = int(parts[5])
                    cds_start = int(parts[6])
                    cds_end = int(parts[7])
                    exon_count = int(parts[8])
                    exon_starts = [int(x) for x in parts[9].strip(',').split(',') if x]
                    exon_ends = [int(x) for x in parts[10].strip(',').split(',') if x]
                    gene_name = parts[12] if len(parts) > 12 else parts[1]

                    # Skip non-standard chromosomes
                    if '_' in chrom and chrom not in ['chrM']:
                        continue

                    # Gene body
                    f_genes.write(f"{chrom}\t{tx_start}\t{tx_end}\t{gene_name}\t0\t{strand}\n")

                    # TSS (promoter region: -500 to +100)
                    if strand == '+':
                        tss = tx_start
                        tss_start = max(0, tss - 500)
                        tss_end = tss + 100
                    else:
                        tss = tx_end
                        tss_start = max(0, tss - 100)
                        tss_end = tss + 500
                    f_tss.write(f"{chrom}\t{tss_start}\t{tss_end}\t{gene_name}_TSS\t0\t{strand}\n")

                    # Exons and introns
                    prev_end = None
                    for i, (e_start, e_end) in enumerate(zip(exon_starts, exon_ends)):
                        f_exons.write(f"{chrom}\t{e_start}\t{e_end}\t{gene_name}_exon{i + 1}\t0\t{strand}\n")

                        # Intron (between this and previous exon)
                        if prev_end is not None and e_start > prev_end:
                            f_introns.write(f"{chrom}\t{prev_end}\t{e_start}\t{gene_name}_intron{i}\t0\t{strand}\n")
                        prev_end = e_end

                    # 5' UTR
                    if strand == '+' and tx_start < cds_start:
                        f_utr5.write(f"{chrom}\t{tx_start}\t{cds_start}\t{gene_name}_5UTR\t0\t{strand}\n")
                    elif strand == '-' and cds_end < tx_end:
                        f_utr5.write(f"{chrom}\t{cds_end}\t{tx_end}\t{gene_name}_5UTR\t0\t{strand}\n")

                    # 3' UTR
                    if strand == '+' and cds_end < tx_end:
                        f_utr3.write(f"{chrom}\t{cds_end}\t{tx_end}\t{gene_name}_3UTR\t0\t{strand}\n")
                    elif strand == '-' and tx_start < cds_start:
                        f_utr3.write(f"{chrom}\t{tx_start}\t{cds_start}\t{gene_name}_3UTR\t0\t{strand}\n")

            return True
        except Exception as e:
            print(f"  ERROR converting: {e}")
            return False

    def convert_ucsc_simple_repeat_to_bed(self, ucsc_path: Path, bed_path: Path) -> bool:
        """Convert UCSC simple repeat format to BED."""
        print(f"  Converting simple repeats to BED...")
        try:
            with open(ucsc_path, 'r') as fin, open(bed_path, 'w') as fout:
                for line in fin:
                    parts = line.strip().split('\t')
                    if len(parts) >= 17:
                        # bin, chrom, chromStart, chromEnd, name, period, copyNum, consensusSize,
                        # perMatch, perIndel, score, A, C, G, T, entropy, sequence
                        chrom = parts[1]
                        start = parts[2]
                        end = parts[3]
                        name = parts[4]
                        period = parts[5]
                        copy_num = parts[6]
                        fout.write(f"{chrom}\t{start}\t{end}\t{name}\t{period}\t.\t{copy_num}\n")
            return True
        except Exception as e:
            print(f"  ERROR converting: {e}")
            return False

    def convert_gtf_to_bed(self, gtf_path: Path, bed_path: Path) -> bool:
        """Convert GTF to multiple feature-specific BED files."""
        print(f"  Converting GTF to BED files...")
        try:
            features = defaultdict(list)

            opener = gzip.open if str(gtf_path).endswith('.gz') else open
            with opener(gtf_path, 'rt') as fin:
                for line in fin:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) < 9:
                        continue

                    chrom = parts[0]
                    feature = parts[2]
                    start = int(parts[3]) - 1  # GTF is 1-based
                    end = int(parts[4])
                    strand = parts[6]
                    attrs = parts[8]

                    # Parse gene name
                    gene_name = "unknown"
                    for attr in attrs.split(';'):
                        attr = attr.strip()
                        if attr.startswith('gene_name'):
                            gene_name = attr.split('"')[1] if '"' in attr else attr.split()[1]
                            break

                    features[feature].append((chrom, start, end, gene_name, strand))

            # Write feature-specific BED files
            for feature, records in features.items():
                feature_bed = bed_path.with_name(f"gencode_{feature}.bed")
                with open(feature_bed, 'w') as fout:
                    for chrom, start, end, name, strand in records:
                        fout.write(f"{chrom}\t{start}\t{end}\t{name}\t0\t{strand}\n")
                print(f"    Created {feature_bed.name} ({len(records)} records)")

            return True
        except Exception as e:
            print(f"  ERROR converting GTF: {e}")
            return False

    def get_annotation_file(self, annotation_name: str) -> Optional[Path]:
        """Get or download an annotation file, return path to sorted BED."""
        if self.genome not in ANNOTATION_SOURCES:
            print(f"  Unknown genome: {self.genome}")
            return None

        if annotation_name not in ANNOTATION_SOURCES[self.genome]:
            print(f"  Unknown annotation: {annotation_name}")
            return None

        source = ANNOTATION_SOURCES[self.genome][annotation_name]
        url = source['url']
        fmt = source['format']
        desc = source['description']

        # Define file paths
        raw_file = self.genome_dir / f"{annotation_name}_raw.txt"
        gz_file = self.genome_dir / f"{annotation_name}_raw.txt.gz"
        bed_file = self.genome_dir / f"{annotation_name}.bed"
        sorted_bed = self.genome_dir / f"{annotation_name}.sorted.bed"

        # Check if sorted BED already exists
        if sorted_bed.exists():
            print(f"  Using cached {annotation_name}: {sorted_bed}")
            return sorted_bed

        # Check if BED exists, just needs sorting
        if bed_file.exists():
            return self.sort_bed_file(bed_file)

        # Download if needed
        download_path = gz_file if url.endswith('.gz') else raw_file
        if url.endswith('.bed'):
            download_path = bed_file

        if not download_path.exists() and not raw_file.exists():
            if not self.download_file(url, download_path, desc):
                return None

        # Decompress if gzipped
        if download_path.suffix == '.gz' and not raw_file.exists():
            if not self.decompress_gzip(download_path, raw_file):
                return None

        # Convert to BED based on format
        if fmt == 'ucsc_rmsk' and not bed_file.exists():
            if not self.convert_ucsc_rmsk_to_bed(raw_file, bed_file):
                return None
        elif fmt == 'ucsc_cpg' and not bed_file.exists():
            if not self.convert_ucsc_cpg_to_bed(raw_file, bed_file):
                return None
        elif fmt == 'ucsc_refseq' and not bed_file.exists():
            if not self.convert_ucsc_refseq_to_bed(raw_file, bed_file):
                return None
            # Return the genes file as main, others are accessible too
            bed_file = bed_file.with_name(bed_file.stem + '_genes.bed')
        elif fmt == 'ucsc_simple_repeat' and not bed_file.exists():
            if not self.convert_ucsc_simple_repeat_to_bed(raw_file, bed_file):
                return None
        elif fmt == 'gtf':
            if not self.convert_gtf_to_bed(raw_file, bed_file):
                return None
            # Return exon file as main
            bed_file = bed_file.with_name('gencode_exon.bed')
        elif fmt == 'bed':
            bed_file = download_path

        # Sort the BED file
        if bed_file.exists():
            return self.sort_bed_file(bed_file)

        return None

    def get_all_available_annotations(self) -> Dict[str, Path]:
        """Download and prepare all available annotations for the genome."""
        annotations = {}

        print(f"\nPreparing annotations for {self.genome}...")

        for name in ANNOTATION_SOURCES.get(self.genome, {}):
            path = self.get_annotation_file(name)
            if path:
                annotations[name] = path

                # For RefSeq, also register the sub-files
                if name == 'refseq_genes':
                    base = path.parent / 'refseq_genes'
                    for suffix in ['_exons', '_tss', '_5utr', '_3utr', '_introns']:
                        sub_file = base.with_name(f'refseq_genes{suffix}.bed')
                        sorted_sub = sub_file.with_suffix('.sorted.bed')
                        if sub_file.exists():
                            sorted_path = self.sort_bed_file(sub_file)
                            annotations[f'refseq{suffix}'] = sorted_path

        return annotations


# =============================================================================
# COMPREHENSIVE GENOMIC ANALYZER
# =============================================================================

class ComprehensiveGenomicAnalyzer:
    """Analyze genomic regions against comprehensive annotations."""

    def __init__(self, output_dir: str = "annotation_results",
                 genome: str = "hg38",
                 cache_dir: str = "~/.genomic_annotations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.genome = genome

        # Initialize annotation manager
        self.annotation_manager = AnnotationFileManager(cache_dir, genome)

        # Results storage
        self.results = {}
        self.stats_results = {}

    def check_bedtools(self) -> bool:
        """Check if bedtools is installed."""
        try:
            result = subprocess.run(['bedtools', '--version'],
                                    capture_output=True, text=True)
            print(f"Using {result.stdout.strip()}")
            return True
        except FileNotFoundError:
            print("ERROR: bedtools not found. Please install:")
            print("  conda install -c bioconda bedtools")
            print("  # or")
            print("  sudo apt-get install bedtools")
            return False

    def create_cluster_bed_files(self, sequences_df: pd.DataFrame,
                                 clusters_df: pd.DataFrame) -> Dict[str, Path]:
        """Create BED files for each cluster."""
        print("\nCreating BED files for clusters...")

        # Identify ID columns
        seq_id_col = 'dhs_id' if 'dhs_id' in sequences_df.columns else sequences_df.columns[0]
        clust_id_col = 'sequence_id' if 'sequence_id' in clusters_df.columns else clusters_df.columns[0]
        cluster_col = 'region' if 'region' in clusters_df.columns else 'cluster'

        # Parse coordinates from sequence IDs if needed
        def parse_seq_id(seq_id):
            parts = str(seq_id).split('_')
            if len(parts) >= 3:
                try:
                    return parts[0], int(parts[1]), int(parts[2])
                except ValueError:
                    pass
            return None, None, None

        # Merge dataframes
        if 'chr' in sequences_df.columns:
            merged = sequences_df.merge(
                clusters_df[[clust_id_col, cluster_col]],
                left_on=seq_id_col, right_on=clust_id_col, how='inner'
            )
        else:
            coords = clusters_df[clust_id_col].apply(parse_seq_id)
            clusters_df = clusters_df.copy()
            clusters_df['chr'] = [c[0] for c in coords]
            clusters_df['start'] = [c[1] for c in coords]
            clusters_df['end'] = [c[2] for c in coords]
            merged = clusters_df

        print(f"  Merged {len(merged)} sequences with cluster assignments")

        # Create BED files
        bed_files = {}
        unique_clusters = sorted(merged[cluster_col].unique())

        for cluster in unique_clusters:
            cluster_data = merged[merged[cluster_col] == cluster]
            bed_path = self.temp_dir / f"cluster_{cluster}.bed"

            with open(bed_path, 'w') as f:
                for _, row in cluster_data.iterrows():
                    chrom = row['chr']
                    start = int(row['start'])
                    end = int(row['end'])
                    name = f"{chrom}_{start}_{end}"
                    f.write(f"{chrom}\t{start}\t{end}\t{name}\t0\t.\n")

            # Sort the BED file
            sorted_path = self.annotation_manager.sort_bed_file(bed_path)
            bed_files[str(cluster)] = sorted_path
            print(f"  Cluster {cluster}: {len(cluster_data)} regions")

        # Combined BED
        all_bed = self.temp_dir / "all_clusters.bed"
        with open(all_bed, 'w') as f:
            for _, row in merged.iterrows():
                f.write(f"{row['chr']}\t{int(row['start'])}\t{int(row['end'])}\t{row[cluster_col]}\t0\t.\n")
        bed_files['all'] = self.annotation_manager.sort_bed_file(all_bed)

        self.merged_df = merged
        self.bed_files = bed_files
        self.cluster_col = cluster_col
        return bed_files

    def run_bedtools_intersect(self, query_bed: Path, annotation_bed: Path,
                               options: str = "-wa -wb") -> pd.DataFrame:
        """Run bedtools intersect."""
        cmd = f"bedtools intersect -a {query_bed} -b {annotation_bed} {options}"
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            if not result.stdout.strip():
                return pd.DataFrame()
            lines = result.stdout.strip().split('\n')
            return pd.DataFrame([l.split('\t') for l in lines])
        except subprocess.CalledProcessError as e:
            return pd.DataFrame()

    def run_bedtools_closest(self, query_bed: Path, annotation_bed: Path) -> pd.DataFrame:
        """Run bedtools closest to find nearest features."""
        cmd = f"bedtools closest -a {query_bed} -b {annotation_bed} -d"
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            if not result.stdout.strip():
                return pd.DataFrame()
            lines = result.stdout.strip().split('\n')
            return pd.DataFrame([l.split('\t') for l in lines])
        except subprocess.CalledProcessError:
            return pd.DataFrame()

    def calculate_overlap_stats(self, cluster: str, overlaps: pd.DataFrame,
                                total_regions: int, name_col: int = 3) -> Dict:
        """Calculate basic overlap statistics."""
        if overlaps.empty:
            return {'overlapping_regions': 0, 'overlap_fraction': 0.0, 'total_overlaps': 0}

        unique_regions = overlaps[name_col].nunique()
        return {
            'overlapping_regions': unique_regions,
            'overlap_fraction': unique_regions / total_regions if total_regions > 0 else 0,
            'total_overlaps': len(overlaps),
        }

    def analyze_repeats(self, repeat_bed: Path) -> pd.DataFrame:
        """Comprehensive repeat element analysis."""
        print("\n[REPEATS] Analyzing repeat elements...")
        results = []

        repeat_classes = ['Alu', 'LINE', 'SINE', 'LTR', 'DNA', 'Simple_repeat',
                          'Low_complexity', 'Satellite', 'RNA', 'L1', 'L2', 'MIR']

        for cluster, bed_file in self.bed_files.items():
            if cluster == 'all':
                continue

            total = len(self.merged_df[self.merged_df[self.cluster_col] == int(cluster)
            if cluster.isdigit() else self.merged_df[self.cluster_col] == cluster])
            overlaps = self.run_bedtools_intersect(bed_file, repeat_bed)

            row = {'cluster': cluster, 'total_regions': total}
            row.update(self.calculate_overlap_stats(cluster, overlaps, total))

            # Count specific repeat types
            if not overlaps.empty:
                for rep_class in repeat_classes:
                    matches = overlaps[overlaps.apply(
                        lambda x: rep_class in str(x.values), axis=1)]
                    row[f'{rep_class}_count'] = len(matches)
                    row[f'{rep_class}_fraction'] = matches[3].nunique() / total if total > 0 else 0

                # Get top repeat families
                if len(overlaps.columns) >= 10:
                    top_repeats = overlaps[9].value_counts().head(5)
                    for i, (name, count) in enumerate(top_repeats.items()):
                        row[f'top_repeat_{i + 1}'] = f"{name}:{count}"

            results.append(row)

        self.results['repeats'] = pd.DataFrame(results)
        return self.results['repeats']

    def analyze_simple_repeats(self, simple_repeat_bed: Path) -> pd.DataFrame:
        """Analyze simple/tandem repeats (microsatellites)."""
        print("\n[SIMPLE REPEATS] Analyzing microsatellites...")
        results = []

        for cluster, bed_file in self.bed_files.items():
            if cluster == 'all':
                continue

            total = len(self.merged_df[self.merged_df[self.cluster_col] == int(cluster)
            if cluster.isdigit() else self.merged_df[self.cluster_col] == cluster])
            overlaps = self.run_bedtools_intersect(bed_file, simple_repeat_bed)

            row = {'cluster': cluster, 'total_regions': total}
            row.update(self.calculate_overlap_stats(cluster, overlaps, total))

            # Analyze repeat periods (dinucleotide, trinucleotide, etc.)
            if not overlaps.empty and len(overlaps.columns) >= 11:
                periods = overlaps[10].astype(int)
                row['mean_period'] = periods.mean()
                row['dinucleotide_count'] = (periods == 2).sum()
                row['trinucleotide_count'] = (periods == 3).sum()
                row['tetranucleotide_count'] = (periods == 4).sum()

            results.append(row)

        self.results['simple_repeats'] = pd.DataFrame(results)
        return self.results['simple_repeats']

    def analyze_cpg_islands(self, cpg_bed: Path) -> pd.DataFrame:
        """Analyze CpG island overlap."""
        print("\n[CPG ISLANDS] Analyzing CpG island overlap...")
        results = []

        for cluster, bed_file in self.bed_files.items():
            if cluster == 'all':
                continue

            total = len(self.merged_df[self.merged_df[self.cluster_col] == int(cluster)
            if cluster.isdigit() else self.merged_df[self.cluster_col] == cluster])
            overlaps = self.run_bedtools_intersect(bed_file, cpg_bed)

            row = {'cluster': cluster, 'total_regions': total}
            row.update(self.calculate_overlap_stats(cluster, overlaps, total))

            # CpG island characteristics
            if not overlaps.empty and len(overlaps.columns) >= 14:
                try:
                    row['mean_cpg_count'] = overlaps[10].astype(float).mean()
                    row['mean_gc_percent'] = overlaps[12].astype(float).mean()
                    row['mean_obs_exp'] = overlaps[13].astype(float).mean()
                except:
                    pass

            results.append(row)

        self.results['cpg_islands'] = pd.DataFrame(results)
        return self.results['cpg_islands']

    def analyze_ccres(self, ccre_bed: Path) -> pd.DataFrame:
        """Analyze ENCODE cCRE overlap."""
        print("\n[CCRES] Analyzing ENCODE regulatory elements...")
        results = []

        ccre_types = {'PLS': 'promoter_like', 'pELS': 'proximal_enhancer',
                      'dELS': 'distal_enhancer', 'CTCF': 'ctcf_bound', 'DNase': 'dnase_only'}

        for cluster, bed_file in self.bed_files.items():
            if cluster == 'all':
                continue

            total = len(self.merged_df[self.merged_df[self.cluster_col] == int(cluster)
            if cluster.isdigit() else self.merged_df[self.cluster_col] == cluster])
            overlaps = self.run_bedtools_intersect(bed_file, ccre_bed)

            row = {'cluster': cluster, 'total_regions': total}
            row.update(self.calculate_overlap_stats(cluster, overlaps, total))

            # Count cCRE types
            if not overlaps.empty:
                for pattern, name in ccre_types.items():
                    matches = overlaps[overlaps.apply(lambda x: pattern in str(x.values), axis=1)]
                    row[f'{name}_count'] = len(matches)
                    row[f'{name}_fraction'] = matches[3].nunique() / total if total > 0 else 0

            results.append(row)

        self.results['ccres'] = pd.DataFrame(results)
        return self.results['ccres']

    def analyze_gene_features(self, annotations: Dict[str, Path]) -> pd.DataFrame:
        """Analyze gene feature overlap (exons, introns, UTRs, TSS)."""
        print("\n[GENE FEATURES] Analyzing gene structure overlap...")
        results = []

        feature_files = {
            'exons': annotations.get('refseq_exons'),
            'introns': annotations.get('refseq_introns'),
            'tss': annotations.get('refseq_tss'),
            'utr5': annotations.get('refseq_5utr'),
            'utr3': annotations.get('refseq_3utr'),
        }

        for cluster, bed_file in self.bed_files.items():
            if cluster == 'all':
                continue

            total = len(self.merged_df[self.merged_df[self.cluster_col] == int(cluster)
            if cluster.isdigit() else self.merged_df[self.cluster_col] == cluster])

            row = {'cluster': cluster, 'total_regions': total}

            for feature, feature_bed in feature_files.items():
                if feature_bed and feature_bed.exists():
                    overlaps = self.run_bedtools_intersect(bed_file, feature_bed)
                    stats = self.calculate_overlap_stats(cluster, overlaps, total)
                    row[f'{feature}_overlap'] = stats['overlapping_regions']
                    row[f'{feature}_fraction'] = stats['overlap_fraction']

            # Distance to nearest exon
            if feature_files['exons'] and feature_files['exons'].exists():
                closest = self.run_bedtools_closest(bed_file, feature_files['exons'])
                if not closest.empty:
                    distances = closest.iloc[:, -1].astype(int)
                    row['median_dist_to_exon'] = distances.median()
                    row['near_splice_100bp'] = (distances <= 100).sum() / total if total > 0 else 0

            results.append(row)

        self.results['gene_features'] = pd.DataFrame(results)
        return self.results['gene_features']

    def analyze_all(self, annotations: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
        """Run all available analyses based on downloaded annotations."""
        print("\n" + "=" * 60)
        print("RUNNING COMPREHENSIVE GENOMIC ANNOTATION ANALYSIS")
        print("=" * 60)

        if 'repeatmasker' in annotations:
            self.analyze_repeats(annotations['repeatmasker'])

        if 'simple_repeats' in annotations:
            self.analyze_simple_repeats(annotations['simple_repeats'])

        if 'cpg_islands' in annotations:
            self.analyze_cpg_islands(annotations['cpg_islands'])

        if 'encode_ccres' in annotations:
            self.analyze_ccres(annotations['encode_ccres'])

        gene_related = {k: v for k, v in annotations.items()
                        if k.startswith('refseq_') or k == 'refseq_genes'}
        if gene_related:
            self.analyze_gene_features(annotations)

        return self.results

    def run_statistical_tests(self) -> Dict:
        """Run chi-square tests comparing clusters for each annotation."""
        print("\n[STATISTICS] Running statistical comparisons...")

        for result_name, df in self.results.items():
            if 'overlap_fraction' not in df.columns:
                continue

            # Chi-square test for overall overlap
            contingency = []
            for _, row in df.iterrows():
                with_overlap = int(row['overlap_fraction'] * row['total_regions'])
                without = row['total_regions'] - with_overlap
                contingency.append([with_overlap, without])

            contingency = np.array(contingency)
            if contingency.sum() > 0 and len(contingency) > 1:
                try:
                    chi2, pval, dof, _ = chi2_contingency(contingency)
                    self.stats_results[f'{result_name}_chi2'] = {
                        'chi2': chi2, 'pvalue': pval, 'dof': dof
                    }
                except:
                    pass

        return self.stats_results

    def generate_summary_heatmap(self) -> Path:
        """Generate comprehensive summary heatmap."""
        print("\n[VISUALIZATION] Generating summary heatmap...")

        summary = defaultdict(dict)
        clusters = sorted([c for c in self.bed_files.keys() if c != 'all'])

        for name, df in self.results.items():
            frac_cols = [c for c in df.columns if c.endswith('_fraction')]
            for col in frac_cols:
                feature = col.replace('_fraction', '').replace('_', ' ').title()
                for _, row in df.iterrows():
                    summary[f"Cluster {row['cluster']}"][f"{name}: {feature}"] = row[col] * 100

        if not summary:
            return None

        summary_df = pd.DataFrame(summary).T

        fig, ax = plt.subplots(figsize=(max(12, len(summary_df.columns) * 0.8),
                                        max(8, len(summary_df) * 0.5)))
        sns.heatmap(summary_df, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                    cbar_kws={'label': '% of Regions'})
        ax.set_title('Comprehensive Genomic Annotation Summary')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        filepath = self.output_dir / 'comprehensive_summary_heatmap.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath

    def generate_report(self) -> str:
        """Generate comprehensive text report."""
        lines = ["=" * 80, "COMPREHENSIVE GENOMIC ANNOTATION REPORT", "=" * 80,
                 f"Genome: {self.genome}", f"Clusters analyzed: {len(self.bed_files) - 1}", ""]

        for name, df in self.results.items():
            lines.extend([f"\n## {name.upper().replace('_', ' ')}", ""])
            for _, row in df.iterrows():
                lines.append(f"### Cluster {row['cluster']} (n={row['total_regions']})")
                for col in df.columns:
                    if col in ['cluster', 'total_regions']:
                        continue
                    val = row[col]
                    if pd.notna(val):
                        if 'fraction' in col:
                            lines.append(f"  - {col}: {val * 100:.1f}%")
                        elif isinstance(val, float):
                            lines.append(f"  - {col}: {val:.2f}")
                        else:
                            lines.append(f"  - {col}: {val}")
                lines.append("")

        # Statistics
        if self.stats_results:
            lines.extend(["\n## STATISTICAL TESTS", ""])
            for name, stats in self.stats_results.items():
                lines.append(f"  {name}: χ²={stats['chi2']:.2f}, p={stats['pvalue']:.2e}")

        return '\n'.join(lines)

    def save_results(self):
        """Save all results to files."""
        for name, df in self.results.items():
            df.to_csv(self.output_dir / f'{name}_results.csv', index=False)

        with open(self.output_dir / 'comprehensive_report.txt', 'w') as f:
            f.write(self.generate_report())

        self.generate_summary_heatmap()
        print(f"\nResults saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive genomic annotation analysis')
    parser.add_argument('--sequences', '-s', required=True, help='Sequences CSV')
    parser.add_argument('--clusters', '-c', required=True, help='Cluster assignments CSV')
    parser.add_argument('--genome', '-g', default='hg38', choices=['hg38', 'hg19', 'mm10'])
    parser.add_argument('--output', '-o', default='annotation_results')
    parser.add_argument('--cache-dir', default='~/.genomic_annotations')
    parser.add_argument('--annotations', '-a', nargs='*', help='Specific annotations to run')
    args = parser.parse_args()

    analyzer = ComprehensiveGenomicAnalyzer(args.output, args.genome, args.cache_dir)

    if not analyzer.check_bedtools():
        return

    # Get annotations
    annotations = analyzer.annotation_manager.get_all_available_annotations()
    if args.annotations:
        annotations = {k: v for k, v in annotations.items() if k in args.annotations}

    if not annotations:
        print("No annotations available!")
        return

    # Load and prepare data
    sequences_df = pd.read_csv(args.sequences)
    clusters_df = pd.read_csv(args.clusters)
    analyzer.create_cluster_bed_files(sequences_df, clusters_df)

    # Run analyses
    analyzer.analyze_all(annotations)
    analyzer.run_statistical_tests()
    analyzer.save_results()

    print("\n" + analyzer.generate_report())


if __name__ == "__main__":
    main()
