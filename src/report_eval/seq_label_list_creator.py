#!/usr/bin/env python3
"""
Generate sequence ID lists based on genomic annotations.

Creates 3 text files with sequence IDs:
1. sequences_with_repeats.txt - sequences overlapping any repeat element
2. sequences_with_alu.txt - sequences overlapping Alu elements
3. sequences_with_line.txt - sequences overlapping LINE elements

These can be used to highlight specific sequences on UMAP plots.

Usage:
    python generate_annotation_lists.py \
        --sequences sequences.csv \
        --repeatmasker ~/annotations/repeatmasker_hg38.bed \
        --output annotation_lists/
"""

import pandas as pd
import subprocess
import tempfile
from pathlib import Path
from typing import Set, Tuple
import argparse
import os

os.environ['PATH'] = os.environ['PATH'] + ":" + '/home/benjaminkroeger/Downloads/bedtools-2.31.1/bedtools2/bin'

def sort_bed_file(bed_file: str, temp_dir: str) -> str:
    """Sort a BED file for bedtools compatibility."""
    sorted_file = Path(temp_dir) / f"{Path(bed_file).stem}_sorted.bed"
    cmd = f"sort -k1,1 -k2,2n {bed_file} > {sorted_file}"
    subprocess.run(cmd, shell=True, check=True)
    return str(sorted_file)


def create_sequences_bed(sequences_df: pd.DataFrame, temp_dir: str) -> Tuple[str, dict]:
    """Create BED file from sequences dataframe and return ID mapping."""

    # Find coordinate columns
    if 'chr' in sequences_df.columns:
        chr_col = 'chr'
    elif 'chrom' in sequences_df.columns:
        chr_col = 'chrom'
    else:
        raise ValueError("No chromosome column found (expected 'chr' or 'chrom')")

    # Find ID column
    if 'dhs_id' in sequences_df.columns:
        id_col = 'dhs_id'
    elif 'sequence_id' in sequences_df.columns:
        id_col = 'sequence_id'
    else:
        id_col = sequences_df.columns[0]

    bed_path = Path(temp_dir) / "sequences.bed"

    with open(bed_path, 'w') as f:
        for _, row in sequences_df.iterrows():
            chrom = row[chr_col]
            start = int(row['start'])
            end = int(row['end'])
            seq_id = row[id_col]
            f.write(f"{chrom}\t{start}\t{end}\t{seq_id}\t0\t.\n")

    print(f"Created BED file with {len(sequences_df)} sequences")
    return str(bed_path), id_col


def get_overlapping_ids(query_bed: str, annotation_bed: str, temp_dir: str,
                        filter_pattern: str = None) -> Set[str]:
    """Get sequence IDs that overlap with annotation."""

    # Sort files
    sorted_query = sort_bed_file(query_bed, temp_dir)
    sorted_annotation = sort_bed_file(annotation_bed, temp_dir)

    cmd = f"bedtools intersect -a {sorted_query} -b {sorted_annotation} -wa -wb"

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"bedtools error: {result.stderr}")
        return set()

    if not result.stdout.strip():
        return set()

    overlapping_ids = set()

    for line in result.stdout.strip().split('\n'):
        parts = line.split('\t')
        seq_id = parts[3]  # 4th column is the sequence ID

        if filter_pattern:
            # Check if any part of the annotation matches the pattern
            annotation_info = '\t'.join(parts[6:])  # Annotation columns
            if filter_pattern.lower() in annotation_info.lower():
                overlapping_ids.add(seq_id)
        else:
            overlapping_ids.add(seq_id)

    return overlapping_ids


def main():
    parser = argparse.ArgumentParser(
        description='Generate sequence ID lists based on repeat annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output files:
  sequences_with_repeats.txt - All sequences overlapping any repeat
  sequences_with_alu.txt     - Sequences overlapping Alu elements
  sequences_with_line.txt    - Sequences overlapping LINE elements
  sequences_no_repeats.txt   - Sequences with NO repeat overlap

Example:
  python generate_annotation_lists.py \\
      --sequences sequences.csv \\
      --repeatmasker ~/annotations/repeatmasker_hg38.bed \\
      --output annotation_lists/
        """
    )

    parser.add_argument('--sequences', '-s', required=True,
                        help='CSV file with sequences (must have chr, start, end columns)')
    parser.add_argument('--repeatmasker', '-r', required=True,
                        help='RepeatMasker BED file')
    parser.add_argument('--output', '-o', default='annotation_lists',
                        help='Output directory for sequence ID lists')

    args = parser.parse_args()

    # Check bedtools
    try:
        subprocess.run(['bedtools', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: bedtools not found. Please install bedtools.")
        return

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create temp directory
    temp_dir = tempfile.mkdtemp()

    # Load sequences
    print(f"Loading sequences from {args.sequences}...")
    seq_df = pd.read_csv(args.sequences)
    print(f"  Loaded {len(seq_df)} sequences")

    # Create BED file
    print("\nCreating BED file...")
    seq_bed, id_col = create_sequences_bed(seq_df, temp_dir)

    # Get all sequence IDs
    all_ids = set(seq_df[id_col].tolist())

    # Get sequences with ANY repeat overlap
    print("\nFinding sequences with repeat overlap...")
    ids_with_repeats = get_overlapping_ids(seq_bed, args.repeatmasker, temp_dir)
    print(f"  Found {len(ids_with_repeats)} sequences with repeat overlap ({len(ids_with_repeats) / len(all_ids) * 100:.1f}%)")

    # Get sequences with Alu overlap
    print("\nFinding sequences with Alu overlap...")
    ids_with_alu = get_overlapping_ids(seq_bed, args.repeatmasker, temp_dir, filter_pattern="Alu")
    print(f"  Found {len(ids_with_alu)} sequences with Alu overlap ({len(ids_with_alu) / len(all_ids) * 100:.1f}%)")

    # Get sequences with LINE overlap
    print("\nFinding sequences with LINE overlap...")
    ids_with_line = get_overlapping_ids(seq_bed, args.repeatmasker, temp_dir, filter_pattern="LINE")
    # Also check for L1, L2 which are LINE elements
    ids_with_l1 = get_overlapping_ids(seq_bed, args.repeatmasker, temp_dir, filter_pattern="L1")
    ids_with_l2 = get_overlapping_ids(seq_bed, args.repeatmasker, temp_dir, filter_pattern="L2")
    ids_with_line = ids_with_line | ids_with_l1 | ids_with_l2
    print(f"  Found {len(ids_with_line)} sequences with LINE overlap ({len(ids_with_line) / len(all_ids) * 100:.1f}%)")

    # Get sequences with SINE overlap
    print("\nFinding sequences with SINE overlap...")
    ids_with_sine = get_overlapping_ids(seq_bed, args.repeatmasker, temp_dir, filter_pattern="SINE")
    ids_with_mir = get_overlapping_ids(seq_bed, args.repeatmasker, temp_dir, filter_pattern="MIR")
    ids_with_sine = ids_with_sine | ids_with_mir | ids_with_alu  # Alu is a SINE
    print(f"  Found {len(ids_with_sine)} sequences with SINE overlap ({len(ids_with_sine) / len(all_ids) * 100:.1f}%)")

    # Get sequences with NO repeats
    ids_no_repeats = all_ids - ids_with_repeats
    print(f"\nSequences with NO repeat overlap: {len(ids_no_repeats)} ({len(ids_no_repeats) / len(all_ids) * 100:.1f}%)")

    # Save lists
    print("\nSaving sequence ID lists...")

    def save_list(ids: Set[str], filename: str):
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            for seq_id in sorted(ids):
                f.write(f"{seq_id}\n")
        print(f"  Saved {len(ids)} IDs to {filepath}")
        return filepath

    save_list(ids_with_repeats, "sequences_with_repeats.txt")
    save_list(ids_with_alu, "sequences_with_alu.txt")
    save_list(ids_with_line, "sequences_with_line.txt")
    save_list(ids_with_sine, "sequences_with_sine.txt")
    save_list(ids_no_repeats, "sequences_no_repeats.txt")

    # Also save summary CSV
    summary_df = pd.DataFrame({
        'sequence_id': list(all_ids),
        'has_repeat': [id in ids_with_repeats for id in all_ids],
        'has_alu': [id in ids_with_alu for id in all_ids],
        'has_line': [id in ids_with_line for id in all_ids],
        'has_sine': [id in ids_with_sine for id in all_ids],
    })
    summary_path = output_dir / "annotation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved summary to {summary_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total sequences: {len(all_ids)}")
    print(f"With ANY repeat: {len(ids_with_repeats)} ({len(ids_with_repeats) / len(all_ids) * 100:.1f}%)")
    print(f"With Alu:        {len(ids_with_alu)} ({len(ids_with_alu) / len(all_ids) * 100:.1f}%)")
    print(f"With LINE:       {len(ids_with_line)} ({len(ids_with_line) / len(all_ids) * 100:.1f}%)")
    print(f"With SINE:       {len(ids_with_sine)} ({len(ids_with_sine) / len(all_ids) * 100:.1f}%)")
    print(f"NO repeats:      {len(ids_no_repeats)} ({len(ids_no_repeats) / len(all_ids) * 100:.1f}%)")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
