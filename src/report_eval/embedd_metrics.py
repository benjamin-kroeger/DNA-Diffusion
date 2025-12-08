#!/usr/bin/env python3
"""
Script 3: Embedding Space Evaluation - GT vs Synthetic

Metrics: MMD, Coverage/Density/Precision, KDE ratio maps
"""

import numpy as np
import pandas as pd
import h5py
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import argparse
import json
import warnings
from tqdm import tqdm
from scipy import linalg
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


@dataclass
class EvalConfig:
    n_neighbors: int = 25
    min_dist: float = 0.1
    metric: str = 'euclidean'
    n_components: int = 2
    random_state: int = 42
    kde_bandwidth: str = 'scott'
    kde_grid_resolution: int = 200
    kde_threshold: float = 1.5
    mmd_kernel: str = 'rbf'
    mmd_gamma: Optional[float] = None
    mmd_n_permutations: int = 1000
    mmd_subsample: Optional[int] = 5000


@dataclass
class DatasetConfig:
    name: str
    filepath: str
    n_samples: Optional[int] = None
    is_reference: bool = False


class EmbeddingLoader:
    @staticmethod
    def load_h5_embeddings(filepath: str, n_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with h5py.File(filepath, 'r') as f:
            top_keys = list(f.keys())
            print(f"  Available keys: {top_keys[:5]}...")
            first_key = top_keys[0] if top_keys else None
            if first_key and isinstance(f[first_key], h5py.Group) and 'embedding' in f[first_key]:
                return EmbeddingLoader._load_id_based_h5(f, n_samples)
            return EmbeddingLoader._load_array_based_h5(f, n_samples)

    @staticmethod
    def _load_id_based_h5(f, n_samples):
        ids = list(f.keys())
        if n_samples and n_samples < len(ids):
            idx = np.random.choice(len(ids), n_samples, replace=False)
            ids = [ids[i] for i in sorted(idx)]
        emb_shape = f[ids[0]]['embedding'][:].shape
        embeddings = np.zeros((len(ids),) + emb_shape, dtype='float32')
        labels = []
        for i, sid in enumerate(tqdm(ids, desc="Loading")):
            grp = f[sid]
            embeddings[i] = grp['embedding'][:]
            lbl = grp.attrs.get('TAG', grp.attrs.get('tag', grp.attrs.get('cell_type', b'unknown')))
            labels.append(lbl.decode() if isinstance(lbl, bytes) else str(lbl))
        return embeddings, np.array(labels), np.array(ids)

    @staticmethod
    def _load_array_based_h5(f, n_samples):
        for key in ['embeddings', 'X', 'embedding', 'data']:
            if key in f and isinstance(f[key], h5py.Dataset):
                embeddings = f[key][:]
                break
        else:
            embeddings = f[list(f.keys())[0]][:]
        labels = np.array(['unknown'] * len(embeddings))
        ids = np.array([f'seq_{i}' for i in range(len(embeddings))])
        if n_samples and n_samples < len(embeddings):
            idx = np.random.choice(len(embeddings), n_samples, replace=False)
            embeddings, labels, ids = embeddings[idx], labels[idx], ids[idx]
        return embeddings, labels, ids


class DistributionalMetrics:

    @staticmethod
    def compute_mmd(gt, synth, kernel='rbf', gamma=None, n_perms=1000, subsample=None):
        print("\nComputing MMD...")
        if subsample:
            if len(gt) > subsample: gt = gt[np.random.choice(len(gt), subsample, replace=False)]
            if len(synth) > subsample: synth = synth[np.random.choice(len(synth), subsample, replace=False)]
        n_gt, n_synth = len(gt), len(synth)
        if gamma is None:
            comb = np.vstack([gt, synth])
            samp = comb[np.random.choice(len(comb), min(1000, len(comb)), replace=False)]
            dists = np.sqrt(((samp[:, None] - samp[None, :]) ** 2).sum(2))
            gamma = 1.0 / (2 * np.median(dists[np.triu_indices(len(samp), k=1)]) ** 2)
        kern = lambda X, Y: rbf_kernel(X, Y, gamma=gamma)
        K_gg, K_ss, K_gs = kern(gt, gt), kern(synth, synth), kern(gt, synth)
        np.fill_diagonal(K_gg, 0);
        np.fill_diagonal(K_ss, 0)
        mmd2 = K_gg.sum() / (n_gt * (n_gt - 1)) + K_ss.sum() / (n_synth * (n_synth - 1)) - 2 * K_gs.mean()
        mmd = np.sqrt(max(0, mmd2))
        comb = np.vstack([gt, synth])
        K_comb = kern(comb, comb);
        np.fill_diagonal(K_comb, 0)
        null = []
        for _ in tqdm(range(n_perms), desc="  Permutation"):
            p = np.random.permutation(len(comb))
            gi, si = p[:n_gt], p[n_gt:]
            m2 = K_comb[np.ix_(gi, gi)].sum() / (n_gt * (n_gt - 1)) + K_comb[np.ix_(si, si)].sum() / (n_synth * (n_synth - 1)) - 2 * K_comb[
                np.ix_(gi, si)].mean()
            null.append(np.sqrt(max(0, m2)))
        pval = np.mean(np.array(null) >= mmd)
        print(f"  MMD: {mmd:.6f}, p-value: {pval:.4f}")
        return {'mmd': float(mmd), 'p_value': float(pval), 'kernel': kernel}

    @staticmethod
    def compute_coverage_density(gt, synth, k=5):
        print("\nComputing Coverage/Density...")
        nn = NearestNeighbors(n_neighbors=k).fit(gt)
        radii = nn.kneighbors(gt)[0][:, -1]
        nn_s = NearestNeighbors().fit(synth)
        cov = sum(1 for i, r in enumerate(radii) if len(nn_s.radius_neighbors([gt[i]], r, return_distance=False)[0]) > 0)
        coverage = cov / len(gt)
        nn_gt = NearestNeighbors(n_neighbors=1).fit(gt)
        precision = np.mean(nn_gt.kneighbors(synth)[0][:, 0] <= np.median(radii))
        print(f"  Coverage: {coverage:.2%}, Precision: {precision:.2%}")
        return {'coverage': float(coverage), 'precision': float(precision), 'k': k}


class KDEAnalysis:
    @staticmethod
    def compute_kde_ratio(gt, synth, resolution=200, bandwidth='scott'):
        print("\nComputing KDE...")
        pad = 1
        xmin, xmax = min(gt[:, 0].min(), synth[:, 0].min()) - pad, max(gt[:, 0].max(), synth[:, 0].max()) + pad
        ymin, ymax = min(gt[:, 1].min(), synth[:, 1].min()) - pad, max(gt[:, 1].max(), synth[:, 1].max()) + pad
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, resolution), np.linspace(ymin, ymax, resolution))
        grid = np.vstack([xx.ravel(), yy.ravel()])
        kde_gt = gaussian_kde(gt.T, bw_method=bandwidth)
        kde_synth = gaussian_kde(synth.T, bw_method=bandwidth)
        d_gt = kde_gt(grid).reshape(xx.shape)
        d_synth = kde_synth(grid).reshape(xx.shape)
        eps = 1e-10
        log_ratio = np.log10((d_synth + eps) / (d_gt + eps))
        return {'xx': xx, 'yy': yy, 'density_gt': d_gt, 'density_synth': d_synth, 'log_ratio': log_ratio}

    @staticmethod
    def identify_problem_regions(kde, threshold=1.5, floor_pct=5.0):
        combined = np.maximum(kde['density_gt'], kde['density_synth'])
        floor = np.percentile(combined[combined > 0], floor_pct)
        meaningful = combined > floor
        over = (kde['log_ratio'] > threshold) & meaningful
        under = (kde['log_ratio'] < -threshold) & meaningful
        well = meaningful & ~over & ~under
        area = meaningful.sum()
        return {
            'meaningful_region': meaningful, 'over_represented': over, 'under_represented': under, 'well_matched': well,
            'over_area_fraction': over.sum() / area if area else 0,
            'under_area_fraction': under.sum() / area if area else 0,
            'well_matched_fraction': well.sum() / area if area else 0,
            'meaningful_area_fraction': area / kde['log_ratio'].size,
        }


class EmbeddingSpaceEvaluator:
    def __init__(self, config=None, output_dir='eval_results'):
        self.config = config or EvalConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.colors = {'gt': '#2ecc71', 'synth': '#e74c3c'}

    def average_pool(self, emb):
        return emb.mean(axis=1) if len(emb.shape) == 3 else emb

    def load_dataset(self, cfg):
        if Path(cfg.filepath).suffix == '.h5':
            return EmbeddingLoader.load_h5_embeddings(cfg.filepath, cfg.n_samples)
        emb = np.load(cfg.filepath)
        if cfg.n_samples and cfg.n_samples < len(emb):
            emb = emb[np.random.choice(len(emb), cfg.n_samples, replace=False)]
        return emb, np.array(['unknown'] * len(emb)), np.array([f'seq_{i}' for i in range(len(emb))])

    def run_evaluation(self, gt_cfg, synth_cfg, skip_mmd=False):
        np.random.seed(self.config.random_state)
        print(f"\nLoading GT: {gt_cfg.name}")
        gt_emb, gt_labels, gt_ids = self.load_dataset(gt_cfg)
        gt = self.average_pool(gt_emb)
        print(f"\nLoading Synth: {synth_cfg.name}")
        synth_emb, synth_labels, synth_ids = self.load_dataset(synth_cfg)
        synth = self.average_pool(synth_emb)
        print(f"\nShapes: GT {gt.shape}, Synth {synth.shape}")

        mmd = {'mmd': None, 'p_value': None, 'skipped': True} if skip_mmd else DistributionalMetrics.compute_mmd(
            gt, synth, self.config.mmd_kernel, self.config.mmd_gamma, self.config.mmd_n_permutations, self.config.mmd_subsample)
        cov = DistributionalMetrics.compute_coverage_density(gt, synth)

        print("\nFitting UMAP...")
        reducer = umap.UMAP(n_neighbors=self.config.n_neighbors, min_dist=self.config.min_dist,
                            random_state=self.config.random_state, verbose=True)
        combined = np.vstack([gt, synth])
        umap_coords = reducer.fit_transform(combined)
        gt_umap, synth_umap = umap_coords[:len(gt)], umap_coords[len(gt):]

        kde = KDEAnalysis.compute_kde_ratio(gt_umap, synth_umap, self.config.kde_grid_resolution)
        problem = KDEAnalysis.identify_problem_regions(kde, self.config.kde_threshold)

        return {
            'gt_umap': gt_umap, 'synth_umap': synth_umap, 'gt_labels': gt_labels, 'synth_labels': synth_labels,
            'gt_ids': gt_ids, 'synth_ids': synth_ids, 'mmd': mmd, 'coverage': cov,
            'kde': kde, 'problem_regions': problem, 'gt_embeddings_pooled': gt, 'synth_embeddings_pooled': synth,
        }

    def plot_umap_gt_vs_synth(self, results, name='umap_gt_vs_synth'):
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(results['gt_umap'][:, 0], results['gt_umap'][:, 1], s=8, alpha=0.5, c=self.colors['gt'], label=f"GT (n={len(results['gt_umap'])})",
                   rasterized=True)
        ax.scatter(results['synth_umap'][:, 0], results['synth_umap'][:, 1], s=8, alpha=0.5, c=self.colors['synth'],
                   label=f"Synth (n={len(results['synth_umap'])})", rasterized=True)
        ax.set_xlabel('UMAP 1');
        ax.set_ylabel('UMAP 2');
        ax.legend(markerscale=3);
        ax.grid(alpha=0.3)
        mmd_str = 'skipped' if results['mmd'].get('skipped') else f"{results['mmd']['mmd']:.4f}"
        ax.text(0.02, 0.98, f"MMD: {mmd_str}\nCoverage: {results['coverage']['coverage']:.1%}",
                transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        path = self.output_dir / f"{name}.png";
        plt.savefig(path);
        plt.close();
        return str(path)

    def plot_umap_by_cell_type(self, results, name='umap_by_cell_type'):
        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        unique = np.unique(np.concatenate([results['gt_labels'], results['synth_labels']]))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique)))
        cmap = {ct: colors[i] for i, ct in enumerate(unique)}
        for ax, umap_c, labels, title in [(axes[0], results['gt_umap'], results['gt_labels'], 'GT'),
                                          (axes[1], results['synth_umap'], results['synth_labels'], 'Synth')]:
            for ct in unique:
                m = labels == ct
                if m.sum(): ax.scatter(umap_c[m, 0], umap_c[m, 1], s=10, alpha=0.6, c=[cmap[ct]], label=f'{ct} ({m.sum()})', rasterized=True)
            ax.set_xlabel('UMAP 1');
            ax.set_ylabel('UMAP 2');
            ax.set_title(title);
            ax.legend(fontsize=8);
            ax.grid(alpha=0.3)
        plt.tight_layout();
        path = self.output_dir / f"{name}.png";
        plt.savefig(path);
        plt.close();
        return str(path)

    def plot_kde_ratio_map(self, results, name='kde_ratio_map'):
        kde, prob = results['kde'], results['problem_regions']
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # A: GT density
        ax = axes[0, 0]
        im = ax.contourf(kde['xx'], kde['yy'], kde['density_gt'], levels=50, cmap='Greens')
        ax.scatter(results['gt_umap'][:, 0], results['gt_umap'][:, 1], s=1, alpha=0.1, c='black', rasterized=True)
        ax.set_xlabel('UMAP 1');
        ax.set_ylabel('UMAP 2');
        ax.set_title('A. Ground Truth Density')
        plt.colorbar(im, ax=ax, label='Density')

        # B: Synth density
        ax = axes[0, 1]
        im = ax.contourf(kde['xx'], kde['yy'], kde['density_synth'], levels=50, cmap='Reds')
        ax.scatter(results['synth_umap'][:, 0], results['synth_umap'][:, 1], s=1, alpha=0.1, c='black', rasterized=True)
        ax.set_xlabel('UMAP 1');
        ax.set_ylabel('UMAP 2');
        ax.set_title('B. Synthetic Density')
        plt.colorbar(im, ax=ax, label='Density')

        # C: Log ratio
        ax = axes[1, 0]
        masked = np.ma.masked_where(~prob['meaningful_region'], kde['log_ratio'])
        vmax = min(3, np.abs(kde['log_ratio'][prob['meaningful_region']]).max())
        im = ax.contourf(kde['xx'], kde['yy'], masked, levels=np.linspace(-vmax, vmax, 51), cmap='RdBu_r', extend='both')
        ax.contour(kde['xx'], kde['yy'], kde['log_ratio'], levels=[-self.config.kde_threshold, self.config.kde_threshold], colors='black',
                   linewidths=2, linestyles='--')
        ax.contour(kde['xx'], kde['yy'], prob['meaningful_region'].astype(float), levels=[0.5], colors='gray', linewidths=1.5)
        ax.set_xlabel('UMAP 1');
        ax.set_ylabel('UMAP 2');
        ax.set_title(f'C. Log₁₀(Synth/GT) Ratio\n(dashed = ±{self.config.kde_threshold})')
        plt.colorbar(im, ax=ax, label='Log₁₀ Ratio')

        # D: Contour overlay
        ax = axes[1, 1]
        ax.contour(kde['xx'], kde['yy'], kde['density_gt'], levels=5, colors='green', linewidths=1.5, alpha=0.7)
        ax.contour(kde['xx'], kde['yy'], kde['density_synth'], levels=5, colors='red', linewidths=1.5, alpha=0.7)
        ax.scatter(results['gt_umap'][:, 0], results['gt_umap'][:, 1], s=3, alpha=0.3, c='green', rasterized=True)
        ax.scatter(results['synth_umap'][:, 0], results['synth_umap'][:, 1], s=3, alpha=0.3, c='red', rasterized=True)
        ax.set_xlabel('UMAP 1');
        ax.set_ylabel('UMAP 2');
        ax.set_title('D. Distribution Overlap')
        ax.legend(handles=[Line2D([0], [0], color='green', lw=2, label='GT'), Line2D([0], [0], color='red', lw=2, label='Synth')], loc='upper right')

        plt.tight_layout();
        path = self.output_dir / f"{name}.png";
        plt.savefig(path);
        plt.close();
        return str(path)



    def plot_per_cell_type_metrics(self, results, name='per_cell_type_metrics'):
        gt_counts = pd.Series(results['gt_labels']).value_counts()
        synth_counts = pd.Series(results['synth_labels']).value_counts()
        all_types = sorted(set(gt_counts.index) | set(synth_counts.index))
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        x = np.arange(len(all_types));
        w = 0.35
        axes[0].bar(x - w / 2, [gt_counts.get(t, 0) for t in all_types], w, label='GT', color=self.colors['gt'])
        axes[0].bar(x + w / 2, [synth_counts.get(t, 0) for t in all_types], w, label='Synth', color=self.colors['synth'])
        axes[0].set_xticks(x);
        axes[0].set_xticklabels(all_types, rotation=45, ha='right');
        axes[0].legend();
        axes[0].set_title('Counts')
        gt_p = np.array([gt_counts.get(t, 0) for t in all_types]) / sum(gt_counts)
        synth_p = np.array([synth_counts.get(t, 0) for t in all_types]) / sum(synth_counts)
        axes[1].bar(x - w / 2, gt_p, w, label='GT', color=self.colors['gt'])
        axes[1].bar(x + w / 2, synth_p, w, label='Synth', color=self.colors['synth'])
        axes[1].set_xticks(x);
        axes[1].set_xticklabels(all_types, rotation=45, ha='right');
        axes[1].legend();
        axes[1].set_title('Proportions')
        plt.tight_layout();
        path = self.output_dir / f"{name}.png";
        plt.savefig(path);
        plt.close();
        return str(path)

    def export_metrics(self, results, name='metrics'):
        metrics = { 'mmd': results['mmd'], 'coverage': results['coverage'],
                   'problem_regions': {k: float(v) if isinstance(v, (int, float, np.floating)) else None
                                       for k, v in results['problem_regions'].items() if not isinstance(v, np.ndarray)}}
        path = self.output_dir / f"{name}.json"
        with open(path, 'w') as f: json.dump(metrics, f, indent=2)
        return str(path)

    def export_umap_coords(self, results, name='umap_coordinates'):
        np.save(self.output_dir / f"{name}_gt.npy", results['gt_umap'])
        np.save(self.output_dir / f"{name}_synth.npy", results['synth_umap'])
        df = pd.concat([
            pd.DataFrame({'id': results['gt_ids'], 'cell_type': results['gt_labels'], 'source': 'gt', 'umap1': results['gt_umap'][:, 0],
                          'umap2': results['gt_umap'][:, 1]}),
            pd.DataFrame({'id': results['synth_ids'], 'cell_type': results['synth_labels'], 'source': 'synth', 'umap1': results['synth_umap'][:, 0],
                          'umap2': results['synth_umap'][:, 1]})
        ])
        path = self.output_dir / f"{name}.csv";
        df.to_csv(path, index=False);
        return str(path)


def main():
    parser = argparse.ArgumentParser(description='Embedding Space Evaluation')
    parser.add_argument('--gt', '-g', required=True, help='GT as name:path[:n]')
    parser.add_argument('--synth', '-s', required=True, help='Synth as name:path[:n]')
    parser.add_argument('--output', '-o', default='eval_results')
    parser.add_argument('--n-neighbors', type=int, default=25)
    parser.add_argument('--min-dist', type=float, default=0.1)
    parser.add_argument('--kde-resolution', type=int, default=200)
    parser.add_argument('--kde-threshold', type=float, default=1.5)
    parser.add_argument('--mmd-kernel', choices=['rbf', 'polynomial', 'multiscale'], default='rbf')
    parser.add_argument('--mmd-permutations', type=int, default=500)
    parser.add_argument('--mmd-subsample', type=int, default=5000)
    parser.add_argument('--skip-mmd', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    def parse(s):
        p = s.split(':')
        return DatasetConfig(p[0], p[1], int(p[2]) if len(p) > 2 else None)

    config = EvalConfig(n_neighbors=args.n_neighbors, min_dist=args.min_dist, random_state=args.seed,
                        kde_grid_resolution=args.kde_resolution, kde_threshold=args.kde_threshold,
                        mmd_kernel=args.mmd_kernel, mmd_n_permutations=args.mmd_permutations,
                        mmd_subsample=args.mmd_subsample if args.mmd_subsample > 0 else None)

    evaluator = EmbeddingSpaceEvaluator(config, args.output)
    results = evaluator.run_evaluation(parse(args.gt), parse(args.synth), args.skip_mmd)

    print("\n" + "=" * 60 + "\nGENERATING VISUALIZATIONS\n" + "=" * 60)
    figs = {
        'umap_gt_vs_synth': evaluator.plot_umap_gt_vs_synth(results),
        'umap_by_cell_type': evaluator.plot_umap_by_cell_type(results),
        'kde_ratio_map': evaluator.plot_kde_ratio_map(results),
        'per_cell_type': evaluator.plot_per_cell_type_metrics(results),
    }
    metrics_file = evaluator.export_metrics(results)
    coords_file = evaluator.export_umap_coords(results)

    print("\n" + "=" * 60 + "\nEVALUATION COMPLETE\n" + "=" * 60)
    if not results['mmd'].get('skipped'):
        print(f"MMD: {results['mmd']['mmd']:.6f} (p={results['mmd']['p_value']:.4f})")
    print(f"Coverage: {results['coverage']['coverage']:.2%}")
    print(f"Precision: {results['coverage']['precision']:.2%}")
    print(f"\nProblem Regions:")
    print(f"  Under-represented: {results['problem_regions']['under_area_fraction'] * 100:.1f}%")
    print(f"  Over-represented: {results['problem_regions']['over_area_fraction'] * 100:.1f}%")
    print(f"\nGenerated: {list(figs.keys())}")
    print(f"Metrics: {metrics_file}")
    print(f"Coords: {coords_file}")


if __name__ == "__main__":
    main()
