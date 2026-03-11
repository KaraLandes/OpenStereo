"""
Evaluate LightStereo models on SceneFlow dataset with disparity-based binning.

Compares two checkpoints:
- Our trained model (max_disp_300_90ep)
- Original developer checkpoint (original)

Bins images by max disparity quartiles (per subset) and generates:
- Visualizations (GT | Predicted | Error map)
- CSV stats (EPE mean, max, median, std)
"""

import sys
import logging
import random
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from easydict import EasyDict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dai.config import load_config
from dai.datasets import DatasetSplitter, SceneFlowDataset, build_transforms
from stereo.modeling.models.lightstereo.lightstereo import LightStereo

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Configuration
SPLITTER_SEED = 42  # Same as training
SAMPLING_SEED = 28  # For random sampling of 1000 images
SAMPLES_PER_SUBSET = 200
VIZ_DPI = 120  # Low DPI for smaller file sizes

CHECKPOINTS = {
    'max_disp_300_90ep': 'src_data/lightstereo_training/training_classic/max_disp_300_90ep/checkpoint_epoch_50.pth',
    'original': 'src_data/checkpoints/lightstereo/LightStereo-S-SceneFlow-General.pth',
}

OUTPUT_DIR = Path('src_data/evaluation/light_stereo_inference')
CACHE_FILE = Path('src_data/evaluation/max_disparity_cache.json')

BIN_NAMES = ['below_q1', 'q1_q2', 'q2_q3', 'above_q3']
NUM_WORKERS = 12  # For parallel max disparity computation


@dataclass
class SampleInfo:
    """Information about a single sample"""
    sample: Dict[str, Any]
    max_disparity: float
    subset: str
    split: str
    bin_name: str = None


def read_pfm(file_path: str) -> np.ndarray:
    """Read PFM disparity file"""
    with open(file_path, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise ValueError(f'Invalid PFM file: {file_path}')
        
        dim_match = f.readline().decode('utf-8')
        width, height = map(int, dim_match.split())
        
        scale = float(f.readline().decode('utf-8').rstrip())
        endian = '<' if scale < 0 else '>'
        
        data = np.fromfile(f, endian + 'f')
        
        if color:
            shape = (height, width, 3)
        else:
            shape = (height, width)
        
        data = np.reshape(data, shape)
        data = np.flipud(data)
        
        return data


def compute_max_disparity(disp_path: str) -> float:
    """Compute max disparity from a disparity file"""
    disp = read_pfm(disp_path)
    # Only consider valid disparities (positive and < 512)
    valid_mask = (disp > 0) & (disp < 512)
    if valid_mask.sum() > 0:
        return float(disp[valid_mask].max())
    return 0.0


def compute_max_disparity_worker(disp_path: str) -> Tuple[str, float]:
    """Worker function for parallel max disparity computation"""
    try:
        max_disp = compute_max_disparity(disp_path)
        return (disp_path, max_disp)
    except Exception as e:
        return (disp_path, 0.0)


def load_or_compute_max_disparities(samples: List[Dict], cache_file: Path) -> Dict[str, float]:
    """Load cached max disparities or compute them in parallel"""
    # Try to load cache
    cache = {}
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            logger.info(f"Loaded {len(cache)} cached max disparity values")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            cache = {}
    
    # Find samples that need computation
    disp_paths_to_compute = []
    for sample in samples:
        disp_path = sample.get('disparity')
        if disp_path and disp_path not in cache:
            disp_paths_to_compute.append(disp_path)
    
    if disp_paths_to_compute:
        logger.info(f"Computing max disparity for {len(disp_paths_to_compute)} samples using {NUM_WORKERS} threads...")
        
        results = []
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(compute_max_disparity_worker, p): p for p in disp_paths_to_compute}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Computing max disparity"):
                results.append(future.result())
        
        # Update cache
        for disp_path, max_disp in results:
            cache[disp_path] = max_disp
        
        # Save cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(cache, f)
        logger.info(f"Saved cache with {len(cache)} entries")
    
    return cache


def compute_quartiles(values: List[float]) -> Tuple[float, float, float]:
    """Compute Q1, Q2 (median), Q3"""
    arr = np.array(values)
    q1 = np.percentile(arr, 25)
    q2 = np.percentile(arr, 50)
    q3 = np.percentile(arr, 75)
    return q1, q2, q3


def assign_bin(max_disp: float, q1: float, q2: float, q3: float) -> str:
    """Assign a sample to a disparity bin"""
    if max_disp < q1:
        return 'below_q1'
    elif max_disp < q2:
        return 'q1_q2'
    elif max_disp < q3:
        return 'q2_q3'
    else:
        return 'above_q3'


def load_model(checkpoint_path: str, model_config: Dict, device: torch.device) -> torch.nn.Module:
    """Load LightStereo model from checkpoint"""
    cfg = EasyDict({
        'MAX_DISP': model_config.get('max_disp', 300),
        'LEFT_ATT': model_config.get('left_att', True),
        'AGGREGATION_BLOCKS': model_config.get('aggregation_blocks', [1, 2, 4]),
        'EXPANSE_RATIO': model_config.get('expanse_ratio', 4),
        'BACKCONE': model_config.get('backbone', 'MobileNetv2'),
    })
    
    model = LightStereo(cfg)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    return model


def compute_epe(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray) -> float:
    """Compute End-Point-Error"""
    error = np.abs(pred - gt)
    error_masked = error * valid_mask
    if valid_mask.sum() > 0:
        return float(error_masked.sum() / valid_mask.sum())
    return 0.0


def generate_visualization(
    gt_disp: np.ndarray,
    pred_disp: np.ndarray,
    valid_mask: np.ndarray,
    epe: float,
    save_path: Path,
    sample_name: str,
    vmax: float = 300.0
):
    """Generate and save visualization (GT | Predicted | Error)"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # GT disparity
    im0 = axes[0].imshow(gt_disp, cmap='turbo', vmin=0, vmax=vmax)
    axes[0].set_title('GT Disparity', fontsize=10)
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Predicted disparity
    im1 = axes[1].imshow(pred_disp, cmap='turbo', vmin=0, vmax=vmax)
    axes[1].set_title('Predicted Disparity', fontsize=10)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Error map
    error = np.abs(pred_disp - gt_disp) * valid_mask
    im2 = axes[2].imshow(error, cmap='hot', vmin=0, vmax=10)
    axes[2].set_title('Error Map', fontsize=10)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    fig.suptitle(f'{sample_name} - EPE: {epe:.2f}px', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=VIZ_DPI, bbox_inches='tight')
    plt.close(fig)


def write_stats_csv(stats: Dict[str, List[float]], save_path: Path, quartile_info: Dict[str, Tuple[float, float]]):
    """Write statistics to CSV file"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['bin', 'num_samples', 'epe_mean', 'epe_max', 'epe_median', 'epe_std', 'disparity_q_low', 'disparity_q_high'])
        
        for bin_name in BIN_NAMES:
            if bin_name in stats and len(stats[bin_name]) > 0:
                epe_values = stats[bin_name]
                q_low, q_high = quartile_info.get(bin_name, (0, 0))
                writer.writerow([
                    bin_name,
                    len(epe_values),
                    f'{np.mean(epe_values):.4f}',
                    f'{np.max(epe_values):.4f}',
                    f'{np.median(epe_values):.4f}',
                    f'{np.std(epe_values):.4f}',
                    f'{q_low:.2f}',
                    f'{q_high:.2f}'
                ])


def write_summary_csv(all_stats: Dict[str, Dict[str, List[float]]], save_path: Path):
    """Write summary statistics across all subsets and splits"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'subset', 'bin', 'num_samples', 'epe_mean', 'epe_max', 'epe_median', 'epe_std'])
        
        for key, stats in all_stats.items():
            split, subset = key.split('/')
            for bin_name in BIN_NAMES:
                if bin_name in stats and len(stats[bin_name]) > 0:
                    epe_values = stats[bin_name]
                    writer.writerow([
                        split,
                        subset,
                        bin_name,
                        len(epe_values),
                        f'{np.mean(epe_values):.4f}',
                        f'{np.max(epe_values):.4f}',
                        f'{np.median(epe_values):.4f}',
                        f'{np.std(epe_values):.4f}'
                    ])


class DisparityBinEvaluator:
    """Main evaluator class"""
    
    def __init__(self, config_path: str, device: str = 'cuda'):
        self.config = load_config(config_path)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Override splitter seed to match training
        self.config['splitter']['seed'] = SPLITTER_SEED
        
        self.splitter = DatasetSplitter(self.config['splitter'])
        self.transforms = build_transforms(self.config.get('transforms', {}), 'val')
        
        # Storage for samples organized by subset and split
        self.samples_by_subset: Dict[str, Dict[str, List[SampleInfo]]] = defaultdict(lambda: defaultdict(list))
        self.global_quartiles: Tuple[float, float, float] = (0, 0, 0)
        self.global_max_disparity: float = 300.0  # Will be computed from data
        
        # Sampled samples (same for both models)
        self.sampled_samples: Dict[str, Dict[str, List[SampleInfo]]] = defaultdict(lambda: defaultdict(list))
    
    def discover_and_split_dataset(self):
        """Discover SceneFlow samples and split them - compute max disparity on FULL dataset for accurate quartiles"""
        logger.info("Discovering and splitting SceneFlow dataset...")
        
        # Collect all samples first
        all_samples = []
        sample_metadata = []  # (split_name, subset) for each sample
        
        for dataset_config in self.config['datasets']:
            if dataset_config['name'] != 'sceneflow':
                continue
            
            root = Path(dataset_config['root'])
            splits = self.splitter.split_dataset('sceneflow', root, dataset_config)
            
            for split_name, samples in splits.items():
                if not samples:
                    continue
                
                logger.info(f"Found {split_name} split with {len(samples)} samples")
                
                for sample in samples:
                    if sample['disparity'] is None:
                        continue
                    all_samples.append(sample)
                    sample_metadata.append((split_name, sample['metadata']['subset']))
        
        # Compute max disparities for ALL samples (cached for future runs)
        logger.info(f"Computing max disparity for FULL dataset ({len(all_samples)} samples)...")
        max_disp_cache = load_or_compute_max_disparities(all_samples, CACHE_FILE)
        
        # Organize all samples with max disparity
        for idx, sample in enumerate(all_samples):
            split_name, subset = sample_metadata[idx]
            disp_path = sample['disparity']
            max_disp = max_disp_cache.get(disp_path, 0.0)
            
            sample_info = SampleInfo(
                sample=sample,
                max_disparity=max_disp,
                subset=subset,
                split=split_name
            )
            
            self.samples_by_subset[subset][split_name].append(sample_info)
        
        # Log sample counts
        for subset in self.samples_by_subset:
            for split in self.samples_by_subset[subset]:
                count = len(self.samples_by_subset[subset][split])
                logger.info(f"{subset}/{split}: {count} samples")
    
    def compute_quartiles_global(self):
        """Compute GLOBAL disparity quartiles across all subsets"""
        logger.info("Computing GLOBAL disparity quartiles (across all subsets)...")
        
        # Collect all max disparities from all subsets and splits
        all_max_disps = []
        for subset in self.samples_by_subset:
            for split in self.samples_by_subset[subset]:
                for sample_info in self.samples_by_subset[subset][split]:
                    all_max_disps.append(sample_info.max_disparity)
        
        if len(all_max_disps) > 0:
            q1, q2, q3 = compute_quartiles(all_max_disps)
            self.global_quartiles = (q1, q2, q3)
            self.global_max_disparity = max(all_max_disps)
            logger.info(f"GLOBAL quartiles: Q1={q1:.2f}, Q2={q2:.2f}, Q3={q3:.2f}")
            logger.info(f"GLOBAL max disparity (for colorbar): {self.global_max_disparity:.2f}")
            logger.info(f"  below_q1: max_disp < {q1:.2f}")
            logger.info(f"  q1_q2: {q1:.2f} <= max_disp < {q2:.2f}")
            logger.info(f"  q2_q3: {q2:.2f} <= max_disp < {q3:.2f}")
            logger.info(f"  above_q3: max_disp >= {q3:.2f}")
            
            # Assign bins to ALL samples using global quartiles
            for subset in self.samples_by_subset:
                for split in self.samples_by_subset[subset]:
                    for sample_info in self.samples_by_subset[subset][split]:
                        sample_info.bin_name = assign_bin(sample_info.max_disparity, q1, q2, q3)
    
    def sample_images(self):
        """Randomly sample 1000 images per subset (same for both models)"""
        logger.info(f"Sampling {SAMPLES_PER_SUBSET} images per subset (seed={SAMPLING_SEED})...")
        
        random.seed(SAMPLING_SEED)
        
        for subset in self.samples_by_subset:
            for split in self.samples_by_subset[subset]:
                samples = self.samples_by_subset[subset][split]
                
                if len(samples) <= SAMPLES_PER_SUBSET:
                    sampled = samples
                else:
                    sampled = random.sample(samples, SAMPLES_PER_SUBSET)
                
                self.sampled_samples[subset][split] = sampled
                logger.info(f"Sampled {len(sampled)} images from {subset}/{split}")
    
    def get_quartile_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get disparity ranges for each bin (global quartiles)"""
        q1, q2, q3 = self.global_quartiles
        return {
            'below_q1': (0, q1),
            'q1_q2': (q1, q2),
            'q2_q3': (q2, q3),
            'above_q3': (q3, float('inf'))
        }
    
    def run_inference(self, model_name: str, checkpoint_path: str):
        """Run inference with a model on all sampled images"""
        logger.info(f"Running inference with {model_name}...")
        
        model = load_model(checkpoint_path, self.config.get('model', {}), self.device)
        
        output_dir = OUTPUT_DIR / model_name
        all_stats: Dict[str, Dict[str, List[float]]] = {}
        
        for subset in self.sampled_samples:
            quartile_ranges = self.get_quartile_ranges()
            
            for split in self.sampled_samples[subset]:
                samples = self.sampled_samples[subset][split]
                
                if not samples:
                    continue
                
                logger.info(f"Processing {subset}/{split} ({len(samples)} samples)...")
                
                # Stats per bin
                bin_stats: Dict[str, List[float]] = defaultdict(list)
                
                for sample_info in tqdm(samples, desc=f"{model_name}/{subset}/{split}"):
                    sample = sample_info.sample
                    bin_name = sample_info.bin_name
                    
                    # Load images and disparity
                    left_img = self._load_image(sample['left'])
                    right_img = self._load_image(sample['right'])
                    gt_disp = read_pfm(sample['disparity']).astype(np.float32)
                    
                    # Compute valid mask
                    valid_mask = (gt_disp > 0) & (gt_disp < 512)
                    
                    # Prepare batch
                    batch = self._prepare_batch(left_img, right_img, gt_disp, valid_mask)
                    
                    # Run inference
                    with torch.no_grad():
                        output = model(batch)
                    
                    pred_disp = output['disp_pred'][0].cpu().numpy().squeeze()
                    
                    # Handle padding (if transforms added padding)
                    orig_h, orig_w = gt_disp.shape[:2]
                    pred_disp = pred_disp[:orig_h, :orig_w]
                    
                    # Compute EPE
                    epe = compute_epe(pred_disp, gt_disp, valid_mask.astype(float))
                    bin_stats[bin_name].append(epe)
                    
                    # Generate visualization
                    scene = sample['metadata']['scene'].replace('/', '_')
                    frame_id = sample['metadata']['frame_id']
                    sample_name = f"{scene}_{frame_id}"
                    
                    viz_path = output_dir / split / subset / bin_name / f"{sample_name}.png"
                    generate_visualization(gt_disp, pred_disp, valid_mask, epe, viz_path, sample_name, 
                                          vmax=self.global_max_disparity)
                
                # Write stats CSV for this subset/split
                stats_path = output_dir / split / subset / 'stats.csv'
                write_stats_csv(bin_stats, stats_path, quartile_ranges)
                
                all_stats[f"{split}/{subset}"] = dict(bin_stats)
        
        # Write summary CSV
        summary_path = output_dir / 'summary.csv'
        write_summary_csv(all_stats, summary_path)
        
        logger.info(f"Completed inference for {model_name}")
        return all_stats
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load image from path"""
        from PIL import Image
        img = Image.open(path).convert('RGB')
        return np.array(img, dtype=np.float32)
    
    def _prepare_batch(self, left_img: np.ndarray, right_img: np.ndarray, 
                       gt_disp: np.ndarray, valid_mask: np.ndarray) -> Dict[str, torch.Tensor]:
        """Prepare a batch for inference"""
        # Apply transforms
        sample = {
            'left': left_img,
            'right': right_img,
            'disparity': gt_disp,
            'valid_mask': valid_mask.astype(np.float32)
        }
        
        if self.transforms:
            sample = self.transforms(sample)
        
        # Add batch dimension and move to device
        batch = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0).to(self.device)
            elif isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(v).unsqueeze(0).to(self.device)
        
        return batch
    
    def write_comparison_csv(self, stats_by_model: Dict[str, Dict[str, Dict[str, List[float]]]]):
        """Write comparison CSV across both models"""
        save_path = OUTPUT_DIR / 'comparison.csv'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'split', 'subset', 'bin', 
                'model1_name', 'model1_epe_mean', 'model1_epe_std',
                'model2_name', 'model2_epe_mean', 'model2_epe_std',
                'epe_diff'
            ])
            
            model_names = list(stats_by_model.keys())
            if len(model_names) < 2:
                return
            
            model1, model2 = model_names[0], model_names[1]
            
            # Get all keys
            all_keys = set(stats_by_model[model1].keys()) | set(stats_by_model[model2].keys())
            
            for key in sorted(all_keys):
                split, subset = key.split('/')
                
                for bin_name in BIN_NAMES:
                    m1_stats = stats_by_model[model1].get(key, {}).get(bin_name, [])
                    m2_stats = stats_by_model[model2].get(key, {}).get(bin_name, [])
                    
                    if not m1_stats and not m2_stats:
                        continue
                    
                    m1_mean = np.mean(m1_stats) if m1_stats else 0
                    m1_std = np.std(m1_stats) if m1_stats else 0
                    m2_mean = np.mean(m2_stats) if m2_stats else 0
                    m2_std = np.std(m2_stats) if m2_stats else 0
                    
                    writer.writerow([
                        split, subset, bin_name,
                        model1, f'{m1_mean:.4f}', f'{m1_std:.4f}',
                        model2, f'{m2_mean:.4f}', f'{m2_std:.4f}',
                        f'{m1_mean - m2_mean:.4f}'
                    ])
        
        logger.info(f"Wrote comparison CSV to {save_path}")
    
    def run(self, model_filter: str = 'both'):
        """Run the full evaluation pipeline
        
        Args:
            model_filter: 'both', 'ours', or 'original'
        """
        logger.info("=" * 80)
        logger.info("Disparity Bin Evaluation Pipeline")
        logger.info("=" * 80)
        
        # Step 1: Discover and split dataset
        self.discover_and_split_dataset()
        
        # Step 2: Compute GLOBAL quartiles (across all subsets)
        self.compute_quartiles_global()
        
        # Step 3: Sample images (same for both models)
        self.sample_images()
        
        # Step 4: Run inference with selected model(s)
        stats_by_model = {}
        for model_name, checkpoint_path in CHECKPOINTS.items():
            # Filter models based on flag
            if model_filter == 'ours' and model_name != 'max_disp_300_90ep':
                continue
            if model_filter == 'original' and model_name != 'original':
                continue
            
            checkpoint_full_path = Path(checkpoint_path)
            if not checkpoint_full_path.exists():
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                continue
            
            stats = self.run_inference(model_name, str(checkpoint_full_path))
            stats_by_model[model_name] = stats
        
        # Step 5: Write comparison CSV (only if both models were run)
        if len(stats_by_model) == 2:
            self.write_comparison_csv(stats_by_model)
        
        logger.info("=" * 80)
        logger.info("Evaluation complete!")
        logger.info(f"Results saved to: {OUTPUT_DIR}")
        logger.info("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate LightStereo with disparity binning')
    parser.add_argument('--config', type=str, default='dai/configs/stereo_flow.yaml', 
                        help='Path to config YAML')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--model', type=str, default='both', choices=['both', 'ours', 'original'],
                        help='Which model(s) to run: both, ours, or original')
    
    args = parser.parse_args()
    
    evaluator = DisparityBinEvaluator(args.config, args.device)
    evaluator.run(model_filter=args.model)


if __name__ == '__main__':
    main()
