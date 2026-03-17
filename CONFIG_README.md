# DAI Configuration Guide

This guide describes how to use YAML configuration files for the DAI (Domain Adaptive Intelligence) stereo training system.

## Table of Contents
- [Overview](#overview)
- [Configuration Sections](#configuration-sections)
  - [Seed](#seed)
  - [Datasets](#datasets)
  - [Splitter](#splitter)
  - [Transforms](#transforms)
  - [Dataloader](#dataloader)
  - [Model](#model)
  - [FoundationStereo](#foundationstereo)
  - [Optimization](#optimization)
  - [Loss](#loss)
  - [Training](#training)
  - [Logging](#logging)
  - [Checkpoints](#checkpoints)
- [Example Configurations](#example-configurations)

---

## Overview

Configuration files are YAML-based and control all aspects of training, from dataset loading to model architecture and optimization settings. Load a config using:

```python
from dai.config import load_config
config = load_config('dai/configs/your_config.yaml')
```

---

## Configuration Sections

### Seed

**Purpose:** Set random seed for reproducibility.

```yaml
seed: 28
```

**Options:**
- Any integer value
- Affects data splitting, shuffling, and model initialization

---

### Datasets

**Purpose:** Define which datasets to use and how to load them.

```yaml
datasets:
  - name: sceneflow
    root: ./src_data/sceneflow
    subsets:
      - monkaa
      - driving
      - flyingthings3d
    pass_type: cleanpass
    resolution: null
    target_source: pseudo-foundationstereo-online
    return_right_disp: false
    
    # Optional: Sequential dataset settings
    sequential:
      enabled: true
      sequence_length: 8
      stride: 4
      skip_incomplete: true
```

**Options:**

| Parameter | Type | Options | Effect |
|-----------|------|---------|--------|
| `name` | string | `sceneflow`, `kitti12`, `kitti15`, `irs` | Dataset type to load |
| `root` | string | Path | Root directory of dataset |
| `subsets` | list | Dataset-specific | Which subsets to include (e.g., monkaa, driving) |
| `pass_type` | string | `cleanpass`, `finalpass` | SceneFlow rendering pass type |
| `resolution` | list/null | `[H, W]` or `null` | Target resolution (null = original) |
| `target_source` | string | See below | Disparity ground truth source |
| `return_right_disp` | bool | `true`, `false` | Whether to return right disparity maps |

**Target Source Options:**
- `provided` - Use ground truth disparity from dataset files
- `pseudo-foundationstereo-presaved` - Load pre-calculated FoundationStereo disparities from disk
- `pseudo-foundationstereo-online` - Generate FoundationStereo disparities in real-time during training
- `ssl` - Self-supervised learning mode (returns zeros)

**Sequential Settings (for temporal models):**
- `enabled` - Enable sequential/temporal data loading
- `sequence_length` - Number of frames per sequence
- `stride` - Sliding window stride between sequences
- `skip_incomplete` - Skip sequences shorter than sequence_length

---

### Splitter

**Purpose:** Control train/val/test data splitting.

```yaml
splitter:
  train:
    enabled: true
    proportion: 0.7
  val:
    enabled: true
    proportion: 0.15
  test:
    enabled: true
    proportion: 0.15
  seed: 42
  strategy: stratified
```

**Options:**

| Parameter | Type | Options | Effect |
|-----------|------|---------|--------|
| `train/val/test.enabled` | bool | `true`, `false` | Enable/disable split |
| `train/val/test.proportion` | float | 0.0-1.0 | Proportion of data for split |
| `seed` | int | Any integer | Random seed for splitting |
| `strategy` | string | `random`, `sequential`, `stratified` | Splitting strategy |

**Strategy Effects:**
- `random` - Random shuffling across entire dataset
- `sequential` - Sequential split (first N% train, next M% val, etc.)
- `stratified` - Preserves dataset structure (e.g., keeps scenes/folders together)

---

### Transforms

**Purpose:** Define data augmentation and preprocessing pipelines for each split.

```yaml
transforms:
  train:
    - {name: RandomCrop, size: [384, 768]}
    - {name: Resize, size: [256, 512]}
    - {name: ToTensor}
    - {name: NormalizeImage, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
  
  val:
    - {name: RightBottomPad, divisor: 32}
    - {name: ToTensor}
    - {name: NormalizeImage, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
  
  test:
    - {name: RightBottomPad, divisor: 32}
    - {name: ToTensor}
    - {name: NormalizeImage, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
```

**Available Transforms:**

| Transform | Parameters | Effect |
|-----------|------------|--------|
| `RandomCrop` | `size: [H, W]` | Random crop to specified size |
| `StridedRandomCrop` | `size: [H, W]`, `stride: int` | Random crop with stride constraint |
| `Resize` | `size: [H, W]` | Resize to specified dimensions |
| `RightBottomPad` | `divisor: int` | Pad to make dimensions divisible by value |
| `ToTensor` | None | Convert to PyTorch tensor |
| `NormalizeImage` | `mean: [R,G,B]`, `std: [R,G,B]` | Normalize with ImageNet stats |
| `StereoColorJitter` | `brightness`, `contrast`, `saturation`, `hue`, `asymmetric_prob` | Color augmentation |
| `RandomErase` | `prob`, `max_time`, `bounds` | Random erasing augmentation |
| `RandomScale` | `min_scale`, `max_scale`, `scale_prob` | Random scaling augmentation |

**Notes:**
- Transforms are applied in order
- `NormalizeImage` with ImageNet stats is typically required
- Training transforms should include augmentation, val/test should not

---

### Dataloader

**Purpose:** Configure PyTorch DataLoader settings.

```yaml
dataloader:
  batch_size: 12
  num_workers: 12
  shuffle: true
  pin_memory: true
  drop_last: true
  persistent_workers: false
  prefetch_factor: 1
```

**Options:**

| Parameter | Type | Effect |
|-----------|------|--------|
| `batch_size` | int | Number of samples per batch |
| `num_workers` | int | Number of parallel data loading workers |
| `shuffle` | bool | Shuffle data each epoch (typically true for train) |
| `pin_memory` | bool | Pin memory for faster GPU transfer (typically true) |
| `drop_last` | bool | Drop incomplete final batch (recommended for training) |
| `persistent_workers` | bool | Keep workers alive between epochs (faster but more memory) |
| `prefetch_factor` | int | Number of batches each worker prefetches |

**Performance Tips:**
- For online pseudo-GT: reduce `batch_size` and increase `prefetch_factor`
- For temporal/sequential: reduce `batch_size` (sequences are larger)
- `num_workers` = number of CPU cores is a good starting point
- `persistent_workers=true` speeds up training but uses more memory

---

### Model

**Purpose:** Define model architecture and parameters.

```yaml
model:
  name: LightStereo
  variant: S
  max_disp: 300
  left_att: true
  aggregation_blocks: [1, 2, 4]
  expanse_ratio: 4
  backbone: MobileNetv2
  freeze_backbone: true
  
  # Optional: Temporal settings (for TemporalLightStereo)
  temporal:
    enabled: true
    fusion_type: convolutional
    hidden_channels: 64
```

**Options:**

| Parameter | Type | Options | Effect |
|-----------|------|---------|--------|
| `name` | string | `LightStereo`, `TemporalLightStereo` | Model architecture |
| `variant` | string | `S`, `M`, `L`, `H` | Model size (S=smallest/fastest) |
| `max_disp` | int | Any positive int | Maximum disparity value to predict |
| `left_att` | bool | `true`, `false` | Enable left attention mechanism |
| `aggregation_blocks` | list | `[int, int, int]` | Blocks at each aggregation level |
| `expanse_ratio` | int | Positive int | Channel expansion ratio |
| `backbone` | string | `MobileNetv2` | Backbone architecture |
| `freeze_backbone` | bool | `true`, `false` | Freeze backbone weights (train only stereo head) |

**Temporal Settings (TemporalLightStereo only):**
- `enabled` - Enable/disable temporal fusion
- `fusion_type` - Fusion strategy:
  - `convolutional` - Confidence-weighted blending (simple, stable)
  - `recurrent` - ConvGRU-based temporal memory (complex, stateful)
  - `residual` - Predict disparity delta/correction (smooth motion)
- `hidden_channels` - Hidden state channels for fusion networks

---

### FoundationStereo

**Purpose:** Configure FoundationStereo for pseudo-GT generation (required for `pseudo-foundationstereo-online` mode).

```yaml
foundationstereo:
  model_size: small
  inference_batch_size: 12
  inference_iters: 12
  inference_amp: true
  prefetch_buffer_size: 400
```

**Options:**

| Parameter | Type | Options | Effect |
|-----------|------|---------|--------|
| `model_size` | string | `small`, `base`, `large` | FoundationStereo model size |
| `inference_batch_size` | int | Positive int | Sub-batch size for inference (reduce if cuDNN errors) |
| `inference_iters` | int | Positive int | GRU iterations (default 32, 12 is ~2.5x faster) |
| `inference_amp` | bool | `true`, `false` | Use float16 autocast for faster inference |
| `prefetch_buffer_size` | int | Positive int | Number of batches to buffer in prefetcher queue (1-5 recommended) |

**Notes:**
- Only used when `target_source: pseudo-foundationstereo-online`
- For presaved mode, only `model_size` and `inference_iters` are used (for generation script)
- Reduce `inference_batch_size` if you encounter cuDNN errors
- Lower `inference_iters` trades quality for speed (~12 is a good balance)

---

### Optimization

**Purpose:** Configure training optimization settings.

```yaml
optimization:
  num_epochs: 90
  executable_epochs: 90
  amp: false
  grad_clip_value: 0.1
  accumulation_steps: 2
  
  optimizer:
    name: AdamW
    lr: 0.001
    weight_decay: 0.00001
    eps: 0.00000001
  
  scheduler:
    name: OneCycleLR
    pct_start: 0.05
    on_epoch: false
```

**Options:**

| Parameter | Type | Options | Effect |
|-----------|------|---------|--------|
| `num_epochs` | int | Positive int | Total training epochs |
| `executable_epochs` | int/null | Positive int or `null` | Early stopping (null = run all epochs) |
| `amp` | bool | `true`, `false` | Use automatic mixed precision (faster but less stable) |
| `grad_clip_value` | float | Positive float | Gradient clipping value (prevents exploding gradients) |
| `accumulation_steps` | int | Positive int | Gradient accumulation steps (effective_batch = batch × steps) |

**Optimizer Options:**

| Parameter | Type | Options | Effect |
|-----------|------|---------|--------|
| `name` | string | `AdamW`, `Adam`, `SGD` | Optimizer algorithm |
| `lr` | float | Positive float | Learning rate |
| `weight_decay` | float | Small float | L2 regularization strength |
| `eps` | float | Small float | Numerical stability epsilon |

**Scheduler Options:**

| Parameter | Type | Options | Effect |
|-----------|------|---------|--------|
| `name` | string | `OneCycleLR`, `StepLR`, `CosineAnnealingLR` | LR schedule type |
| `pct_start` | float | 0.0-1.0 | Percentage of cycle for warmup (OneCycleLR only) |
| `on_epoch` | bool | `true`, `false` | Update per epoch (true) or per iteration (false) |

**Notes:**
- `amp=false` recommended for stability with large `max_disp` values
- Use `accumulation_steps` to simulate larger batch sizes with limited memory
- `OneCycleLR` with `on_epoch=false` provides smooth per-iteration updates

---

### Loss

**Purpose:** Configure loss function weights.

```yaml
loss:
  loss_weights:
    disp_pred: 1.0
    disp_4: 0.3
```

**Options:**

| Parameter | Type | Effect |
|-----------|------|--------|
| `disp_pred` | float | Weight for final disparity prediction loss |
| `disp_4` | float | Weight for 1/4 resolution intermediate prediction loss |

**Notes:**
- Multi-scale supervision helps training stability
- Typical ratio is 1.0 for final, 0.3 for intermediate

---

### Training

**Purpose:** Specify training mode.

```yaml
training:
  mode: normal
```

**Options:**

| Mode | Effect |
|------|--------|
| `normal` | Standard single-frame stereo training |
| `sequential_fused_features` | Temporal training with feature fusion |
| `sequential_warped_disparity` | Temporal training with disparity warping |

**Notes:**
- `normal` mode is used for most training
- Sequential modes require `datasets[].sequential.enabled: true`
- Sequential modes require `TemporalLightStereo` model

---

### Logging

**Purpose:** Configure Weights & Biases logging and checkpoint saving intervals.

```yaml
logging:
  wandb:
    project: lightstereo
    entity: null
    run_name: my_experiment
    group: training_group
    tags: [tag1, tag2]
  
  log_interval: 10
  save_interval: 1
```

**Options:**

| Parameter | Type | Effect |
|-----------|------|--------|
| `wandb.project` | string | W&B project name |
| `wandb.entity` | string/null | W&B entity/team (null = personal account) |
| `wandb.run_name` | string/null | Run name (null = auto-generated) |
| `wandb.group` | string/null | Group name for organizing runs |
| `wandb.tags` | list | Tags for filtering/organizing runs |
| `log_interval` | int | Log metrics every N iterations |
| `save_interval` | int | Save checkpoint every N epochs |

---

### Checkpoints

**Purpose:** Configure checkpoint saving behavior.

```yaml
checkpoints:
  save_dir: src_data/lightstereo_training/my_experiment
  save_best: true
  save_periodic: true
```

**Options:**

| Parameter | Type | Effect |
|-----------|------|--------|
| `save_dir` | string | Directory to save checkpoints |
| `save_best` | bool | Save best model based on validation EPE |
| `save_periodic` | bool | Save periodic checkpoints every `save_interval` epochs |

---

## Example Configurations

### Basic Training with Ground Truth

```yaml
seed: 28

datasets:
  - name: sceneflow
    root: ./src_data/sceneflow
    subsets: [monkaa, driving, flyingthings3d]
    pass_type: cleanpass
    resolution: null
    target_source: provided
    return_right_disp: false

splitter:
  train: {enabled: true, proportion: 0.7}
  val: {enabled: true, proportion: 0.15}
  test: {enabled: true, proportion: 0.15}
  seed: 42
  strategy: stratified

transforms:
  train:
    - {name: RandomCrop, size: [384, 768]}
    - {name: Resize, size: [256, 512]}
    - {name: ToTensor}
    - {name: NormalizeImage, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
  val:
    - {name: RightBottomPad, divisor: 32}
    - {name: ToTensor}
    - {name: NormalizeImage, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
  test:
    - {name: RightBottomPad, divisor: 32}
    - {name: ToTensor}
    - {name: NormalizeImage, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}

dataloader:
  batch_size: 12
  num_workers: 12
  shuffle: true
  pin_memory: true
  drop_last: true

model:
  name: LightStereo
  variant: S
  max_disp: 192
  left_att: true
  aggregation_blocks: [1, 2, 4]
  expanse_ratio: 4
  backbone: MobileNetv2
  freeze_backbone: true

optimization:
  num_epochs: 90
  amp: false
  grad_clip_value: 0.1
  accumulation_steps: 1
  optimizer:
    name: AdamW
    lr: 0.001
    weight_decay: 0.00001
  scheduler:
    name: OneCycleLR
    pct_start: 0.05
    on_epoch: false

loss:
  loss_weights:
    disp_pred: 1.0
    disp_4: 0.3

training:
  mode: normal

logging:
  wandb:
    project: lightstereo
    entity: null
    run_name: sceneflow_baseline
    tags: [sceneflow, baseline]
  log_interval: 50
  save_interval: 10

checkpoints:
  save_dir: src_data/checkpoints/baseline
  save_best: true
  save_periodic: false
```

### Online Pseudo-GT Training

```yaml
seed: 28

datasets:
  - name: irs
    root: ./src_data/IRS
    subsets: [Home_1, Home_2, Office_1, Office_2, Restaurant, Store]
    resolution: null
    target_source: pseudo-foundationstereo-online
    return_right_disp: false

splitter:
  train: {enabled: true, proportion: 0.7}
  val: {enabled: true, proportion: 0.15}
  test: {enabled: true, proportion: 0.15}
  seed: 42
  strategy: stratified

transforms:
  train:
    - {name: RandomCrop, size: [384, 768]}
    - {name: Resize, size: [256, 512]}
    - {name: ToTensor}
    - {name: NormalizeImage, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
  val:
    - {name: RightBottomPad, divisor: 32}
    - {name: ToTensor}
    - {name: NormalizeImage, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
  test:
    - {name: RightBottomPad, divisor: 32}
    - {name: ToTensor}
    - {name: NormalizeImage, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}

dataloader:
  batch_size: 12
  num_workers: 12
  shuffle: true
  pin_memory: true
  drop_last: true
  persistent_workers: false
  prefetch_factor: 1

model:
  name: LightStereo
  variant: S
  max_disp: 300
  left_att: true
  aggregation_blocks: [1, 2, 4]
  expanse_ratio: 4
  backbone: MobileNetv2
  freeze_backbone: true

foundationstereo:
  model_size: small
  inference_batch_size: 12
  inference_iters: 12
  inference_amp: true
  prefetch_buffer_size: 400

optimization:
  num_epochs: 90
  amp: false
  grad_clip_value: 0.1
  accumulation_steps: 2
  optimizer:
    name: AdamW
    lr: 0.001
    weight_decay: 0.00001
  scheduler:
    name: OneCycleLR
    pct_start: 0.05
    on_epoch: false

loss:
  loss_weights:
    disp_pred: 1.0
    disp_4: 0.3

training:
  mode: normal

logging:
  wandb:
    project: lightstereo
    entity: null
    run_name: irs_online_pseudo_gt
    tags: [irs, online_pseudo_gt]
  log_interval: 10
  save_interval: 1

checkpoints:
  save_dir: src_data/checkpoints/irs_online
  save_best: true
  save_periodic: true
```

### Presaved Pseudo-GT Training

```yaml
seed: 28

datasets:
  - name: irs
    root: ./src_data/IRS
    subsets: [Home_1, Home_2, Office_1, Office_2, Restaurant, Store]
    resolution: null
    target_source: pseudo-foundationstereo-presaved
    return_right_disp: false

splitter:
  train: {enabled: true, proportion: 0.7}
  val: {enabled: true, proportion: 0.15}
  test: {enabled: true, proportion: 0.15}
  seed: 42
  strategy: stratified

transforms:
  train:
    - {name: StridedRandomCrop, size: [384, 768], stride: 50}
    - {name: Resize, size: [256, 512]}
    - {name: ToTensor}
    - {name: NormalizeImage, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
  val:
    - {name: RightBottomPad, divisor: 32}
    - {name: ToTensor}
    - {name: NormalizeImage, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
  test:
    - {name: RightBottomPad, divisor: 32}
    - {name: ToTensor}
    - {name: NormalizeImage, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}

dataloader:
  batch_size: 12
  num_workers: 12
  shuffle: true
  pin_memory: true
  drop_last: true

model:
  name: LightStereo
  variant: S
  max_disp: 300
  left_att: true
  aggregation_blocks: [1, 2, 4]
  expanse_ratio: 4
  backbone: MobileNetv2
  freeze_backbone: true

foundationstereo:
  model_size: small
  inference_iters: 12
  inference_amp: true

optimization:
  num_epochs: 90
  amp: false
  grad_clip_value: 0.1
  accumulation_steps: 2
  optimizer:
    name: AdamW
    lr: 0.001
    weight_decay: 0.00001
  scheduler:
    name: OneCycleLR
    pct_start: 0.05
    on_epoch: false

loss:
  loss_weights:
    disp_pred: 1.0
    disp_4: 0.3

training:
  mode: normal

logging:
  wandb:
    project: lightstereo
    entity: null
    run_name: irs_presaved_pseudo_gt
    tags: [irs, presaved_pseudo_gt]
  log_interval: 10
  save_interval: 1

checkpoints:
  save_dir: src_data/checkpoints/irs_presaved
  save_best: true
  save_periodic: true
```

### Temporal/Sequential Training

```yaml
seed: 28

datasets:
  - name: sceneflow
    root: ./src_data/sceneflow
    subsets: [monkaa, driving, flyingthings3d]
    pass_type: cleanpass
    resolution: null
    target_source: provided
    return_right_disp: false
    sequential:
      enabled: true
      sequence_length: 8
      stride: 4
      skip_incomplete: true

splitter:
  train: {enabled: true, proportion: 0.7}
  val: {enabled: true, proportion: 0.15}
  test: {enabled: true, proportion: 0.15}
  seed: 42
  strategy: stratified

transforms:
  train:
    - {name: RandomCrop, size: [320, 736]}
    - {name: ToTensor}
    - {name: NormalizeImage, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
  val:
    - {name: RightBottomPad, divisor: 32}
    - {name: ToTensor}
    - {name: NormalizeImage, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
  test:
    - {name: RightBottomPad, divisor: 32}
    - {name: ToTensor}
    - {name: NormalizeImage, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}

dataloader:
  batch_size: 2
  num_workers: 12
  shuffle: true
  pin_memory: true
  drop_last: true
  persistent_workers: true
  prefetch_factor: 12

model:
  name: TemporalLightStereo
  variant: S
  max_disp: 192
  left_att: true
  aggregation_blocks: [1, 2, 4]
  expanse_ratio: 4
  backbone: MobileNetv2
  temporal:
    enabled: true
    fusion_type: convolutional
    hidden_channels: 64

optimization:
  num_epochs: 90
  amp: true
  grad_clip_value: 0.1
  accumulation_steps: 12
  optimizer:
    name: AdamW
    lr: 0.0025
    weight_decay: 0.00001
  scheduler:
    name: OneCycleLR
    pct_start: 0.01
    on_epoch: false

loss:
  loss_weights:
    disp_pred: 1.0
    disp_4: 0.3

training:
  mode: sequential_fused_features

logging:
  wandb:
    project: lightstereo
    entity: null
    run_name: temporal_sceneflow
    tags: [sceneflow, temporal]
  log_interval: 50
  save_interval: 10

checkpoints:
  save_dir: src_data/checkpoints/temporal
  save_best: true
  save_periodic: false
```

---

## Tips and Best Practices

### Performance Optimization

1. **Online Pseudo-GT Mode:**
   - Reduce `batch_size` (e.g., 12)
   - Set `persistent_workers: false`
   - Adjust `prefetch_buffer_size` based on memory (400-1500)
   - Lower `inference_iters` to 12 for 2.5x speedup

2. **Temporal/Sequential Mode:**
   - Reduce `batch_size` significantly (e.g., 2) - sequences are large
   - Increase `accumulation_steps` to maintain effective batch size
   - Use `persistent_workers: true` for faster epoch transitions

3. **Memory Management:**
   - Use `accumulation_steps` to simulate larger batches
   - Enable `amp: true` for mixed precision (faster, uses less memory)
   - Reduce `max_disp` if running out of memory

### Training Stability

1. Always use `grad_clip_value: 0.1` to prevent exploding gradients
2. Disable `amp` for large `max_disp` values (>256) for stability
3. Use `OneCycleLR` with `on_epoch: false` for smooth learning rate updates
4. Start with `freeze_backbone: true` for faster convergence

### Data Splitting

1. Use `strategy: stratified` to preserve dataset structure
2. Set `seed` for reproducible splits
3. Typical split: 70% train, 15% val, 15% test

### Logging and Debugging

1. Use descriptive `run_name` and `tags` for organization
2. Set `log_interval` low (10-50) during debugging
3. Increase `log_interval` (250+) for production runs
4. Always enable `save_best: true` to keep best checkpoint
