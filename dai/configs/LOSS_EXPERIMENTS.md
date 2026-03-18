# Extended Loss Components - Experiment Recommendations

## Quick Start

**Best default for sharp edges + clean surfaces:**
```yaml
loss:
  loss_weights:
    disp_pred: 1.0
    disp_4: 0.3
    gradient: 0.4
    edge_smooth: 0.1
    second_order_smooth: 0.05
```

---

## Experiment Progression

### Phase 1: Baseline (Current)
**Config:** All extended losses disabled (gradient=0, edge_smooth=0, second_order_smooth=0)

**Purpose:** Establish baseline metrics (EPE, D1-all) for comparison.

**Expected:** Reasonable accuracy but potentially blurry edges, noisy flat regions.

---

### Phase 2: Add Gradient Loss (Edge Sharpening)
**Config:**
```yaml
gradient: 0.4
edge_smooth: 0.0
second_order_smooth: 0.0
```

**Purpose:** Test if gradient matching alone improves edge sharpness.

**Expected improvements:**
- Sharper object boundaries
- Better thin structure preservation
- Reduced edge thickness

**Tune if:**
- Edges still blurry → increase to 0.5-0.6
- Edges too harsh/artifacts → decrease to 0.2-0.3

**WandB metrics to watch:**
- `loss_gradient` (should be stable, not dominating)
- D1-all at object boundaries (if you have per-region metrics)

---

### Phase 3: Add Edge-Aware Smoothness (Noise Reduction)
**Config:**
```yaml
gradient: 0.4
edge_smooth: 0.1
second_order_smooth: 0.0
```

**Purpose:** Reduce noise in flat regions while preserving edges from Phase 2.

**Expected improvements:**
- Smoother flat surfaces (walls, floors, sky)
- Less speckle noise
- Maintained edge sharpness from gradient loss

**Tune if:**
- Still noisy in flat regions → increase to 0.15-0.2
- Edges becoming blurry → decrease to 0.05-0.08

**WandB metrics to watch:**
- `loss_edge_smooth` (should be moderate, not overwhelming)
- EPE in flat regions (if you can measure separately)

---

### Phase 4: Add Second-Order Smoothness (Planar Surfaces)
**Config:**
```yaml
gradient: 0.4
edge_smooth: 0.1
second_order_smooth: 0.05
```

**Purpose:** Encourage planar surfaces, straighten long edges.

**Expected improvements:**
- Straighter walls and floors
- Less waviness on large flat surfaces
- Better geometric consistency

**Tune if:**
- Surfaces still wavy → increase to 0.08-0.1
- Losing fine detail → decrease to 0.02-0.03

**WandB metrics to watch:**
- `loss_second_order_smooth` (should be smallest of the three)
- Visual inspection of large planar regions

---

## Experiment Matrix (Sweep)

If using WandB sweeps, try this grid:

```yaml
# wandb sweep config
parameters:
  gradient:
    values: [0.0, 0.3, 0.4, 0.5]
  edge_smooth:
    values: [0.0, 0.05, 0.1, 0.15]
  second_order_smooth:
    values: [0.0, 0.03, 0.05, 0.08]
```

**Total runs:** 4 × 4 × 4 = 64 (can be reduced with random search)

**Recommended subset (12 runs):**
1. Baseline: (0, 0, 0)
2. Gradient only: (0.3, 0, 0), (0.4, 0, 0), (0.5, 0, 0)
3. Gradient + smooth: (0.4, 0.05, 0), (0.4, 0.1, 0), (0.4, 0.15, 0)
4. Full combo low: (0.3, 0.05, 0.03)
5. Full combo mid: (0.4, 0.1, 0.05)
6. Full combo high: (0.5, 0.15, 0.08)
7. Smooth-focused: (0.3, 0.15, 0.08)
8. Edge-focused: (0.5, 0.05, 0.03)
9. Planar-focused: (0.4, 0.1, 0.1)

---

## Problem-Specific Tuning

### Problem: Edges are blurry
**Solution:** Increase `gradient`
```yaml
gradient: 0.5  # or 0.6
edge_smooth: 0.1
second_order_smooth: 0.05
```

### Problem: Flat regions are noisy
**Solution:** Increase `edge_smooth`
```yaml
gradient: 0.4
edge_smooth: 0.15  # or 0.2
second_order_smooth: 0.05
```

### Problem: Walls/floors are wavy
**Solution:** Increase `second_order_smooth`
```yaml
gradient: 0.4
edge_smooth: 0.1
second_order_smooth: 0.08  # or 0.1
```

### Problem: Thin objects disappear
**Solution:** Reduce smoothness, keep gradient high
```yaml
gradient: 0.5
edge_smooth: 0.05
second_order_smooth: 0.02
```

### Problem: Training becomes unstable (NaN loss)
**Solution:** Start with only gradient, add others gradually
```yaml
# Step 1: Train with gradient only
gradient: 0.3
edge_smooth: 0.0
second_order_smooth: 0.0

# Step 2: After stable, add smoothness
gradient: 0.3
edge_smooth: 0.05
second_order_smooth: 0.0

# Step 3: Add second-order last
gradient: 0.3
edge_smooth: 0.05
second_order_smooth: 0.03
```

---

## Scene-Specific Recommendations

### Indoor scenes (IRS dataset - your current case)
**Characteristics:** Lots of planar surfaces (walls, floors), sharp furniture edges

**Recommended:**
```yaml
gradient: 0.4          # Sharp furniture edges
edge_smooth: 0.1       # Smooth walls/floors
second_order_smooth: 0.05  # Straight walls
```

### Outdoor driving scenes
**Characteristics:** Large planar roads, distant objects, sky

**Recommended:**
```yaml
gradient: 0.3          # Moderate edge sharpness
edge_smooth: 0.15      # Smooth roads/sky
second_order_smooth: 0.08  # Very straight roads
```

### Close-range objects
**Characteristics:** Complex geometry, less planar surfaces

**Recommended:**
```yaml
gradient: 0.5          # High edge detail
edge_smooth: 0.05      # Minimal smoothing
second_order_smooth: 0.02  # Minimal planarity constraint
```

---

## Monitoring During Training

### Key WandB metrics to track:

1. **Loss components ratio:**
   - `loss_gradient / loss_total` should be ~20-40%
   - `loss_edge_smooth / loss_total` should be ~5-15%
   - `loss_second_order_smooth / loss_total` should be ~2-8%
   - If any component dominates (>60%), reduce its weight

2. **Loss stability:**
   - All losses should decrease smoothly
   - If `loss_gradient` spikes → reduce weight or learning rate
   - If `loss_edge_smooth` explodes → reduce weight

3. **Validation metrics:**
   - EPE (End-Point Error) - lower is better
   - D1-all (% pixels with >3px error) - lower is better
   - Visual inspection of validation samples

### Red flags:
- NaN in any loss component → reduce all extended weights by 50%
- One loss component >> others → rebalance weights
- Validation metrics worse than baseline → try lower weights

---

## Example Experiment Configs

### Experiment 1: Baseline
File: `presaved_pseudo_baseline.yaml`
```yaml
loss:
  loss_weights:
    disp_pred: 1.0
    disp_4: 0.3
    gradient: 0.0
    edge_smooth: 0.0
    second_order_smooth: 0.0
```

### Experiment 2: Sharp Edges
File: `presaved_pseudo_sharp.yaml`
```yaml
loss:
  loss_weights:
    disp_pred: 1.0
    disp_4: 0.3
    gradient: 0.5
    edge_smooth: 0.05
    second_order_smooth: 0.02
```

### Experiment 3: Smooth Surfaces
File: `presaved_pseudo_smooth.yaml`
```yaml
loss:
  loss_weights:
    disp_pred: 1.0
    disp_4: 0.3
    gradient: 0.3
    edge_smooth: 0.15
    second_order_smooth: 0.08
```

### Experiment 4: Balanced (Recommended)
File: `presaved_pseudo_balanced.yaml`
```yaml
loss:
  loss_weights:
    disp_pred: 1.0
    disp_4: 0.3
    gradient: 0.4
    edge_smooth: 0.1
    second_order_smooth: 0.05
```

---

## Expected Results

Based on stereo matching literature and the loss formulations:

| Configuration | Edge Sharpness | Surface Smoothness | Planar Geometry | Training Stability |
|---------------|----------------|--------------------|-----------------|--------------------|
| Baseline | ⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| + Gradient | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| + Edge Smooth | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| + 2nd Order | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## Next Steps

1. **Run baseline** (current config with all extended losses = 0)
2. **Run balanced** (gradient=0.4, edge_smooth=0.1, second_order_smooth=0.05)
3. **Compare metrics** (EPE, D1-all, visual quality)
4. **Fine-tune** based on results using the problem-specific tuning guide above
5. **Optional:** Run full sweep if you have compute budget

Good luck! 🚀
