# Multi-Head Attention Maps Visualization for Explainable Temporal Correlation in Time Series Analysis

## Project Overview

- This project demonstrates the **explainable capability** of **Multi-Head Self-Attention layers** by combining their well-known parallelization power while maintaining sequence order through positional encoding and outperforming classical sequential models in time-series forecasting
- The foundation model used is the **Vision Transformer (ViT)**, state-of-the-art in time series prediction models. REF: **[Paper](https://openreview.net/pdf?id=XInsJDBIkp) | [Code](https://github.com/IkeYang/ViTime)**
  - *Model_name*: `ViTime_Model.pth`
  - *Model_weights*: [Google Drive](https://drive.google.com/file/d/1ex5ZrIKhsnLj2EuUkP9We3Bpcr1kVh5d/view?usp=sharing)
- Model hidden architecture has been analyzed to describe the sequence-image-sequence transformations and their tensors' dimensionality
- The **MultiHeadAttentionRollout** class has been implemented:
  - Provides deep insights into how the model attends to different temporal patterns
  - Extracts and analyzes attention weights from the transformer's multi-head self-attention mechanism
- Model's 2D *spatial-attention* has been traduced in temporal attention patterns by extracting heads' **Attention Maps**
- **Attention Maps** have been scaled to real input time sample through patch conversion
- **Attention Maps** have been analyzed and visualized to demonstrate how the model identifies **periodicity** and **lagged correlations** in **zero-learning** time series data.

## Key Features
- **Attention Weight Extraction**: Custom hook mechanism to capture attention weights from transformer layers
- **Token-to-Time Conversion**: Transforms 2D patch-based attention into 1D temporal attention maps
- **Temporal Dependency Analysis**: Computes mean attention vs. time lag to identify periodic patterns
- **Multi-Head Visualization**: Visualizes attention patterns from specific attention heads
- **Experiment Framework**: Compare model behavior across different input signals

## Experiments

All experiments use synthetic signals of length **512 samples** and analyze **Head 7**, which shows strong sensitivity to periodic patterns.

### Experiment 1: Three Harmonics (Full Signal)

**Input Signal:**
```python
input_tensor_full = sin(n/10) + sin(n/5 + 50) + cos(n + 50)
```

#### True Periods

For **sin(ωn)**, the period is: **P = 2π/ω**

| Component | ω | Period |
|-----------|---|--------|
| sin(n/10) | 1/10 | ~63 samples |
| sin(n/5) | 1/5 | ~31 samples |
| cos(n) | 1 | ~6 samples |

Thus the signal contains periodic structure at:
- **~63 samples** (longest period)
- **~31 samples** (medium period)
- **~6 samples** (shortest period)

#### Results

**Attention Heatmap:**

![Experiment 1 - Temporal Dependencies](experiments/attention_maps/experiment_1_temporal_dependencies_heads_contrast.png)
*Figure 1: Time-time attention map for three-harmonic signal. Diagonal patterns reveal temporal dependencies.*

**Time-Lag Correlation:**

![Experiment 1 - Lag Analysis](experiments/attention_maps/experiment_1_time-lag correlation.png)
*Figure 2: Mean attention vs. time lag. Peaks appear at multiples of the fundamental period.*

#### Observations

The lag-attention curve shows **distinct peaks** at:
- **≈ 63 samples** (fundamental period)
- **≈ 126 samples** (2× fundamental)
- **≈ 189 samples** (3× fundamental)
- etc.

These correspond to **integer multiples** of the fundamental period.

✅ **Transformer attention successfully captures all three sinusoidal temporal dependencies.**

---

### Experiment 2: Reduced Harmonics (Missing Component)

**Input Signal:**
```python
input_tensor_reduced = sin(n/5 + 50) + cos(n + 50)
```

**Removed Component:** `sin(n/10)` — the **~63 sample period**

#### Results

**Attention Heatmap:**

![Experiment 2 - Temporal Dependencies](experiments/attention_maps/experiment_2_temporal_dependencies_heads_contrast.png)
*Figure 3: Time-time attention map for two-harmonic signal. Notice the absence of long-range periodic structure.*

**Time-Lag Correlation:**

![Experiment 2 - Lag Analysis](experiments/attention_maps/experiment_2_time-lag correlation.png)
*Figure 4: Mean attention vs. time lag. The ~63 sample peaks have disappeared.*

#### Observations

- **Peaks corresponding to ~63 samples disappear** ❌
- Only **shorter-period structures** remain (from the 31-sample and 6-sample components)
- **Periodic peak spacing** changes — no longer shows multiples of 63

✅ **This confirms attention peaks are NOT artifacts — they directly correspond to true signal periodicity.**

---

### Key Findings

This experiment demonstrates:

1. **Direct Harmonic Detection**: The transformer's attention mechanism identifies and responds to specific frequency components in the input
2. **Component Specificity**: Removing a harmonic component from the input causes its corresponding attention peaks to vanish
3. **No Spurious Patterns**: Attention peaks are not computational artifacts but genuine reflections of temporal structure
4. **Causal Awareness**: The model attends to past patterns at intervals matching the signal's periodicity

**Head 7** appears to specialize in detecting periodic patterns, making it ideal for analyzing temporal correlations and validating that the model has learned meaningful time-domain relationships.



## Conclusion and Next steps:
- retraning


## Implementation Details
### File Structure

```
.
├── main.py    # Main experiment script
├── config.py  # Model weights path
├── model/
│   └── attention_rollout.py      # AttentionRollout implementation
└── experiments/
    └── attention_maps/           # Generated visualizations
```

### Core Components

#### 1. AttentionRollout Class (`attention_rollout.py`)

The main visualization engine that:
- Registers forward hooks on attention dropout layers
- Captures attention weights during model inference
- Converts spatial token attention to temporal attention
- Generates visualization plots with enhanced contrast options

#### 2. Visualization Script (`main.py`)

Orchestrates experiments comparing how the model attends to signals with different harmonic compositions:
- **Experiment 1 (full)**: Three harmonics - `sin(t/10) + sin(t/5 + 50) + cos(t + 50)`
- **Experiment 2 (reduced_harmonic)**: Two harmonics - `sin(t/5 + 50) + cos(t + 50)`

## Critical Implementation Details

###  Fused Attention Deactivation

**This step is essential for hook registration to work properly.**

Before registering hooks, the code explicitly disables the fused attention optimization in the Vision Transformer:

```python
for name, module in self.model.predictor._iface.model.named_modules():
    if name.endswith(self.attention_layer_name) and isinstance(module, Attention):
        module.fused_attn = False  # CRITICAL: Disable fused attention
```

**Why?** The `fused_attn` optimization bypasses the standard attention computation path, making it impossible to hook into the attention dropout layer. By setting `fused_attn = False`, we ensure the attention computation follows the standard forward pass where our hooks can intercept the outputs.

**Verification:** The code includes a check to confirm fusion is disabled:
```python
for name, module in self.model.predictor._iface.model.named_modules():
    if name.endswith(self.attention_layer_name) and isinstance(module, Attention):
        print(name, module.fused_attn)  # Should print False
```

### 🪝 Hook Mechanism

The hook system captures attention weights during the forward pass:

1. **Hook Registration:**
   ```python
   def register_softmax_hooks(self):
       self.attentions = {}
       name_to_module = dict(self.model.predictor._iface.model.named_modules())
       hook_handles = []
       for name in self.hooks:
           m = name_to_module[name]
           h = m.register_forward_hook(self.get_attn_hook(name))
           hook_handles.append(h)
   ```

2. **Hook Function:**
   ```python
   def get_attn_hook(self, name: str):
       def hook(module, inputs, output):
           if torch.is_tensor(output):
               self.attentions[name] = output.detach().cpu()
       return hook
   ```

The hooks are attached to `attn_drop` layers (attention dropout), which output the post-softmax attention weights with shape `(Batch, Heads, Sequence, Sequence)`.

### 🔄 Token-to-Time Attention Conversion

The model uses a 2D grid of patches (height × width) to process 1D time series. The conversion process:

1. **Input Layout**: Time series is reshaped into 2D grid (`grid_h × grid_w`)
2. **Attention Shape**: `(Npatch, Npatch)` where `Npatch = grid_h × grid_w`
3. **CLS Token Handling**: Removes the CLS token (if present) from attention maps
4. **Dimensionality Reduction**: Reshapes to `(grid_h, grid_w, grid_h, grid_w)` then averages over height dimensions
5. **Output**: Time-to-time attention matrix `(grid_w, grid_w)`

```python
def token_attn_to_time_attn(self, A, drop_cls="auto"):
    # Remove CLS token if present
    if S == Npatch + 1:
        A = A[1:, 1:]
    
    # Reshape to 4D: (query_height, query_width, key_height, key_width)
    A4 = A.reshape(grid_h, grid_w, grid_h, grid_w)
    
    # Average over height dimensions -> (query_time, key_time)
    A_time = A4.mean(axis=(0, 2))
    return A_time
```

### 📊 Time-Lag Correlation Analysis

Computes how attention changes with temporal distance (lag) between query and key:

```python
def mean_attention_vs_time_lag(self, A_time, max_lag=40):
    lags = np.arange(1, min(max_lag, A_time.shape[0] - 1) + 1)
    vals = []
    for lag in lags:
        d = np.diag(A_time, k=-lag)  # Causal: query attends to earlier keys
        vals.append(d.mean())
    return lags, np.array(vals)
```

This reveals periodic patterns in the model's attention mechanism.

## Usage

### Basic Usage

```python
import numpy as np
from main import ViTimePrediction
from model.attention_rollout import AttentionRollout

# Load model
model = ViTimePrediction(
    device='cuda:0',
    model_name='MAE',
    lookbackRatio=None,
    tempature=1
)

# Initialize visualizer
visualizer = AttentionRollout(model)

# Create input signal
input_tensor = np.sin(np.arange(512) / 10) + np.sin(np.arange(512) / 5 + 50)

# Visualize specific attention heads
visualizer.plot_attention_heads(
    input_tensor,
    heads_to_plot=[7],
    powermap=True,
    experiment='my_experiment'
)
```

### Running the Experiments

```bash
python main.py
```

This will generate:
- Attention heatmaps (with and without contrast enhancement)
- Time-lag correlation plots
- Saved in `experiments/attention_maps/`

## Visualization Options

### Contrast Enhancement (`powermap=True`)

When enabled, applies several enhancements for better visualization:

- **Gamma correction**: Power-law normalization (`γ=0.1` emphasizes high values)
- **Robust clipping**: Uses 5th-99th percentile to avoid outliers
- **Log scaling**: Optional `log1p` transformation for peaky distributions
- **Top-value overlay**: Contour lines around top 99.9% attention values

### Output Files

For each experiment, generates:
1. `experiment_{name}_temporal_dependencies_heads_contrast.png` - Enhanced visualization
2. `experiment_{name}_temporal_dependencies_heads_no_contrast.png` - Raw attention
3. `experiment_{name}_time-lag correlation.png` - Lag analysis plot

## Parameters

### AttentionRollout Constructor

- `model`: ViTimePrediction model instance
- `attention_layer_name`: Layer name pattern to match (default: `'attn'`)
- `attention_drop_layer_name`: Dropout layer name (default: `'attn_drop'`)
- `head_fusion`: How to fuse heads - "mean", "max", or "min"
- `discard_ratio`: Ratio for rollout computation (default: `0.9`)

### plot_attention_heads Parameters

- `input_tensor`: Input time series (numpy array)
- `save_dir`: Directory for output plots
- `heads_to_plot`: Tuple of head indices to visualize
- `experiment`: Experiment name (used in filenames)
- `powermap`: Enable contrast enhancement (default: `True`)
- `robust_percentiles`: Percentile range for clipping (default: `(5, 99)`)
- `gamma`: Gamma correction factor (default: `0.1`)
- `show_log`: Apply log transformation (default: `True`)
- `top_overlay_percent`: Percentile for contour overlay (default: `99.9`)

## Dependencies

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from timm.models.vision_transformer import Attention
```

Additional requirements:
- Custom `ViTimePrediction` model implementation
- CUDA-capable GPU (for `device='cuda:0'`)


## 🧠 Model Context

The underlying model is `ViTimePrediction(model_name='MAE')`.

### Attention Mechanism

Internally, the MAE model:
- Tokenizes the input into a 2D grid of patches
- Computes self-attention over all tokens
- Each attention head produces an attention matrix: **A ∈ ℝ^(S×S)**

Where:
- **S = grid_h × grid_w**
- In this setup:
  - `grid_h = H / patch_h`
  - `grid_w = W / patch_w`
  - Example: **32 × 77 = 2464 tokens**

Thus, initial attention is: **A ∈ ℝ^(2464×2464)**

This is **token-token attention**, not directly time-time attention.

### 🔄 Token → Time Attention Mapping

Since time is encoded along the horizontal axis (`grid_w`), we convert:

**A_token ∈ ℝ^((grid_h·grid_w) × (grid_h·grid_w))**

into:

**A_time ∈ ℝ^(grid_w × grid_w)**

#### Conversion Procedure:

1. **Remove CLS token** (if present)
2. **Reshape** attention to 4D:
   ```
   A → (grid_h, grid_w, grid_h, grid_w)
   ```
3. **Average** across vertical spatial bins:
   ```
   A_time(c_q, c_k) = (1/grid_h²) Σ_{r_q,r_k} A(r_q, c_q, r_k, c_k)
   ```

This yields **pure time-time attention**.

### 📊 Time-Lag Correlation Analysis

To detect periodic dependencies, we compute:

```
MeanAttention(lag) = mean of A_time(i, i-lag)
```

This extracts attention along sub-diagonals (causal direction: queries attending to earlier keys).

Then we convert **patch lag** to **real time lag**:

```
real_lag = lag × samples_per_patch
```

In this implementation: `num_sample_per_patch = 16`

So: **real_lag = lag × 16**


## Scientific Insights

This visualization tool reveals:

1. **Temporal Dependencies**: Which time lags the model focuses on
2. **Periodic Pattern Detection**: How different heads specialize in different periodicities
3. **Harmonic Sensitivity**: Comparing full vs. reduced harmonic inputs shows which harmonics drive attention
4. **Causal Structure**: Diagonal patterns show how queries attend to past keys

## Notes

- The model processes sequences of length 512 with 16 samples per patch
- Grid dimensions are determined by model architecture (`h`, `patch_size`)
- CLS tokens are automatically detected and removed from analysis
- All attention maps are saved as high-resolution PNG files (200 DPI)

## Troubleshooting

**No attention maps captured:**
- Verify `fused_attn = False` is set correctly
- Check layer names match your model architecture
- Ensure hooks are registered before calling `model.prediction()`

**Unexpected attention map shapes:**
- Confirm grid dimensions match your model's patch configuration
- Check if CLS token is being handled correctly (`drop_cls="auto"`)

---

**Author**: Built for analyzing Vision Transformer attention in time series prediction tasks  
**Purpose**: Research tool for understanding temporal dependencies learned by ViT-based forecasting models
