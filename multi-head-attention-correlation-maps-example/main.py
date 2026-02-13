import os
import numpy as np

import matplotlib.pyplot as plt
from tools import ViTimePrediction
from timm.models.vision_transformer import Attention
import matplotlib.colors as colors

# ------- custom modules -------
from model.multi_head_attention_rollout import MultiHeadAttentionRollout


# ======================================================================================================================
#   Global Variables
# ======================================================================================================================
torch_extensions: [str] = ['.pth']
np_extensions: [str] = ['.npy']

# ======================================================================================================================
#   Analytical Functions
# ======================================================================================================================

class MultiHeadAttentionRolloutPlotter(MultiHeadAttentionRollout):
    """
    Extended class for multi-head attention analysis and visualization.
    Provides deep insights into how the model attends to different temporal patterns
    by extracting and analyzing attention weights from the transformer's multi-head self-attention mechanism.
    """

    def __init__(self, model, attention_layer_name='attn', attention_drop_layer_name='attn_drop'):
        """
        Args:
            model: The PyTorch model (e.g., MAE/ViT).
            attention_layer_name: String to match the attention layer modules (e.g., 'attn' or 'self_attn').
            attention_drop_layer_name: String to match attention dropout layers.
            head_fusion: "mean", "max", or "min" to fuse heads for rollout (not used for specific head plotting).
            discard_ratio: Ratio of elements to discard in rollout (optional).
        """
        super().__init__(model, attention_layer_name, attention_drop_layer_name)


    def mean_attention_vs_time_lag(self, A_time, max_lag=40):
        """
        Compute lagged mean value attention over lagged A_time entries considering causal attention (keys delayed).

        Args:
            A_time: Time-time attention matrix
            max_lag: Maximum time lag to analyze

        Returns:
            lags: Array of lag values
            vals: Mean attention values at each lag
        """
        # A_time is (77,77)

        lags = np.arange(1, min(max_lag, A_time.shape[0] - 1) + 1)
        vals = []
        for lag in lags:
            d = np.diag(A_time, k=-lag)  # query attends to earlier keys
            vals.append(d.mean() if d.size else 0.0)
        return lags, np.array(vals)

    def plot_attention_heads(
            self,
            input_tensor,
            prediction_horizon=720,
            save_dir="experiments/attention_maps",
            heads_to_plot=(0, 1, 2),
            # visualization controls
            experiment='1',
            powermap=True,
            robust_percentiles=(5, 99),  # clip extremes for better contrast
            gamma=1,  # <1 boosts high values, >1 boosts low values
            show_log=False,  # log view can help if values are very peaky
            top_overlay_percent=99.99  # draw contour around top X% (set None to disable)
    ):
        """
        Plot attention patterns from multiple heads with advanced visualization options.
        Each head gets one figure with 2 subplots: heatmap and lagged correlation.

        Args:
            input_tensor: Input data tensor
            prediction_horizon: Number of prediction steps
            save_dir: Directory to save visualization plots
            heads_to_plot: Tuple of head indices to visualize
            experiment: Experiment name for labeling
            powermap: Whether to use power normalization for contrast
            robust_percentiles: Percentile range for clipping outliers
            gamma: Power law gamma for normalization
            show_log: Whether to apply log transformation
            top_overlay_percent: Percentile threshold for highlighting top attention
        """

        print(f"\n{'=' * 60}")
        print(f"EXPERIMENT: {experiment}")
        print(f"{'=' * 60}")

        self.forward(x=input_tensor, prediction_horizon=prediction_horizon)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        num_heads = self.multi_heads_tensor.shape[0]
        print(f"Total heads available: {num_heads}")
        print(f"Plotting heads: {heads_to_plot}\n")
        for head_idx in heads_to_plot:
            if head_idx >= num_heads:
                print(f"⚠ Head {head_idx} out of range (max: {num_heads - 1}), skipping...")
                continue

            print(f"Processing Head {head_idx}...")

            # Create figure with 2 subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Get attention map
            attn_map = self.multi_heads_tensor[head_idx].float().numpy()
            attn_map = self.token_attn_to_time_attn(attn_map)

            # ===== LEFT SUBPLOT: HEATMAP =====
            if powermap:
                view = attn_map.copy()
                if show_log:
                    view = np.log1p(view)

                # Robust vmin/vmax to avoid outliers flattening the colormap
                lo, hi = np.percentile(view, robust_percentiles)
                if np.isclose(lo, hi):
                    lo, hi = float(view.min()), float(view.max()) + 1e-9

                # Gamma (power-law) normalization to emphasize high values
                norm = colors.PowerNorm(gamma=gamma, vmin=lo, vmax=hi)

                im = ax1.imshow(view, aspect="auto", cmap="magma", norm=norm, origin='lower')
                ax1.set_title(f"Head {head_idx} - Attention Heatmap", fontsize=12, fontweight='bold')

                ax1.set_xlabel("Key (Patch)", fontsize=10)
                ax1.set_ylabel("Query (Patch)", fontsize=10)

                # Optional: outline top X% values
                if top_overlay_percent is not None:
                    thr = np.percentile(view, top_overlay_percent)
                    mask = (view >= thr).astype(float)
                    ax1.contour(mask, levels=[0.5], colors='cyan', linewidths=1.5, alpha=0.8)

                cbar1 = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
                cbar1.set_label("Attention (scaled)", fontsize=9)

            else:
                im = ax1.imshow(attn_map, aspect="auto", cmap="magma", origin='lower')
                ax1.set_title(f"Head {head_idx} - Attention Heatmap", fontsize=12, fontweight='bold')

                ax1.set_xlabel("Key (Patch)", fontsize=10)
                ax1.set_ylabel("Query (Patch)", fontsize=10)

                cbar1 = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
                cbar1.set_label("Mean Attention", fontsize=9)

            # ===== RIGHT SUBPLOT: TIME-LAG CORRELATION =====
            print(f"Max possible lag (patches): {self.attention_patched.shape[0] - 1}")
            print(f"Max possible lag (real timesteps): {(self.attention_patched.shape[0] - 1) * self.num_sample_per_patch} (Input length + Prediction length)")

            lags, values = self.mean_attention_vs_time_lag(attn_map)
            real_lags = lags * self.num_sample_per_patch

            ax2.plot(real_lags, values, linewidth=2, marker='o', markersize=4, color='#E64626')
            ax2.set_xlabel("Lag (Real Timesteps)", fontsize=10)
            ax2.set_ylabel("Mean Attention", fontsize=10)
            ax2.set_title(f"Head {head_idx} - Time-Lag Correlation: Lagged Keys", fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')

            # Overall figure title
            fig.suptitle(f"Experiment {experiment} - Attention Analysis", fontsize=14, fontweight='bold', y=1.00)

            plt.tight_layout()

            # Save figure
            save_filename = f"experiment_{experiment}_head_{head_idx}_correlations.png"
            save_path = os.path.join(save_dir, save_filename)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)  # Close the figure to free memory

            print(f"     ✓ Saved: {save_filename}")

        print(f"\n{'=' * 60}")
        print(f"✅ Experiment {experiment} completed!")
        print(f"{'=' * 60}\n")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Load model:
    model = ViTimePrediction(device='cuda:0', model_name='MAE', lookbackRatio=None, tempature=1)

    # Initialize Visualizer:
    visualizer = MultiHeadAttentionRolloutPlotter(model=model,attention_layer_name='attn', attention_drop_layer_name='attn_drop')

    # We are looking for periodicity lagged-time correlation:
    heads_to_plot = [7] # selected after correlation analysis

    # Experiment 1 - three harmonics:
    experiment = '1'
    samples = 512

    input_tensor = np.sin(np.arange(samples) / 10) + np.sin(np.arange(samples) / 5 + 50) + np.cos(np.arange(samples) + 50)
    visualizer.plot_attention_heads(input_tensor=input_tensor, prediction_horizon=720, heads_to_plot=heads_to_plot,
                                    experiment=experiment)
    # Experiment 2 - 2 harmonics:
    experiment = '2'
    model = ViTimePrediction(device='cuda:0', model_name='MAE', lookbackRatio=None, tempature=1)
    visualizer = MultiHeadAttentionRolloutPlotter(model=model,attention_layer_name='attn', attention_drop_layer_name='attn_drop')
    input_tensor = np.sin(np.arange(samples) / 5 + 50) + np.cos(np.arange(samples) + 50)
    visualizer.plot_attention_heads(input_tensor=input_tensor,prediction_horizon=720, heads_to_plot=heads_to_plot,
                                    experiment=experiment)
