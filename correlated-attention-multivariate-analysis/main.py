import os
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# ------- custom modules -------
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
from config import get_config
from attention_analyzer import AttentionAnalyzer


# ======================================================================================================================
#   Global Variables
# ======================================================================================================================
torch_extensions: [str] = ['.pth']
np_extensions: [str] = ['.npy']

# ======================================================================================================================
#   Preparation Functions
# ======================================================================================================================

def set_seed(seed=2023):
    """Set random seed for reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def experiment_setting(args, checkpoint_path):
    """Define experiments settings for correct class loading
    plot and test share the same experiment class
    """
    if args.exp_name == 'partial_train':
        exp = Exp_Long_Term_Forecast_Partial(args)
    else:
        exp = MultiHeadAttentionRolloutPlotter(args)

    print(f'Verifying pretrained model from {checkpoint_path}')
    # Load the checkpoint
    if os.path.exists(checkpoint_path):
        print(f'Checkpoint found at {checkpoint_path}')
        #checkpoint = torch.load(checkpoint_path, map_location=exp.device)
        #exp.model.load_state_dict(checkpoint)
        #exp.model.eval()
        return exp
    else:
        print(f'Checkpoint not found at {checkpoint_path}')
        return None


def find_checkpoint(args):
    """Find the correct checkpoint based on the configuration"""
    # Use the exact setting string format
    setting = args.get_setting_string()

    checkpoint_dir = os.path.join(args.checkpoints, setting)
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')

    print(f'Looking for checkpoint at: {checkpoint_path}')

    # If the exact path doesn't exist, try to find any .pth file in the directory
    if not os.path.exists(checkpoint_path) and os.path.exists(checkpoint_dir):
        pth_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if pth_files:
            checkpoint_path = os.path.join(checkpoint_dir, pth_files[0])
            print(f'Using checkpoint: {checkpoint_path}')

    return checkpoint_path, setting


# ======================================================================================================================
#   Analytical Functions
# ======================================================================================================================

class MultiHeadAttentionRolloutPlotter(Exp_Long_Term_Forecast):
    """
    Extended class for multi-head attention analysis and visualization.
    Provides deep insights into how the model attends to different temporal patterns
    by extracting and analyzing attention weights from the transformer's multi-head self-attention mechanism.
    """

    def __init__(self, args):
        super(MultiHeadAttentionRolloutPlotter, self).__init__(args)
        self.batch_size = None
        self.n_heads = None
        self.n_variates = None
        self.multi_heads_tensor = []


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

    def _get_heads(self,test_loader, test_length):
        self.model.eval()
        attentions = []
        with (torch.no_grad()):
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if i >= test_length:
                    print(f"✓ Stopping test loop at {i} batch")
                    break
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, attention_weights = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs, attention_weights = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs, attention_weights = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs, attention_weights = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                print(f"✓ Batch {i} done")

                print(f"✓ Captured {len(attention_weights)} attention layers")
                self.batch_size = attention_weights[0].size(0)
                self.n_heads = attention_weights[0].size(1)
                self.n_variates = attention_weights[0].size(2)
                assert self.n_variates == attention_weights[0].size(3)
                print(f"✓ Attention Map dim: B={self.batch_size} x H={self.n_heads} x N={self.n_variates} x N={self.n_variates}")
                self.multi_heads_tensor.append(attention_weights)
        return

    def plot_attention_heads(
            self,
            settings,
            test =0,
            test_length = 1000,
            save_dir='results/attention_maps/',
            # visualization controls
            experiment='1',
            layer_idx='separate',
    ):
        """
        Plot attention patterns from multiple heads with advanced visualization options.
        Each head gets one figure with 2 subplots: heatmap and lagged correlation.

        Args:
            settings: TBD
            test: flag
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

        os.makedirs(save_dir, exist_ok=True)

        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            self.model.output_attention_map = True
            self.model.output_attention = True

            def set_attention_recursive(module):
                for child in module.children():
                    if hasattr(child, 'output_attention'):
                        child.output_attention = True
                    if hasattr(child, 'inner_attention') and hasattr(child.inner_attention, 'output_attention'):
                        child.inner_attention.output_attention = True
                    set_attention_recursive(child)

            set_attention_recursive(self.model)

        # Get All Layers' Attention Maps: [test size x Attention Layers]
        self._get_heads(test_loader,test_length)

        analyzer = AttentionAnalyzer(self.multi_heads_tensor)

        # Analyze Global Flow as a Network and Rollout as paper:
        analyzer.get_average_feature_correlation(layer_idx=None)

        top_nodes, effective_adj, core_subgraph = analyzer.analyze_global_flow(top_k=10)

        # Get average correlations
        avg_attention = analyzer.get_average_feature_correlation(layer_idx=layer_idx)
        # Visualize
        analyzer.visualize_average_correlation(
            avg_attention, top_k_features=10,
            save_path=os.path.join(save_dir, 'average_correlations.png')
        )


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Set random seed
    set_seed(2023)

    # Configuration - Easy to change dataset and prediction length
    dataset_name = 'solar'  # Options: 'solar', 'electricity' (add more as needed)
    pred_len = 96  # Options: 96, 192, 336 (based on available checkpoints)

    # Get accurate configuration
    args = get_config(dataset_name, pred_len)

    print('=' * 50)
    print('CONFIGURATION')
    print('=' * 50)
    print(f'Model: {args.model}')
    print(f'Dataset: {args.data}')
    print(f'Data path: {args.root_path}{args.data_path}')
    print(f'Prediction length: {args.pred_len}')
    print(f'Model dimensions - d_model: {args.d_model}, n_heads: {args.n_heads}')
    print(f'Model layers - e_layers: {args.e_layers}, d_layers: {args.d_layers}')
    print(f'Input/Output size: enc_in={args.enc_in}, c_out={args.c_out}')
    print(f'Using GPU: {args.use_gpu}')
    print('=' * 50)

    # Find the correct checkpoint
    checkpoint_path, setting = find_checkpoint(args)

    # Load the pretrained model
    exp = experiment_setting(args, checkpoint_path)
    exp.plot_attention_heads(setting, test=1, test_length = 1)
