# iTransformer Attention Analysis
# Analyzing feature-feature correlations from attention weights

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import pandas as pd
import networkx as nx

class AttentionAnalyzer:
    """
    Analyzes attention weights from iTransformer to extract feature correlations
    
    Tensor structure: 
    - self.multi_heads_tensor[test_iter][layer_idx] → [Batch, Heads, N_variates, N_variates]
    
    Full structure:
    - test_iteration x number_of_layers x Batch x Heads x N x N
    """
    
    def __init__(self, multi_heads_tensor, feature_names=None):
        """
        Args:
            multi_heads_tensor: List of test iterations, each containing list of layer attentions
            feature_names: List of feature names (e.g., ['Temp', 'Humid', 'Press', ...])
        """
        self.attention_dynamic_graph = []
        self.multi_heads_tensor = multi_heads_tensor
        
        # Extract dimensions
        self.n_test_iters = len(multi_heads_tensor)
        self.n_layers = len(multi_heads_tensor[0])  # Number of layers
        self.batch_size = multi_heads_tensor[0][0].size(0)
        self.n_heads = multi_heads_tensor[0][0].size(1)
        self.n_variates = multi_heads_tensor[0][0].size(2)
        
        # Feature names
        if feature_names is None:
            self.feature_names = [f'Feature_{i}' for i in range(self.n_variates)]
        else:
            assert len(feature_names) == self.n_variates
            self.feature_names = feature_names
        
        print("="*60)
        print("ATTENTION ANALYZER INITIALIZED")
        print("="*60)
        print(f"Test iterations: {self.n_test_iters}")
        print(f"Layers: {self.n_layers}")
        print(f"Batch size: {self.batch_size}")
        print(f"Heads: {self.n_heads}")
        print(f"Variables: {self.n_variates}")
        print(f"Feature names: {self.feature_names}")
        print("="*60)
    
    # ============================================================================
    # QUESTION 1: AVERAGE FEATURE CORRELATIONS (GLOBAL)
    # ============================================================================
    
    def get_average_feature_correlation(self, layer_idx=None):
        """
        Question 1: In average, which are the most correlated features?
        
        Aggregation strategy:
        1. Average across all heads (multi-head attention should be averaged)
        2. Average across all batches (get general behavior)
        3. Average across all test iterations (overall pattern)
        4. Optionally average across layers OR analyze per layer
        
        Args:
            layer_idx: If None, average across all layers. 
                      If int, use specific layer.
                      If 'separate', return dict with each layer
        
        Returns:
            avg_attention: [N, N] matrix of average feature-feature attention
        """
        print("\n" + "="*60)
        print("QUESTION 1: AVERAGE FEATURE CORRELATIONS")
        print("="*60)
        
        if layer_idx == 'separate':
            # Return separate matrix for each layer
            result = {}
            for layer in range(self.n_layers):
                result[f'layer_{layer}'] = self._compute_average(layer_idx=layer)
            return result
        else:
            # Average across all layers or use specific layer
            return self._compute_average(layer_idx=layer_idx)
    
    def _compute_average(self, layer_idx=None):
        """
        Internal method to compute average attention
        """
        attention_sum = torch.zeros(self.n_variates, self.n_variates)
        attention_over_layer = torch.zeros(self.n_variates, self.n_variates)
        count = 0
        
        # Iterate over test iterations
        for test_iter in range(self.n_test_iters):
            # Determine which layers to process
            if layer_idx is None:
                layers_to_process = range(self.n_layers)
            else:
                layers_to_process = [layer_idx]
            
            # Iterate over layers
            for layer in layers_to_process:
                # Get attention: [Batch, Heads, N, N]
                attn = self.multi_heads_tensor[test_iter][layer]
                
                # Average over heads: [Batch, Heads, N, N] → [Batch, N, N]
                attn_avg_heads = attn.mean(dim=1)
                
                # Average over batch: [Batch, N, N] → [N, N]
                attn_avg_batch = attn_avg_heads.mean(dim=0)
                
                # Accumulate
                attention_sum += attn_avg_batch
                count += 1

            attention_over_layer = attention_sum / self.n_layers
            self.attention_dynamic_graph.append(attention_over_layer.cpu().numpy())
        
        # Final average
        avg_attention = attention_sum / count
        
        print(f"\nAggregated over:")
        print(f"  - Test iterations: {self.n_test_iters}")
        print(f"  - Layers: {len(layers_to_process) if layer_idx is None else 1}")
        print(f"  - Batches: {self.batch_size}")
        print(f"  - Heads: {self.n_heads}")
        print(f"  - Total samples: {count * self.batch_size}")
        
        return avg_attention.cpu().numpy()
    
    def visualize_average_correlation(self, avg_attention, save_path=None,
                                      top_k_features=None):
        """
        Visualize the average feature-feature correlation matrix
        
        Args:
            avg_attention: Either:
                - [N, N] average attention matrix (single matrix)
                - dict with layer-specific matrices: {'layer_0': [N, N], 'layer_1': [N, N], ...}
            save_path: Path to save the figure
            top_k_features: If int, show only top K most correlated features (default: None = show all)
        """
        # Check if avg_attention is a dictionary (layer-specific)
        if isinstance(avg_attention, dict):
            # Multiple layers - create subplot for each
            n_layers = len(avg_attention)
            
            # Determine subplot layout
            if n_layers == 1:
                n_rows, n_cols = 1, 1
                figsize = (10, 8)
            elif n_layers == 2:
                n_rows, n_cols = 1, 2
                figsize = (20, 8)
            elif n_layers <= 4:
                n_rows, n_cols = 2, 2
                figsize = (20, 16)
            else:
                n_rows = (n_layers + 2) // 3
                n_cols = 3
                figsize = (30, 8 * n_rows)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            
            # Flatten axes array for easy iteration
            if n_layers == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if n_layers > 1 else [axes]
            
            # Find global min/max for consistent color scale (before filtering)
            all_values = np.concatenate([layer_attn.flatten() for layer_attn in avg_attention.values()])
            vmin, vmax = all_values.min(), all_values.max()
            
            # Plot each layer
            for idx, (layer_name, layer_attn) in enumerate(avg_attention.items()):
                ax = axes[idx]
                
                # Filter to top K features if requested
                if top_k_features is not None:
                    filtered_attn, filtered_names = self._get_top_k_features(
                        layer_attn, top_k_features
                    )
                else:
                    filtered_attn = layer_attn
                    filtered_names = self.feature_names
                
                sns.heatmap(
                    filtered_attn,
                    annot=True,
                    fmt='.3f',
                    cmap='RdYlBu_r',
                    xticklabels=filtered_names,
                    yticklabels=filtered_names,
                    cbar_kws={'label': 'Attention'},
                    ax=ax,
                    square=True,
                    linewidths=0.5,
                    vmin=vmin,
                    vmax=vmax
                )
                
                # Extract layer number from layer name (e.g., 'layer_0' -> 0)
                layer_num = layer_name.split('_')[-1]
                title = f'Layer {layer_num}'
                if top_k_features is not None:
                    title += f' (Top {top_k_features} Features)'
                ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
                ax.set_xlabel('Feature (Attends To)', fontsize=11)
                ax.set_ylabel('Feature (Query)', fontsize=11)
            
            # Hide unused subplots
            for idx in range(n_layers, len(axes)):
                axes[idx].set_visible(False)
            
            # Overall title
            fig.suptitle('Average Feature-Feature Correlations by Layer', 
                        fontsize=16, fontweight='bold', y=0.995)
            
        else:
            # Single matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Filter to top K features if requested
            if top_k_features is not None:
                filtered_attn, filtered_names = self._get_top_k_features(
                    avg_attention, top_k_features
                )
            else:
                filtered_attn = avg_attention
                filtered_names = self.feature_names
            
            sns.heatmap(
                filtered_attn,
                annot=True,
                fmt='.3f',
                cmap='RdYlBu_r',
                xticklabels=filtered_names,
                yticklabels=filtered_names,
                cbar_kws={'label': 'Average Attention'},
                ax=ax,
                square=True,
                linewidths=0.5
            )
            
            title = 'Average Feature-Feature Correlations (Global)'
            if top_k_features is not None:
                title += f'\n(Top {top_k_features} Most Correlated Features)'
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Feature (Attends To)', fontsize=12)
            ax.set_ylabel('Feature (Query)', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved figure to: {save_path}")
        
        plt.show()
        return fig
    
    def _get_top_k_features(self, attn_matrix, top_k):
        """
        Get the top K features with highest total correlation
        
        Args:
            attn_matrix: [N, N] attention matrix
            top_k: Number of top features to select
        
        Returns:
            filtered_matrix: [K, K] attention matrix with top K features
            filtered_names: List of K feature names
        """
        # Calculate total attention for each feature (sum of row + column, excluding diagonal)
        n_features = attn_matrix.shape[0]
        total_attention = np.zeros(n_features)
        
        for i in range(n_features):
            # Sum of attention TO this feature (column sum, excluding diagonal)
            incoming = np.sum(attn_matrix[:, i]) - attn_matrix[i, i]
            # Sum of attention FROM this feature (row sum, excluding diagonal)
            outgoing = np.sum(attn_matrix[i, :]) - attn_matrix[i, i]
            # Total
            total_attention[i] = incoming + outgoing
        
        # Get indices of top K features
        top_k = min(top_k, n_features)  # Don't exceed total number of features
        top_indices = np.argsort(total_attention)[::-1][:top_k]
        
        # Sort indices to maintain original order
        top_indices = np.sort(top_indices)
        
        # Extract submatrix and feature names
        filtered_matrix = attn_matrix[np.ix_(top_indices, top_indices)]
        filtered_names = [self.feature_names[i] for i in top_indices]
        
        print(f"\n📊 Top {top_k} Most Correlated Features:")
        for idx, i in enumerate(top_indices):
            print(f"  {idx+1}. {self.feature_names[i]} (total attention: {total_attention[i]:.4f})")
        
        # Treats the attention-map as graph:

        return filtered_matrix, filtered_names
    
    def _plot_attention_map(self, attn_map, ax, title='Attention Map',
                           powermap=True, show_log=False,
                           robust_percentiles=(5, 99), gamma=1.0,
                           top_overlay_percent=99.9):
        """
        Internal method to plot a single attention map with power normalization
        
        Args:
            attn_map: [N, N] attention matrix
            ax: Matplotlib axis
            title: Plot title
            powermap: Use power normalization
            show_log: Apply log transformation
            robust_percentiles: Percentile clipping range
            gamma: Power law gamma
            top_overlay_percent: Percentile for top value overlay
        """
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
            
            im = ax.imshow(view, aspect="auto", cmap="magma", norm=norm, origin='lower')
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            ax.set_xlabel("Feature (Key)", fontsize=10)
            ax.set_ylabel("Feature (Query)", fontsize=10)
            
            # Optional: outline top X% values
            if top_overlay_percent is not None:
                thr = np.percentile(view, top_overlay_percent)
                mask = (view >= thr).astype(float)
                ax.contour(mask, levels=[0.5], colors='cyan', linewidths=1.5, alpha=0.8)
            
            cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Attention (scaled)", fontsize=9)
            
        else:
            im = ax.imshow(attn_map, aspect="auto", cmap="magma", origin='lower')
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            ax.set_xlabel("Feature (Key)", fontsize=10)
            ax.set_ylabel("Feature (Query)", fontsize=10)
            
            cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Mean Attention", fontsize=9)
        
        # Set feature name tick labels
        n_features = attn_map.shape[0]
        ax.set_xticks(range(n_features))
        ax.set_yticks(range(n_features))
        ax.set_xticklabels(self.feature_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(self.feature_names, fontsize=8)
        
        # Set tick labels
        n_features = attn_map.shape[0]
        if n_features <= 20:  # Only show labels for reasonable number of features
            ax.set_xticks(np.arange(n_features))
            ax.set_yticks(np.arange(n_features))
            ax.set_xticklabels(self.feature_names, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(self.feature_names, fontsize=8)
        else:
            # Too many features - show indices instead
            tick_spacing = max(1, n_features // 10)
            ticks = np.arange(0, n_features, tick_spacing)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels([str(i) for i in ticks], rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels([str(i) for i in ticks], fontsize=8)
    
    def get_top_correlations(self, avg_attention, top_k=10, exclude_diagonal=True):
        """
        Find the top K feature-feature correlations
        
        Args:
            avg_attention: [N, N] average attention matrix
            top_k: Number of top correlations to return
            exclude_diagonal: Whether to exclude self-attention (diagonal)
        
        Returns:
            DataFrame with top correlations
        """
        # Get all correlations
        correlations = []
        
        for i in range(self.n_variates):
            for j in range(self.n_variates):
                if exclude_diagonal and i == j:
                    continue
                
                correlations.append({
                    'Feature_From': self.feature_names[i],
                    'Feature_To': self.feature_names[j],
                    'From_Idx': i,
                    'To_Idx': j,
                    'Attention': avg_attention[i, j]
                })
        
        # Create DataFrame and sort
        df = pd.DataFrame(correlations)
        df_sorted = df.sort_values('Attention', ascending=False).head(top_k)
        
        print("\n" + "="*60)
        print(f"TOP {top_k} FEATURE CORRELATIONS (Excluding diagonal)")
        print("="*60)
        print(df_sorted.to_string(index=False))
        print("="*60)
        
        return df_sorted
    
    def analyze_feature_importance(self, avg_attention):
        """
        Analyze which features are most important overall
        
        Two metrics:
        1. Incoming attention: How much other features attend TO this feature
        2. Outgoing attention: How much this feature attends TO other features
        
        Args:
            avg_attention: [N, N] average attention matrix
        
        Returns:
            DataFrame with importance scores
        """
        # Incoming attention (column sum - how much others attend to this feature)
        incoming = avg_attention.sum(axis=0)  # Sum over query dimension
        
        # Outgoing attention (row sum - how much this feature attends to others)
        outgoing = avg_attention.sum(axis=1)  # Sum over key dimension
        
        # Self-attention (diagonal)
        self_attn = np.diag(avg_attention)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Incoming_Attention': incoming,
            'Outgoing_Attention': outgoing,
            'Self_Attention': self_attn,
            'Total_Attention': incoming + outgoing
        })
        
        importance_df = importance_df.sort_values('Total_Attention', ascending=False)
        
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        print(importance_df.to_string(index=False))
        print("="*60)
        print("\nInterpretation:")
        print("- Incoming: How much other features depend on this feature")
        print("- Outgoing: How much this feature depends on other features")
        print("- Self: How much the feature attends to itself")
        print("="*60)
        
        return importance_df

    def analyze_global_flow(self, top_k=5):
        """
            attn_layers: List or array of matrices [A1, A2, ... AT]
            """
        # 1. Compute Effective Adjacency Matrix (Attention Rollout)
        # We start with identity and multiply through the layers
        attn_layers = self.attention_dynamic_graph
        effective_adj = np.eye(self.n_variates)
        for A in attn_layers:
            # 1. Residual Handling: 0.5 * I + 0.5 * A
            # This prevents the original token meaning from being "washed away"
            A_residual = 0.5 * np.eye(self.n_variates) + 0.5 * A

            # 2. Sequential Matmul (Flow from layer to layer)
            effective_adj = np.matmul(A_residual, effective_adj)

            # 3. Thresholding (Sparsification)
            # We zero out weak connections to prevent a "complete graph" hairball
            effective_adj[effective_adj < 0.01] = 0

        # 2. Treat this 'Total Flow' as a Graph
        # Remove diagonal to see inter-token influence only
        np.fill_diagonal(effective_adj, 0)
        G = nx.from_numpy_array(effective_adj, create_using=nx.DiGraph)
        G = nx.relabel_nodes(G, {i: n for i, n in enumerate(self.feature_names)})

        # 3. Calculate Global Importance (PageRank)
        global_importance = nx.pagerank(G, weight='weight')

        # 4. Extract Global Core Subgraph
        top_nodes = sorted(global_importance, key=global_importance.get, reverse=True)[:top_k]
        core_subgraph = G.subgraph(top_nodes)

        return top_nodes, effective_adj, core_subgraph
    
    # ============================================================================
    # QUESTION 2: SAMPLE-SPECIFIC CORRELATIONS
    # ============================================================================
    
    def get_sample_correlation(self, test_iter_idx=0, batch_idx=0, layer_idx=None):
        """
        Question 2: Which is the correlation among features in a particular sample?
        
        Args:
            test_iter_idx: Which test iteration to use
            batch_idx: Which sample in the batch
            layer_idx: Which layer to use (None = average all layers)
        
        Returns:
            sample_attention: [N, N] attention matrix for this specific sample
        """
        print("\n" + "="*60)
        print("QUESTION 2: SAMPLE-SPECIFIC CORRELATIONS")
        print("="*60)
        print(f"Test iteration: {test_iter_idx}/{self.n_test_iters}")
        print(f"Batch sample: {batch_idx}/{self.batch_size}")
        
        if layer_idx is None:
            # Average across all layers
            attention_sum = torch.zeros(self.n_variates, self.n_variates)
            
            for layer in range(self.n_layers):
                # Get attention for this sample: [Heads, N, N]
                attn = self.multi_heads_tensor[test_iter_idx][layer][batch_idx]
                
                # Average over heads: [Heads, N, N] → [N, N]
                attn_avg_heads = attn.mean(dim=0)
                
                attention_sum += attn_avg_heads
            
            sample_attention = attention_sum / self.n_layers
            print(f"Averaged over: {self.n_layers} layers, {self.n_heads} heads")
        else:
            # Use specific layer
            attn = self.multi_heads_tensor[test_iter_idx][layer_idx][batch_idx]
            sample_attention = attn.mean(dim=0)  # Average over heads
            print(f"Using: Layer {layer_idx}, averaged over {self.n_heads} heads")
        
        print("="*60)
        
        return sample_attention.cpu().numpy()
    
    def visualize_sample_correlation(self, sample_attention, test_iter_idx, 
                                    batch_idx, save_path=None):
        """
        Visualize sample-specific feature correlations
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            sample_attention,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            xticklabels=self.feature_names,
            yticklabels=self.feature_names,
            cbar_kws={'label': 'Attention'},
            ax=ax,
            square=True,
            linewidths=0.5
        )
        
        ax.set_title(f'Feature Correlations\nTest Iter: {test_iter_idx}, Sample: {batch_idx}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Feature (Attends To)', fontsize=12)
        ax.set_ylabel('Feature (Query)', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved figure to: {save_path}")
        
        plt.show()
        return fig
    
    def compare_samples(self, n_samples=4, layer_idx=None):
        """
        Compare feature correlations across multiple samples
        
        Args:
            n_samples: Number of random samples to compare
            layer_idx: Which layer to use (None = average all)
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
        
        for i in range(min(n_samples, 4)):
            # Random sample
            test_iter = np.random.randint(0, self.n_test_iters)
            batch_idx = np.random.randint(0, self.batch_size)
            
            # Get correlation
            sample_attn = self.get_sample_correlation(test_iter, batch_idx, layer_idx)
            
            # Plot
            sns.heatmap(
                sample_attn,
                annot=True,
                fmt='.2f',
                cmap='RdYlBu_r',
                xticklabels=self.feature_names,
                yticklabels=self.feature_names,
                cbar_kws={'label': 'Attention'},
                ax=axes[i],
                square=True,
                linewidths=0.5
            )
            
            axes[i].set_title(f'Sample: Test {test_iter}, Batch {batch_idx}', 
                            fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        return fig
    
    # ============================================================================
    # LAYER-SPECIFIC ANALYSIS
    # ============================================================================
    
    def compare_layers(self, save_path=None):
        """
        Compare feature correlations across different layers
        Shows how correlations evolve through the network
        """
        fig, axes = plt.subplots(1, self.n_layers, figsize=(8*self.n_layers, 6))
        
        if self.n_layers == 1:
            axes = [axes]
        
        for layer in range(self.n_layers):
            # Get average attention for this layer
            layer_avg = self._compute_average(layer_idx=layer)
            
            # Plot
            sns.heatmap(
                layer_avg,
                annot=True,
                fmt='.3f',
                cmap='RdYlBu_r',
                xticklabels=self.feature_names,
                yticklabels=self.feature_names,
                cbar_kws={'label': 'Attention'},
                ax=axes[layer],
                square=True,
                linewidths=0.5,
                vmin=0,
                vmax=layer_avg.max()
            )
            
            axes[layer].set_title(f'Layer {layer}', fontsize=14, fontweight='bold')
        
        fig.suptitle('Feature Correlations Across Layers', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved figure to: {save_path}")
        
        plt.show()
        return fig
    
    # ============================================================================
    # STATISTICAL ANALYSIS
    # ============================================================================
    
    def compute_correlation_statistics(self):
        """
        Compute statistics of feature correlations across all samples
        
        Returns mean, std, min, max for each feature pair
        """
        # Collect all attention matrices
        all_attentions = []
        
        for test_iter in range(self.n_test_iters):
            for layer in range(self.n_layers):
                attn = self.multi_heads_tensor[test_iter][layer]
                # Average over heads: [Batch, Heads, N, N] → [Batch, N, N]
                attn_avg_heads = attn.mean(dim=1)
                all_attentions.append(attn_avg_heads)
        
        # Stack: [Total_samples, N, N]
        all_attentions = torch.cat(all_attentions, dim=0)
        
        # Compute statistics
        mean = all_attentions.mean(dim=0).cpu().numpy()
        std = all_attentions.std(dim=0).cpu().numpy()
        min_val = all_attentions.min(dim=0)[0].cpu().numpy()
        max_val = all_attentions.max(dim=0)[0].cpu().numpy()
        
        print("\n" + "="*60)
        print("CORRELATION STATISTICS")
        print("="*60)
        print(f"Total samples: {all_attentions.shape[0]}")
        print(f"Mean attention range: [{mean.min():.4f}, {mean.max():.4f}]")
        print(f"Std attention range: [{std.min():.4f}, {std.max():.4f}]")
        print("="*60)
        
        return {
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val
        }
    
    def visualize_correlation_uncertainty(self, stats, save_path=None):
        """
        Visualize mean ± std for correlations
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot mean
        sns.heatmap(
            stats['mean'],
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            xticklabels=self.feature_names,
            yticklabels=self.feature_names,
            cbar_kws={'label': 'Mean Attention'},
            ax=axes[0],
            square=True,
            linewidths=0.5
        )
        axes[0].set_title('Mean Feature Correlations', fontsize=14, fontweight='bold')
        
        # Plot std
        sns.heatmap(
            stats['std'],
            annot=True,
            fmt='.3f',
            cmap='viridis',
            xticklabels=self.feature_names,
            yticklabels=self.feature_names,
            cbar_kws={'label': 'Std Deviation'},
            ax=axes[1],
            square=True,
            linewidths=0.5
        )
        axes[1].set_title('Correlation Uncertainty (Std)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved figure to: {save_path}")
        
        plt.show()
        return fig


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def analyze_itransformer_attention(multi_heads_tensor, feature_names=None, 
                                  save_dir='results/attention_analysis/'):
    """
    Complete analysis pipeline for iTransformer attention weights
    
    Args:
        multi_heads_tensor: The attention weights from your model
        feature_names: Names of your features
        save_dir: Directory to save results
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = AttentionAnalyzer(multi_heads_tensor, feature_names)
    
    print("\n\n")
    print("#"*60)
    print("# QUESTION 1: AVERAGE FEATURE CORRELATIONS")
    print("#"*60)
    
    # Get average correlations
    avg_attention = analyzer.get_average_feature_correlation(layer_idx=None)
    
    # Visualize
    analyzer.visualize_average_correlation(
        avg_attention, 
        save_path=os.path.join(save_dir, 'average_correlations.png')
    )
    
    # Get top correlations
    top_corr = analyzer.get_top_correlations(avg_attention, top_k=10)
    
    # Analyze feature importance
    importance = analyzer.analyze_feature_importance(avg_attention)
    
    # Compare layers
    if analyzer.n_layers > 1:
        analyzer.compare_layers(
            save_path=os.path.join(save_dir, 'layer_comparison.png')
        )
    
    print("\n\n")
    print("#"*60)
    print("# QUESTION 2: SAMPLE-SPECIFIC CORRELATIONS")
    print("#"*60)
    
    # Get sample-specific correlation
    sample_attn = analyzer.get_sample_correlation(
        test_iter_idx=0, 
        batch_idx=0, 
        layer_idx=None
    )
    
    # Visualize
    analyzer.visualize_sample_correlation(
        sample_attn, 
        test_iter_idx=0, 
        batch_idx=0,
        save_path=os.path.join(save_dir, 'sample_correlation.png')
    )
    
    # Compare multiple samples
    analyzer.compare_samples(n_samples=4)
    
    print("\n\n")
    print("#"*60)
    print("# STATISTICAL ANALYSIS")
    print("#"*60)
    
    # Compute statistics
    stats = analyzer.compute_correlation_statistics()
    
    # Visualize uncertainty
    analyzer.visualize_correlation_uncertainty(
        stats,
        save_path=os.path.join(save_dir, 'correlation_uncertainty.png')
    )
    
    print("\n\n")
    print("="*60)
    print("✅ ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to: {save_dir}")
    
    return analyzer, {
        'average_attention': avg_attention,
        'top_correlations': top_corr,
        'feature_importance': importance,
        'statistics': stats
    }
