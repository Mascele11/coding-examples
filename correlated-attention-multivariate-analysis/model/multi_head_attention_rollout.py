import torch


class MultiHeadAttentionRollout:
    """
    Base class for extracting and processing attention weights from Vision Transformer models.
    Handles hook registration and basic attention transformations.
    """

    def __init__(self, model, attention_layer_name='attn', attention_drop_layer_name='attn_drop'):
        """
        Args:
            model: The PyTorch model (e.g., MAE/ViT).
            attention_layer_name: String to match the attention layer modules (e.g., 'attn' or 'self_attn').
            attention_drop_layer_name: String to match attention dropout layers.
        """
        self.model = model
        self.hooks = []
        self.attentions = []
        self.attention_layer_name = attention_layer_name
        self.attention_drop_layer_name = attention_drop_layer_name
        self.num_sample_per_patch = 16
        self.multi_heads_tensor = torch.tensor([0, 1, 2])
        self.attention_patched = torch.tensor([0, 1, 2])

        # Find and Deactivate the Fused Attention Branch -> Necessary for register_forward_hook:
        for name, module in self.model.predictor._iface.model.named_modules():
            if name.endswith(self.attention_layer_name) and isinstance(module, torch.nn.Module):
                module.fused_attn = False

        # Register Attention Dropout Layers Hooks:
        for name, module in self.model.predictor._iface.model.named_modules():
            if name.endswith(self.attention_drop_layer_name) and isinstance(module, torch.nn.Module):
                self.hooks.append(name)

    def get_attn_hook(self, name: str):
        """Create a forward hook to capture attention weights."""

        def hook(module, inputs, output):
            # output is the forward output of the hooked module (e.g., attn_drop)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if torch.is_tensor(output):
                self.attentions[name] = output.detach().cpu()
            else:
                for x in (output if isinstance(output, (tuple, list)) else [output]):
                    if torch.is_tensor(x):
                        self.attentions[name] = x.detach().cpu()
                        break

        return hook

    def register_softmax_hooks(self):
        """Register forward hooks to capture attention weights during forward pass."""
        self.attentions = {}  # name -> tensor
        name_to_module = dict(self.model.predictor._iface.model.named_modules())

        hook_handles = []
        for name in self.hooks:  # self.hooks contains *module names* of ViTime
            m = name_to_module[name]
            h = m.register_forward_hook(self.get_attn_hook(name))
            hook_handles.append(h)

        self.hooks = hook_handles

    def token_attn_to_time_attn(self, A, drop_cls="auto"):
        """
        Convert token-token attention A (S,S) into time-time attention (grid_w, grid_w)
        by averaging over the height/bin axis.

        The implemented model MAE uses a 2D grid of tokens (e.g., 32x77) to compute attention.
        The total sequence length is S = grid_w * grid_h.

        Token layout assumed:
            token t -> (r = t // grid_w, c = t % grid_w)

        Args:
            A: Attention matrix of shape (S, S)
            drop_cls: How to handle CLS token. "auto", True, or False

        Returns:
            Time-time attention matrix of shape (grid_w, grid_w)
        """

        grid_h = self.model.predictor._iface.model.args.h // self.model.predictor._iface.model.args.patch_size[0]
        grid_w = ((self.model.predictor._iface.model.args.size[0] + self.model.predictor._iface.model.args.size[-1]) //
                  self.model.predictor._iface.model.args.patch_size[1])

        S = A.shape[0]
        assert A.shape[0] == A.shape[1], f"A must be square, got {A.shape}"

        Npatch = grid_h * grid_w

        # Handle CLS additional token introduced by ViTime -> Not representative of the real sequence
        if drop_cls == "auto":
            if S == Npatch + 1:
                A = A[1:, 1:]
            elif S == Npatch:
                pass
            else:
                raise ValueError(f"Unexpected S={S}. Expected {Npatch} or {Npatch + 1}.")
        elif drop_cls:
            A = A[1:, 1:]
        else:
            print("Warning: CLS token kept in attention map.")
            pass

        # Now A is (Npatch, Npatch) -> (2464,2464)
        assert A.shape == (Npatch, Npatch), f"After CLS handling expected {(Npatch, Npatch)} got {A.shape}"

        # Reshape tokens back to 2D grid: (rq, cq, rk, ck), expressing both queries and keys in 2D matrices
        A4 = A.reshape(grid_h, grid_w, grid_h, grid_w)

        # Average over image height both for queries and keys:
        A_time = A4.mean(axis=(0, 2))  # -> (cq, ck) = (grid_w, grid_w)

        # Got the 77 patches of queries and keys
        self.attention_patched = A_time
        return A_time

    def forward(self, x, prediction_horizon):
        """
        """
        # Check Fuse Attention Deactivation:
        fused_attn_status = []
        for name, module in self.model.predictor._iface.model.named_modules():
            if name.endswith(self.attention_layer_name) and isinstance(module, Attention):
                fused_attn_status.append(module.fused_attn)

        if fused_attn_status and all(status == False for status in fused_attn_status):
            print(f"✓ All {len(fused_attn_status)-1} attention layers have fused_attn = False")
        else:
            print(f"⚠ Warning: Not all attention layers have fused_attn = False")

        # Ensure hooks are active
        self.register_softmax_hooks()
        _ = self.model.prediction(x, prediction_horizon)

        if not self.attentions:
            print("⚠ Warning: No attention maps captured. Check layer names.")
            return

        # Use last captured module output
        self.multi_heads_tensor = list(self.attentions.values())[-1]

        # shape: (B,H,S,S) or (H,S,S) = Batch x Heads x Sequence x Sequence
        if self.multi_heads_tensor.dim() == 4:
            self.multi_heads_tensor = self.multi_heads_tensor[0]  # (Heads,S,S)

        print(f"✓ Captured Multi Heads Attentions Map shape (Heads,S,S): {self.multi_heads_tensor.shape}")
        return


