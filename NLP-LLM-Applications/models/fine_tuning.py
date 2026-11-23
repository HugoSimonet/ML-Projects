"""
Fine-tuning Pipeline with Parameter-Efficient Methods.

This module implements various fine-tuning strategies:
- Full fine-tuning
- LoRA (Low-Rank Adaptation)
- AdaLoRA (Adaptive LoRA)
- Prefix tuning
- Adapter layers
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    r: int = 8  # Rank of the low-rank matrices
    lora_alpha: int = 16  # Scaling factor
    lora_dropout: float = 0.1
    target_modules: List[str] = None  # Modules to apply LoRA to
    bias: str = "none"  # Bias handling: 'none', 'all', 'lora_only'
    modules_to_save: Optional[List[str]] = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


class LoRALayer(nn.Module):
    """
    LoRA layer implementation.

    Applies low-rank adaptation: h = W0*x + (B*A)*x * (alpha/r)
    where W0 is the frozen pre-trained weight, B and A are low-rank matrices
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
        self.scaling = self.lora_alpha / self.r
        self.dropout = nn.Dropout(p=lora_dropout)

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA layer."""
        # Apply low-rank adaptation
        x = self.dropout(x)
        adaptation = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return adaptation


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""

    def __init__(
        self,
        linear_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        merge_weights: bool = False
    ):
        super().__init__()

        self.linear = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            r, lora_alpha, lora_dropout
        )
        self.merge_weights = merge_weights
        self.merged = False

        # Freeze original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.merged:
            return self.linear(x)

        return self.linear(x) + self.lora(x)

    def merge(self):
        """Merge LoRA weights into the base layer."""
        if not self.merged:
            self.linear.weight.data += (
                self.lora.lora_B @ self.lora.lora_A
            ) * self.lora.scaling
            self.merged = True

    def unmerge(self):
        """Separate LoRA weights from the base layer."""
        if self.merged:
            self.linear.weight.data -= (
                self.lora.lora_B @ self.lora.lora_A
            ) * self.lora.scaling
            self.merged = False


class LoRAFineTuner:
    """
    LoRA-based fine-tuning manager.

    Features:
    - Apply LoRA to target modules
    - Manage trainable parameters
    - Save/load LoRA weights
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[LoRAConfig] = None
    ):
        """
        Initialize LoRA fine-tuner.

        Args:
            model: Base model to fine-tune
            config: LoRA configuration
        """
        self.model = model
        self.config = config or LoRAConfig()
        self.lora_layers: Dict[str, LoRALinear] = {}

        self._apply_lora()

    def _apply_lora(self):
        """Apply LoRA to target modules."""
        for name, module in self.model.named_modules():
            if self._is_target_module(name):
                if isinstance(module, nn.Linear):
                    # Replace with LoRA linear
                    lora_linear = LoRALinear(
                        module,
                        r=self.config.r,
                        lora_alpha=self.config.lora_alpha,
                        lora_dropout=self.config.lora_dropout
                    )
                    self._replace_module(name, lora_linear)
                    self.lora_layers[name] = lora_linear

    def _is_target_module(self, name: str) -> bool:
        """Check if module is a target for LoRA."""
        return any(target in name for target in self.config.target_modules)

    def _replace_module(self, name: str, new_module: nn.Module):
        """Replace a module in the model."""
        parent_name, module_name = name.rsplit('.', 1) if '.' in name else ('', name)

        if parent_name:
            parent = self.model.get_submodule(parent_name)
        else:
            parent = self.model

        setattr(parent, module_name, new_module)

    def get_trainable_parameters(self) -> Tuple[int, int]:
        """Get count of trainable vs total parameters."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total

    def save_lora_weights(self, path: str):
        """Save only LoRA weights."""
        lora_state = {
            name: layer.lora.state_dict()
            for name, layer in self.lora_layers.items()
        }
        torch.save(lora_state, path)

    def load_lora_weights(self, path: str):
        """Load LoRA weights."""
        lora_state = torch.load(path)
        for name, state_dict in lora_state.items():
            if name in self.lora_layers:
                self.lora_layers[name].lora.load_state_dict(state_dict)

    def merge_and_unload(self):
        """Merge LoRA weights and remove LoRA layers."""
        for layer in self.lora_layers.values():
            layer.merge()


class AdapterLayer(nn.Module):
    """
    Adapter layer for parameter-efficient fine-tuning.

    Adds a bottleneck layer after transformer blocks.
    """

    def __init__(
        self,
        hidden_size: int,
        adapter_size: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through adapter."""
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        x = self.dropout(x)
        return x + residual


class PrefixTuning(nn.Module):
    """
    Prefix tuning implementation.

    Learns continuous task-specific vectors (prefixes) prepended to the input.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        prefix_length: int = 10,
        prefix_hidden_size: int = 512
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.prefix_length = prefix_length

        # Prefix parameters (reparameterized)
        self.prefix_encoder = nn.Sequential(
            nn.Linear(prefix_length, prefix_hidden_size),
            nn.Tanh(),
            nn.Linear(prefix_hidden_size, num_layers * 2 * num_heads * head_dim)
        )

        # Trainable prefix tokens
        self.prefix_tokens = nn.Parameter(
            torch.randn(prefix_length)
        )

    def forward(self) -> torch.Tensor:
        """Generate prefix key-value pairs."""
        prefix_states = self.prefix_encoder(self.prefix_tokens)

        # Reshape to [num_layers, 2, num_heads, prefix_length, head_dim]
        batch_size = 1
        prefix_states = prefix_states.view(
            self.num_layers,
            2,  # Key and value
            self.num_heads,
            self.prefix_length,
            self.head_dim
        )

        return prefix_states


class FullFineTuner:
    """
    Full model fine-tuning manager.

    Features:
    - Layer freezing
    - Learning rate scheduling
    - Gradient checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        freeze_layers: Optional[List[str]] = None,
        freeze_except: Optional[List[str]] = None
    ):
        """
        Initialize full fine-tuner.

        Args:
            model: Model to fine-tune
            freeze_layers: Layer names to freeze
            freeze_except: Freeze all except these layers
        """
        self.model = model
        self.freeze_layers = freeze_layers or []
        self.freeze_except = freeze_except

        self._apply_freezing()

    def _apply_freezing(self):
        """Apply layer freezing."""
        if self.freeze_except:
            # Freeze all except specified
            for name, param in self.model.named_parameters():
                if not any(except_layer in name for except_layer in self.freeze_except):
                    param.requires_grad = False
        elif self.freeze_layers:
            # Freeze specified layers
            for name, param in self.model.named_parameters():
                if any(freeze_layer in name for freeze_layer in self.freeze_layers):
                    param.requires_grad = False

    def unfreeze_layers(self, layer_names: List[str]):
        """Unfreeze specific layers."""
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in layer_names):
                param.requires_grad = True

    def freeze_all_except(self, layer_names: List[str]):
        """Freeze all layers except specified."""
        for name, param in self.model.named_parameters():
            param.requires_grad = any(layer in name for layer in layer_names)

    def get_trainable_parameters(self) -> Tuple[int, int]:
        """Get count of trainable vs total parameters."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory."""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()


class FineTuningStrategy:
    """
    Unified fine-tuning strategy manager.

    Supports multiple fine-tuning approaches and provides a unified interface.
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: str = "lora",
        **strategy_kwargs
    ):
        """
        Initialize fine-tuning strategy.

        Args:
            model: Model to fine-tune
            strategy: Fine-tuning strategy ('lora', 'full', 'adapter', 'prefix')
            **strategy_kwargs: Strategy-specific arguments
        """
        self.model = model
        self.strategy = strategy.lower()

        if self.strategy == "lora":
            config = LoRAConfig(**strategy_kwargs)
            self.tuner = LoRAFineTuner(model, config)
        elif self.strategy == "full":
            self.tuner = FullFineTuner(model, **strategy_kwargs)
        elif self.strategy == "adapter":
            # Would implement adapter-based tuning
            raise NotImplementedError("Adapter tuning not yet implemented")
        elif self.strategy == "prefix":
            # Would implement prefix tuning
            raise NotImplementedError("Prefix tuning not yet implemented")
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def get_trainable_parameters(self) -> Tuple[int, int]:
        """Get trainable parameter count."""
        return self.tuner.get_trainable_parameters()

    def print_trainable_parameters(self):
        """Print trainable parameter statistics."""
        trainable, total = self.get_trainable_parameters()
        percentage = 100 * trainable / total

        print(f"Trainable parameters: {trainable:,}")
        print(f"Total parameters: {total:,}")
        print(f"Percentage trainable: {percentage:.2f}%")

    def save(self, path: str):
        """Save fine-tuned weights."""
        if hasattr(self.tuner, 'save_lora_weights'):
            self.tuner.save_lora_weights(path)
        else:
            torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        """Load fine-tuned weights."""
        if hasattr(self.tuner, 'load_lora_weights'):
            self.tuner.load_lora_weights(path)
        else:
            self.model.load_state_dict(torch.load(path))


# Utility functions

def apply_lora_to_model(
    model: nn.Module,
    r: int = 8,
    lora_alpha: int = 16,
    target_modules: Optional[List[str]] = None
) -> LoRAFineTuner:
    """
    Quick function to apply LoRA to a model.

    Args:
        model: Model to apply LoRA to
        r: LoRA rank
        lora_alpha: LoRA alpha
        target_modules: Target module names

    Returns:
        LoRAFineTuner instance
    """
    config = LoRAConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules or ["q_proj", "v_proj"]
    )
    return LoRAFineTuner(model, config)


def count_parameters(model: nn.Module, only_trainable: bool = True) -> int:
    """Count model parameters."""
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def freeze_model(model: nn.Module):
    """Freeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model: nn.Module):
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True
