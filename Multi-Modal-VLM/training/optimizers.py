"""
Optimizer and learning rate scheduler utilities
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, List
import math


def create_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = 'adamw',
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    momentum: float = 0.9,
    layer_decay: Optional[float] = None,
    freeze_params: Optional[List[str]] = None
) -> torch.optim.Optimizer:
    """
    Create optimizer with optional layer-wise learning rate decay

    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer ('adamw', 'adam', 'sgd')
        learning_rate: Base learning rate
        weight_decay: Weight decay factor
        betas: Adam betas
        eps: Adam epsilon
        momentum: SGD momentum
        layer_decay: Layer-wise LR decay factor
        freeze_params: List of parameter name prefixes to freeze

    Returns:
        Optimizer instance
    """
    # Filter frozen parameters
    if freeze_params:
        parameters = [
            p for n, p in model.named_parameters()
            if p.requires_grad and not any(prefix in n for prefix in freeze_params)
        ]
    else:
        parameters = [p for p in model.parameters() if p.requires_grad]

    # Create parameter groups with layer-wise LR decay
    if layer_decay is not None and layer_decay < 1.0:
        param_groups = []

        # Group parameters by layer
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if freeze_params and any(prefix in name for prefix in freeze_params):
                continue

            # Determine layer depth
            if 'vision_encoder' in name:
                depth = _get_layer_depth(name, 'vision')
            elif 'language_encoder' in name:
                depth = _get_layer_depth(name, 'language')
            elif 'fusion' in name:
                depth = _get_layer_depth(name, 'fusion')
            else:
                depth = 0

            # Compute scaled learning rate
            lr_scale = layer_decay ** (20 - depth)  # Assuming max 20 layers
            lr = learning_rate * lr_scale

            param_groups.append({
                'params': [param],
                'lr': lr,
                'weight_decay': weight_decay if len(param.shape) > 1 else 0.0
            })
    else:
        # No layer-wise decay - differentiate only by weight decay
        no_decay = ['bias', 'LayerNorm', 'layer_norm', 'bn']
        param_groups = [
            {
                'params': [p for n, p in model.named_parameters()
                          if p.requires_grad and not any(nd in n for nd in no_decay)
                          and not (freeze_params and any(prefix in n for prefix in freeze_params))],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters()
                          if p.requires_grad and any(nd in n for nd in no_decay)
                          and not (freeze_params and any(prefix in n for prefix in freeze_params))],
                'weight_decay': 0.0
            }
        ]

    # Create optimizer
    if optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(
            param_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer


def _get_layer_depth(param_name: str, module_type: str) -> int:
    """Get layer depth for layer-wise LR decay"""
    if 'embedding' in param_name or 'pos_' in param_name:
        return 0

    # Extract layer number
    if 'layer' in param_name:
        try:
            parts = param_name.split('.')
            for i, part in enumerate(parts):
                if 'layer' in part and i + 1 < len(parts):
                    return int(parts[i + 1])
        except:
            pass

    return 10  # Default middle depth


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'cosine',
    num_training_steps: int = 10000,
    num_warmup_steps: int = 1000,
    min_lr: float = 1e-6,
    **kwargs
) -> _LRScheduler:
    """
    Create learning rate scheduler

    Args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler ('cosine', 'linear', 'polynomial', 'constant')
        num_training_steps: Total training steps
        num_warmup_steps: Warmup steps
        min_lr: Minimum learning rate
        **kwargs: Additional scheduler arguments

    Returns:
        Scheduler instance
    """
    if scheduler_type == 'cosine':
        scheduler = CosineSchedulerWithWarmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr=min_lr
        )
    elif scheduler_type == 'linear':
        scheduler = LinearSchedulerWithWarmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr=min_lr
        )
    elif scheduler_type == 'polynomial':
        power = kwargs.get('power', 1.0)
        scheduler = PolynomialSchedulerWithWarmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            power=power,
            min_lr=min_lr
        )
    elif scheduler_type == 'constant':
        scheduler = ConstantSchedulerWithWarmup(
            optimizer,
            num_warmup_steps=num_warmup_steps
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


class CosineSchedulerWithWarmup(_LRScheduler):
    """
    Cosine learning rate scheduler with warmup
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            # Linear warmup
            return [
                base_lr * self.last_epoch / max(1, self.num_warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.num_warmup_steps) / \
                      max(1, self.num_training_steps - self.num_warmup_steps)
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class LinearSchedulerWithWarmup(_LRScheduler):
    """
    Linear learning rate scheduler with warmup
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            # Linear warmup
            return [
                base_lr * self.last_epoch / max(1, self.num_warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            # Linear decay
            progress = (self.last_epoch - self.num_warmup_steps) / \
                      max(1, self.num_training_steps - self.num_warmup_steps)
            return [
                self.min_lr + (base_lr - self.min_lr) * (1 - progress)
                for base_lr in self.base_lrs
            ]


class PolynomialSchedulerWithWarmup(_LRScheduler):
    """
    Polynomial learning rate scheduler with warmup
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        power: float = 1.0,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            # Linear warmup
            return [
                base_lr * self.last_epoch / max(1, self.num_warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            # Polynomial decay
            progress = (self.last_epoch - self.num_warmup_steps) / \
                      max(1, self.num_training_steps - self.num_warmup_steps)
            return [
                self.min_lr + (base_lr - self.min_lr) * (1 - progress) ** self.power
                for base_lr in self.base_lrs
            ]


class ConstantSchedulerWithWarmup(_LRScheduler):
    """
    Constant learning rate with warmup
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        last_epoch: int = -1
    ):
        self.num_warmup_steps = num_warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            return [
                base_lr * self.last_epoch / max(1, self.num_warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            return self.base_lrs


if __name__ == "__main__":
    # Test optimizer and scheduler creation
    model = torch.nn.Linear(10, 10)

    optimizer = create_optimizer(
        model,
        optimizer_type='adamw',
        learning_rate=1e-4,
        weight_decay=0.01
    )
    print(f"Created optimizer: {type(optimizer).__name__}")

    scheduler = create_scheduler(
        optimizer,
        scheduler_type='cosine',
        num_training_steps=10000,
        num_warmup_steps=1000
    )
    print(f"Created scheduler: {type(scheduler).__name__}")

    # Test LR schedule
    lrs = []
    for step in range(100):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    print(f"Initial LR: {lrs[0]:.6f}")
    print(f"LR at step 50: {lrs[50]:.6f}")
    print(f"LR at step 99: {lrs[99]:.6f}")
