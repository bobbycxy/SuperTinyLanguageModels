import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
import math


class LoraLinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) implementation for Linear layers.
    
    This replaces a standard nn.Linear layer with a LoRA-adapted version that
    keeps the original weights frozen and adds trainable low-rank matrices.
    
    Args:
        original_layer (nn.Linear): The original linear layer to adapt
        rank (int): The rank of the low-rank adaptation (typically 4-64)
        alpha (float): Scaling factor for LoRA weights (typically rank * 2)
        dropout (float): Dropout probability for LoRA layers
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.merged = False
        
        # Store the original layer (frozen)
        self.original_layer = original_layer
        
        # Freeze original parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # LoRA matrices: W = W_0 + (alpha/rank) * B * A
        # A: (rank, in_features) - down projection
        # B: (out_features, rank) - up projection
        self.lora_A = nn.Linear(self.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False)
        
        # Dropout for LoRA path
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Scaling factor
        self.scaling = alpha / rank
        
        # Initialize LoRA weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize LoRA weights using Kaiming normal for A and zeros for B."""
        # Initialize A with small random values (Kaiming normal)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        
        # Initialize B with zeros so that initially LoRA has no effect
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA-adapted linear layer."""
        
        if self.merged:
            # If weights are merged, just use the original layer
            return self.original_layer(x)
        else:
            # Compute original output + LoRA adaptation
            original_output = self.original_layer(x)
            
            # LoRA path: x -> A -> dropout -> B -> scale
            lora_output = self.lora_A(x)
            lora_output = self.lora_dropout(lora_output)
            lora_output = self.lora_B(lora_output)
            lora_output = lora_output * self.scaling
            
            return original_output + lora_output
    
    def merge_weights_into_original(self):
        """Merge LoRA weights into the original layer for inference efficiency."""
        if not self.merged:
            # Compute the LoRA weight matrix: scaling * B.weight @ A.weight
            lora_weight = self.scaling * (self.lora_B.weight @ self.lora_A.weight)
            
            # Add to original weights
            self.original_layer.weight.data += lora_weight
            self.merged = True
    
    def unmerge_weights_from_original(self):
        """Unmerge LoRA weights from the original layer."""
        if self.merged:
            # Compute the LoRA weight matrix: scaling * B.weight @ A.weight
            lora_weight = self.scaling * (self.lora_B.weight @ self.lora_A.weight)
            
            # Subtract from original weights
            self.original_layer.weight.data -= lora_weight
            self.merged = False
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'rank={self.rank}, alpha={self.alpha}, dropout={self.dropout}'


class LoraEmbedding(nn.Module):
    """
    LoRA adaptation for Embedding layers.
    
    Args:
        original_layer (nn.Embedding): The original embedding layer to adapt
        rank (int): The rank of the low-rank adaptation
        alpha (float): Scaling factor for LoRA weights
    """
    
    def __init__(
        self,
        original_layer: nn.Embedding,
        rank: int = 16,
        alpha: float = 32.0
    ):
        super().__init__()
        
        self.num_embeddings = original_layer.num_embeddings
        self.embedding_dim = original_layer.embedding_dim
        self.rank = rank
        self.alpha = alpha
        self.merged = False
        
        # Store the original layer (frozen)
        self.original_layer = original_layer
        
        # Freeze original parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # LoRA matrices for embeddings
        self.lora_A = nn.Parameter(torch.zeros(rank, self.embedding_dim))
        self.lora_B = nn.Parameter(torch.zeros(self.num_embeddings, rank))
        
        # Scaling factor
        self.scaling = alpha / rank
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LoRA weights."""
        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA-adapted embedding layer."""
        
        if self.merged:
            return self.original_layer(input_ids)
        else:
            # Original embedding
            original_output = self.original_layer(input_ids)
            
            # LoRA adaptation
            lora_output = F.embedding(input_ids, self.lora_B @ self.lora_A) * self.scaling
            
            return original_output + lora_output
    
    def merge_weights_into_original(self):
        """Merge LoRA weights into the original embedding layer."""
        if not self.merged:
            lora_weight = self.scaling * (self.lora_B @ self.lora_A)
            self.original_layer.weight.data += lora_weight
            self.merged = True
    
    def unmerge_weights_from_original(self):
        """Unmerge LoRA weights from the original embedding layer."""
        if self.merged:
            lora_weight = self.scaling * (self.lora_B @ self.lora_A)
            self.original_layer.weight.data -= lora_weight
            self.merged = False


def apply_lora_to_model(
    model: nn.Module,
    target_modules: Optional[list] = None,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.0,
    exclude_modules: Optional[list] = None
) -> nn.Module:
    """
    Apply LoRA adaptation to specified modules in a model.
    
    Args:
        model: The model to apply LoRA to
        target_modules: List of module name patterns to apply LoRA to.
                       If None, applies to common transformer modules.
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        dropout: Dropout rate for LoRA layers
        exclude_modules: List of module names to exclude from LoRA adaptation
    
    Returns:
        Modified model with LoRA layers
    """
    
    if target_modules is None:
        # Default target modules for common transformer architectures
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention projections
            "gate_proj", "up_proj", "down_proj",     # MLP projections (LLaMA style)
            "fc1", "fc2",                            # MLP projections (GPT style)
            "c_attn", "c_proj",                      # GPT-2 style
            "query", "key", "value", "dense",        # BERT style
        ]
    
    if exclude_modules is None:
        exclude_modules = []
    
    def should_apply_lora(name: str) -> bool:
        """Check if LoRA should be applied to this module."""
        # Check if any target pattern matches
        for target in target_modules:
            if target in name:
                # Check if it's not excluded
                for exclude in exclude_modules:
                    if exclude in name:
                        return False
                return True
        return False
    
    # Collect modules to replace
    modules_to_replace = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and should_apply_lora(name):
            modules_to_replace[name] = LoraLinear(
                original_layer=module,
                rank=rank,
                alpha=alpha,
                dropout=dropout
            )
        elif isinstance(module, nn.Embedding) and should_apply_lora(name):
            modules_to_replace[name] = LoraEmbedding(
                original_layer=module,
                rank=rank,
                alpha=alpha
            )
    
    # Replace modules
    for name, new_module in modules_to_replace.items():
        # Navigate to parent module
        parent = model
        parts = name.split('.')
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        # Replace the module
        setattr(parent, parts[-1], new_module)
    
    # print(f"Applied LoRA to {len(modules_to_replace)} modules: {list(modules_to_replace.keys())}")
    
    return model


def get_lora_parameters(model: nn.Module):
    """Get only the LoRA parameters from a model."""
    lora_params = []
    for module in model.modules():
        if isinstance(module, (LoraLinear, LoraEmbedding)):
            if isinstance(module, LoraLinear):
                lora_params.extend([module.lora_A.weight, module.lora_B.weight])
            else:  # LoraEmbedding
                lora_params.extend([module.lora_A, module.lora_B])
    return lora_params


def count_lora_parameters(model: nn.Module) -> tuple:
    """
    Count LoRA parameters vs total parameters.
    
    Returns:
        tuple: (lora_params, total_params, percentage)
    """
    lora_params = sum(p.numel() for p in get_lora_parameters(model))
    total_params = sum(p.numel() for p in model.parameters())
    percentage = (lora_params / total_params) * 100 if total_params > 0 else 0
    
    return lora_params, total_params, percentage


def merge_all_lora_weights(model: nn.Module):
    """Merge all LoRA weights in the model for inference."""
    for module in model.modules():
        if isinstance(module, (LoraLinear, LoraEmbedding)):
            module.merge_weights_into_original()


def unmerge_all_lora_weights(model: nn.Module):
    """Unmerge all LoRA weights in the model."""
    for module in model.modules():
        if isinstance(module, (LoraLinear, LoraEmbedding)):
            module.unmerge_weights_from_original()


def save_lora_weights(model: nn.Module, path: str):
    """Save only the LoRA weights to a file."""
    lora_state_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, LoraLinear):
            lora_state_dict[f"{name}.lora_A.weight"] = module.lora_A.weight
            lora_state_dict[f"{name}.lora_B.weight"] = module.lora_B.weight
        elif isinstance(module, LoraEmbedding):
            lora_state_dict[f"{name}.lora_A"] = module.lora_A
            lora_state_dict[f"{name}.lora_B"] = module.lora_B
    
    torch.save(lora_state_dict, path)


def load_lora_weights(model: nn.Module, path: str):
    """Load LoRA weights from a file."""
    lora_state_dict = torch.load(path, map_location='cpu')
    
    # Create a mapping for LoRA modules
    lora_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, (LoraLinear, LoraEmbedding)):
            lora_modules[name] = module
    
    # Load weights
    for key, weight in lora_state_dict.items():
        # Parse the key to get module name and parameter name
        parts = key.split('.')
        param_name = '.'.join(parts[-2:])  # e.g., "lora_A.weight" or "lora_A"
        module_name = '.'.join(parts[:-2])  # Everything before the parameter
        
        if module_name in lora_modules:
            module = lora_modules[module_name]
            if param_name == "lora_A.weight":
                module.lora_A.weight.data = weight
            elif param_name == "lora_B.weight":
                module.lora_B.weight.data = weight
            elif param_name == "lora_A":
                module.lora_A.data = weight
            elif param_name == "lora_B":
                module.lora_B.data = weight