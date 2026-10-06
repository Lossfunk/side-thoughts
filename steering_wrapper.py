import torch
import torch.nn as nn


def bake_steering_into_weights(model, layer_idx, steering_vector, strength):
    """
    Bakes steering into down_proj bias before vLLM initialization.
    Effect is unconditional (all tokens), but survives fast_generate.
    """
    layer = model.model.layers[layer_idx]
    down_proj = layer.mlp.down_proj
    
    steer = (strength * steering_vector).to(down_proj.weight.dtype).to(down_proj.weight.device)
    
    if down_proj.bias is None:
        bias_dtype = down_proj.weight.dtype
        if not bias_dtype.is_floating_point:
            bias_dtype = torch.float16  # or torch.float16 / torch.float32

        down_proj.bias = torch.nn.Parameter(
            torch.zeros(down_proj.out_features, 
                       dtype=bias_dtype,
                       device=down_proj.weight.device)
        )
    
    with torch.no_grad():
        down_proj.bias.data += steer
        print(f"steering: applied to bias, {steer.shape}")