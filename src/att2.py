# src/att2.py

"""
Installs BAP2 into a diffusers UNet.

BAP2 = Child-Anchor BAP with a contextual Heredity Gate.

Only attn2.processor (cross-attention) is replaced.
attn1.processor (self-attention) is left untouched.

Branch weights can be copied from:
1. Official IP-Adapter: to_k_ip / to_v_ip
2. BAP0 / No-Gate: to_k_child, to_k_father, to_k_mother
3. BAP1 / simple gate: to_k_child, to_k_father, to_k_mother

The BAP2 contextual gate is kept in float32.
"""

import torch
import torch.nn as nn

from .bap2 import BAP2


def _infer_hidden_size(unet, name: str):
    if name.startswith("mid_block"):
        return unet.config.block_out_channels[-1]

    if name.startswith("up_blocks"):
        block_id = int(name[len("up_blocks."):].split(".")[0])
        return list(reversed(unet.config.block_out_channels))[block_id]

    if name.startswith("down_blocks"):
        block_id = int(name[len("down_blocks."):].split(".")[0])
        return unet.config.block_out_channels[block_id]

    raise ValueError(f"Could not infer hidden_size for: {name}")


def attach_bap2(
    unet,
    num_tokens: int = 4,
    scale: float = 1.0,
    child_scale: float = 1.0,
    parent_scale: float = 0.1,
    gate_groups: int = 16,
    gate_hidden: int = 256,
    temperature: float = 1.0,
    device="cuda",
    dtype=torch.float16,
    copy_ip_weights: bool = True,
):
    """
    Replace only attn2.processor with BAP2.

    attn1.processor is left untouched.

    Important:
        The BAP2 contextual gate runs in fp32.
        The attention branches stay in `dtype`, usually fp16.
    """

    old_attn_procs = dict(unet.attn_processors)
    new_attn_procs = {}

    bap2_modules = nn.ModuleDict()

    copied_from_ip = 0
    copied_from_bap = 0
    replaced = 0
    kept = 0

    for name, old_proc in old_attn_procs.items():

        # Keep self-attention untouched
        if name.endswith("attn1.processor"):
            new_attn_procs[name] = old_proc
            kept += 1
            continue

        # Replace cross-attention only
        if not name.endswith("attn2.processor"):
            new_attn_procs[name] = old_proc
            kept += 1
            continue

        hidden_size = _infer_hidden_size(unet, name)
        cross_attention_dim = unet.config.cross_attention_dim

        proc = BAP2(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            num_tokens=num_tokens,
            scale=scale,
            child_scale=child_scale,
            parent_scale=parent_scale,
            gate_groups=gate_groups,
            gate_hidden=gate_hidden,
            temperature=temperature,
        ).to(device=device, dtype=dtype)

        # The contextual gate must stay in fp32
        proc.gate.float()

        if copy_ip_weights:
            with torch.no_grad():

                # Case 1: coming from the official IP-Adapter-FaceID
                if hasattr(old_proc, "to_k_ip") and hasattr(old_proc, "to_v_ip"):
                    k_weight = old_proc.to_k_ip.weight.detach().to(device=device, dtype=dtype)
                    v_weight = old_proc.to_v_ip.weight.detach().to(device=device, dtype=dtype)

                    proc.to_k_child.weight.copy_(k_weight)
                    proc.to_v_child.weight.copy_(v_weight)

                    proc.to_k_father.weight.copy_(k_weight)
                    proc.to_v_father.weight.copy_(v_weight)

                    proc.to_k_mother.weight.copy_(k_weight)
                    proc.to_v_mother.weight.copy_(v_weight)

                    copied_from_ip += 1

                # Case 2: coming from BAP0/BAP1/BAP2
                elif (
                    hasattr(old_proc, "to_k_child") and hasattr(old_proc, "to_v_child")
                    and hasattr(old_proc, "to_k_father") and hasattr(old_proc, "to_v_father")
                    and hasattr(old_proc, "to_k_mother") and hasattr(old_proc, "to_v_mother")
                ):
                    proc.to_k_child.weight.copy_(
                        old_proc.to_k_child.weight.detach().to(device=device, dtype=dtype)
                    )
                    proc.to_v_child.weight.copy_(
                        old_proc.to_v_child.weight.detach().to(device=device, dtype=dtype)
                    )

                    proc.to_k_father.weight.copy_(
                        old_proc.to_k_father.weight.detach().to(device=device, dtype=dtype)
                    )
                    proc.to_v_father.weight.copy_(
                        old_proc.to_v_father.weight.detach().to(device=device, dtype=dtype)
                    )

                    proc.to_k_mother.weight.copy_(
                        old_proc.to_k_mother.weight.detach().to(device=device, dtype=dtype)
                    )
                    proc.to_v_mother.weight.copy_(
                        old_proc.to_v_mother.weight.detach().to(device=device, dtype=dtype)
                    )

                    copied_from_bap += 1

        safe_name = (
            name.replace(".", "_")
                .replace("processor", "")
                .strip("_")
        )

        new_attn_procs[name] = proc
        bap2_modules[safe_name] = proc
        replaced += 1

    unet.set_attn_processor(new_attn_procs)

    print("✓ BAP2 installed")
    print(f"  attn2 replaced:          {replaced}")
    print(f"  attn1/untouched:         {kept}")
    print(f"  weights copied from IP:  {copied_from_ip}/{replaced}")
    print(f"  weights copied from BAP: {copied_from_bap}/{replaced}")
    print(f"  child_scale:             {child_scale}")
    print(f"  parent_scale:            {parent_scale}")
    print(f"  gate_groups:             {gate_groups}")
    print(f"  gate_hidden:             {gate_hidden}")
    print(f"  temperature:             {temperature}")
    print("  expected initial alpha:  0.5")
    print("  expected gate dtype:     float32")

    if copy_ip_weights and copied_from_ip == 0 and copied_from_bap == 0:
        print("WARNING: no weights were copied. Check which processor the UNet had before.")

    return bap2_modules