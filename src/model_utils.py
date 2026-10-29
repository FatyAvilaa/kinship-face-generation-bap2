from collections import Counter
from pathlib import Path
import sys

import torch

from .att2 import attach_bap2
from .config import ProjectPaths, RuntimeConfig


def processor_counts(unet) -> Counter:
    return Counter(type(proc).__name__ for proc in unet.attn_processors.values())


def count_ip_processors(unet) -> int:
    return sum(
        1
        for proc in unet.attn_processors.values()
        if hasattr(proc, "to_k_ip") and hasattr(proc, "to_v_ip")
    )


def get_bap2_modules(unet) -> dict:
    return {
        name: proc
        for name, proc in unet.attn_processors.items()
        if type(proc).__name__ == "BAP2"
    }


def load_diffusion_pipeline(
    runtime: RuntimeConfig,
    model_id: str = "SG161222/Realistic_Vision_V4.0_noVAE",
    vae_id: str = "stabilityai/sd-vae-ft-mse",
    safety_checker=None,
):
    from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline

    vae = AutoencoderKL.from_pretrained(
        vae_id,
        torch_dtype=runtime.dtype,
    ).to(runtime.device)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=runtime.dtype,
        safety_checker=safety_checker,
    ).to(runtime.device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe


def download_ip_adapter_faceid_checkpoint() -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id="h94/IP-Adapter-FaceID",
        filename="ip-adapter-faceid_sd15.bin",
    )


def load_ip_adapter_faceid(
    pipe,
    runtime: RuntimeConfig,
    checkpoint_path: str | Path | None = None,
    num_tokens: int = 4,
):
    try:
        from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
    except ModuleNotFoundError:
        ip_adapter_repo = "/content/IP-Adapter"
        if ip_adapter_repo not in sys.path:
            sys.path.insert(0, ip_adapter_repo)
        from ip_adapter.ip_adapter_faceid import IPAdapterFaceID

    checkpoint_path = checkpoint_path or download_ip_adapter_faceid_checkpoint()
    ip_model = IPAdapterFaceID(
        pipe,
        str(checkpoint_path),
        device=runtime.device,
        num_tokens=num_tokens,
    )
    ip_model.image_proj_model = ip_model.image_proj_model.to(dtype=runtime.dtype)

    n_ip = count_ip_processors(pipe.unet)
    if n_ip != 16:
        raise RuntimeError(
            f"IP-Adapter was not installed correctly: expected 16 IP processors, got {n_ip}."
        )
    return ip_model


def load_arcface_torch(runtime: RuntimeConfig, onnx_path: str | Path | None = None):
    import onnx
    import onnx2torch
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320))

    onnx_path = onnx_path or "/root/.insightface/models/buffalo_l/w600k_r50.onnx"
    onnx_model = onnx.load(str(onnx_path))
    arcface_torch = onnx2torch.convert(onnx_model).to(runtime.device).eval().float()
    for param in arcface_torch.parameters():
        param.requires_grad = False
    return arcface_torch


def install_bap2(
    pipe,
    runtime: RuntimeConfig,
    parent_scale: float = 0.1,
    scale: float = 1.0,
    child_scale: float = 1.0,
    gate_groups: int = 16,
    gate_hidden: int = 256,
    temperature: float = 1.0,
    copy_ip_weights: bool = True,
) -> dict:
    attach_bap2(
        pipe.unet,
        num_tokens=4,
        scale=scale,
        child_scale=child_scale,
        parent_scale=parent_scale,
        gate_groups=gate_groups,
        gate_hidden=gate_hidden,
        temperature=temperature,
        device=runtime.device,
        dtype=runtime.dtype,
        copy_ip_weights=copy_ip_weights,
    )
    bap2_modules = get_bap2_modules(pipe.unet)
    if len(bap2_modules) != 16:
        raise RuntimeError(f"Expected 16 BAP2 modules, got {len(bap2_modules)}.")
    return bap2_modules


def load_bap2_gate_checkpoint(
    pipe,
    checkpoint_path: str | Path,
    map_location: str = "cpu",
) -> tuple[dict, dict]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    bap2_modules = get_bap2_modules(pipe.unet)
    if len(bap2_modules) != 16:
        raise RuntimeError("Install BAP2 before loading gate checkpoint.")

    ckpt = torch.load(checkpoint_path, map_location=map_location)
    gate_state = ckpt["bap2_gate_state"]

    missing = []

    for name, proc in bap2_modules.items():

        safe_name = (
            name.replace(".", "_")
                .replace("processor", "")
                .strip("_")
        )

        if safe_name in gate_state:
            proc.gate.load_state_dict(
                gate_state[safe_name]
            )
        else:
            missing.append(name)

    if missing:
        raise RuntimeError(f"Missing gate layers in checkpoint: {missing[:5]}")

    return bap2_modules, ckpt


def load_final_bap2(
    pipe,
    runtime: RuntimeConfig,
    paths: ProjectPaths,
    parent_scale: float = 0.1,
) -> tuple[dict, dict]:
    bap2_modules = install_bap2(pipe, runtime, parent_scale=parent_scale)
    bap2_modules, ckpt = load_bap2_gate_checkpoint(pipe, paths.final_bap2_checkpoint)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    return bap2_modules, ckpt


def load_generation_stack(runtime: RuntimeConfig):
    pipe = load_diffusion_pipeline(runtime)
    ip_model = load_ip_adapter_faceid(pipe, runtime)
    return pipe, ip_model
