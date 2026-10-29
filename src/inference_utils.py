from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms as T


DEFAULT_NEGATIVE_PROMPT = (
    "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
)

CHILD_NEGATIVE_PROMPT = (
    "monochrome, lowres, bad anatomy, worst quality, low quality, blurry, "
    "adult, middle aged, old, elderly, beard, mustache, facial hair, wrinkles, "
    "cropped, close-up, extreme close-up, out of frame, cut off face, "
    "partial face, face not fully visible, forehead cropped, chin cropped, "
    "profile view, side view, studio lighting, dramatic lighting, overprocessed skin"
)


def ensure_2d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.unsqueeze(0)
    return x


def get_embedding(embeddings: dict[str, Any], key: str) -> torch.Tensor:
    key = str(key)
    if key not in embeddings:
        raise KeyError(f"Missing embedding for {key}")
    return torch.tensor(embeddings[key], dtype=torch.float32).unsqueeze(0)


def set_parent_scale_bap2(unet, value: float, expected: int | None = 16) -> int:
    count = 0
    for proc in unet.attn_processors.values():
        if type(proc).__name__ == "BAP2":
            proc.parent_scale = value
            count += 1
    if expected is not None and count != expected:
        raise RuntimeError(f"Expected {expected} BAP2 processors, found {count}.")
    return count


def collect_bap2_alpha_stats(unet) -> dict[str, float | bool | None]:
    vals = []
    for proc in unet.attn_processors.values():
        if type(proc).__name__ == "BAP2" and getattr(proc, "last_alpha", None) is not None:
            vals.append(proc.last_alpha.detach().float().cpu())

    if not vals:
        return {
            "alpha_mean": None,
            "alpha_std": None,
            "alpha_min": None,
            "alpha_max": None,
            "alpha_has_nan": None,
        }

    x = torch.cat(vals, dim=0)
    return {
        "alpha_mean": x.mean().item(),
        "alpha_std": x.std().item(),
        "alpha_min": x.min().item(),
        "alpha_max": x.max().item(),
        "alpha_has_nan": torch.isnan(x).any().item(),
    }


def subject_from_gender(gender: str, age: int) -> str:
    gender = str(gender).strip().lower()
    if age <= 12:
        if gender == "man":
            return "boy"
        if gender == "woman":
            return "girl"
        return "child"
    if gender == "man":
        return "teenage boy"
    if gender == "woman":
        return "teenage girl"
    return "teenager"


def build_basic_prompt(gender: str, age: int | float) -> str:
    gender = gender if gender in ["man", "woman", "person"] else "person"
    return f"portrait photo of a {gender}, age {int(age)}, realistic, neutral expression"


def build_child_prompt(gender: str, age: int | float) -> tuple[str, int]:
    age_int = int(round(float(age)))
    subject = subject_from_gender(gender, age_int)
    prompt = (
        f"realistic portrait of a {subject}, age {age_int}, "
        f"full face visible, centered, head and shoulders, looking at camera, "
        f"neutral expression, natural appearance, soft natural lighting, "
        f"visible hair, visible forehead, visible chin, simple background"
    )
    return prompt, age_int


def build_bap2_prompt_embeds(
    pipe,
    ip_model,
    h_c: torch.Tensor,
    h_p: torch.Tensor,
    h_m: torch.Tensor,
    prompt: str,
    negative_prompt: str,
    device: str,
    dtype: torch.dtype,
):
    h_c = ensure_2d(h_c).to(device=device, dtype=dtype)
    h_p = ensure_2d(h_p).to(device=device, dtype=dtype)
    h_m = ensure_2d(h_m).to(device=device, dtype=dtype)

    child_tokens = ip_model.image_proj_model(h_c)
    father_tokens = ip_model.image_proj_model(h_p)
    mother_tokens = ip_model.image_proj_model(h_m)

    zero_child_tokens = ip_model.image_proj_model(torch.zeros_like(h_c))
    zero_father_tokens = ip_model.image_proj_model(torch.zeros_like(h_p))
    zero_mother_tokens = ip_model.image_proj_model(torch.zeros_like(h_m))

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=[prompt],
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=[negative_prompt],
    )

    prompt_embeds = torch.cat(
        [prompt_embeds, child_tokens, father_tokens, mother_tokens],
        dim=1,
    )
    negative_prompt_embeds = torch.cat(
        [negative_prompt_embeds, zero_child_tokens, zero_father_tokens, zero_mother_tokens],
        dim=1,
    )
    return prompt_embeds, negative_prompt_embeds


@torch.inference_mode()
def generate_bap2_image(
    pipe,
    ip_model,
    h_c: torch.Tensor,
    h_p: torch.Tensor,
    h_m: torch.Tensor,
    gender: str,
    age: int | float,
    device: str,
    dtype: torch.dtype,
    parent_scale: float = 0.1,
    seed: int = 42,
    guidance_scale: float = 4.0,
    num_inference_steps: int = 30,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    child_prompt: bool = False,
) -> tuple[Any, str, dict[str, float | bool | None]]:
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    ip_model.image_proj_model.eval()

    set_parent_scale_bap2(pipe.unet, parent_scale)

    if child_prompt:
        prompt, _ = build_child_prompt(gender, age)
    else:
        prompt = build_basic_prompt(gender, age)

    prompt_embeds, negative_prompt_embeds = build_bap2_prompt_embeds(
        pipe=pipe,
        ip_model=ip_model,
        h_c=h_c,
        h_p=h_p,
        h_m=h_m,
        prompt=prompt,
        negative_prompt=negative_prompt,
        device=device,
        dtype=dtype,
    )

    generator = torch.Generator(device=device).manual_seed(int(seed))
    image = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]

    return image, prompt, collect_bap2_alpha_stats(pipe.unet)


def make_arcface_embedder(arcface_torch, device: str):
    preprocess_arcface = T.Compose(
        [
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    @torch.inference_mode()
    def arcface_embed_from_pil(img):
        x = preprocess_arcface(img.convert("RGB")).unsqueeze(0).to(device).float()
        emb = arcface_torch(x)
        return F.normalize(emb, p=2, dim=1)

    return arcface_embed_from_pil
