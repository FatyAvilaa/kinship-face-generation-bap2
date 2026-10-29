# Usage guide

The notebooks do not repeat imports, paths, data loading, or the Stable Diffusion, IP-Adapter and BAP2 setup:
all of that is loaded with the helpers in `src/`.

## 1. Prepare Google Colab

The project was run on Google Colab with a GPU, with this repository copied to
`/content/drive/MyDrive/Notebook4` on Google Drive. You can use any other folder: just pass its path as
`drive` to the setup functions below.

```python
from google.colab import drive
drive.mount("/content/drive")

!pip install -q -r /content/drive/MyDrive/Notebook4/requirements.txt
!git clone -q https://github.com/tencent-ailab/IP-Adapter.git /content/IP-Adapter

import sys
sys.path[:0] = ["/content/drive/MyDrive/Notebook4", "/content/IP-Adapter"]
```

The `drive` folder must be the root of this repository, with `data/` and `results/` complete (see
[`data/README.md`](../data/README.md)).

## 2. Load everything step by step

```python
from src.notebook_setup import setup_runtime, setup_data, setup_generation_models, setup_final_bap2

ctx = setup_runtime("/content/drive/MyDrive/Notebook4", require_cuda=True)  # device and paths
ctx = setup_data(ctx)                    # FIW embeddings, datasets and DataLoaders
ctx = setup_generation_models(ctx)       # SD 1.5 + VAE + IP-Adapter-FaceID
ctx = setup_final_bap2(ctx, parent_scale=0.1)  # installs BAP2 and loads results/ckpt/bap2_4689/bap2_s4689.pt
```

Or in a single call:

```python
from src.notebook_setup import setup_full_generation_context

ctx = setup_full_generation_context(
    "/content/drive/MyDrive/Notebook4",
    load_data=True,
    load_arcface=False,   # True only if you will compute identity similarities
    parent_scale=0.1,
)
```

## 3. Generate an image

```python
from src.inference_utils import generate_bap2_image, get_embedding, CHILD_NEGATIVE_PROMPT

sample = ctx.data.test_ds[0]
h_c = get_embedding(ctx.data.embeddings, sample["key_c"])
h_p = get_embedding(ctx.data.embeddings, sample["key_p"])
h_m = get_embedding(ctx.data.embeddings, sample["key_m"])

img, prompt, alpha_stats = generate_bap2_image(
    pipe=ctx.pipe,
    ip_model=ctx.ip_model,
    h_c=h_c, h_p=h_p, h_m=h_m,
    gender=sample["gender"],
    age=10,
    device=ctx.runtime.device,
    dtype=ctx.runtime.dtype,
    parent_scale=0.1,      # 0.0 = child identity only
    seed=42,
    child_prompt=True,
    negative_prompt=CHILD_NEGATIVE_PROMPT,
)

display(img)
print(prompt)
print(alpha_stats)   # mean / std / min / max of α across the 16 gates
```

## Main parameters

| Parameter | Where | Effect |
|---|---|---|
| `parent_scale` | `generate_bap2_image`, `attach_bap2` | Weight of the parent mix (0.1 in the final model; 0 turns it off). |
| `child_scale` | `attach_bap2` | Weight of the child branch (1.0). |
| `gate_groups` | `attach_bap2` | Number of channel groups with an independent α (16). |
| `temperature` | `attach_bap2` | Divides the gate logits before the sigmoid. |
| `child_prompt` | `generate_bap2_image` | Uses the detailed child/teen prompt instead of the basic prompt. |

## Install BAP2 into your own UNet

```python
from src.att2 import attach_bap2

bap2_modules = attach_bap2(
    pipe.unet,            # a UNet that already has IP-Adapter-FaceID installed
    num_tokens=4,
    parent_scale=0.1,
    device="cuda",
    dtype=torch.float16,
    copy_ip_weights=True, # copies to_k_ip / to_v_ip into the child, father and mother branches
)
```

The `encoder_hidden_states` passed to the UNet must be laid out as
`[text (77) | child (4) | father (4) | mother (4)]` tokens; see `build_bap2_prompt_embeds` in
[`src/inference_utils.py`](../src/inference_utils.py).
