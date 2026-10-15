# Guía de uso del código

Los notebooks no repiten imports, rutas, carga de datos, Stable Diffusion, IP-Adapter ni BAP2: todo se
carga con los helpers de `src/`.

## Preparar Colab

```python
from google.colab import drive
drive.mount("/content/drive")

!pip install -q -r /content/drive/MyDrive/Notebook4/requirements.txt
!git clone -q https://github.com/tencent-ailab/IP-Adapter.git /content/IP-Adapter

import sys
sys.path[:0] = ["/content/drive/MyDrive/Notebook4", "/content/IP-Adapter"]
```

`DRIVE` debe apuntar a la raíz de este repositorio (con `data/` y `results/` completos, ver
[`data/README.md`](../data/README.md)).

## Carga paso a paso

```python
from src.notebook_setup import setup_runtime, setup_data, setup_generation_models, setup_final_bap2

ctx = setup_runtime("/content/drive/MyDrive/Notebook4", require_cuda=True)  # dispositivo y rutas
ctx = setup_data(ctx)                    # embeddings, datasets y DataLoaders de FIW
ctx = setup_generation_models(ctx)       # SD 1.5 + VAE + IP-Adapter-FaceID
ctx = setup_final_bap2(ctx, parent_scale=0.1)  # instala BAP2 y carga results/ckpt/bap2_4689/bap2_s4689.pt
```

O todo en una sola llamada:

```python
from src.notebook_setup import setup_full_generation_context

ctx = setup_full_generation_context(
    "/content/drive/MyDrive/Notebook4",
    load_data=True,
    load_arcface=False,   # True solo si se van a calcular similitudes de identidad
    parent_scale=0.1,
)
```

## Generar una imagen

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
    parent_scale=0.1,      # 0.0 = solo identidad del hijo
    seed=42,
    child_prompt=True,
    negative_prompt=CHILD_NEGATIVE_PROMPT,
)

display(img)
print(prompt)
print(alpha_stats)   # media / desv. / min / max de α en los 16 gates
```

## Parámetros principales

| Parámetro | Dónde | Efecto |
|---|---|---|
| `parent_scale` | `generate_bap2_image`, `attach_bap2` | Peso de la mezcla de padres (0.1 en el modelo final; 0 la desactiva). |
| `child_scale` | `attach_bap2` | Peso de la rama del hijo (1.0). |
| `gate_groups` | `attach_bap2` | Grupos de canales con α independiente (16). |
| `temperature` | `attach_bap2` | Divide los logits del gate antes del sigmoid. |
| `child_prompt` | `generate_bap2_image` | Usa el prompt detallado para niños/adolescentes en vez del prompt básico. |

## Instalar BAP2 en un UNet propio

```python
from src.att2 import attach_bap2

bap2_modules = attach_bap2(
    pipe.unet,            # UNet que ya tiene IP-Adapter-FaceID instalado
    num_tokens=4,
    parent_scale=0.1,
    device="cuda",
    dtype=torch.float16,
    copy_ip_weights=True, # copia to_k_ip / to_v_ip a las ramas hijo, padre y madre
)
```

El `encoder_hidden_states` que recibe el UNet debe tener la forma
`[texto (77) | hijo (4) | padre (4) | madre (4)]` tokens; ver `build_bap2_prompt_embeds` en
[`src/inference_utils.py`](../src/inference_utils.py).
