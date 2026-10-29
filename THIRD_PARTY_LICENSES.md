# Licenses

This project has three kinds of content, each under its own terms.

| Content | Files | License |
|---|---|---|
| Source code | `src/`, `tests/`, code cells in `notebooks/`, `setup.py` | [MIT](LICENSE) |
| Documentation, figures and metric tables | `*.md`, `docs/`, `results/figures/`, `results/eval/`, notebook text and outputs | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Trained Heredity Gate checkpoint | `results/ckpt/bap2_4689/bap2_s4689.pt` | **Non-commercial research use only** (inherited from FIW, IP-Adapter-FaceID and InsightFace, see below) |
| Split metadata | `data/splits/`, `data/ages/`, `data/gender_dict.pkl` | Derived from FIW: **non-commercial research use only**, and you need your own copy of FIW to use them |

## Third-party components

This repository does **not** redistribute any of the following. They are downloaded by the code or must
be obtained from their authors, and each keeps its own license. Check the original terms before any use,
especially commercial use.

| Component | Used for | License / terms |
|---|---|---|
| [Families in the Wild (FIW)](https://web.northeastern.edu/smilelab/fiw/) | Face images of families | Research use only, by request to the authors |
| [Realistic Vision V4.0 (noVAE)](https://huggingface.co/SG161222/Realistic_Vision_V4.0_noVAE) | Stable Diffusion 1.5 base model | CreativeML OpenRAIL-M |
| [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse) | VAE | MIT |
| [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) (code) | Identity adapter | Apache 2.0 |
| [IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID) (weights) | Identity adapter weights | Non-commercial research use only |
| [InsightFace](https://github.com/deepinsight/insightface) (code) | Face detection and ArcFace embeddings | MIT |
| InsightFace pretrained models (`buffalo_l`) | ArcFace embeddings | Non-commercial research use only |
| [MiVOLO](https://github.com/WildChlamydia/MiVOLO) / [`iitolstykh/mivolo_v2`](https://huggingface.co/iitolstykh/mivolo_v2) | Age labels for the children | Apache 2.0 |
| [pytorch-DEX](https://github.com/siriusdemon/pytorch-DEX) | Age estimation for E_age | No license stated in the repository; check with its author |
| [diffusers](https://github.com/huggingface/diffusers), [transformers](https://github.com/huggingface/transformers) | Diffusion pipeline | Apache 2.0 |
| [pytorch-fid](https://github.com/mseitzer/pytorch-fid) | FID | Apache 2.0 |

Licenses can change over time; the links above are the source of truth.
