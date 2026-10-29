from dataclasses import dataclass
from pathlib import Path

import torch


DEFAULT_DRIVE = "/content/drive/MyDrive/Notebook4"


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    dtype: torch.dtype


@dataclass(frozen=True)
class ProjectPaths:
    drive: Path

    @property
    def data_dir(self) -> Path:
        return self.drive / "data"

    @property
    def results_dir(self) -> Path:
        return self.drive / "results"

    @property
    def embeddings_path(self) -> Path:
        return self.data_dir / "embeddings" / "arcface_insightface.pkl"

    @property
    def gender_cache_path(self) -> Path:
        return self.data_dir / "gender_dict.pkl"

    @property
    def fiw_root(self) -> Path:
        return self.data_dir / "raw" / "FIDs"

    @property
    def train_csv(self) -> Path:
        return self.data_dir / "splits" / "train_identity.csv"

    @property
    def val_csv(self) -> Path:
        return self.data_dir / "splits" / "val_identity.csv"

    @property
    def test_csv(self) -> Path:
        return self.data_dir / "splits" / "test_identity.csv"

    @property
    def ages_csv(self) -> Path:
        return self.data_dir / "ages" / "mivolo_ages.csv"

    @property
    def final_bap2_checkpoint(self) -> Path:
        return (
            self.results_dir
            / "ckpt"
            / "bap2_4689"
            / "bap2_s4689.pt"
        )


def get_runtime(require_cuda: bool = False) -> RuntimeConfig:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if require_cuda and device != "cuda":
        raise RuntimeError("CUDA is required for this workflow.")
    dtype = torch.float16 if device == "cuda" else torch.float32
    return RuntimeConfig(device=device, dtype=dtype)


def get_paths(drive: str | Path = DEFAULT_DRIVE) -> ProjectPaths:
    return ProjectPaths(drive=Path(drive))


def print_runtime(runtime: RuntimeConfig, paths: ProjectPaths | None = None) -> None:
    print("DEVICE:", runtime.device)
    print("DTYPE:", runtime.dtype)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("GPU: NO GPU")
    if paths is not None:
        print("DRIVE:", paths.drive)
