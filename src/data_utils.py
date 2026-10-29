import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from .config import ProjectPaths
from .dataset import FIWTriadsDataset


@dataclass
class FIWDataBundle:
    embeddings: dict[str, Any]
    gender_dict: dict[str, Any]
    train_ds: FIWTriadsDataset
    val_ds: FIWTriadsDataset
    test_ds: FIWTriadsDataset
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def load_pickle(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_embeddings(paths: ProjectPaths) -> dict[str, Any]:
    return load_pickle(paths.embeddings_path)


def load_gender_dict(paths: ProjectPaths) -> dict[str, Any]:
    if not paths.gender_cache_path.exists():
        return {}
    return load_pickle(paths.gender_cache_path)


def build_fiw_datasets(
    paths: ProjectPaths,
    embeddings: dict[str, Any] | None = None,
    gender_dict: dict[str, Any] | None = None,
    image_size: int = 512,
) -> tuple[FIWTriadsDataset, FIWTriadsDataset, FIWTriadsDataset]:
    embeddings = embeddings if embeddings is not None else load_embeddings(paths)
    gender_dict = gender_dict if gender_dict is not None else load_gender_dict(paths)
    ages_csv = str(paths.ages_csv) if paths.ages_csv.exists() else None

    kwargs = {
        "embeddings": embeddings,
        "fiw_root": str(paths.fiw_root),
        "gender_dict": gender_dict,
        "ages_csv_path": ages_csv,
        "image_size": image_size,
    }
    train_ds = FIWTriadsDataset(csv_path=str(paths.train_csv), **kwargs)
    val_ds = FIWTriadsDataset(csv_path=str(paths.val_csv), **kwargs)
    test_ds = FIWTriadsDataset(csv_path=str(paths.test_csv), **kwargs)
    return train_ds, val_ds, test_ds


def build_loaders(
    train_ds,
    val_ds,
    test_ds,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def load_fiw_bundle(
    paths: ProjectPaths,
    image_size: int = 512,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> FIWDataBundle:
    embeddings = load_embeddings(paths)
    gender_dict = load_gender_dict(paths)
    train_ds, val_ds, test_ds = build_fiw_datasets(
        paths,
        embeddings=embeddings,
        gender_dict=gender_dict,
        image_size=image_size,
    )
    train_loader, val_loader, test_loader = build_loaders(
        train_ds,
        val_ds,
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return FIWDataBundle(
        embeddings=embeddings,
        gender_dict=gender_dict,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
