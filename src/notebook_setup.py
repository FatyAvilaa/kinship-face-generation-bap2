from dataclasses import dataclass

from .config import DEFAULT_DRIVE, ProjectPaths, RuntimeConfig, get_paths, get_runtime, print_runtime
from .data_utils import FIWDataBundle, load_fiw_bundle
from .model_utils import (
    load_arcface_torch,
    load_final_bap2,
    load_generation_stack,
    processor_counts,
)


@dataclass
class NotebookContext:
    paths: ProjectPaths
    runtime: RuntimeConfig
    data: FIWDataBundle | None = None
    pipe: object | None = None
    ip_model: object | None = None
    bap2_modules: dict | None = None
    checkpoint: dict | None = None
    arcface_torch: object | None = None


def setup_runtime(
    drive: str = DEFAULT_DRIVE,
    require_cuda: bool = False,
    verbose: bool = True,
) -> NotebookContext:
    paths = get_paths(drive)
    runtime = get_runtime(require_cuda=require_cuda)
    if verbose:
        print_runtime(runtime, paths)
    return NotebookContext(paths=paths, runtime=runtime)


def setup_data(
    ctx: NotebookContext,
    image_size: int = 512,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> NotebookContext:
    ctx.data = load_fiw_bundle(
        ctx.paths,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    print(f"Train: {len(ctx.data.train_ds):,} triads")
    print(f"Val:   {len(ctx.data.val_ds):,} triads")
    print(f"Test:  {len(ctx.data.test_ds):,} triads")
    return ctx


def setup_generation_models(ctx: NotebookContext) -> NotebookContext:
    ctx.pipe, ctx.ip_model = load_generation_stack(ctx.runtime)
    print("Processors:", processor_counts(ctx.pipe.unet))
    return ctx


def setup_final_bap2(
    ctx: NotebookContext,
    parent_scale: float = 0.1,
) -> NotebookContext:
    if ctx.pipe is None:
        raise RuntimeError("Call setup_generation_models(ctx) before setup_final_bap2(ctx).")
    ctx.bap2_modules, ctx.checkpoint = load_final_bap2(
        ctx.pipe,
        ctx.runtime,
        ctx.paths,
        parent_scale=parent_scale,
    )
    print("BAP2 modules:", len(ctx.bap2_modules))
    print("Checkpoint step:", ctx.checkpoint.get("step", "NA"))
    return ctx


def setup_arcface(ctx: NotebookContext) -> NotebookContext:
    ctx.arcface_torch = load_arcface_torch(ctx.runtime)
    print("ArcFace loaded")
    return ctx


def setup_full_generation_context(
    drive: str = DEFAULT_DRIVE,
    load_data: bool = True,
    load_arcface: bool = False,
    parent_scale: float = 0.1,
) -> NotebookContext:
    ctx = setup_runtime(drive=drive, require_cuda=True)
    if load_data:
        setup_data(ctx)
    setup_generation_models(ctx)
    setup_final_bap2(ctx, parent_scale=parent_scale)
    if load_arcface:
        setup_arcface(ctx)
    return ctx
