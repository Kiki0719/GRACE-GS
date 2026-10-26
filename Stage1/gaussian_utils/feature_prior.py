"""
feature_prior.py
================

Frozen DINOv2 feature prior for single-view Stage-1 Gaussian refinement.

Design
------
This module deliberately does NOT change GaussianModel or the rasterizer.

Instead:

    current Gaussian model
        -> render RGB
        -> frozen DINOv2
        -> patch features

    target RGB
        -> frozen DINOv2
        -> cached target patch features

Then Stage-1 minimizes a foreground-masked cosine feature distance.

Because the DINOv2 network is frozen but the rendered RGB remains connected to
the renderer graph, gradients flow:

    DINO feature loss
        -> rendered RGB
        -> Gaussian xyz / scale / rotation / opacity / SH-DC

This makes the module compatible with the existing E-D3DGS / graphdeco
Gaussian renderer without adding high-dimensional feature channels to each
Gaussian.

The target and rendered backgrounds are neutralized identically before feature
extraction so the real target background does not become a semantic shortcut.

Backends
--------
1) torchhub (recommended):
       model = dinov2_vits14
       repo  = facebookresearch/dinov2

   If `stage1_feature_repo_path` points to a local DINOv2 repository, it is
   loaded with torch.hub source="local" and requires no network.

2) timm:
       any timm ViT whose forward_features exposes patch tokens.

The feature prior is DEFAULT OFF so an existing Stage-1 configuration continues
to run even when DINOv2 weights are not installed/cached.

Enable with:
    stage1_enable_feature_prior = True
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _cfg_float(cfg: Any, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


def _cfg_int(cfg: Any, name: str, default: int) -> int:
    return int(getattr(cfg, name, default))


def _cfg_bool(cfg: Any, name: str, default: bool) -> bool:
    value = getattr(cfg, name, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _cfg_str(cfg: Any, name: str, default: str) -> str:
    value = getattr(cfg, name, default)
    if value is None:
        return str(default)
    return str(value).strip()


@dataclass
class FeaturePriorState:
    enabled: bool
    model: Optional[torch.nn.Module]
    backend: str
    model_name: str
    input_size: int
    patch_size: int
    target_patch: Optional[torch.Tensor]
    target_cls: Optional[torch.Tensor]
    patch_mask: Optional[torch.Tensor]
    grid_hw: Tuple[int, int]
    device: torch.device
    dtype: torch.dtype


def _imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 3, 1, 1)
    return (x - mean) / std


def _neutralize_background(
    image: torch.Tensor,
    mask: torch.Tensor,
    neutral_value: float,
) -> torch.Tensor:
    """
    image: CHW or BCHW in [0,1]
    mask : HW or B1HW / BHW
    """
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None]

    mask = mask.to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0)
    neutral = torch.full_like(image, float(neutral_value))
    return image * mask + neutral * (1.0 - mask)


def _prepare_input(
    image: torch.Tensor,
    mask: torch.Tensor,
    input_size: int,
    neutral_value: float,
) -> torch.Tensor:
    x = _neutralize_background(image, mask, neutral_value)
    x = F.interpolate(
        x,
        size=(input_size, input_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    x = x.clamp(0.0, 1.0)
    return _imagenet_normalize(x)


def _load_torchhub_model(cfg: Any, device: torch.device) -> Tuple[torch.nn.Module, str]:
    model_name = _cfg_str(cfg, "stage1_feature_model", "dinov2_vits14")
    repo_path = _cfg_str(cfg, "stage1_feature_repo_path", "")
    repo_name = _cfg_str(
        cfg,
        "stage1_feature_torchhub_repo",
        "facebookresearch/dinov2",
    )

    if repo_path:
        local = Path(repo_path).expanduser().resolve()
        if not local.is_dir():
            raise FileNotFoundError(
                f"stage1_feature_repo_path does not exist: {local}"
            )
        model = torch.hub.load(
            str(local),
            model_name,
            source="local",
            pretrained=True,
        )
        source = f"torchhub-local:{local}"
    else:
        model = torch.hub.load(
            repo_name,
            model_name,
            pretrained=True,
            trust_repo=True,
        )
        source = f"torchhub:{repo_name}"

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, source


def _load_timm_model(cfg: Any, device: torch.device) -> Tuple[torch.nn.Module, str]:
    try:
        import timm
    except Exception as exc:
        raise RuntimeError(
            "stage1_feature_backend=timm requires timm. "
            "Install it with: pip install timm"
        ) from exc

    model_name = _cfg_str(
        cfg,
        "stage1_feature_timm_model",
        "vit_small_patch14_dinov2.lvd142m",
    )
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,
    )
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, f"timm:{model_name}"


def _load_model(cfg: Any, device: torch.device) -> Tuple[torch.nn.Module, str, str]:
    backend = _cfg_str(cfg, "stage1_feature_backend", "torchhub").lower()

    if backend == "torchhub":
        model, source = _load_torchhub_model(cfg, device)
        model_name = _cfg_str(cfg, "stage1_feature_model", "dinov2_vits14")
        return model, source, model_name

    if backend == "timm":
        model, source = _load_timm_model(cfg, device)
        model_name = _cfg_str(
            cfg,
            "stage1_feature_timm_model",
            "vit_small_patch14_dinov2.lvd142m",
        )
        return model, source, model_name

    raise ValueError(
        "stage1_feature_backend must be either 'torchhub' or 'timm', "
        f"got: {backend}"
    )


def _infer_patch_size(model: torch.nn.Module, cfg: Any) -> int:
    configured = max(1, _cfg_int(cfg, "stage1_feature_patch_size", 14))

    patch_embed = getattr(model, "patch_embed", None)
    patch_size = getattr(patch_embed, "patch_size", None)

    if patch_size is None:
        return configured

    if isinstance(patch_size, (tuple, list)):
        if len(patch_size) >= 1:
            return int(patch_size[0])

    try:
        return int(patch_size)
    except Exception:
        return configured


def _tensor_tokens_to_patch(
    value: torch.Tensor,
    input_size: int,
    patch_size: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[int, int]]:
    """
    Normalize common ViT feature layouts to:
        patch: B,N,C
        cls  : B,C or None
    """
    if value.ndim == 4:
        # BCHW feature map
        b, c, h, w = value.shape
        patch = value.flatten(2).transpose(1, 2)
        return patch, None, (int(h), int(w))

    if value.ndim != 3:
        raise RuntimeError(
            f"Unsupported feature tensor shape: {tuple(value.shape)}"
        )

    b, n, c = value.shape
    grid = max(1, int(round(input_size / max(patch_size, 1))))
    expected = grid * grid

    if n == expected + 1:
        cls = value[:, 0]
        patch = value[:, 1:]
        return patch, cls, (grid, grid)

    if n == expected:
        return value, None, (grid, grid)

    # Last-resort square-token inference.
    root = int(round(n ** 0.5))
    if root * root == n:
        return value, None, (root, root)

    if n > 1:
        root = int(round((n - 1) ** 0.5))
        if root * root == n - 1:
            return value[:, 1:], value[:, 0], (root, root)

    raise RuntimeError(
        "Could not infer DINO patch-token grid from tensor shape "
        f"{tuple(value.shape)}."
    )


def _extract_features(
    model: torch.nn.Module,
    x: torch.Tensor,
    input_size: int,
    patch_size: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[int, int]]:
    """
    Support DINOv2 torch.hub and common timm ViT forward_features outputs.
    """
    if hasattr(model, "forward_features"):
        out = model.forward_features(x)
    else:
        out = model(x)

    if isinstance(out, dict):
        # Official DINOv2 torch.hub path.
        patch = out.get("x_norm_patchtokens", None)
        cls = out.get("x_norm_clstoken", None)

        if torch.is_tensor(patch):
            n = int(patch.shape[1])
            root = int(round(n ** 0.5))
            if root * root == n:
                grid_hw = (root, root)
            else:
                grid = max(1, int(round(input_size / max(patch_size, 1))))
                grid_hw = (grid, max(1, n // grid))
            return patch, cls if torch.is_tensor(cls) else None, grid_hw

        # timm-like dictionary fallbacks.
        for key in (
            "x",
            "last_hidden_state",
            "features",
            "feat",
        ):
            value = out.get(key, None)
            if torch.is_tensor(value):
                return _tensor_tokens_to_patch(
                    value,
                    input_size,
                    patch_size,
                )

        tensor_values = [v for v in out.values() if torch.is_tensor(v)]
        if tensor_values:
            return _tensor_tokens_to_patch(
                tensor_values[-1],
                input_size,
                patch_size,
            )

    if isinstance(out, (tuple, list)):
        for value in reversed(out):
            if torch.is_tensor(value):
                return _tensor_tokens_to_patch(
                    value,
                    input_size,
                    patch_size,
                )

    if torch.is_tensor(out):
        return _tensor_tokens_to_patch(
            out,
            input_size,
            patch_size,
        )

    raise RuntimeError(
        f"Unsupported DINO feature output type: {type(out)!r}"
    )


def _build_patch_mask(
    target_mask: torch.Tensor,
    grid_hw: Tuple[int, int],
    threshold: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = target_mask
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None]

    pooled = F.interpolate(
        mask.float(),
        size=grid_hw,
        mode="area",
    ).flatten(2).squeeze(1)

    # Soft support with a minimum occupancy gate.
    keep = pooled >= float(threshold)
    weights = pooled.clamp(0.0, 1.0)
    weights = torch.where(keep, weights, torch.zeros_like(weights))
    return weights.to(device=device, dtype=dtype)


@torch.no_grad()
def prepare_feature_prior(
    target_rgb: torch.Tensor,
    target_mask: torch.Tensor,
    cfg: Any,
) -> FeaturePriorState:
    """
    Load frozen DINOv2 and cache target patch features once.

    If disabled, returns a lightweight disabled state and imports/loads no model.
    """
    device = target_rgb.device
    enabled = _cfg_bool(cfg, "stage1_enable_feature_prior", False)

    if not enabled:
        return FeaturePriorState(
            enabled=False,
            model=None,
            backend="disabled",
            model_name="disabled",
            input_size=0,
            patch_size=0,
            target_patch=None,
            target_cls=None,
            patch_mask=None,
            grid_hw=(0, 0),
            device=device,
            dtype=target_rgb.dtype,
        )

    input_size = max(
        112,
        _cfg_int(cfg, "stage1_feature_input_size", 224),
    )

    model, source, model_name = _load_model(cfg, device)
    patch_size = _infer_patch_size(model, cfg)

    # Keep spatial size compatible with the ViT patch size.
    input_size = max(
        patch_size,
        (input_size // patch_size) * patch_size,
    )

    neutral = min(
        max(_cfg_float(cfg, "stage1_feature_background_value", 0.5), 0.0),
        1.0,
    )

    target_input = _prepare_input(
        target_rgb,
        target_mask,
        input_size,
        neutral,
    )

    patch, cls, grid_hw = _extract_features(
        model,
        target_input,
        input_size,
        patch_size,
    )

    patch = F.normalize(patch.float(), dim=-1).detach()
    if cls is not None:
        cls = F.normalize(cls.float(), dim=-1).detach()

    mask_threshold = min(
        max(_cfg_float(cfg, "stage1_feature_mask_threshold", 0.10), 0.0),
        1.0,
    )
    patch_mask = _build_patch_mask(
        target_mask,
        grid_hw,
        mask_threshold,
        device=patch.device,
        dtype=patch.dtype,
    ).detach()

    if float(patch_mask.sum().item()) <= 0.0:
        patch_mask = torch.ones(
            (patch.shape[0], patch.shape[1]),
            device=patch.device,
            dtype=patch.dtype,
        )

    state = FeaturePriorState(
        enabled=True,
        model=model,
        backend=source,
        model_name=model_name,
        input_size=input_size,
        patch_size=patch_size,
        target_patch=patch,
        target_cls=cls,
        patch_mask=patch_mask,
        grid_hw=grid_hw,
        device=device,
        dtype=target_rgb.dtype,
    )

    return state


def feature_prior_summary(state: FeaturePriorState) -> str:
    if not state.enabled:
        return "DINO feature prior: disabled"

    n_tokens = (
        int(state.target_patch.shape[1])
        if state.target_patch is not None
        else 0
    )
    dim = (
        int(state.target_patch.shape[2])
        if state.target_patch is not None
        else 0
    )
    active = (
        int((state.patch_mask > 0).sum().item())
        if state.patch_mask is not None
        else 0
    )

    return (
        f"DINO feature prior: backend={state.backend} | "
        f"model={state.model_name} | input={state.input_size} | "
        f"patch={state.patch_size} | grid={state.grid_hw[0]}x{state.grid_hw[1]} | "
        f"tokens={n_tokens} | dim={dim} | foreground_tokens={active}"
    )


def feature_prior_loss(
    rendered_rgb: torch.Tensor,
    target_mask: torch.Tensor,
    state: Optional[FeaturePriorState],
    cfg: Any,
    iteration: int,
    *,
    force: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Differentiable DINO patch-feature loss for the rendered RGB.

    The DINO parameters are frozen. Do NOT wrap this function in no_grad during
    training, because gradients must reach rendered_rgb.
    """
    zero = rendered_rgb.sum() * 0.0
    diag: Dict[str, Any] = {
        "active": False,
        "patch": 0.0,
        "global": 0.0,
        "foreground_tokens": 0,
        "mean_cosine": 0.0,
    }

    if state is None or not state.enabled or state.model is None:
        return zero, diag

    every = max(
        1,
        _cfg_int(cfg, "stage1_feature_every", 10),
    )
    start_iter = max(
        1,
        _cfg_int(cfg, "stage1_feature_start_iter", 1),
    )
    stop_iter = _cfg_int(
        cfg,
        "stage1_feature_stop_iter",
        10**9,
    )

    if not force:
        if int(iteration) < start_iter or int(iteration) > stop_iter:
            return zero, diag
        if int(iteration) % every != 0:
            return zero, diag

    neutral = min(
        max(_cfg_float(cfg, "stage1_feature_background_value", 0.5), 0.0),
        1.0,
    )

    x = _prepare_input(
        rendered_rgb,
        target_mask,
        state.input_size,
        neutral,
    )

    # Mixed precision is intentionally optional.  float32 is safer and the
    # default because this branch runs only every N iterations.
    use_amp = (
        _cfg_bool(cfg, "stage1_feature_amp", False)
        and x.is_cuda
    )

    if use_amp:
        amp_dtype_name = _cfg_str(
            cfg,
            "stage1_feature_amp_dtype",
            "float16",
        ).lower()
        amp_dtype = (
            torch.bfloat16
            if amp_dtype_name in ("bf16", "bfloat16")
            else torch.float16
        )
        amp_ctx = torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
        )
    else:
        from contextlib import nullcontext
        amp_ctx = nullcontext()

    with amp_ctx:
        patch, cls, grid_hw = _extract_features(
            state.model,
            x,
            state.input_size,
            state.patch_size,
        )

    patch = F.normalize(patch.float(), dim=-1)

    if patch.shape != state.target_patch.shape:
        raise RuntimeError(
            "DINO rendered/target token shapes differ: "
            f"rendered={tuple(patch.shape)} "
            f"target={tuple(state.target_patch.shape)}. "
            "Use a fixed stage1_feature_input_size."
        )

    cosine = (patch * state.target_patch).sum(dim=-1).clamp(-1.0, 1.0)
    patch_distance = 1.0 - cosine

    weights = state.patch_mask.to(
        device=patch_distance.device,
        dtype=patch_distance.dtype,
    )
    denom = weights.sum().clamp_min(1e-6)
    patch_loss = (patch_distance * weights).sum() / denom

    global_loss = zero
    global_fraction = max(
        0.0,
        _cfg_float(cfg, "stage1_feature_global_fraction", 0.10),
    )

    if (
        global_fraction > 0.0
        and cls is not None
        and state.target_cls is not None
    ):
        cls = F.normalize(cls.float(), dim=-1)
        global_loss = (
            1.0
            - (cls * state.target_cls).sum(dim=-1).clamp(-1.0, 1.0)
        ).mean()

    total = patch_loss + global_fraction * global_loss

    diag["active"] = True
    diag["patch"] = float(patch_loss.detach().item())
    diag["global"] = float(
        global_loss.detach().item()
        if torch.is_tensor(global_loss)
        else global_loss
    )
    diag["foreground_tokens"] = int((weights > 0).sum().item())
    diag["mean_cosine"] = float(
        ((cosine * weights).sum() / denom).detach().item()
    )

    return total, diag
