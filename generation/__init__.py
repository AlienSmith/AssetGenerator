"""Single-layer static asset generation microservice.

Turn ComfyUI into the generation backend that HomeBot already calls over
HTTP-over-Unix-socket (`/run/feishu-bot/gen.sock`), producing single-layer
static game assets (weapon, prop, armor, background). This is intentionally a
thin layer: it builds Flux + ControlNet workflow graphs and drives them through
ComfyUI's own in-process `/prompt` + `/history` machinery, reusing the existing
GGUF/Flux/ControlNet loaders so model loading and VRAM stay under ComfyUI's
control (dynamic VRAM fits the ~12 GB Flux unet into 16 GB).

Public entry point:

    from generation.service import GenerationService
"""

from generation.asset_types import AssetType, ASSET_TYPES, resolve_asset_type
from generation.pipeline import build_workflow

__all__ = [
    "AssetType",
    "ASSET_TYPES",
    "resolve_asset_type",
    "build_workflow",
]