"""Prompt assembly for single-layer static assets.

A single positive block is built from:

    GRPZA, <user's core description>, <asset-type desc>,
    <shared style block>

and a single negative block from:

    <shared negative block>, <asset-type extra negatives>

Keeping the style/negative fragments here (instead of inline in the pipeline)
means the Flux prompt stays coherent across all four asset types and only the
type descriptor differs.
"""
from __future__ import annotations

from generation.asset_types import AssetType

# 2D game asset LoRA trigger (GRPZA) — model-only LoRA, kept in the positive.
TRIGGER = "GRPZA"

# Shared style block for every type that does not override it (see
# AssetType.style in asset_types.py): flat 2D sprite asset, single layer, clean
# edges, nothing behind it. Background assets carry their own block because a
# scene fills the canvas — "single object on a white background" fights it.
# NOTE: keep the LoRA's proven style tokens ("flat cel shading", "vibrant flat
# colors") — swapping them for generic vector-icon wording (tried 2026-09-05)
# breaks the trained 2D game-art texture. The no-lighting/no-shadow goal is met
# by the flat guide image + prompt-word hygiene (e.g. no "floating"), NOT by
# restyling this block; negatives are dead weight at Flux cfg=1.0 anyway.
STYLE = (
    "flat 2D sprite asset, flat cel shading, bold clean outlines, "
    "vibrant flat colors, single object on a white background, "
    "white background, game asset"
)

# Shared negative block: kills photorealism, extra layers, text and clutter.
# "colored background" / "solid color background" keep the model from painting
# its own flat tone behind the object instead of the requested white. Types
# that override the negative (background) drop those tokens via their own
# AssetType.negative block.
NEGATIVE = (
    "photorealistic, realistic, 3d, messy, blurry, lowres, text, watermark, "
    "signature, multiple objects, layered composition, gradient, shadow, "
    "colored background, solid color background, scale mismatch"
)


def build_positive(prompt: str, asset_type: AssetType) -> str:
    """Positive prompt fragment for one asset: trigger + user prompt + type.

    The style block is the type's own override when it has one (background),
    otherwise the shared STYLE block.
    """
    content = (prompt or "").strip()
    parts = [TRIGGER]
    if content:
        parts.append(content.rstrip(",.;"))
    parts.append(asset_type.desc)
    parts.append(asset_type.style or STYLE)
    return ", ".join(p for p in parts if p)


def build_negative(asset_type: AssetType) -> str:
    """Negative prompt fragment for one asset type.

    Same override rule as the positive: AssetType.negative replaces the shared
    NEGATIVE block entirely (so a background never inherits the shared
    "colored background" ban), then the type's extra_neg is appended.
    """
    base = asset_type.negative or NEGATIVE
    extra = asset_type.extra_neg
    if extra:
        return f"{base}, {extra}"
    return base