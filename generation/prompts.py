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

# Shared style block reused verbatim for every asset type: flat 2D sprite asset,
# single layer, clean edges, nothing behind it. Background assets lean on the
# same flat-game tone so they read as one layer, not a composite scene.
STYLE = (
    "flat 2D sprite asset, flat cel shading, bold clean outlines, "
    "vibrant flat colors, simple flat background, single object on a "
    "solid flat background, game asset"
)

# Shared negative block: kills photorealism, extra layers, text and clutter.
NEGATIVE = (
    "photorealistic, realistic, 3d, messy, blurry, lowres, text, watermark, "
    "signature, multiple objects, layered composition, gradient, shadow, "
    "busy background, scale mismatch"
)


def build_positive(prompt: str, asset_type: AssetType) -> str:
    """Positive prompt fragment for one asset: trigger + user prompt + type. """
    content = (prompt or "").strip()
    parts = [TRIGGER]
    if content:
        parts.append(content.rstrip(",.;"))
    parts.append(asset_type.desc)
    parts.append(STYLE)
    return ", ".join(p for p in parts if p)


def build_negative(asset_type: AssetType) -> str:
    """Negative prompt fragment for one asset type."""
    extra = asset_type.extra_neg
    if extra:
        return f"{NEGATIVE}, {extra}"
    return NEGATIVE