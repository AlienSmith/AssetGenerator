"""Per-asset-type registry for single-layer static assets.

The HomeBot payload (`POST /generations`) has no `asset_type` field today, so
the type is derived heuristically from the prompt (keyword match) with a
`safe` default. Each entry carries the knobs the pipeline needs for that type:

    canvas      canvas size (pixels, square)
    strength    ControlNet strength to use for the color-edge hint
    guidance    Flux guidance scale
    steps       sampling steps
    denoise     denoise amount (1.0 = fresh sample from noise)
    desc        prompt fragment describing the type (injected into positives)
    extra_neg   negatives specific to this type

Keeping the mapping in one place makes it trivial to extend the later
(optionally via a future contract field) without touching the pipeline or the
HTTP layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AssetType:
    """Static knobs + prompt fragments for one asset type."""

    key: str
    canvas: int
    strength: float
    guidance: float
    steps: int
    denoise: float
    desc: str
    extra_neg: str = ""
    # A token to match against the user prompt when deriving the type.
    keywords: "tuple[str, ...]" = field(default_factory=tuple)


ASSET_TYPES: "dict[str, AssetType]" = {
    "weapon": AssetType(
        key="weapon",
        canvas=768,
        strength=0.6,
        guidance=3.5,
        steps=14,
        denoise=1.0,
        desc=(
            "single weapon, one weapon only, weapon centered, held off to the "
            "side, clean silhouette, item icon"
        ),
        extra_neg="hands, fingers, character, person, body, multiple weapons, shadow on ground",
        keywords=("weapon", "sword", "axe", "gun", "blade", "dagger", "bow", "staff", "mace", "spear"),
    ),
    "prop": AssetType(
        key="prop",
        canvas=768,
        strength=0.5,
        guidance=3.5,
        steps=14,
        denoise=1.0,
        desc=(
            "single prop, one object only, prop centered, floating icon, "
            "clean silhouette, game asset icon"
        ),
        extra_neg="character, person, body, hands, multiple objects, shadow on ground, background detail",
        keywords=(
            "prop", "object", "item", "potion", "key", "coin", "chest",
            "lamp", "book", "shield", "helmet",
        ),
    ),
    "armor": AssetType(
        key="armor",
        canvas=768,
        strength=0.55,
        guidance=3.5,
        steps=14,
        denoise=1.0,
        desc=(
            "single piece of armor, one piece only, chest armor, pauldron, or "
            "plating, centered, item icon, clean silhouette"
        ),
        extra_neg="character, person, body, face, hands, multiple pieces, full suit of armor, shadow on ground",
        keywords=("armor", "pauldron", "chestplate", "plate", "helm", "gauntlet", "bracer", "cuirass"),
    ),
    "background": AssetType(
        key="background",
        canvas=1024,
        strength=0.4,
        guidance=3.5,
        steps=16,
        denoise=1.0,
        desc="scene background, wide establishing shot, environment, landscape, atmospheric",
        extra_neg="character, person, foreground object, text, watermark, busy composition",
        keywords=("background", "scene", "landscape", "environment", "skyline", "backdrop", "sky"),
    ),
}


DEFAULT_ASSET_TYPE = "prop"

# Ordering matters: more specific keywords first so "background scene" wins over
# a stray "scene" in some other context (weapons live indoors, etc.).
_KEYWORD_RANK = (
    "background", "landscape", "environment", "skyline", "scene",
    "weapon", "sword", "axe", "gun", "blade", "dagger", "bow", "staff", "mace", "spear",
    "armor", "pauldron", "chestplate", "plate", "helm", "gauntlet", "bracer", "cuirass",
    "prop", "object", "item", "potion", "key", "coin", "chest", "lamp", "book", "shield", "helmet",
)


def resolve_asset_type(prompt: str) -> AssetType:
    """Derive an AssetType from a free-text prompt using keyword matching.

    Falls back to DEFAULT_ASSET_TYPE when no configured keyword matches. The
    prompt is lowercased before matching so the word red "Door" still matches
    door-ish keywords.
    """
    text = (prompt or "").lower()
    for word in _KEYWORD_RANK:
        for at in ASSET_TYPES.values():
            if word in at.keywords and word in text:
                return at
    return ASSET_TYPES[DEFAULT_ASSET_TYPE]