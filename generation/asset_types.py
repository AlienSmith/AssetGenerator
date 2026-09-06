"""Per-asset-type registry for single-layer static assets.

The HomeBot payload (`POST /generations`) has no `asset_type` field today, so
the type is derived heuristically from the prompt (keyword match) with a
`safe` default. Each entry carries the knobs the pipeline needs for that type:

    canvas      canvas size (pixels, square)
    strength    ControlNet strength (ignored when guide_mode="none")
    guidance    Flux guidance scale
    steps       sampling steps
    denoise     denoise amount (1.0 = fresh sample from noise)
    desc        prompt fragment describing the type (injected into positives)
    extra_neg   negatives specific to this type
    guide_mode  how the payload image becomes a ControlNet hint:
                  "edge_map" — mask/line art -> white-line edge map (default)
                  "raw"      — plain resize, fed to ControlNet as-is
                  "none"     — no hint at all; the graph runs pure txt2img
    style       optional per-type replacement for the shared STYLE block
    negative    optional per-type replacement for the shared NEGATIVE block

Keeping the mapping in one place makes it trivial to extend the later
(optionally via a future contract field) without touching the pipeline or the
HTTP layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# How a type's payload image becomes a ControlNet hint. Values are consumed by
# guides.prepare_guide(); see the module docstring for semantics.
GUIDE_MODES = ("edge_map", "raw", "none")


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
    # How the payload image becomes a ControlNet hint (see module docstring).
    guide_mode: str = "edge_map"
    # Per-type prompt block overrides; None falls back to the shared blocks in
    # prompts.py. Backgrounds override both: a scene IS the image, so the
    # shared "single object on a white background" wording fights it.
    style: str | None = None
    negative: str | None = None

    def __post_init__(self) -> None:
        if self.guide_mode not in GUIDE_MODES:
            raise ValueError(
                f"unknown guide_mode {self.guide_mode!r}; expected one of {GUIDE_MODES}"
            )


# --- per-type prompt block overrides ----------------------------------------
# The shared STYLE/NEGATIVE blocks (prompts.py) are written for single objects
# on white. A type whose imagery contradicts that wording carries its own
# blocks here; prompts.py falls back to the shared ones when the override is
# None. Background is the first such type: a scene fills the whole canvas, so
# "white background" / "single object" / "colored background" (negative) all
# fight the requested output.

_BACKGROUND_STYLE = (
    "flat 2D game background art, flat cel shading, vibrant flat colors, "
    "clean bold shapes, cohesive scene, game background"
)
_BACKGROUND_NEGATIVE = (
    "photorealistic, realistic, 3d, messy, blurry, lowres, text, watermark, "
    "signature, character, person, foreground object, busy composition, "
    "shadow, white background, single object, item icon"
)

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
            "single prop, one object only, prop centered, item icon, "
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
        strength=0.4,  # unused: guide_mode="none" never builds a ControlNet node
        guidance=3.5,
        steps=16,
        denoise=1.0,
        desc="scene background, wide establishing shot, environment, landscape, atmospheric",
        # Type-specific negatives live in _BACKGROUND_NEGATIVE (the override
        # replaces the shared block wholesale, so extra_neg stays empty to keep
        # the shared "colored background" wording from leaking back in).
        extra_neg="",
        keywords=("background", "scene", "landscape", "environment", "skyline", "backdrop", "sky"),
        guide_mode="none",
        style=_BACKGROUND_STYLE,
        negative=_BACKGROUND_NEGATIVE,
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