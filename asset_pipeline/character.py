"""Shared character/style config injected into every body-part prompt.

Consistency across body parts comes from a single character description and
style block that each part's prompt reuses, plus a common negative block that
stops other body parts or a full body from leaking into a single-part render.
"""

# Editable in one place: describe the character once, and all parts share it.
# OUTFit is intentionally concrete and deterministic so every part is forced
# toward the same clothing (arms with the same sleeve fabric, legs).

PALETTE = "light skin, brown hair"

OUTFIT = (
    "clothes: light blue top, white short sleeves, white shorts, "
    "matching same-color fabric, uniform outfit"
)

STYLE = (
    "flat 2D sprite asset, flat cel shading, bold clean outlines, "
    "vibrant flat colors, simple flat background, white background, "
    "game asset, pixel art"
)

NEGATIVE = (
    "photorealistic, realistic, 3d, messy, blurry, lowres, text, watermark, "
    "signature, bad anatomy, multiple parts, full body, extra body parts, "
    "scale mismatch, gradient, shadow"
)