"""Reference-image preprocessing for the single-layer pipeline.

HomeBot sends the ControlNet reference as base64 PNG in the `/generations`
payload. We decode it, normalize it (RGBA -> RGB), and write it into ComfyUI's
input folder so the graph's `LoadImage` node can read it by filename. Keeping
the pipeline graph pure (a `LoadImage` node) means the batch is driven entirely
through ComfyUI's engine; this module only does the import step.
"""
from __future__ import annotations

import base64
import io
import math
import uuid

from PIL import Image, ImageDraw

import folder_paths


def _as_l(im: "Image.Image") -> "Image.Image":
    """Return `im` as a single-channel L mask (for compositing)."""
    return im.convert("L")


def _ellipse(im: "Image.Image", x0, y0, x1, y1, *, fill: int) -> None:
    ImageDraw.Draw(im).ellipse([x0, y0, x1, y1], fill=fill)


def _polygon(im: "Image.Image", pts, *, fill: tuple) -> None:
    ImageDraw.Draw(im).polygon(pts, fill=fill)


def _arc(im: "Image.Image", bbox, start, end, *, width: int, fill: tuple) -> None:
    ImageDraw.Draw(im).arc(bbox, start, end, width=width, fill=fill)


def decode_image(base64_str: str) -> Image.Image:
    """Decode a base64 PNG/JPEG into an RGB PIL image."""
    raw = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(raw))
    img.load()
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def persist_guide(image: Image.Image, *, width: int | None = None, height: int | None = None) -> str:
    """Save an RGB PIL image into the input folder, returning its filename.

    Optionally resizes to a fixed canvas first (LANCZOS). The returned name is
    safe to feed straight into the graph's `LoadImage` node (no subfolder).
    """
    if width and height and (image.size != (width, height)):
        image = image.resize((width, height), Image.LANCZOS)

    filename = f"gen_guide_{uuid.uuid4().hex}.png"
    input_dir = folder_paths.get_input_directory()
    path = folder_paths.get_annotated_filepath(filename)  # resolves input dir
    image.save(path, "PNG")
    return filename


def make_guide_from_base64(base64_str: str, *, width: int | None = None, height: int | None = None) -> str:
    """Decode + persist a base64 reference image for the graph.

    Returns the input-folder filename for the graph's `LoadImage` node.
    """
    img = decode_image(base64_str)
    return persist_guide(img, width=width, height=height)


def make_guide_with_detail(
    base64_str: str,
    *,
    width: int | None = None,
    height: int | None = None,
) -> str:
    """Decode a base64 mask and synthesize a *material-bearing* ControlNet guide.

    The flat black/white silhouette mask forces the model to paint its region one
    flat color (below the flat-threshold the medal came out as a single red blob,
    with no golden relief). This variant repaints the mask with material cues so
    the Canny edge hint carries *surface detail*, not just a shape boundary:

      - a golden radial-gradient disc (the medal body),
      - a distinct red five-point star inset (the emblem),
      - a bevel rim / bright top-left highlight for the edge,
      - all on a white background.

    The returned hint reuses the exact same `LoadImage` + `Canny` graph path, so
    the pipeline contract is unchanged — we only change *what* the reference
    image shows. Expected outcome: the model keeps the white background but draws
    a golden medal with visible highlights and shaded relief instead of a flat
    red disc.

    Returns the input-folder filename for the graph's `LoadImage` node.
    """
    base = decode_image(base64_str).convert("RGBA")
    if width and height and base.size != (width, height):
        base = base.resize((width, height), Image.LANCZOS)
    W, H = base.size
    cx, cy = W / 2.0, H / 2.0
    R = min(W, H) * 0.39  # match runbook §2 body diameter (~78% of canvas)

    body_mask = Image.new("L", (W, H), 0)
    _ellipse(body_mask, cx - R, cy - R, cx + R, cy + R, fill=255)
    body = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    # Radial "gold" gradient across the disc (bright upper-left -> deep gold
    # lower-right) so the model has a highlight->shadow range to render.
    top_left = (236, 200, 102, 255)
    bottom_right = (122, 74, 18, 255)
    for y in range(H):
        for x in range(W):
            t = (x / max(W - 1, 1) + y / max(H - 1, 1)) / 2.0  # 0..1 diag
            col = tuple(int(a + (b - a) * t) for a, b in zip(top_left, bottom_right))
            body.putpixel((x, y), col)

    # Red five-point star as the emblem (distinct from the gold body).
    star = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    R2 = R * 0.62
    pts = []
    inner = []
    for i in range(10):
        rr = R2 if i % 2 == 0 else R2 * 0.30
        ang = math.pi * i / 5.0 - math.pi / 2.0
        pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
    _polygon(star, pts, fill=(200, 36, 30, 255))

    # Bevel rim: bright top-left arc (highlight) + darker bottom-right (shadow).
    rim = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rim_w = max(2, int(R * 0.03))
    _arc(rim, (cx - R, cy - R, cx + R, cy + R), 160, 340, width=rim_w,
         fill=(250, 240, 180, 255))  # top-left bright
    _arc(rim, (cx - R, cy - R, cx + R, cy + R), -20, 160, width=rim_w,
         fill=(90, 54, 12, 255))  # lower-right dark

    guide = Image.composite(star, body, _as_l(body_mask))
    guide = Image.composite(rim, guide, _as_l(rim.split()[3]))
    # Everything composited onto white.
    out = Image.new("RGB", (W, H), (255, 255, 255))
    out.paste(guide, (0, 0), guide)
    return persist_guide(out)