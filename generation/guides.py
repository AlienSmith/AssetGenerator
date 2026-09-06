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
import uuid

from PIL import Image, ImageFilter, ImageOps

import folder_paths

from generation.asset_types import AssetType


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


# Mean-luminance cutoff separating the two supported reference kinds: line art
# is mostly black with thin white strokes (mean ≈ 10-20), a filled region mask
# is mostly bright areas (mean ≈ 190 for the medal mask).
_LINE_ART_MEAN_MAX = 64
# FIND_EDGES response above this becomes a solid line; the medal mask's band
# steps (255->128, 128->26, 26->255) all respond ≈ 100-230, flat areas ≈ 0.
_EDGE_THRESHOLD = 64
# Dilate 1px gradient ridges into a Canny-like stroke width so the hint
# survives the downsampling inside the ControlNet pyramid.
_EDGE_DILATE = 5


def _mask_to_line_art(gray: Image.Image) -> Image.Image:
    """Filled region mask -> white line art on black with solid strokes."""
    # Pad with the corner color first: masks usually have a WHITE background,
    # and FIND_EDGES treats the white-to-nothing transition at the canvas
    # boundary as an edge, drawing a frame around the whole hint. The frame
    # lands on the padded image's outermost row, and the dilation below reaches
    # _EDGE_DILATE//2 pixels back — so the pad must exceed that radius or the
    # crop re-includes the frame.
    pad = _EDGE_DILATE // 2 + 1
    border = gray.getpixel((0, 0))
    padded = ImageOps.expand(gray, border=pad, fill=border)
    edges = padded.filter(ImageFilter.FIND_EDGES)
    edges = edges.point(lambda v: 255 if v >= _EDGE_THRESHOLD else 0)
    edges = edges.filter(ImageFilter.MaxFilter(_EDGE_DILATE))
    w, h = gray.size
    return edges.crop((pad, pad, pad + w, pad + h))


def _binarize(gray: Image.Image) -> Image.Image:
    """Snap anti-aliased strokes to a pure binary edge map.

    Canny output (what flux_canny saw in training) is binary; keeping the hint
    binary stays in-distribution.
    """
    return gray.point(lambda v: 255 if v >= 128 else 0)


def make_guide_with_detail(
    base64_str: str,
    *,
    width: int | None = None,
    height: int | None = None,
) -> str:
    """Turn the user's reference into a white-line edge map for ControlNet.

    The graph (pipeline.py) feeds the guide straight into ControlNetApplyAdvanced
    with the flux_canny model, which was trained on Canny-style edge maps: white
    lines on a black background. This function guarantees that format.

    History: the guide used to be a colored repaint of the mask (gold body /
    red detail / dark outline) and the graph ran the Canny node on it. Two
    problems: the repaint created DOUBLE edges a few px apart (white-to-outline
    and outline-to-red) that interfere after Canny's internal Gaussian blur —
    the visible symptom was dashed contours and missing segments (the medal
    ribbon's top edge vanished) — and the raw mask's luminance steps sit at or
    below the Canny thresholds after that blur, so even a raw pass-through
    mask produced dotted edges at every threshold pair tested (0.1-0.8).
    Producing explicit line art sidesteps edge detection entirely.

    Two reference kinds are supported (auto-detected by mean luminance):
    * already line art: binarized and passed through unchanged;
    * a filled region mask: FIND_EDGES + threshold + dilation derives solid
      single-stroke lines around every region boundary.

    Returns the input-folder filename for the graph's `LoadImage` node.
    """
    base = decode_image(base64_str)
    if width and height and base.size != (width, height):
        base = base.resize((width, height), Image.LANCZOS)
    gray = base.convert("L")
    hist = gray.histogram()
    total = sum(hist) or 1
    mean = sum(i * count for i, count in enumerate(hist)) / total
    line_art = _binarize(gray) if mean < _LINE_ART_MEAN_MAX else _mask_to_line_art(gray)
    return persist_guide(line_art.convert("RGB"))


def prepare_guide(
    asset_type: AssetType,
    base64_str: str,
    *,
    width: int | None = None,
    height: int | None = None,
) -> str | None:
    """Turn the payload image into the hint the graph should consume.

    Single dispatch point for the per-type `guide_mode` (see asset_types.py):

    * "edge_map" — filled mask or line art -> white-line edge map in
      flux_canny's training format (`make_guide_with_detail`);
    * "raw"      — plain resize, ControlNet consumes the pixels as-is
      (`make_guide_from_base64`);
    * "none"     — no hint at all: returns None and the graph runs pure
      txt2img (background assets).

    Returns the input-folder filename for the graph's `LoadImage` node, or
    None when the asset type needs no guide.
    """
    mode = asset_type.guide_mode
    if mode == "none":
        return None
    if mode == "raw":
        return make_guide_from_base64(base64_str, width=width, height=height)
    if mode == "edge_map":
        return make_guide_with_detail(base64_str, width=width, height=height)
    raise ValueError(f"unknown guide_mode {mode!r} for asset type {asset_type.key!r}")