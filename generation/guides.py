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

from PIL import Image

import folder_paths


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