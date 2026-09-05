"""Build a single-layer static-asset Flux + ControlNet workflow graph.

The graph is a plain ComfyUI prompt dict (`{"<node_id>": {"class_type": ...}}`)
so it can be validated and queued by ComfyUI's own `/prompt` endpoint. We reuse
the same loader/sampler node classes used by `asset_pipeline/generate_body_part.py`
(GGUF model, DualCLIPLoaderGGUF, VAELoader, LoraLoaderModelOnly, CLIPTextEncode,
FluxGuidance, ControlNetLoader, ControlNetApplyAdvanced, KSampler, VAEDecode,
SaveImage). The reference image is supplied through the graph via a `LoadImage`
node (saved into the input folder by the service). guides.py normalizes it into
a white-line edge map — flux_canny's training format — which feeds the
ControlNet directly so the produced asset follows the user's silhouette.

This module is offline-testable: `build_workflow()` only combines plain strings
and numbers and returns a plain dict.
"""
from __future__ import annotations

import re
import time
from datetime import datetime

from generation.asset_types import AssetType
from generation.prompts import build_negative, build_positive

# Model files referenced by the graph. Kept here so the workflow is
# self-describing and the loader can warm the same names.
UNET_NAME = "flux1-dev-Q8_0.gguf"
CLIP_L_NAME = "clip_l.safetensors"
T5_NAME = "t5-v1_1-xxl-encoder-Q8_0.gguf"
VAE_NAME = "ae.safetensors"
LORA_NAME = "game_assets_v3.safetensors"
CONTROLNET_NAME = "flux_canny_instantx.safetensors"


def batch_output_slug(prompt: str, *, when: float | None = None) -> str:
    """Per-batch output folder name: `<sanitized prompt>_<YYYYMMDD-HHMMSS>`.

    Used as a path segment of the SaveImage prefix so every variant of one
    batch lands in `output/flux_<type>/<slug>/` instead of mixing with other
    runs. `when` defaults to now; pass a fixed value for reproducible tests.
    """
    words = re.sub(r"[^a-z0-9]+", "_", (prompt or "").lower())
    words = "_".join(w for w in words.split("_") if w)[:48].rstrip("_")
    moment = time.time() if when is None else when
    stamp = datetime.fromtimestamp(moment).strftime("%Y%m%d-%H%M%S")
    return f"{words or 'batch'}_{stamp}"


def build_workflow(
    *,
    asset_type: AssetType,
    prompt: str,
    seed: int,
    strength: float | None = None,
    filename_prefix: str | None = None,
    guide_image_name: str | None = None,
    cfg: float = 1.0,
) -> dict:
    """Build a single-variant Flux + ControlNet workflow graph.

    Parameters
    ----------
    asset_type : AssetType
        The type descriptor (canvas, strength, guidance, steps, denoise).
    prompt : str
        The user's free-text description injected into the positive block.
    seed : int
        Sampling seed for this variant.
    strength : float | None
        Override for the ControlNet strength (defaults to asset_type.strength).
    filename_prefix : str | None
        SaveImage prefix (defaults to `flux_<asset_type.key>`).
    guide_image_name : str | None
        Name of the reference image in the input folder used as the ControlNet
        hint. If None, usage is unconditioned (txt2img) — for background assets
        that have no single-layer silhouette to guide off.
    cfg : float
        Sampler cfg. Flux normally runs at 1.0.

    Returns
    -------
    dict
        A ComfyUI prompt graph with numbered node ids.
    """
    p = asset_type
    strength = p.strength if strength is None else strength
    prefix = filename_prefix or f"flux_{p.key}"

    positive = build_positive(prompt, p)
    negative = build_negative(p)

    graph: dict = {}
    nid = 1

    def add(class_type: str, inputs: dict) -> str:
        nonlocal nid
        node_id = str(nid)
        nid += 1
        graph[node_id] = {"class_type": class_type, "inputs": inputs}
        # Link references must use the SAME (string) id space as the graph
        # keys: ComfyUI's validation does `prompt[link_id]` verbatim, and the
        # HTTP API always carries string node ids.
        return node_id

    # --- loaders ---
    unet = add("UnetLoaderGGUF", {"unet_name": UNET_NAME})
    clip = add("DualCLIPLoaderGGUF", {
        "clip_name1": CLIP_L_NAME,
        "clip_name2": T5_NAME,
        "type": "flux",
    })
    vae = add("VAELoader", {"vae_name": VAE_NAME})
    model_lora = add("LoraLoaderModelOnly", {
        "model": [unet, 0],
        "lora_name": LORA_NAME,
        "strength_model": 0.8,
    })
    controlnet = add("ControlNetLoader", {"control_net_name": CONTROLNET_NAME})

    # --- conditioning ---
    pos_enc = add("CLIPTextEncode", {"clip": [clip, 0], "text": positive})
    neg_enc = add("CLIPTextEncode", {"clip": [clip, 0], "text": negative})
    pos_guid = add("FluxGuidance", {"conditioning": [pos_enc, 0], "guidance": p.guidance})

    # --- latent + control hint ---
    latent = add("EmptySD3LatentImage", {
        "width": p.canvas, "height": p.canvas, "batch_size": 1,
    })

    if guide_image_name is None:
        # background: no silhouette hint, pure txt2img.
        pos_final = [pos_guid, 0]
        neg_final = [neg_enc, 0]
    else:
        guide = add("LoadImage", {"image": guide_image_name})
        # The guide arrives as a finished edge map (white lines on black —
        # guaranteed by guides.make_guide_with_detail), which is exactly the
        # format flux_canny was trained on, so it feeds ControlNet directly.
        # Running it through the Canny node would re-derive edges FROM the
        # lines, doubling every stroke. Persist the exact hint image the model
        # consumes, next to this variant's render (QA: makes guide-vs-output
        # mismatches visible).
        add("SaveImage", {
            "images": [guide, 0],
            "filename_prefix": f"{prefix}_canny_hint",
        })
        cn = add("ControlNetApplyAdvanced", {
            "positive": [pos_guid, 0],
            "negative": [neg_enc, 0],
            "control_net": [controlnet, 0],
            "image": [guide, 0],
            "strength": strength,
            "start_percent": 0.0,
            "end_percent": 1.0,
            "vae": [vae, 0],
        })
        pos_final = [cn, 0]
        neg_final = [cn, 1]

    # --- sampler + decode + save ---
    samples = add("KSampler", {
        "model": [model_lora, 0],
        "seed": seed,
        "steps": p.steps,
        "cfg": cfg,
        "sampler_name": "euler",
        "scheduler": "simple",
        "positive": pos_final,
        "negative": neg_final,
        "latent_image": [latent, 0],
        "denoise": p.denoise,
    })
    decoded = add("VAEDecode", {"samples": [samples, 0], "vae": [vae, 0]})
    add("SaveImage", {"images": [decoded, 0], "filename_prefix": prefix})

    return graph