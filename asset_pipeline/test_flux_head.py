#!/usr/bin/env python3
"""Generate flat-vector PuyoPuyo-style heads with Flux (GGUF) + ControlNet.

Flux is the fallback base model because Pony V6 kept failing on the blob guide
(neck / pure mask / no face) due to its portrait prior. Flux (MMDiT) has a
different prior and obeys "head only, no neck" constraints much better.

Components (all GGUF-format to fit the 16GB RTX 5060 Ti):
  flux1-dev-Q8_0.gguf            -> transformer (UnetLoaderGGUF)
  clip_l + t5-v1_1-xxl Q8        -> text encoders (DualCLIPLoaderGGUF, type flux)
  ae.safetensors                 -> Flux VAE
  game_assets_v3.safetensors     -> 2D game asset LoRA (trigger: GRPZA)
  flux_canny_instantx.safetensors-> Flux ControlNet (latent-input, canny)

The hand-drawn guide is turned into a canny edge map (ControlNet hint) and into
a latent mask (VAEEncodeForInpaint) that confines diffusion to the head region,
so the neck/shoulders have nowhere to form. Only ControlNet strength varies.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib.util
import numpy as np
import torch
import nodes

CANVAS = 768
GUIDE_PATH = "input/face_mask.png"


def load_module_by_path(name, path):
    """Import a standalone module file (no package-relative imports)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_gguf_package():
    """Import custom_nodes/ComfyUI-GGUF as a package, resolving its relative
    imports (from .ops / .loader / .dequant import ...). Works despite the
    hyphenated folder name by registering a synthetic package on sys.modules."""
    root = os.path.abspath("custom_nodes")
    # Create a package object for the GGUF extension.
    pkg = importlib.util.module_from_spec(
        importlib.machinery.ModuleSpec("ComfyUIGguf", None))
    pkg.__path__ = [os.path.join(root, "ComfyUI-GGUF")]
    pkg.__package__ = "ComfyUIGguf"
    sys.modules["ComfyUIGguf"] = pkg

    # Load its submodules in dependency order so nodes.py's relative imports work.
    for mod_name in ("ops", "loader", "dequant", "nodes"):
        path = os.path.join(root, "ComfyUI-GGUF", f"{mod_name}.py")
        spec = importlib.util.spec_from_file_location(f"ComfyUIGguf.{mod_name}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"ComfyUIGguf.{mod_name}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["ComfyUIGguf.nodes"]




def load_guide(path=GUIDE_PATH, size=CANVAS):
    """Load the hand-drawn guide as an IMAGE tensor (1,H,W,3)."""
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def make_head_mask(guide):
    """1 where non-white (the drawn blobs), 0 on the white background."""
    return (guide < 0.95).any(dim=-1).float()


def make_color_edges(guide, thresh=0.1):
    """Edge hint from local color differences between neighboring pixels.

    Catches every blob boundary, including facial features drawn on top of the
    head fill, regardless of their brightness or size. Grayscale Canny drops
    features whose luminance is close to the head fill they sit on, so we use
    the per-channel RGB difference instead.
    """
    img = guide[0].numpy()  # (H,W,3) float in [0,1]
    padded = np.pad(img, ((0, 1), (0, 1), (0, 0)), constant_values=1.0)
    # Max channelwise difference vs the right and down neighbors (white border).
    right = np.abs(padded[:-1, :-1] - padded[:-1, 1:]).max(-1)
    down = np.abs(padded[:-1, :-1] - padded[1:, :-1]).max(-1)
    edges = (np.maximum(right, down) > thresh).astype(np.float32)
    return torch.from_numpy(np.repeat(edges[..., None], 3, axis=-1)).unsqueeze(0)


def main():
    # ---- GGUF loaders (custom_nodes/ComfyUI-GGUF) ----
    gguf_nodes = load_gguf_package()

    # ---- Flux transformer (GGUF) ----
    model = gguf_nodes.UnetLoaderGGUF().load_unet("flux1-dev-Q8_0.gguf")[0]
    print("flux unet:", type(model.model).__name__)

    # ---- Flux text encoders: CLIP-L + T5-XXL (type flux) ----
    clip = gguf_nodes.DualCLIPLoaderGGUF().load_clip(
        "clip_l.safetensors", "t5-v1_1-xxl-encoder-Q8_0.gguf", type="flux")[0]
    print("flux clip loaded:", clip is not None)

    # ---- Flux VAE ----
    vae = nodes.VAELoader().load_vae("ae.safetensors")[0]
    print("flux vae loaded:", vae is not None)

    # ---- Apply the 2D game asset LoRA (model-only, GRPZA trigger) ----
    model_lora = nodes.LoraLoaderModelOnly().load_lora(
        model, clip, "game_assets_v3.safetensors", strength_model=0.8,
        strength_clip=0.0)[0]
    print("game asset lora applied:", model_lora is not None)

    # ---- Encode positive / negative (flat game asset head, no neck) ----
    positive = nodes.CLIPTextEncode().encode(
        clip,
        "GRPZA, 1girl, face, head, cute anime face, face centered, head only, "
        "face only, no neck, no shoulders, no torso, no body, flat 2D sprite "
        "asset, flat cel shading, bold clean outlines, vibrant flat colors, "
        "simple flat background, white background, game asset, pixel art")[0]
    negative = nodes.CLIPTextEncode().encode(
        clip,
        "photorealistic, realistic, 3d, messy, blurry, lowres, text, watermark, "
        "signature, bad anatomy, multiple heads, full body, neck, shoulders, "
        "torso, hair below chin, gradient, shadow")[0]

    # ---- Flux guidance (cfg is always 1.0; guidance via conditioning) ----
    flux_nodes = load_module_by_path("flux_nodes", os.path.join("comfy_extras", "nodes_flux.py"))
    positive = flux_nodes.FluxGuidance().execute(positive, 3.5).args[0]

    # ---- Load Flux ControlNet (canny instantx, latent-input) ----
    control_net = nodes.ControlNetLoader().load_controlnet("flux_canny_instantx.safetensors")[0]
    print("flux controlnet loaded:", control_net is not None)

    # ---- Load guide, derive color-edge hint + head mask ----
    guide = load_guide()
    print("guide:", tuple(guide.shape))
    hint = make_color_edges(guide, thresh=0.06)

    mask = make_head_mask(guide)  # (1,H,W)
    blank = torch.ones((1, CANVAS, CANVAS, 3), dtype=torch.float32)
    latent = nodes.VAEEncodeForInpaint().encode(vae, blank, mask, grow_mask_by=6)[0]

    # ---- Sweep ControlNet strength, only strength varies ----
    for strength in (0.2, 0.35, 0.5, 0.65, 0.8):
        # Re-apply controlnet to fresh conditioning copies each iteration.
        pos_cn, neg_cn = nodes.ControlNetApplyAdvanced().apply_controlnet(
            positive, negative, control_net, hint, strength=strength,
            start_percent=0.0, end_percent=1.0, vae=vae)

        out = nodes.KSampler().sample(
            model_lora, seed=1, steps=20, cfg=1.0, sampler_name="euler",
            scheduler="simple", positive=pos_cn, negative=neg_cn,
            latent_image=latent, denoise=1.0)
        latent_out = out[0]

        image = nodes.VAEDecode().decode(vae, latent_out)[0]
        result = nodes.SaveImage().save_images(
            image, f"flux_head_canny_s{strength}")
        print(f"strength={strength} saved:",
              [i["filename"] for i in result["ui"]["images"]])

    # ---- Save the guide + hint for reference ----
    nodes.SaveImage().save_images(guide, "flux_head_guide")
    nodes.SaveImage().save_images(hint, "flux_head_edges_hint")


if __name__ == "__main__":
    main()