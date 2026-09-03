#!/usr/bin/env python3
"""Generate flat-vector PuyoPuyo-style body parts from a hand-drawn guide.

Workflow:
  hand-drawn guide (filled head + eyes + mouth blobs on white) -> scribble
  union ControlNet -> Pony V6 base + PuyoPuyo LoRA fills in the head style.

The guide is a simple, low-detail blob drawing (not a clean outline). Scribble
mode reads the filled regions and their boundaries loosely, so the model turns
the rough guide into a clean head instead of rigidly tracing edges (unlike
canny, which produced a bare circle from a clean outline).

This script sweeps several ControlNet strengths so we can pick one where the
model fills in the head while still respecting the guide. All seeds/weights are
fixed so only the strength varies.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import nodes

CANVAS = 768  # square canvas, matches the union ControlNet training aspect
GUIDE_PATH = "input/face_mask.png"  # your hand-drawn blob guide on white


def load_guide(path=GUIDE_PATH, size=CANVAS):
    """Load a drawn guide image and return it as an IMAGE tensor (1,H,W,3)."""
    import comfy.utils
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # (1,H,W,3)


def make_head_mask(guide):
    """Derive a head mask from the guide: 1 where non-white (the drawn blobs),
    0 on the white background. Returns a MASK tensor (1,H,W)."""
    head = (guide < 0.95).any(dim=-1).float()  # any channel not near-white
    return head


def main():
    # ---- Load the native SDXL UNet ----
    model = nodes.UNETLoader().load_unet("autismmix_pony_sdxl_unet.safetensors", weight_dtype="default")[0]
    print("unet:", type(model.model).__name__)

    # ---- Load clip encoders ----
    clip = nodes.DualCLIPLoader().load_clip(
        "pony_clip_g.safetensors", "pony_clip_l.safetensors", type="sdxl")[0]
    print("clip loaded:", clip is not None)

    # ---- Apply the PuyoPuyo LoRA ----
    model_lora, clip_lora = nodes.LoraLoader().load_lora(
        model, clip, "PuyoPuyo_XL_Styles.safetensors", strength_model=1.0, strength_clip=1.0)
    print("puyo lora applied:", model_lora is not None, clip_lora is not None)

    # ---- Encode positive / negative (head-focused, no neck/shoulders) ----
    positive = nodes.CLIPTextEncode().encode(
        clip_lora,
        "score_9, score_8_up, score_7_up, _XL_PStyles, 1girl, face, head, "
        "cute anime face, face centered, head only, face only, "
        "no neck, no shoulders, no torso, no body, "
        "flat cel shading, vibrant flat colors, bold clean outlines, "
        "simple flat background, extremely quality extremely detailed, illustration")[0]
    negative = nodes.CLIPTextEncode().encode(
        clip_lora,
        "score_6, score_5, score_4, lowres, bad anatomy, bad hands, text, error, "
        "missing fingers, extra digit, fewer digits, cropped, worst quality, "
        "low quality, normal quality, jpeg artifacts, signature, watermark, "
        "username, blurry, artist name, 3d, photorealistic, realistic, "
        "multiple heads, full body, neck, shoulders, torso, hair below chin")[0]

    # ---- Load union ControlNet, set scribble type (base model supports it) ----
    control_net = nodes.ControlNetLoader().load_controlnet("controlnet-union-sdxl.safetensors")[0]
    st = __import__("comfy_extras.nodes_controlnet", fromlist=["SetUnionControlNetType"])
    control_net = st.SetUnionControlNetType().execute(control_net, "hed/pidi/scribble/ted").args[0]
    print("controlnet loaded, scribble type set")

    # ---- Load your hand-drawn guide (filled blobs on white) ----
    guide = load_guide()  # (1,H,W,3)
    print("guide:", tuple(guide.shape))
    vae = nodes.VAELoader().load_vae("pony_VAE_SDXL.safetensors")[0]

    # ---- Latent confined to the head mask ----
    # The scribble guide gives facial structure, but alone Pony's portrait prior
    # grows neck/shoulders. Hard-confining diffusion to the masked head region
    # (via VAEEncodeForInpaint's noise_mask) cuts off anything below the chin.
    mask = make_head_mask(guide)  # (1,H,W), 1 where the drawn blobs are
    blank = torch.ones((1, CANVAS, CANVAS, 3), dtype=torch.float32)  # white canvas
    latent = nodes.VAEEncodeForInpaint().encode(vae, blank, mask, grow_mask_by=6)[0]

    # ---- Sweep ControlNet strength, keeping the simple scribble guide ----
    for strength in (0.2, 0.35, 0.5, 0.65, 0.8):
        # Re-apply controlnet to fresh conditioning copies (apply_controlnet
        # mutates the conditioning with a control hint).
        pos_cn, neg_cn = nodes.ControlNetApplyAdvanced().apply_controlnet(
            positive, negative, control_net, guide, strength=strength,
            start_percent=0.0, end_percent=1.0)

        out = nodes.KSampler().sample(
            model_lora, seed=1, steps=25, cfg=6.0, sampler_name="dpmpp_2m",
            scheduler="karras", positive=pos_cn, negative=neg_cn,
            latent_image=latent, denoise=1.0)
        latent_out = out[0]

        image = nodes.VAEDecode().decode(vae, latent_out)[0]
        result = nodes.SaveImage().save_images(
            image, f"pony_puyo_head_scribble_s{strength}")
        print(f"strength={strength} saved:",
              [i["filename"] for i in result["ui"]["images"]])

    # ---- Also save the guide for reference ----
    nodes.SaveImage().save_images(guide, "pony_puyo_head_scribble_guide")

if __name__ == "__main__":
    main()