#!/usr/bin/env python3
"""Generate one flat-vector game character body part with Flux (GGUF) + ControlNet.

Generalizes test_flux_head.py to any body part (torso, arms, legs, head). Each
part is drawn from a colored-blob guide (see docs/head_mask_guide.md). Cross-part
consistency comes from the shared character config in character.py.

Usage:
  python generate_body_part.py --part torso [--strengths 0.2,0.5] [--seed 1] [--count 3] [--steps 12]

--strengths: comma-separated ControlNet strengths to sweep.
--count: how many seed variants to sample per strength (pick the best).
--steps:  sampling steps per image (lower = faster; default 12).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib.util
import argparse
import numpy as np
import torch
import nodes

import character

CANVAS = 768

# Each part: guide path, part-specific prompt description, output prefix.
PARTS = {
    "head": {
        "guide": "input/face_mask.png",
        "desc": "1girl, face, head, cute anime face, face centered, head only, face only, no neck, no shoulders, no torso, no body",
        "prefix": "flux_head",
        # head is a face; negate the rest of the body explicitly.
        "extra_neg": "neck, shoulders, torso, hair below chin",
    },
    "torso": {
        "guide": "input/parts/torso_mask.png",
        "desc": "1girl, torso, chest, upper body, trunk, centered, alone, no head, no neck, no arms, no legs, loose shirt",
        "prefix": "flux_torso",
        "extra_neg": "head, face, neck, shoulders, arms, legs, hips, full body",
    },
    "upper_arm": {
        "guide": "input/parts/upper_arm_mask.png",
        "desc": "1girl, one upper arm, upper arm only, shoulder to elbow, sleeve, held away from body, alone",
        "prefix": "flux_upper_arm",
        "extra_neg": "hand, forearm, wrist, shoulder, torso, head, two arms",
    },
    "lower_arm": {
        "guide": "input/parts/lower_arm_mask.png",
        "desc": "1girl, one lower arm, forearm only, elbow to wrist, forearm, held away from body, alone",
        "prefix": "flux_lower_arm",
        "extra_neg": "hand, elbow, upper arm, torso, head, two arms",
    },
    "upper_leg": {
        "guide": "input/parts/upper_leg_mask.png",
        "desc": "1girl, one upper leg, thigh only, hip to knee, alone, shorts",
        "prefix": "flux_upper_leg",
        "extra_neg": "foot, lower leg, shin, hip, torso, two legs, full body",
    },
    "lower_leg": {
        "guide": "input/parts/lower_leg_mask.png",
        "desc": "1girl, one lower leg, calf only, knee to ankle, alone, stocking/shoe-less leg",
        "prefix": "flux_lower_leg",
        "extra_neg": "foot, upper leg, thigh, knee, torso, two legs, full body",
    },
}

DEFAULT_STRENGTHS = "0.2,0.35,0.5,0.65,0.8"


def load_module_by_path(name, path):
    """Import a standalone module file (no package-relative imports)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_gguf_package():
    """Import custom_nodes/ComfyUI-GGUF as a package, resolving its relative
    imports. Works despite the hyphenated folder name by registering a
    synthetic package on sys.modules."""
    root = os.path.abspath("custom_nodes")
    pkg = importlib.util.module_from_spec(
        importlib.machinery.ModuleSpec("ComfyUIGguf", None))
    pkg.__path__ = [os.path.join(root, "ComfyUI-GGUF")]
    pkg.__package__ = "ComfyUIGguf"
    sys.modules["ComfyUIGguf"] = pkg
    for mod_name in ("ops", "loader", "dequant", "nodes"):
        path = os.path.join(root, "ComfyUI-GGUF", f"{mod_name}.py")
        spec = importlib.util.spec_from_file_location(f"ComfyUIGguf.{mod_name}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"ComfyUIGguf.{mod_name}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["ComfyUIGguf.nodes"]


def load_guide(path, size=CANVAS):
    """Load a hand-drawn guide as an IMAGE tensor (1,H,W,3)."""
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def make_part_mask(guide):
    """1 where non-white (the drawn blob), 0 on the white background."""
    return (guide < 0.95).any(dim=-1).float()


def make_color_edges(guide, thresh=0.1):
    """Edge hint from local color differences between neighboring pixels.

    Catches every blob boundary regardless of shape, brightness, or size.
    """
    img = guide[0].numpy()  # (H,W,3) float in [0,1]
    padded = np.pad(img, ((0, 1), (0, 1), (0, 0)), constant_values=1.0)
    right = np.abs(padded[:-1, :-1] - padded[:-1, 1:]).max(-1)
    down = np.abs(padded[:-1, :-1] - padded[1:, :-1]).max(-1)
    edges = (np.maximum(right, down) > thresh).astype(np.float32)
    return torch.from_numpy(np.repeat(edges[..., None], 3, axis=-1)).unsqueeze(0)


def build_prompts(part):
    """Positive is shared character block (skin/hair/outfit) + part desc +
    shared style. The outfit terms are shared so every part gets the same
    clothing colors/materials."""
    positive = (
        f"GRPZA, {character.PALETTE}, {character.OUTFIT}, "
        f"{part['desc']}, {character.STYLE}"
    )
    negative = (
        f"{character.NEGATIVE}, {part['extra_neg']}"
    )
    return positive, negative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=list(PARTS), required=True)
    parser.add_argument("--strengths", default=DEFAULT_STRENGTHS,
                        help="comma-separated ControlNet strengths")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--count", type=int, default=3,
                        help="seed variants to sample per strength")
    parser.add_argument("--steps", type=int, default=12,
                        help="sampling steps per image (lower = faster)")
    args = parser.parse_args()

    part = PARTS[args.part]
    strengths = [float(s) for s in args.strengths.split(",")]

    # ---- GGUF loaders ----
    gguf_nodes = load_gguf_package()
    model = gguf_nodes.UnetLoaderGGUF().load_unet("flux1-dev-Q8_0.gguf")[0]
    print("flux unet:", type(model.model).__name__)

    # ---- Flux text encoders: CLIP-L + T5-XXL (type flux) ----
    clip = gguf_nodes.DualCLIPLoaderGGUF().load_clip(
        "clip_l.safetensors", "t5-v1_1-xxl-encoder-Q8_0.gguf", type="flux")[0]
    print("flux clip loaded:", clip is not None)

    # ---- Flux VAE ----
    vae = nodes.VAELoader().load_vae("ae.safetensors")[0]
    print("flux vae loaded:", vae is not None)

    # ---- 2D game asset LoRA (model-only, GRPZA trigger) ----
    model_lora = nodes.LoraLoaderModelOnly().load_lora(
        model, clip, "game_assets_v3.safetensors", strength_model=0.8,
        strength_clip=0.0)[0]

    # ---- Encode positive / negative (shared char + part desc) ----
    positive, negative = build_prompts(part)
    print("positive:", positive)
    positive = nodes.CLIPTextEncode().encode(clip, positive)[0]
    negative = nodes.CLIPTextEncode().encode(clip, negative)[0]

    # ---- Flux guidance (cfg is always 1.0; guidance via conditioning) ----
    flux_nodes = load_module_by_path("flux_nodes", os.path.join("comfy_extras", "nodes_flux.py"))
    positive = flux_nodes.FluxGuidance().execute(positive, 3.5).args[0]

    # ---- Flux ControlNet (canny instantx, latent-input) ----
    control_net = nodes.ControlNetLoader().load_controlnet("flux_canny_instantx.safetensors")[0]

    # ---- Load guide, derive color-edge hint + part mask ----
    guide_path = part["guide"]
    if not os.path.exists(guide_path):
        sys.exit(f"guide not found: {guide_path}. Draw input/parts/{args.part}_mask.png per docs/head_mask_guide.md")
    guide = load_guide(guide_path)
    print("guide:", tuple(guide.shape))
    hint = make_color_edges(guide, thresh=0.06)

    mask = make_part_mask(guide)  # (1,H,W)
    blank = torch.ones((1, CANVAS, CANVAS, 3), dtype=torch.float32)
    latent = nodes.VAEEncodeForInpaint().encode(vae, blank, mask, grow_mask_by=6)[0]

    # ---- Sweep ControlNet strength, N seed variants each ----
    for strength in strengths:
        pos_cn, neg_cn = nodes.ControlNetApplyAdvanced().apply_controlnet(
            positive, negative, control_net, hint, strength=strength,
            start_percent=0.0, end_percent=1.0, vae=vae)

        saved = []
        for i in range(args.count):
            seed = args.seed + i
            out = nodes.KSampler().sample(
                model_lora, seed=seed, steps=args.steps, cfg=1.0,
                sampler_name="euler", scheduler="simple",
                positive=pos_cn, negative=neg_cn,
                latent_image=latent, denoise=1.0)
            image = nodes.VAEDecode().decode(vae, out[0])[0]
            result = nodes.SaveImage().save_images(
                image, f"{part['prefix']}_canny_s{strength}")
            saved.append(result["ui"]["images"][0]["filename"])
        print(f"strength={strength} ({args.count} variants) saved:", saved)

    # ---- Save the guide + hint for reference ----
    nodes.SaveImage().save_images(guide, f"{part['prefix']}_guide")
    nodes.SaveImage().save_images(hint, f"{part['prefix']}_edges_hint")


if __name__ == "__main__":
    main()