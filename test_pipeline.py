#!/usr/bin/env python3
"""Test the autismmix-pony native SDXL UNet with the IPAdapter pipeline.

Builds the same graph as pony_sdxl_gguf_workflow.json but swaps the GGUF
UnetLoaderGGUF for the stock UNetLoader pointed at the native SDXL model.
Runs through ComfyUI's own node implementations so real inference runs.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nodes
from comfy.cli_args import args

import custom_nodes.ComfyUI_IPAdapter_plus.IPAdapterPlus as IAP

def main():
    # ---- Load the reference image ----
    img_pixels = nodes.LoadImage().load_image("example.png")[0]
    print("reference image:", tuple(img_pixels.shape))

    # ---- Load the native SDXL UNet ----
    model = nodes.UNETLoader().load_unet("autismmix_pony_sdxl_unet.safetensors", weight_dtype="default")[0]
    print("unet:", type(model.model).__name__)

    # ---- Load clip encoders ----
    clip = nodes.DualCLIPLoader().load_clip(
        "pony_clip_g.safetensors", "pony_clip_l.safetensors", type="sdxl")[0]
    print("clip loaded:", clip is not None)

    # ---- Encode positive / negative ----
    positive = nodes.CLIPTextEncode().encode(
        clip, "score_9, score_8_up, score_7_up, 1girl, portrait, blue eyes, detailed, masterpiece")[0]
    negative = nodes.CLIPTextEncode().encode(
        clip, "score_6, score_5, score_4, lowres, bad anatomy, bad hands, cropped")[0]

    # ---- Load IPAdapter components ----
    ipadapter = IAP.IPAdapterModelLoader().load_ipadapter_model("ip-adapter_sdxl.safetensors")[0]
    clip_vision = nodes.CLIPVisionLoader().load_clip("ip_adapter_sdxl_vit_h.safetensors")[0]

    # ---- Apply IPAdapter to the model ----
    patched = IAP.IPAdapterAdvanced().apply_ipadapter(
        model, ipadapter, image=img_pixels, clip_vision=clip_vision,
        weight=0.8, weight_type="linear", combine_embeds="concat",
        start_at=0.0, end_at=1.0, embeds_scaling="V only")[0]
    print("ipadapter applied:", patched is not None)

    # ---- Latent + sampler ----
    latent = nodes.EmptyLatentImage().generate(1024, 1024, 1)[0]
    out = nodes.KSampler().sample(patched, 42, 20, 7.0, "euler", "normal", 1.0,
                                  positive, negative, latent)
    latent_out = out[0]
    print("sampled latent:", tuple(latent_out["samples"].shape))

    # ---- Decode ----
    vae = nodes.VAELoader().load_vae("pony_VAE_SDXL.safetensors")[0]
    image = nodes.VAEDecode().decode(vae, latent_out)[0]
    print("decoded image:", tuple(image.shape))

    # ---- Save ----
    result = nodes.SaveImage().save_images(image, "ipadapter_test")
    print("saved:", result["ui"]["images"])

if __name__ == "__main__":
    main()