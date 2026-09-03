#!/usr/bin/env python3
"""Generate a PuyoPuyo-style flat-vector game asset with the Pony V6 base +
PuyoPuyo LoRA + ControlNet++ (union) guided by canny edges.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nodes

def main():
    # ---- Load the native SDXL UNet ----
    model = nodes.UNETLoader().load_unet("autismmix_pony_sdxl_unet.safetensors", weight_dtype="default")[0]
    print("unet:", type(model.model).__name__)

    # ---- Load clip encoders ----
    clip = nodes.DualCLIPLoader().load_clip(
        "pony_clip_g.safetensors", "pony_clip_l.safetensors", type="sdxl")[0]
    print("clip loaded:", clip is not None)

    # ---- Apply the PuyoPuyo LoRA to model + clip ----
    model_lora, clip_lora = nodes.LoraLoader().load_lora(
        model, clip, "PuyoPuyo_XL_Styles.safetensors", strength_model=1.0, strength_clip=1.0)
    print("puyo lora applied:", model_lora is not None, clip_lora is not None)

    # ---- Encode positive / negative ----
    positive = nodes.CLIPTextEncode().encode(
        clip_lora,
        "score_9, score_8_up, score_7_up, _XL_PStyles, 1girl, cute petite, "
        "school uniform, fang, smile, extremely quality extremely detailed, "
        "illustration, flat cel shading, vibrant flat colors, cute anime face, "
        "simple background")[0]
    negative = nodes.CLIPTextEncode().encode(
        clip_lora,
        "score_6, score_5, score_4, lowres, bad anatomy, bad hands, text, error, "
        "missing fingers, extra digit, fewer digits, cropped, worst quality, "
        "low quality, normal quality, jpeg artifacts, signature, watermark, "
        "username, blurry, artist name, 3d, photorealistic, realistic")[0]

    # ---- Load ControlNet (union) and set canny/lineart type ----
    control_net = nodes.ControlNetLoader().load_controlnet("controlnet-union-sdxl.safetensors")[0]
    st = __import__("comfy_extras.nodes_controlnet", fromlist=["SetUnionControlNetType"])
    control_net = st.SetUnionControlNetType().execute(control_net, "canny/lineart/anime_lineart/mlsd").args[0]
    print("controlnet loaded, union type set")

    # ---- Load reference image and compute canny edges ----
    img = nodes.LoadImage().load_image("example.png")[0]
    print("reference image:", tuple(img.shape))
    from comfy_extras.nodes_canny import Canny
    canny_img = Canny.execute(img, 0.4, 0.8).args[0]
    print("canny edges:", tuple(canny_img.shape))

    # ---- Apply ControlNet to conditioning ----
    positive, negative = nodes.ControlNetApplyAdvanced().apply_controlnet(
        positive, negative, control_net, canny_img, strength=0.8,
        start_percent=0.0, end_percent=1.0)
    print("controlnet applied to conditioning")

    # ---- Latent + sampler ----
    latent = nodes.EmptyLatentImage().generate(832, 1216, 1)[0]
    out = nodes.KSampler().sample(
        model_lora, seed=1, steps=25, cfg=6.0, sampler_name="dpmpp_2m",
        scheduler="karras", positive=positive, negative=negative,
        latent_image=latent, denoise=1.0)
    latent_out = out[0]
    print("sampled latent:", tuple(latent_out["samples"].shape))

    # ---- Decode ----
    vae = nodes.VAELoader().load_vae("pony_VAE_SDXL.safetensors")[0]
    image = nodes.VAEDecode().decode(vae, latent_out)[0]
    print("decoded image:", tuple(image.shape))

    # ---- Save ----
    result = nodes.SaveImage().save_images(image, "pony_puyo_controlnet")
    print("saved:", result["ui"]["images"])

if __name__ == "__main__":
    main()