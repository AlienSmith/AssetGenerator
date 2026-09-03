# Game Asset Generation Process

We generate flat, vector-style game character heads (PuyoPuyo-style) and cut them out
onto a transparent background. This doc summarizes the end-to-end process.

## Overview

1. Draw a **guide image** (colored blobs on pure white) — see
   [`docs/head_mask_guide.md`](docs/head_mask_guide.md).
2. Run [`test_flux_head.py`](../test_flux_head.py) to generate the asset via Flux (GGUF)
   + a game-asset LoRA, steered by a ControlNet and a head mask.
3. Remove the background with [`remove_bg.py`](../remove_bg.py) (rembg + color-key).
4. Manually clean any tiny leftover region (e.g. in Photoshop).

## Guides

- [`docs/head_mask_guide.md`](head_mask_guide.md) — how to draw a proper guide/mask image.
- [`test_flux_head.py`](../test_flux_head.py) — the Flux generation pipeline.

## The pipeline

- **GPU:** RTX 5060 Ti (16 GB). Uses GGUF-quantized models to fit VRAM.
- **Base:** `flux1-dev-Q8_0.gguf` (Flux MMDiT).
- **Text encoders:** `clip_l.safetensors` + `t5-v1_1-xxl-encoder-Q8_0.gguf`.
- **VAE:** `ae.safetensors`.
- **LoRA:** `game_assets_v3.safetensors` (2D game asset style, trigger `GRPZA`).
- **ControlNet:** `flux_canny_instantx.safetensors` (latent-input, canny).

The guide is turned into two things:

- an **edge hint** via [`make_color_edges()`](../test_flux_head.py:77) — a color-aware
  edge detector that marks any pixel differing in RGB from a neighbor, capturing every
  blob boundary (eyes, mouth, head) regardless of brightness or size;
- a **head mask** via [`make_head_mask()`](../test_flux_head.py:72) that confines
  diffusion to the head region.

A ControlNet strength sweep (0.2 – 0.8) produces several candidate renders; the user
picks the preferred one.

## Background removal

[`remove_bg.py`](../remove_bg.py) does two stages:

1. **rembg** (U2Net neural matte) finds the rough subject silhouette.
2. **Color-key cleanup** estimates the background color from the image corners and turns
   nearby-colored pixels transparent, dropping small specks and keeping the largest
   connected component (the head).

```
./venv/bin/python remove_bg.py output/<render>.png transparent_asset.png
```

Output is an RGBA PNG (768×768).

### Known limitation: faint gray ring

Because the head does not fully fill the drawn guide circle, a faint **gray ring** can
remain inside the silhouette where the background bleeds in. This ring is a *different*
color from the pure-white background (roughly `0.9 gray`), so neither rembg nor the
color-key reliably removes it without risking near-white head details (eye whites,
highlights).

**Resolution:** finish the cutout manually. Select and delete the small gray region in
Photoshop (or any editor). This is fast and safe because you can see exactly what is
removed. This is the recommended final step.

## Environment

- Python venv at `./venv/bin/python`.
- `rembg` installed with GPU onnxruntime (`rembg[gpu]`) and CLI extras (`rembg[cli]`).
- Note: onnxruntime currently falls back to CPU for rembg (missing
  `libcublasLt.so.13`); correct but not GPU-accelerated. Fine for single images.