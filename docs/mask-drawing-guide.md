# Mask / guide drawing guide (nakkas-canvas + svg-mcp)

How to draw the ControlNet mask/guide PNG for the asset pipeline. The runbook
([`asset-generation-runbook.md`](asset-generation-runbook.md)) references this
file from its "Draw the ControlNet mask/guide" step.

## Toolchain

| Step | Tool | What it does |
|---|---|---|
| Draw | **nakkas-canvas** `render_svg` | Renders an SVG (from JSON config) and returns a preview image to iterate on |
| Convert / verify | **svg-mcp** `viewSVG` / `viewSVGFile` | Converts an SVG string / file to a rendered image (raster preview) |
| Persist | nakkas-canvas `save` | Writes the final PNG into `ComfyUI_t/input/` |

## Workflow

1. **Draw with nakkas-canvas** — `render_svg` with a 768×768 canvas:
   - background rect `#ffffff` (white — see polarity rules below)
   - body region filled `#000000` (e.g. a `circle` r=300 at 384,384)
   - inner detail regions filled `#808080` (e.g. a five-point star emblem)
   - that's the whole palette: white + black + gray. See the two-color rule
     below — Canny only needs luminance steps, not distinct colors per region.
2. **Always use a primitive when nakkas-canvas provides one** — `circle`,
   `polygon`, `parametric` (`fn: "star"` for a five-point star), `arc-group`,
   etc. — instead of hand-rolled `path` data. Primitives render predictably
   and keep the geometry editable.
3. **Inspect the preview** returned by `render_svg`; iterate on the config if
   the shape is off. Optionally cross-check the exact SVG with svg-mcp
   `viewSVG` before saving.
4. **Save the final PNG** with nakkas-canvas `save`, `format: "png"`,
   `width: 768`, output path inside `ComfyUI_t/input/`.
   ⚠️ `save` does **not** create parent directories — `mkdir -p` first, and it
   appends `-1`, `-2`… instead of overwriting.
5. **Convert RGBA → RGB** (the saved PNG is RGBA; the pipeline expects RGB):
   ```bash
   ComfyUI_t/venv/bin/python -c "
   from PIL import Image
   im = Image.open('ComfyUI_t/input/<name>.png').convert('RGB')
   im.save('ComfyUI_t/input/<name>.png', 'PNG')"
   ```
6. **Verify the Canny outline** (see probe snippet in the runbook §2) — every
   region boundary must appear in the intermediate edge image before you
   spend a generation run on it.

## Mask polarity & design rules (learned the hard way)

- **White background; every non-background region gets a dark fill.** The
  flux-canny hint convention expects a white background. Inner regions must
  NOT be white: a white star inside a black disc reads as background and the
  emblem shape is lost.
- **Canny is region-agnostic — only TWO non-white colors are needed.** Canny
  does not care what a region represents; it responds purely to luminance
  *steps* between adjacent regions. Whether neighboring regions use the same
  color or different colors changes nothing. So the whole palette is:
  **white `#ffffff` (background) + black `#000000` (body) + gray `#808080`
  (inner detail)**. Reuse black/gray for every region regardless of meaning.
- **Luminance step is what matters, not hue.** The pipeline's Canny node
  (`kornia.filters.canny`, thresholds 0.4/0.8) converts to **grayscale
  first** (`rgb_to_grayscale`), then Gaussian-blurs (5×5, σ=1) before the
  Sobel gradient. Two dark colors with similar brightness produce NO edge.
  Measured on this exact pipeline (768² medal mask, star-in-disc, live
  server probe):

  | Fill (gray level) | Step vs black body | Inner edge pixels | Verdict |
  |---|---|---|---|
  | `#404040` (64) | 64/255 (~0.25) | 0 | ❌ emblem vanishes |
  | `#606060` (96) | 96/255 (~0.38) | 1168 | ✅ outline survives |
  | `#808080` (128) | 128/255 (~0.50) | 1066 | ✅ **default** |
  | `#a0a0a0` (160) | 160/255 (~0.63) | 1064 | ✅ works |
  | `#c0c0c0` (192) | 192/255 (~0.75) | 1217 | ✅ works |

  Rule of thumb: **adjacent regions must differ by ≥ 96/255 (~0.38) in gray
  level**; `#808080` on `#000000` is the standard pair.
- The guide goes through **Canny**, so only *edges* matter: every fill is
  reduced to its outline. Flat color regions = clean single outlines.
- Keep the body region large in frame (≈78% of canvas diameter for medals);
  tiny regions produce weak Canny edges that ControlNet ignores.
- Canvas must match the asset type's canvas (768 for weapon/prop/armor,
  1024 for background — though backgrounds run without a guide).
