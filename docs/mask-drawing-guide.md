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
6. **(Optional) QA: verify the Canny outline** — routine masks that follow the
   palette rules above don't need this step. Run the probe snippet (runbook
   §8) only for unusual masks or when a finished asset is missing an element;
   every region boundary must appear in the intermediate edge image.

## What the pipeline accepts (2026-09-05: line-art hint, Canny node removed)

The graph no longer runs a Canny node. `guides.make_guide_with_detail()`
normalizes whatever you send into a **white-line edge map on black** —
flux_canny's exact training format — and `pipeline.build_workflow()` feeds it
straight into ControlNet. Two reference kinds are auto-detected by mean
luminance:

1. **Line art (preferred): white strokes on black background.** Mean
   luminance < 64 → binarized and passed through unchanged. Draw it directly
   with nakkas-canvas (see workflow above): outline every region with ~8px
   white strokes on `#000000`. This is the highest-fidelity option — thin
   features (medal ribbon) survive because the strokes ARE the geometry, not
   derived edges. Verified live: ribbon present in output.
2. **Filled region mask: white background + dark fills.** Mean luminance ≥ 64
   → converted via FIND_EDGES + threshold + dilation into solid single-stroke
   lines around every region boundary. Simpler to draw, but thin features can
   still fade in the render at ControlNet strength 0.5.

## Mask polarity & design rules (learned the hard way)

- **Line art: black background, white strokes.** Inverted from the old
  region-mask convention. Inner regions must NOT be white: a white star inside
  a black disc reads as background and the emblem shape is lost.
- **Region masks: white background; every non-background region gets a dark
  fill.** Only TWO non-white colors are needed — the converter is
  region-agnostic and responds purely to luminance *steps* between adjacent
  regions: **white `#ffffff` (background) + black `#000000` (body) + gray
  `#808080` (inner detail)**. Adjacent regions must differ by ≥ 96/255 in gray
  level or no line is drawn between them (`#808080` on `#000000` is the
  standard pair; `#404040` on `#000000` produces nothing).
- Only *boundaries* matter: every region is reduced to its outline. Flat
  color regions = clean single strokes.
- Keep the body region large in frame (≈78% of canvas diameter for medals);
  tiny regions produce thin strokes that ControlNet ignores.
- Canvas must match the asset type's canvas (768 for weapon/prop/armor,
  1024 for background — though backgrounds run without a guide).
