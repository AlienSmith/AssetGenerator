# Making a Good Guide Image

The pipeline draws a flat game-asset body part (head, torso, arm, leg) by feeding a
hand-drawn **guide** into Flux through a ControlNet. The guide is not a painting — it is a
set of **colored blobs on a white background**. The pipeline derives two things from it:

- an **edge hint** (drives the ControlNet) via [`make_color_edges()`](../generate_body_part.py:86)
- a **part mask** (confines diffusion to the blob region) via [`make_part_mask()`](../generate_body_part.py:81)

Both are computed automatically. Your job is only to make the guide image right.

## How the edge hint works

[`make_color_edges()`](../test_flux_head.py:77) marks a pixel as an edge whenever its RGB
color differs from a neighboring pixel. This is **color-aware**, so it does not care about
brightness or size — but it **does** care about color contrast against the pixel next to it.

A note on why we don't use Canny anymore: Canny converts to grayscale first, so a feature
whose brightness matches its surroundings gets dropped. The left eye and mouth kept
disappearing for exactly that reason. The color-difference detector avoids this, but it
means **two regions must actually differ in color** for an edge to appear.

## Rules for drawing the guide

1. **White background only.** The background must be pure white (`255,255,255`). Any
   smudge, gradient, or anti-aliased edge on the background becomes unwanted edges and
   noise.

2. **Flat, solid colors.** Fill each blob with a single flat color. No gradients, no
   soft shading, no blur. Gradual color change produces weak or missing edges.

3. **Every feature must differ in color from its neighbor.** The rule that matters:
   - The **head fill** must differ in color from the white background.
   - Each **facial feature** (eyes, mouth) must differ in color from the head fill it
     sits on.
   - Two features must differ from each other if they touch.
   The difference does not need to be large — a modest color change is enough for the
   `thresh` used (default 0.06). But identical colors produce **no edge**.

4. **Vivid colors are fine now.** Because the detector is color-aware, a bright yellow
   mouth works — it just needs to be a different color than the head fill around it.

5. **Thicker strokes are better.** A thin 1–2 px line can be filtered as noise or
   swallowed by the mask. Make eyes and the mouth clearly visible filled blobs, not hairlines.

6. **Keep everything inside the canvas.** The head should be centered and not touch the
   canvas border, so the mask cleanup has clean edges.

## Two features that confuse beginners

- **Same color for left and right eye is OK** *only if* the eyes are fully surrounded by
  a different-colored head fill. The edge hint will outline both identically. If the eyes
  touch another same-colored region, their shared boundary vanishes.
- **One large blob reads as "one thing".** If the whole face is one color, the edge hint
  produces a single outline (a "large circle"). Break the face into distinct colored
  regions — skin fill, eye blobs, mouth blob — so the outlines appear.

## Example layout

```
                     [white background]
            ┌──────────────────────────────┐
            │         head fill (skin)     │
            │        ┌─────┐   ┌─────┐    │
            │        │ eye │   │ eye │    │   <- eye color != skin
            │        └─────┘   └─────┘    │
            │             ┌──────┐        │
            │             │mouth │        │   <- mouth color != skin
            │             └──────┘        │
            └──────────────────────────────┘
```

Each labeled region is a flat, solid color; the background is pure white.

## Body parts (torso / arms / legs)

The generalized generator [`generate_body_part.py`](../generate_body_part.py) draws any part
from a colored-blob guide. Draw each part's guide at `input/parts/{part}_mask.png`:

- `input/parts/torso_mask.png`
- `input/parts/upper_arm_mask.png`
- `input/parts/lower_arm_mask.png`
- `input/parts/upper_leg_mask.png`
- `input/parts/lower_leg_mask.png`

The **head** still uses `input/face_mask.png`.

Follow the same rules as the head, with a few body-specific notes:

1. **One part per guide.** A torso guide should be only the torso blob; an arm guide should
   be only one arm. Use the negative prompt (already set per part) to forbid other parts,
   but the mask still confines diffusion to the drawn blob.

2. **Avoid joint/overlap hints.** For `upper_arm`, end the blob at the elbow; for `lower_arm`,
   end it at the wrist. Do **not** draw the hand or shoulder into the blob, or the model
   will try to include them.

3. **Scale for consistency.** Since the parts must assemble into one character, keep each
   part covering a **similar fraction of the canvas** so output sizes stay proportional.
   A torso blob and an upper-leg blob should have comparable footprint to the head blob.

4. **Clothing hinting.** Because the automatic color/edge hint finds every boundary, you
   can add a flat clothing-color region (e.g. a sleeve band) inside the limb blob to steer
   outfit color. Keep it a single flat color, clearly different from the limb fill.

At low ControlNet strength (~0.2) the guide is mostly a **positioning hint**: the shared
character config in [`character.py`](../character.py) (palette + style) is what makes the
parts share skin/outfit colors and shading.

## Iterating

After a run, the pipeline saves the extracted hint as `flux_head_edges_hint.png`. Always
check it:

- **Feature missing in the hint** → its color is too close to its neighbor. Change the
  color (more contrast) or, if it's on the white background, make it less bright.
- **Too much noise / background texture** → raise the `thresh` in
  [`make_color_edges()`](../test_flux_head.py:141) (e.g. 0.06 → 0.12).
- **Faint features** → lower `thresh` (e.g. 0.03).

Updating the hint thresholds is a one-line change in `main()`:

```python
hint = make_color_edges(guide, thresh=0.06)