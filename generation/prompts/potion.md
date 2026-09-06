# Health potion

A round red health potion in a glass flask, classic RPG prop icon. Exercises
the `prop` asset type; a clean icon-shaped subject for validating that the
white-background + single-layer prompt holds on a smaller, more detailed object
than the medal.

## Type & knobs

| Key | Value |
|---|---|
| type | `prop` |
| canvas | `768` |
| strength | `0.5` |
| guidance | `3.5` |
| steps | `14` |
| denoise | `1.0` |

## Subject

a red health potion in a glass flask

## Positive

```
GRPZA, a red health potion in a glass flask, single prop, prop centered, floating icon, clean silhouette, game asset icon, white background, game asset
```

Short and concrete; "floating icon / clean silhouette" from the `prop` `desc`
in `asset_types.py`. The name carries color (red) and material (glass); nothing
more, so the LoRA isn't overloaded.

## Negative

```
photorealistic, realistic, 3d, messy, blurry, lowres, text, watermark,
signature, multiple objects, layered composition, gradient, shadow,
busy background, scale mismatch,
character, person, body, hands, multiple objects, shadow on ground, background detail
```

(`prop` extra_neg in `asset_types.py`.)

## Iterations

### `2026-09-06` — line-art hint + long color-bound prompt: colors bind, shadow stalks the flask

- **Setup:** line-art hint `input/potion_line_art.png` (white 8px strokes on
  black 1024²: flask bulb circle r=300, neck, cork rect, curved liquid line,
  2 bubbles, glass highlight). Draft 1 read as a SAD FACE (straight liquid
  line + wave = pareidolia) — curved line + bubbles fixed it.
- **Prompt:** `a red health potion in a clear glass flask, the liquid is
  bright red, the cork is light brown, the flask is transparent glass`
  (harness prepends GRPZA + prop desc + STYLE via `build_positive()`).
- **Batch 1** (`..._025921`, seeds 1788634761590-593): red liquid + brown
  cork bind on every variant, but glass tint flips between clear and
  pink-washed by seed, and 3/4 have a pink ground shadow.
- **Batch 2** (`..._030805`, seeds 1788635285090-093, same prompt): 4/4 have
  a ground shadow (grey or pink). 7/8 across both batches.
- **Diagnosis:** the flask bulb sits ~10% from the canvas bottom in the
  guide, so the model reads "sitting on the ground" and paints a contact
  shadow. The medal never had this — it floats center-canvas with big
  margins. Prompt negatives are dead weight at Flux cfg=1.0, so the fix has
  to come from the guide geometry or the prompt.

### `2026-09-06` — short prompt VERIFIED: winner found (seed 1788635634214)

- **User idea:** cut the prompt to the bone — `red health potion` (final
  positive: `GRPZA, red health potion, a small prop item, flat 2D sprite
  asset, ...`). Less material language ("clear glass flask") for the model
  to over-read.
- **Batch 3** (`red_health_potion_20260906-031354`, seeds
  1788635634211-214): shadow rate drops to 2/4, and seed...214 is the
  clean winner — brown cork, clear glass, bright red liquid, curved
  liquid line respected, bubbles kept, NO ground shadow.
- **Verdict:** line-art routine generalizes to props (3rd asset type after
  medal + sword). Two lessons for bottom-heavy subjects: (1) keep the
  subject off the canvas bottom edge in the guide, (2) prefer the short
  prompt — the LoRA + type desc already carry style; every extra material
  word is another seed-dependent coin flip. Reproduce the winner with
  `--count 1` and the same seed.