# Iron sword

A single medieval iron sword, fantasy game weapon icon. Exercises the `weapon`
asset type (its `extra_neg` bans hands/character).

## Type & knobs

| Key | Value |
|---|---|
| type | `weapon` |
| canvas | `768` |
| strength | `0.6` |
| guidance | `3.5` |
| steps | `14` |
| denoise | `1.0` |

## Subject

a single iron sword

## Positive

```
GRPZA, a single iron sword, held off to the side, weapon centered, clean silhouette, item icon, white background, game asset
```

Kept short: names the object and sparse attributes (single, iron). The "held
off to the side / centered / clean silhouette" phrasing comes from the `weapon`
`desc` in `asset_types.py`; it stays in the positive as the LoRA responds to it
there, not in the negative.

## Negative

```
photorealistic, realistic, 3d, messy, blurry, lowres, text, watermark,
signature, multiple objects, layered composition, gradient, shadow,
busy background, scale mismatch,
hands, fingers, character, person, body, multiple weapons, shadow on ground
```

(`weapon` extra_neg in `asset_types.py`.)

## Iterations

### `2026-09-05` — first-try success: line-art hint + color-bound prompt

**Setup.** Drew the sword as white line art on black with nakkas-canvas
(3 drafts: draft 1-2 read as daggers — blade too narrow / fully triangular;
draft 3 gave parallel blade edges + short triangular tip, a fuller line,
rounded crossguard, wrapped grip, ring pommel). Saved to
`input/sword_line_art.png`. Fed straight to ControlNet (no Canny node —
see medal.md "line-art hint" iteration).

**Prompt** (color binding per medal.md lesson — restate each color on its part):

```
a single iron sword, the blade is steel grey, the crossguard and pommel are
dark bronze, brown leather grip
```

**Result** (`output/sword_lineart_00001_.png`, seed 1788632556323, strength 0.6):
first-try success. Geometry follows the hint exactly (parallel edges, tip,
fuller, crossguard, grip wraps, ring pommel); all three colors bound
correctly — steel-grey blade, dark bronze crossguard, brown leather grip.
Flat colors, white background, no shadow, no hands/character.

**Verdict.** The line-art routine generalizes beyond the medal: draw clean
line art → hint feeds ControlNet directly → bind every color explicitly in
the prompt. No pipeline changes needed for a new asset type.

### `2026-09-05` — 6-variant batch, per-batch folder + seed in filename

**Setup.** Same line-art hint + prompt as above, run through the harness's
new `--count 6` flag: one guide file, six graphs, seeds `base+0..5`, all
sharing one per-batch subfolder.

**Output layout** (`output/flux_weapon/<prompt>_<timestamp>/`):

```
seed1788633599230_00001_.png   ← render; filename carries the SEED
seed1788633599231_00001_.png
...
seed1788633599235_00001_.png
seed1788633599230_canny_hint_00001_.png  ← QA hint dump (one per variant)
```

The seed in the filename is deliberate: pick a favorite style, then
re-run with that exact seed to reproduce it. The harness also cleans up its
`gen_guide_*.png` after the batch (the microservice's `_cleanup_guide`
equivalent, since the harness bypasses the service).

**Result.** All 6 variants: geometry identical to the hint, colors bound
(steel blade / bronze guard / brown grip), flat, white bg, no shadow.
Variation across seeds is subtle — grip shading, pommel ring thickness,
blade bevel — which is what we want from a fixed silhouette + fixed palette.

**Verdict.** Batch reliability confirmed: 6/6 usable, zero failures, ~44 s
per variant. The `--count` flag is the standard way to explore styles.