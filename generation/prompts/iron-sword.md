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

`_no successful run yet — pending.`