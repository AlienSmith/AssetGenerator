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

`_no successful run yet — pending.`