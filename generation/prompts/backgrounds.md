# Game backgrounds (theme set)

Full-canvas 2D scene backgrounds for level art — the first validated use of
the `background` asset type's **pure txt2img path** (`guide_mode="none"`, no
ControlNet, payload image ignored). Six themes generated in one session, all
landing in the LoRA's flat 2D game style with layered depth.

## Type & knobs

| Key | Value |
|---|---|
| type | `background` |
| canvas | `1024` |
| strength | n/a (no ControlNet — `guide_mode="none"`) |
| guidance | `3.5` |
| steps | `16` |
| denoise | `1.0` |

## Subject

Free-text scene description. The type keyword (`background`, `scene`,
`landscape`, `environment`, `skyline`, `backdrop`, `sky`) must appear in the
prompt so `resolve_asset_type` picks the background type.

## Positive

Assembled by `build_positive` as `GRPZA, <subject>, background.desc,
_BACKGROUND_STYLE`. The background type carries its own style block (no
"white background" / "single object" wording — those fight a full-canvas
scene):

```
GRPZA, <subject>, scene background, wide establishing shot, environment,
landscape, atmospheric, flat 2D game background art, flat cel shading,
vibrant flat colors, clean bold shapes, cohesive scene, game background
```

## Negative

`_BACKGROUND_NEGATIVE` (replaces the shared block wholesale — the shared
"colored background" ban would fight the scene). Note: Flux runs at cfg=1.0
so the negative is dead weight; the positive phrasing does the steering.

## Iterations

### `2026-09-06` — first txt2img batch, 6 themes × 2 variants, all reliable
- **Changed:** pipeline refactor — `guide_mode="none"` for background
  (graph has no LoadImage/ControlNet nodes), per-type style/negative
  overrides in `../asset_types.py`, `prepare_guide()` dispatch in
  `../guides.py`. Server restarted to load it.
- **How it ran:** harness (`python -m generation.harness --type background
  --prompt "<subject>" --count 2`), no mask. ~2 min per variant (1024²,
  16 steps).
- **Prompts + results (all 👍 flat 2D style, layered depth, no white bg):**
  - `misty forest at dawn, tall pine trees` — sun + layered blue pines,
    storybook parallax look. Output
    `output/flux_background/misty_forest_at_dawn_tall_pine_trees_20260906-171631/`
  - `dark dungeon interior, stone walls, torches` — arch corridor, columns,
    torch bowls, moonlight. `..._dark_dungeon_interior..._20260906-172002/`
  - `cozy medieval village, cobblestone road, small houses` — timber-frame
    houses, castle towers behind. `..._cozy_medieval_village..._20260906-172217/`
  - `sunny desert with dunes and ruins` — canyon dunes, warm palette; ruins
    subtle. `..._sunny_desert..._20260906-172432/`
  - `snowy mountain village at night, aurora sky` — aurora, cabins, pines;
    two tiny skier figures slipped in (negative is inert at cfg=1.0 — add
    "no people" style positive phrasing if it matters).
    `..._snowy_mountain_village..._20260906-203019/`
  - `volcanic wasteland, lava rivers, dark rocks` — lava valley, purple sky,
    strong silhouette layering. `..._volcanic_wasteland..._20260906-203234/`
- **Verdict:** 👍👍 background type production-ready for themed scene art.
  Keep subjects short ("<place>, <2-3 concrete features>"); the style block
  carries the look. Watch for stray small figures in populated scenes.
