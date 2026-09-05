# Golden medal

A golden medallion with a red five-point star emblem, an in-game collectible
prop. This is the first **verified** asset — the exact prompt below was run
end-to-end through the live microservice and produced a clean, single-layer
result on a (unintentionally non-white, see iterations) background.

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

a golden medal

## Positive

```
GRPZA, red star on a golden medal, white background, game asset
```

The best result to date used this positive. Note it drops a heavy type
descriptor and "medal" does the type resolution via the `prop` keyword path in
`asset_types.py`. Keep the subject short; the star-vs-medal arrangement came
from the ControlNet guide (a black disc outline with a white star inside, see
the runbook §2).

## Negative

```
photorealistic, realistic, 3d, messy, blurry, lowres, text, watermark,
signature, multiple objects, layered composition, gradient, shadow,
busy background, scale mismatch,
character, person, body, hands, multiple objects, shadow on ground, background detail
```

(`prop` extra_neg in `asset_types.py`.)

## Iterations

### `2026-09-05` — verified run, non-white background
- **Changed:** first real runbook run (no manual white-bg enrichment).
- **Prompt:** `GRPZA, red star on a golden medal, white background, game asset`
  (as assembled by `build_positive`: trigger + this prompt + `prop.desc` +
  `STYLE`).
- **Result:** job `gen_db156887adfd` → `completed` in ~50 s, 14 steps,
  output `ComfyUI_t/output/flux_prop/v01_00002_.png` (768² RGB). Clean medal
  with star; but the **background came out as a flat olive/tan tone, not
  white** — the shared `STYLE` block says `flat background`/`solid flat
  background`, never literally `white background`, so the model chose its own
  flat color.
- **Verdict:** 👍 silhouette & subject reliable; 🔁 background needs the shared
  template to say `white background` (todo: fix `STYLE` in `../prompts.py`),
  then re-run to confirm it honors white.

### `2026-09-05` — white-bg OK, but *flat* (red blob diagnosis)
- **Changed:** fixed `STYLE` to literal `white background` + negative
  `colored background, solid color background`; re-ran medal.
- **Prompt:** `GRPZA, red star on a golden medal, single prop, ... white
  background, game asset` (new STYLE).
- **Result:** job `gen_f6bc445c32cc` → `completed` (83 s), output
  `flux_prop/v01_00003_.png`. Background **fixed** (border avg RGB
  `(254,254,254)` vs old `(237,184,91)`). BUT the *medal itself* is flat:
  programmatic core scan showed **0.0% highlight (>225)**, ~77% dark (≤120),
  ~78% red. The model paints the whole black-body disc **red** (driven by
  `red star`), ignoring that the disc should be **golden** with a red star on
  top — reads as a flat red blob, not a crafted golden medal with relief.
- **Verdict:** 👍 white-bg confirmed reliable; 👎 surface detail fails.
  Root cause is the **binary black/white mask polarity** feeding Canny: the
  body + emblem both read as "leave white", so the only semantic left is "a
  big dark shape with red". Fix = give the guide *material detail* (gold
  disc + red star + edge bevels + rim shading), i.e. a detail-bearing mask,
  not a flat silhouette. Document in runbook §2.

### `2026-09-05` — detail-guide fix VERIFIED (gold body + rim relief)
- **Changed:** `make_guide_with_detail()` in `../guides.py` now synthesizes a
  material-bearing guide (gold radial-gradient disc + red star + bevel rim on
  white) instead of passing the flat silhouette through. Wired into
  `../service.py` for all non-background types.
- **How it ran:** via the new harness interface `../harness.py` (submits the
  current-on-disk graph to the RUNNING server over `POST /prompt` — no
  restart). `prompt_id 807c3e76-037e-441a-8264-8c4ff351731d`, executed in
  ~74 s, 14 steps.
- **Result:** output `output/flux_prop/harness_medal_00001_.png` (768²).
  The medal now renders as a **golden disc with a beveled rim** (darker outer
  ring + lighter inner ring), a **red five-point star** centered on the body,
  and a clean **white background**. The flat red-blob failure mode is gone:
  the gold/red split and rim shading from the guide carried through Canny
  into the render.
- **Verdict:** 👍👍 subject, materials, and background all reliable with the
  detail guide. This is the new baseline guide path for every non-background
  asset type. Remaining polish (optional): star could sit slightly larger /
  rim contrast could be stronger — tune `R2`/`rim_w` in `make_guide_with_detail`
  if a future spec needs it.