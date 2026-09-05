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

### `2026-09-05` — 4-variant batch, per-batch output folder VERIFIED
- **Changed:** `batch_output_slug()` in `../pipeline.py` — every batch now
  writes to `output/flux_<type>/<prompt>_<YYYYMMDD-HHMMSS>/` instead of mixing
  into `flux_<type>/`. Wired into `../service.py` (`_submit_variant` prefix)
  and `../harness.py`. Server restarted to load it (approved this time).
- **Prompt:** `red star on a golden medal`, count=4, mask
  `input/medal_star_mask.png` (black disc + `#808080` star).
- **Result:** job `gen_3e0ce1339d78` → `completed` 4/4 in ~4 min (~45–83 s per
  variant). Output
  `output/flux_prop/red_star_on_a_golden_medal_20260905-170154/`:
  `v01`–`v04_00001_.png`. All four are clean golden medals with a red
  five-point star on white — v01 flat-icon with beveled rim, v02 faceted star
  with long-shadow, v03 soft 3D relief, v04 golden rim + red inner disc.
  Zero failures, zero off-guide stars (the `#808080` mask fix holds at batch
  scale).
- **Verdict:** 👍👍 batch reliability confirmed; per-batch folder works
  end-to-end (job `message` reports the folder path).

### `2026-09-05` — synthetic-guide bug found and fixed: guide now TRACES the mask
- **Bug (user spotted):** the generated star "looks very different from our
  mask image". Root cause: `make_guide_with_detail()` in `../guides.py` used the
  mask ONLY for canvas size, then painted a fully synthetic medal (own disc
  `R=0.39*min(W,H)`, own star polygon `R2=0.62*R`, own rim arcs). The drawn
  star geometry never reached ControlNet; earlier "matches" were the prompt
  (`red star on a golden medal`) doing the work, not the hint.
- **Fix 1 (trace, don't paint):** rewrote the function to extract luminance
  bands from the mask itself: below 48 = body (gold gradient fill), 48-200 =
  detail (solid red), 200+ = white background; bevel rim derived from the
  body band's own edge (MinFilter erosion ring, diagonal bright/dark split).
  All-white mask passes through; no-dark-band masks treat the whole non-white
  area as body. Zero invented geometry.
- **Fix 2 (Canny contrast):** first live check (`gen_477f11fbc7c2`, folder
  `..._223903`) showed the hint had the disc circle but NO star: red-on-gold is
  only a ~0.22 luminance step, below the graph's Canny 0.4/0.8 defaults (the
  old synthetic guide's star edges were dashed, barely passing). Added a dark
  inner outline around detail shapes (detail band minus its own erosion,
  `(40,16,12)`), still 100% mask-derived. Offline edge count at 0.4/0.8 went
  2271 to 3833 (the raw mask itself scores 3669).
- **Verified:** job `gen_1d1f0130b003` = completed 90.8 s, folder
  `output/flux_prop/red_star_on_a_golden_medal_20260905-225722/`. The
  `v01_canny_hint` now shows the complete solid star contour matching the drawn
  mask, and the render's star proportions follow the mask geometry.
- **Verdict:** guide is now faithful to the mask by construction. The per-run
  `*_canny_hint` save (added earlier) is what made this bug visible; keep
  checking it whenever a render "doesn't match".

### `2026-09-05` — no-lighting / no-shadow requirement (engine lights assets at runtime)
- **Request (user):** the game applies lighting and shadows at runtime, so the
  asset itself must not contain baked-in shading.
- **Constraint discovered:** Flux runs at cfg=1.0 (`../pipeline.py`), so the
  negative prompt has ZERO effect - every "shadow" entry in NEGATIVE is dead
  weight. All steering must be positive phrasing + the guide image.
- **Fix A (flat guide):** `../guides.py` rewritten to solid fills only - body
  solid mid-gold `(184,124,42)` (white-to-body luminance step ~0.48 stays above
  the Canny 0.4 default), red detail + dark outline kept, gold gradient and
  bevel rim REMOVED. A shaded guide teaches the model to bake lighting in.
  Offline check: exactly 4 colors, 3751 edges at 0.4/0.8.
- **Fix B (STYLE v1, negation list):** added "no lighting, no shadows, no drop
  shadow, no gradients" to STYLE. Batch `gen_a1581c656838` (`..._231902`):
  flatter, but a soft drop shadow survived. Literal "no X" is weak for Flux.
- **Fix C (STYLE v2, positive phrasing):** STYLE now reads "flat vector icon
  style, uniform solid colors, evenly lit, minimal shading". Batch
  `gen_841d1b9f6847` (`..._232436`): still a LONG drop shadow - worse.
- **Fix D (the one that worked):** the prop type descriptor said "floating
  icon" (`../asset_types.py`). "Floating" invites illustrators to draw a drop
  shadow to communicate float. Changed to "item icon". Batch
  `gen_52248a3efda4` (`..._233710`): drop shadow GONE, disc nearly uniform,
  only trace shading at the star's lower edges.
- **Verdict:** requirement met for this subject. Lesson: hunt for
  shadow-inviting words in the prompt itself before blaming the model; and
  never rely on the negative prompt with Flux at cfg=1.0.

### `2026-09-05` — STYLE restore (user: "texture looks weird") + richer mask design
- **User feedback:** batch 5 texture looked off for a 2D game asset; asked
  whether GRPZA / white background / game asset were even in the prompt. They
  were (verified by printing `build_positive()` output), but the diagnosis
  showed my STYLE v2 had REPLACED the LoRA's proven style tokens
  ("flat cel shading", "vibrant flat colors") with generic vector-icon wording
  ("flat vector icon style, uniform solid colors, evenly lit, minimal
  shading") - that is what broke the texture.
- **Fix E:** STYLE restored verbatim to the proven block ("flat 2D sprite
  asset, flat cel shading, bold clean outlines, vibrant flat colors, single
  object on a white background, white background, game asset") with a comment
  warning never to swap the LoRA's style tokens again. Batch `gen_bff0146968f8`
  (`..._234833`): LoRA texture back, shadow only a subtle hint.
- **Fix F (user: "medal is too simple"):** drew a richer mask
  (`input/medal_rich_mask.png`): swallowtail ribbon strap + gold disc +
  decorative inner ring + star, using the same 3-band palette (white bg /
  #1a1a1a body / #808080 detail). Tracer handled it unchanged: 4 flat colors,
  ring+star+ribbon all in the guide. Batch `gen_e258be36771e`
  (`..._235657`): proper game-asset medal - red ring, gold disc, red star,
  flat cel-shaded, no drop shadow. The ribbon itself was dropped by the model
  (ControlNet 0.5 lets background elements fade) but the ring alone carries
  the richness.
- **Verdict:** current production config = proven STYLE + "item icon" (no
  "floating") + flat tracing guide. Design richness comes from the MASK, not
  the prompt - draw more bands/shapes and the tracer picks them up for free.

### `2026-09-05` — line-art hint (user idea): white lines on black, Canny node removed
- **User feedback:** "there is one issue take a look at canny hint image, the
  line is broken and the top part seems to be missing a big chunk" (ribbon top
  edge gone, side contours dashed). Then: "I think using the color guide image
  for canny is the reason for broken outline" and finally: "how about you use
  the svg tool to generate the white line on the black background image
  directly then we can use that image."
- **Diagnosis (offline, kornia canny):** the colored repaint created DOUBLE
  edges a few px apart (white-to-outline and outline-to-red) that interfere
  after Canny's internal Gaussian blur -> dashed contours. A raw gray-mask
  pass-through was no better: its luminance steps (255->128 = 0.50,
  128->26 = 0.40) sit at/below the 0.4 threshold after blur -> dotted edges at
  EVERY threshold pair swept (0.1-0.8, six pairs, identical counts).
- **Fix G (user's idea, generalized):** the guide is now ALWAYS a finished
  edge map - white lines on black, flux_canny's exact training format:
  * `guides.make_guide_with_detail()` auto-detects the reference kind by mean
    luminance (<64 = line art -> binarize; else region mask -> FIND_EDGES +
    threshold 64 + MaxFilter(5) dilation, padded with the corner color so the
    white background doesn't draw a frame around the canvas).
  * `pipeline.build_workflow()` DELETED the Canny node: LoadImage feeds
    ControlNetApplyAdvanced directly. Running Canny on line art would
    re-derive edges FROM the lines and double every stroke. The QA save now
    dumps the raw hint the model consumes.
- **Verified offline:** mask-derived guide 100% connected strokes (ribbon-top
  1054 px vs 100 dotted px before), no frame; drawn line art 100% connected.
- **Verified live:** batch `red_star_on_a_golden_medal_20260906-014644`
  (mask-derived hint): solid hint, clean medal, ribbon still dropped by the
  model (strength 0.5, unchanged). Batch `lineart_test` (drawn
  `input/medal_line_art.png` as reference): **ribbon PRESENT** - red
  swallowtail strap + gold disc + ring + yellow star, flat colors, white bg,
  no shadow. Best medal so far; the drawn line art carries the ribbon because
  its strokes are explicit geometry, not derived edges.
- **Verdict:** production guide = white-line edge map. Two ways to supply it:
  send line art directly (full control, ribbon survives) or send the region
  mask (auto-converted; simpler but thin features can still fade at strength
  0.5). The Canny node is gone from the graph; thresholds are no longer a knob.

### `2026-09-05` — color binding on the line-art hint (user: "I thought the star suppose to be red?")
- **Problem:** the line-art guide carries NO color info (pure geometry), so
  color assignment is model+seed. With "red star on a golden medal" the seed
  put red on the ribbon and left the star gold.
- **Prompt wording sweep (line-art reference, one variant each):**
  * "golden medal with a red star emblem, red ribbon" -> ribbon red, star
    still GOLD ("golden" bled onto the star).
  * "red star on a golden medal, red ribbon strap" -> ribbon red, star gold.
  * "red star on a golden medal, the star is red, red ribbon strap" ->
    **red star + red ribbon + gold disc** (`lineart_redstar3_00001_.png`).
    Explicit restatement ("the star is red") is what binds the color.
- **Lesson:** with a colorless (line-art) guide, bind each color to its part
  explicitly in the positive prompt; a single leading color word is not
  enough and "golden medal" actively bleeds onto inner details.