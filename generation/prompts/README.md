# Prompt library — 2D game assets

This folder is the **source of truth for prompt tuning**. Each asset is one
`.md` file (see [`_SPEC_TEMPLATE.md`](_SPEC_TEMPLATE.md)) describing the
**positive** prompt, the **negative** prompt, the knobs that produced a good
result, and the iteration record — what we tried, what made it worse, what made
it reliable.

> **Status: manual tuning phase.** The microservice still assembles prompts from
> the hardcoded blocks in [`../prompts.py`](../prompts.py) and
> [`../asset_types.py`](../asset_types.py). The specs here are the *target* we
> are converging on. Once enough are validated, `prompts.py` can either read
> these files directly or the validated fragments can be lifted back into code.
> Do **not** wire them into the pipeline yet — reliability first, automation
> second.

---

## The LoRA formula (game_assets_v3)

The model was trained with a specific positive structure. Everything below
starts from it:

```
GRPZA, <subject>, <type/context>, white background, game asset
```

- **`GRPZA`** is the LoRA trigger token, always first.
- The subject is the concrete object (readability and silhouettes come from the
  ControlNet guide; the prompt names *what* it is, not how to draw it).
- **`white background`** is essential for clean, composable single-layer assets.
  It is the single biggest reliability lever — the shared style block in
  `../prompts.py` `STYLE` currently says `flat background`/`solid flat
  background` but never literally `white background`. That is the first fix in
  the shared template.
- **`game asset`** is the trailing task token.

## Anatomy of a good spec

Each `.md` captures, in order:

| Section | What it's for |
|---|---|
| `type` | which asset class (weapon / prop / armor / background) |
| `canvas` | size that type runs at (weapon/prop/armor 768², background 1024²) |
| `strength` | ControlNet strength that gave the best silhouette adherence |
| `steps` / `guidance` | sampler settings for this subject |
| `subject` | the one-line thing being drawn |
| `positive` | the assembled positive prompt (LoRA formula + details) |
| `negative` | the negative prompt (shared + type + subject-specific) |
| `iterations` | reverse-chronological log: prompt change → result → verdict |

## Guiding rules (learned so far)

1. **Lead with the trigger, then the subject, then `white background`, then the
   task word.** Order matters to this LoRA.
2. **Name the object, don't describe material first.** Let the LoRA handle
   "2D game art style"; the prompt adds *what* it is (`golden medal`, `iron
   sword`) and optional *sparse* attributes (color, one material) — keep it short.
3. **Keep the positive short and concrete.** Long prose fights the LoRA. The
   detailed composition/quality lives in negatives and in the shared style, not
   the positive.
4. **White background consistently.** Every background-agnostic asset targets
   `white background`; only the *background* type deviates, and it is exempt
   from the guide entirely.
5. **Record everything.** A spec with no `iterations` isn't validated. Only
   prompts with at least one successful run and a `reliable: yes` mark are
   considered done.

## Index

| Asset | File | Type | Status |
|---|---|---|---|
| Golden medal | [`medal.md`](medal.md) | prop | proven (one successful run) |
| Iron sword | [`iron-sword.md`](iron-sword.md) | weapon | authored, pending run |
| Health potion | [`potion.md`](potion.md) | prop | authored, pending run |