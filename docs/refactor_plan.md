# Asset Pipeline Refactor Plan

> Phase 1 (done): models pulled inside the project so `ComfyUI_t` is self-contained.
> Phase 2 (planned, NOT yet applied): split the pipeline code into a clean package layout.
> The code refactor is documented here and will only be executed after this plan is
> reviewed and approved. No service-breaking deletes.

## 1. Goal

The Flux + LoRA + ControlNet pipeline currently mixes two concerns in one file:

- **Pure logic**: the `PARTS` table, guide loading, color-edge hint, prompt building.
  These need no ComfyUI.
- **ComfyUI glue**: `import nodes`, the GGUF package hack (`load_gguf_package`), and the
  dynamic `FluxGuidance` import (`load_module_by_path`). These only work inside a ComfyUI
  checkout.

We want the pure code separated from the glue so the scripts are readable and committable
without dragging ComfyUI internals alongside them.

## 2. Current state

- `generate_body_part.py` holds everything: `PARTS`, `DEFAULT_STRENGTHS`, `build_prompts`,
  `load_guide`, `make_part_mask`, `make_color_edges`, `load_module_by_path`,
  `load_gguf_package`, and `main()`.
- `character.py` is pure config (moved into `asset_pipeline/` during Phase 1).
- `remove_bg.py` is standalone (moved into `asset_pipeline/` during Phase 1).
- `models/` is git-ignored; the six pipeline weights were symlinks to external installs
  (now replaced by real files copied in — original installs left intact).

## 3. Target layout

```
asset_pipeline/
  character.py           # moved, unchanged (shared palette/outfit/style/negative)
  parts.py               # NEW: PARTS table + DEFAULT_STRENGTHS + build_prompts (pure, no torch/nodes)
  guide.py               # NEW: CANVAS + load_guide + make_part_mask + make_color_edges (pure PIL/numpy/torch)
  runtime.py             # NEW: the ONLY ComfyUI-touching module (import nodes, load_gguf_package, load_module_by_path, build_runtime)
  generate_body_part.py  # NEW: thin CLI entry that imports parts/guide/runtime and orchestrates
  remove_bg.py           # moved, unchanged
  test_flux_head.py      # moved history / reference
  test_puyo_*.py         # moved history / reference
```

## 4. Refactor steps

1. **`parts.py`** — move `PARTS`, `DEFAULT_STRENGTHS`, `build_prompts`, and
   `character` import. No `torch`/`nodes` imports.
2. **`guide.py`** — move `CANVAS`, `load_guide`, `make_part_mask`, `make_color_edges`.
   Pure PIL/numpy/torch only, no `nodes`.
3. **`runtime.py`** — absorb `load_module_by_path`, `load_gguf_package`, and all loader
   glue into a single `build_runtime()` returning the loaded unet/clip/vae + the
   `FluxGuidance` module. Only this file touches ComfyUI.
4. **`generate_body_part.py`** — keep argparse + orchestration only: import from
   `parts`/`guide`/`runtime`, call `build_runtime()` once, run the strength/count/steps sweep,
   save outputs.
5. **Delete duplicated top-level files** — `character.py`, `remove_bg.py`,
   `generate_body_part.py` at repo root (they are superceded by `asset_pipeline/`).
6. **Update `docs/asset_generation.md`** — point commands at
   `asset_pipeline/generate_body_part.py`, describe the multi-part workflow.

## 5. Models: pulled inside first (Phase 1)

The six pipeline weights were symlinks pointing outside the repo:

| Repo path                  | External target                      |
|----------------------------|--------------------------------------|
| `models/unet/flux1-dev-Q8_0.gguf`              | `/home/zombie/ComfyUI/models/unet/...`        |
| `models/clip/clip_l.safetensors`               | `/home/zombie/ComfyUI/models/clip/...`        |
| `models/clip/t5-v1_1-xxl-encoder-Q8_0.gguf`    | `/home/zombie/ComfyUI/models/clip/...`        |
| `models/loras/game_assets_v3.safetensors`      | `/home/zombie/bender/ComfyUI/storage/basedir/models/loras/...` |
| `models/vae/ae.safetensors`                    | `/home/zombie/bender/ComfyUI/storage/basedir/models/vae/...`   |
| `models/controlnet/flux_canny_instantx.safetensors` | `/home/zombie/bender/ComfyUI/storage/basedir/models/controlnet/...` |

Phase 1 moved the weights inside: for each symlink, the repo symlink was removed and the
external target file was `mv`'d into the repo path (same disk, no extra copy). The six old
external weights are no longer left behind under `/home/zombie/ComfyUI` or
`/home/zombie/bender/ComfyUI` — they now live solely under `ComfyUI_t/models/`.

`models/` remains git-ignored: the weights are needed locally but are not committed.

## 6. Safety rules

- Never `rm`/`mv` the external installs `/home/zombie/ComfyUI` or
  `/home/zombie/bender/ComfyUI`.
- Copy (not move) model data into the project.
- Stop and review before deleting any repo directory.