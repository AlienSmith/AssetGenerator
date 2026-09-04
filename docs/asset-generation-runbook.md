# Asset Generation Runbook — ControlNet + Flux + LoRA pipeline

Step-by-step playbook for generating a single-layer game asset on this machine
(RTX 5060 Ti 16 GB) using the [`ComfyUI_t/generation/`](../ComfyUI_t/generation/README.md)
microservice. Written from the verified "red star on a golden medal" run —
follow it top to bottom and nothing needs to be figured out twice.

---

## 0. What the pipeline is

```
guide/mask image ──► LoadImage ──► Canny ──► ControlNetApplyAdvanced ──┐
                                                                       ├─► KSampler ──► VAEDecode ──► SaveImage
prompt ──► CLIPTextEncode ──► FluxGuidance ────────────────────────────┘
Flux Q8 gguf + game_assets_v3 LoRA + DualCLIP (clip_l + T5 Q8) + ae.safetensors VAE
```

- Graph builder: [`ComfyUI_t/generation/pipeline.py`](../ComfyUI_t/generation/pipeline.py)
- The guide image is used through a **Canny edge** hint, so the mask must have
  clean, high-contrast region boundaries (see §2).
- Asset type (canvas size, ControlNet strength, steps) is derived from prompt
  keywords in [`ComfyUI_t/generation/asset_types.py`](../ComfyUI_t/generation/asset_types.py):
  `weapon` / `prop` / `armor` → 768², `background` → 1024² (no guide used).

### Models required (all present in `ComfyUI_t/models/`)

| Role | File |
|---|---|
| Unet | `unet/flux1-dev-Q8_0.gguf` |
| CLIP | `clip/clip_l.safetensors` + `clip/t5-v1_1-xxl-encoder-Q8_0.gguf` |
| VAE | `vae/ae.safetensors` |
| LoRA | `loras/game_assets_v3.safetensors` (strength 0.8, model-only) |
| ControlNet | `controlnet/flux_canny_instantx.safetensors` |

---

## 1. Start the server

```bash
mkdir -p /tmp/feishu-gen
cd ComfyUI_t && ./venv/bin/python main.py \
  --generation-socket /tmp/feishu-gen/gen.sock \
  --listen 127.0.0.1 --port 8188
```

- Takes ~30–60 s to boot. Ready when the log prints
  `generation service listening on /tmp/feishu-gen/gen.sock`.
- **Do NOT** add `--highvram`/`--gpu-only`/`--novram` — the ~12 GB Flux unet
  only fits 16 GB VRAM with ComfyUI's default dynamic offloading.
- `/run/feishu-bot/gen.sock` is the production default but needs root to
  create; use a `/tmp` path for ad-hoc runs.
- Stop it later with: `pkill -f "generation-socket"` (also delete a stale
  `/tmp/feishu-gen/gen.sock` before the next start).

### Health check

```bash
curl -s --unix-socket /tmp/feishu-gen/gen.sock http://localhost/health
# → {"ok": true}
```

Note: generation routes exist **only on the Unix socket**, not on the TCP port
(TCP `http://127.0.0.1:8188` is the normal ComfyUI web UI and returns 404 for
`/health`).

---

## 2. Draw the ControlNet mask/guide

Two options.

### Option A — nakkas-canvas MCP (visual, iterative)

1. `render_svg` with a 768×768 canvas:
   - background rect `#ffffff` (white — see polarity note below)
   - body region filled `#000000` (e.g. a `circle` r=300 at 384,384)
   - inner detail region filled `#ffffff` (e.g. a 5-point star `path`)
2. Inspect the returned preview; iterate if the shape is off.
3. `save` with `format: "png"`, `width: 768`, output path inside
   `ComfyUI_t/input/`. ⚠️ `save` does **not** create parent directories —
   `mkdir -p` first, and it appends `-1`, `-2`… instead of overwriting.
4. The saved PNG is RGBA; convert to RGB:
   ```bash
   ComfyUI_t/venv/bin/python -c "
   from PIL import Image
   im = Image.open('ComfyUI_t/input/<name>.png').convert('RGB')
   im.save('ComfyUI_t/input/<name>.png', 'PNG')"
   ```

### Option B — one-liner PIL (fast, no MCP)

```bash
ComfyUI_t/venv/bin/python - <<'EOF'
from PIL import Image, ImageDraw
import math
S = 768
img = Image.new("RGB", (S, S), "white")
d = ImageDraw.Draw(img)
cx = cy = S // 2
r = int(S * 0.39)                      # medal body
d.ellipse([cx-r, cy-r, cx+r, cy+r], fill="black")
pts = []
for i in range(10):                    # star emblem
    rr = int(r*0.62) if i % 2 == 0 else int(r*0.26)
    a = math.pi*i/5 - math.pi/2
    pts.append((cx+rr*math.cos(a), cy+rr*math.sin(a)))
d.polygon(pts, fill="white")
img.save("ComfyUI_t/input/medal_star_mask.png", "PNG")
EOF
```

### Mask polarity & design rules (learned the hard way)

- **White background, black body region, white inner regions.** The flux-canny
  hint convention expects a white background; a black background + white body
  also "works" but the model treats the surrounding black as part of the scene.
- The guide goes through **Canny**, so only *edges* matter: every fill is
  reduced to its outline. Flat color regions = clean single outlines.
- Keep the body region large in frame (≈78% of canvas diameter for medals);
  tiny regions produce weak Canny edges that ControlNet ignores.
- Canvas must match the asset type's canvas (768 for weapon/prop/armor,
  1024 for background — though backgrounds run without a guide).

---

## 3. Submit the generation job

The contract is `POST /generations` with `{image, prompt, count}` where
`image` is the **base64 of the guide PNG**. Build the payload and send it over
the socket:

```bash
# 3a. build payload (base64-encode the mask)
ComfyUI_t/venv/bin/python -c "
import base64, json
with open('ComfyUI_t/input/medal_star_mask.png', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()
json.dump({'image': b64, 'prompt': 'red star on a golden medal', 'count': 1},
          open('/tmp/feishu-gen/payload.json', 'w'))
"

# 3b. POST it
curl -s --unix-socket /tmp/feishu-gen/gen.sock -X POST \
  -H "Content-Type: application/json" \
  -d @/tmp/feishu-gen/payload.json \
  -w "\nHTTP %{http_code}\n" http://localhost/generations
# → {"job_id": "gen_xxxxxxxxxxxx"} + HTTP 201
```

- `count` = number of variants (each is a separate queued prompt with its own
  seed; capped at `GEN_MAX_BATCH_COUNT`, default 8).
- Prompt tips: type keywords steer the type resolution (`medal`/`coin`/`key`
  → prop, `sword`/`axe` → weapon, `chestplate`/`helm` → armor). The `GRPZA`
  LoRA trigger + style block are prepended automatically
  ([`prompts.py`](../ComfyUI_t/generation/prompts.py)).
- ⚠️ Don't use `urllib` with a raw `socket` object for this — it dials TCP and
  fails with `Connection refused`. Use `curl --unix-socket` (or a proper
  `HTTPConnection` subclass).

---

## 4. Poll until done

```bash
JOB=gen_xxxxxxxxxxxx   # from step 3b
while true; do
  curl -s --unix-socket /tmp/feishu-gen/gen.sock http://localhost/generations/$JOB
  echo
  sleep 10
done
```

Status lifecycle: `pending → running → completed | failed | canceled`.
Payload: `{status, variant, progress, stage, message}`. A 768² prop takes
~2 min end-to-end on this GPU (~110 s: ~40 s model load on first run, then
14 steps × ~2 s). `completed` looks like:

```json
{"status": "completed", "variant": "1/1", "progress": 1.0,
 "stage": "done", "message": "saved 1 image(s) to output dir"}
```

Cancel mid-run with:
`curl -s --unix-socket /tmp/feishu-gen/gen.sock -X DELETE http://localhost/generations/$JOB`

---

## 5. Collect the output

Files land under `ComfyUI_t/output/flux_<type>/` (e.g. `flux_prop/`), named
`v01_00001_.png`, `v02_00001_.png`, … per variant:

```bash
ls -t ComfyUI_t/output/flux_prop/ | head
```

View the result: open the PNG directly (VS Code preview / `read_file` on the
image). The `svg-mcp` MCP (`viewSVG` / `viewSVGFile`) is for **SVG** files
only — it will not render PNGs.

---

## 6. Troubleshooting (symptoms seen in the first real run)

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeWarning: coroutine 'validate_prompt' was never awaited` then `TypeError: cannot unpack non-iterable coroutine object` in the server log, variant fails instantly | [`service.py`](../ComfyUI_t/generation/service.py) called the now-async `execution.validate_prompt` synchronously | Fixed (2026-09): `_validate_prompt` hops the coroutine onto the main loop via `run_coroutine_threadsafe`. If it reappears after a ComfyUI sync, re-check that helper. |
| `Failed to validate prompt ... Exception when validating node: 14` (details = a bare number) | Graph link ids were ints while graph keys are strings (`prompt[o_id]` KeyError) | Fixed (2026-09): [`pipeline.py`](../ComfyUI_t/generation/pipeline.py) `add()` returns string ids. Any new graph-builder code must keep both id spaces strings. |
| `POST /generations` returns 201 but no variant ever appears | Same as above — job created, variant submission failed; check server log | Read the ComfyUI terminal; the real exception is printed there. |
| `Connection refused` from a Python client | `urllib` ignored the unix socket | Use `curl --unix-socket` or subclass `http.client.HTTPConnection.connect`. |
| `ENOENT` when saving a mask with nakkas-canvas | `save` doesn't mkdir | `mkdir -p` the target directory first. |
| Mask saved as `name-1.png` instead of overwriting | nakkas-canvas never overwrites | Delete/rename the old file, or use the `-1` name. |
| OOM / model thrashing | `--highvram`-style flags passed | Restart with only the flags in §1. |
| 404 on `http://127.0.0.1:8188/health` | Generation routes are unix-socket-only | Curl the socket path, not the TCP port. |

---

## 7. Quick copy-paste sequence (whole run)

```bash
# 1. server (leave running in its own terminal)
mkdir -p /tmp/feishu-gen && cd ComfyUI_t && ./venv/bin/python main.py \
  --generation-socket /tmp/feishu-gen/gen.sock --listen 127.0.0.1 --port 8188

# 2. mask (PIL one-liner from §2 Option B) — or nakkas-canvas per §2 Option A

# 3. submit
ComfyUI_t/venv/bin/python -c "
import base64, json
b64 = base64.b64encode(open('ComfyUI_t/input/medal_star_mask.png','rb').read()).decode()
json.dump({'image': b64, 'prompt': '<YOUR PROMPT>', 'count': 1},
          open('/tmp/feishu-gen/payload.json','w'))"
curl -s --unix-socket /tmp/feishu-gen/gen.sock -X POST \
  -H "Content-Type: application/json" -d @/tmp/feishu-gen/payload.json \
  http://localhost/generations

# 4. poll (replace JOB id)
curl -s --unix-socket /tmp/feishu-gen/gen.sock http://localhost/generations/gen_XXXX

# 5. collect
ls -t ComfyUI_t/output/flux_prop/
```
