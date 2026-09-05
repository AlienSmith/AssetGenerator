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
rm -f /tmp/feishu-gen/gen.sock            # clear any stale socket from a prior run
cd ComfyUI_t && ./venv/bin/python main.py \
  --generation-socket /tmp/feishu-gen/gen.sock \
  --listen 127.0.0.1 --port 8188 --cpu-vae
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

See the standalone drawing guide:
**[`mask-drawing-guide.md`](mask-drawing-guide.md)** — toolchain
(nakkas-canvas to draw, svg-mcp to convert/verify), the step-by-step
workflow, and the mask polarity & design rules (white background; black body
+ `#808080` inner details — two non-white colors is all Canny needs, with
≥ ~0.38 gray separation between adjacent regions; always prefer a provided
primitive). The optional Canny-outline QA probe lives in §8 — routine masks
that follow the palette don't need it.

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
- Prompt tips: type keywords steer the type resolution (`coin`/`key`/`potion`
  → prop, `sword`/`axe` → weapon, `chestplate`/`helm` → armor); unmatched
  prompts fall back to the default type `prop`
  ([`asset_types.py`](../ComfyUI_t/generation/asset_types.py)). The `GRPZA`
  LoRA trigger + style block are prepended automatically
  ([`prompts.py`](../ComfyUI_t/generation/prompts.py)).
- **Prompt library:** the per-asset prompt specs live in
  [`generation/prompts/`](../ComfyUI_t/generation/prompts/README.md) — one
  `.md` per asset (subject, positive/negative, knobs, iteration log), written
  from the template
  [`_SPEC_TEMPLATE.md`](../ComfyUI_t/generation/prompts/_SPEC_TEMPLATE.md) and
  indexed in the folder's
  [`README.md`](../ComfyUI_t/generation/prompts/README.md). Copy the asset's
  **core fragment** (its positive minus the leading `GRPZA`, e.g. `red star on
  a golden medal`) into the payload's `prompt` field — the service prepends
  the trigger and appends the type descriptor + style block itself, so pasting
  a spec's full assembled positive would duplicate those tokens.
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
Payload: `{status, variant, progress, stage, message}`. A 768² prop verified at
~50 s end-to-end on this GPU (14 steps × ~2 s denoising + ~10 s model init).
A truly cold boot / first run may add more for disk model load. `completed`
looks like:

```json
{"status": "completed", "variant": "1/1", "progress": 1.0,
 "stage": "done", "message": "saved 1 image(s) to output dir"}
```

Cancel mid-run with:
`curl -s --unix-socket /tmp/feishu-gen/gen.sock -X DELETE http://localhost/generations/$JOB`

---

## 5. Collect the output

Each batch gets its own folder: `ComfyUI_t/output/flux_<type>/<prompt>_<timestamp>/`
(e.g. `flux_prop/red_star_on_a_golden_medal_20260905-170154/`), built by
[`batch_output_slug()`](../ComfyUI_t/generation/pipeline.py) from the sanitized
prompt (≤48 chars) + `YYYYMMDD-HHMMSS`. Inside, the leading `v01`/`v02` is the
variant index; the trailing `00001` is a **global** image counter that keeps
incrementing across runs (it never resets per run):

```bash
ls -td ComfyUI_t/output/flux_prop/*/ | head      # newest batch first
ls -t ComfyUI_t/output/flux_prop/<batch_dir>/    # variants of one batch
```

The completed job's `message` field reports the exact batch folder. Harness
runs (`generation/harness.py`) use the same scheme unless `--prefix` overrides
it.

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
mkdir -p /tmp/feishu-gen && rm -f /tmp/feishu-gen/gen.sock && cd ComfyUI_t && \
  ./venv/bin/python main.py --generation-socket /tmp/feishu-gen/gen.sock \
  --listen 127.0.0.1 --port 8188 --cpu-vae

# 2. mask — draw per docs/mask-drawing-guide.md (nakkas-canvas + svg-mcp)

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

---

## 8. QA — verify the intermediate Canny outline (only when needed)

Not part of the routine. If the mask follows the drawing guide's palette
(white background, black body, `#808080` inner details), the outline is
guaranteed — skip straight to §3. Run this probe only when a mask is unusual
(novel geometry, very small regions) or a finished asset is missing an
element and you suspect the guide.

The pipeline reduces the guide to edges with `Canny` (0.4/0.8). Check what
ControlNet will actually see by running the same nodes on the live server:

```bash
ComfyUI_t/venv/bin/python -c "
import json, urllib.request, time, uuid
graph = {
  '1': {'class_type': 'LoadImage', 'inputs': {'image': 'medal_star_mask.png'}},
  '2': {'class_type': 'Canny', 'inputs': {'image': ['1', 0], 'low_threshold': 0.4, 'high_threshold': 0.8}},
  '3': {'class_type': 'SaveImage', 'inputs': {'images': ['2', 0], 'filename_prefix': 'canny_probe/<mask_name>'}},
}
pid = str(uuid.uuid4())
req = urllib.request.Request('http://127.0.0.1:8188/prompt',
    data=json.dumps({'prompt': graph, 'client_id': 'canny-probe', 'prompt_id': pid}).encode(),
    headers={'Content-Type': 'application/json'}, method='POST')
urllib.request.urlopen(req)
for i in range(40):
    time.sleep(2)
    with urllib.request.urlopen(f'http://127.0.0.1:8188/history/{pid}') as r:
        h = json.loads(r.read())
    if pid in h and 'outputs' in h[pid]:
        print('OUT:', [i['filename'] for o in h[pid]['outputs'].values() for i in o.get('images', [])]); break
"
```

The result lands in `ComfyUI_t/output/canny_probe/`. **Every region boundary
must be visible** in that outline — if an inner region is missing, its fill
is too close in luminance to its neighbor (see the drawing guide's gray-step
rule).
