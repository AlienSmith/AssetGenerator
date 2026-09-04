# Single-layer static asset generation service

A thin HomeBot-compatible HTTP-over-Unix-socket service that runs inside the
ComfyUI process. It turns ComfyUI into the generation backend HomeBot already
calls, producing **single-layer static game assets** (weapon, prop, armor,
background) from a free-text prompt plus a base64 ControlNet reference.

This is intentionally separate from `asset_pipeline/` (the layered pipeline) —
that directory is left untouched.

## Contract (mirrors HomeBot)

| Endpoint | Method | Request | Response |
|---|---|---|---|
| `/health` | GET | — | `200` |
| `/generations` | POST | `{image, prompt, count}` | `201 {job_id}` |
| `/generations/{id}` | GET | — | `200 {status, variant, progress, stage, message}` |
| `/generations/{id}` | DELETE | — | `204` |

- `image`: base64-encoded PNG ControlNet reference (the single-layer silhouette).
- `count`: number of variants in the batch.
- `status`: `pending -> running -> completed | failed | canceled`.
- On `completed`, produced PNGs are persisted as assets by ComfyUI's own
  `prompt_worker` via `register_output_files(paths, job_id=prompt_id)`; the
  orchestrator never fetches images back.
- Unknown `/generations/{id}` on GET/DELETE -> `404`.

## Enable

```bash
python main.py --generation-socket --preload-gen-models \
  --listen 127.0.0.1 --port 8188
```

- `--generation-socket`: bind the Contract routes on a Unix socket, default
  `/run/feishu-bot/gen.sock`. An optional value overrides the path. Implies
  `--enable-assets` so produced assets are persisted.
- `--preload-gen-models`: warm the model stack at startup (VRAM/disk budget
  check). Off by default to avoid paying the ~12 GB Flux unet read at boot.
- **Do NOT pass `--highvram`/`--gpu-only`/`--novram`.** ComfyUI's default
  dynamic VRAM is what fits the ~12 GB Flux unet into 16 GB
  (`comfy/model_management.load_models_gpu` sees every model and offloads
  inactive ones to CPU). See `plans/asset-generation-microservice.md` §4.1.

## Config / env

| Var | Meaning | Default |
|---|---|---|
| `GEN_SOCKET_PATH` | Unix socket to bind | `/run/feishu-bot/gen.sock` |
| `GEN_SOCKET_MODE` | socket file mode (octal) | `0o660` |
| `GEN_MAX_BATCH_COUNT` | cap on `count` per `/generations` | `8` |
| `GEN_DEFAULT_BATCH_COUNT` | `count` when omitted/<=0 | `1` |
| `GEN_MAX_PROMPT_LEN` | prompt length cap | `500` |

## How it works

- `asset_types.py` — per-type knobs (canvas, ControlNet strength, guidance,
  steps, denoise) + keywords used to derive the type from a free-text prompt.
- `prompts.py` — shared style/negative blocks + per-type descriptors.
- `pipeline.py` — builds a plain ComfyUI workflow graph (GGUF Flux unet,
  DualCLIPLoaderGGUF, VAE, LoRA, ControlNet, FluxGuidance, Canny, KSampler,
  SaveImage). Pure dict; offline-testable.
- `guides.py` — decodes the base64 reference and saves it into the input folder
  so the graph's `LoadImage` node can read it.
- `loader.py` — resolves each model file into a `ModelPatcher` once (identity
  cache) so dynamic VRAM reuses the loaded weights across a batch's variants.
- `service.py` — batch job manager: one `POST` = N variants, each submitted to
  the in-process queue as its own `prompt_id`; polls `/history` to map
  progress/variant/status; `DELETE` cancels the queued prompt_ids.
- `server.py` — aiohttp `web.UnixSite` route layer implementing the four
  contract endpoints.

## Deploy

A systemd unit is provided at `deploy/feishu-gen.service`. Both services must
share the group that owns `/run/feishu-bot` (HomeBot's unit creates it, mode
0755, group `feishubot`). Start ComfyUI with the generation flag; the socket
defaults to the same path HomeBot reads.