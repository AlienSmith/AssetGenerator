"""Standalone interface to exercise the generation pipeline against a RUNNING server.

Why this exists
---------------
The ComfyUI process boots the generation service from whatever `service.py` /
`guides.py` were on disk at process start. Editing those files does not hot-reload
them, and we deliberately do NOT restart the long-booted server to pick up new
code (the user's rule). So to test pipeline code changes against the live engine
without touching the running process, this harness:

  1. builds the ControlNet guide with the *current* `make_guide_with_detail()`
     (from the current files on disk) and persists it into the input folder,
  2. builds the workflow graph with the *current* `build_workflow()`, and
  3. submits it to the running server over the standard ComfyUI HTTP `/prompt`
     endpoint, then polls `/history/{prompt_id}` until completion.

The running server is used purely as the inference engine: its `LoadImage` node
reads the freshly-written guide from the input folder at execution time, and its
queued `/prompt` path runs the graph. No restart, no in-process access needed.

Usage
-----
    cd ComfyUI_t
    ./venv/bin/python generation/harness.py \
        --mask input/medal_star_mask.png \
        --prompt "GRPZA, red star on a golden medal, white background, game asset" \
        --type prop

The optional `--prefix` overrides the `SaveImage` filename prefix entirely.
By default the run writes to `output/flux_<type>/<prompt>_<timestamp>/` — the
same per-batch folder scheme the microservice uses (see
`pipeline.batch_output_slug`).
"""
from __future__ import annotations

import argparse
import base64
import io
import time
import urllib.request
import urllib.error
import uuid

from PIL import Image

# Must run from the ComfyUI_t directory so `folder_paths` and the `generation`
# package resolve; the server also ran from there, so the input folder matches.
import folder_paths  # noqa: E402

from generation import pipeline as pl  # noqa: E402
from generation import asset_types  # noqa: E402
from generation.guides import make_guide_with_detail  # noqa: E402


DEFAULT_URL = "http://127.0.0.1:8188"


def _http_json(method: str, url: str, payload: dict | None = None, timeout: int = 30):
    data = None
    headers = {}
    if payload is not None:
        data = __import__("json").dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        return resp.status, __import__("json").loads(body.decode("utf-8")) if body else {}


def _mask_base64(mask_path: str) -> str:
    with open(mask_path, "rb") as fh:
        raw = fh.read()
    # Re-encode as PNG so the guide path matches how HomeBot sends it.
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def run(
    *,
    mask_path: str,
    prompt: str,
    asset_key: str,
    base_url: str,
    prefix: str | None,
) -> dict:
    asset_type = asset_types.resolve_asset_type(prompt)
    if asset_key:
        asset_type = asset_types.ASSET_TYPES[asset_key]

    # 1. Build the detail-bearing guide with the current code on disk.
    guide_name = make_guide_with_detail(
        _mask_base64(mask_path),
        width=asset_type.canvas,
        height=asset_type.canvas,
    )
    print(f"[harness] guide persisted: {guide_name}")

    # 2. Build the workflow graph with the current pipeline code. Outputs land
    #    in output/flux_<type>/<prompt>_<timestamp>/ unless --prefix overrides.
    seed = int(time.time() * 1000)
    batch_dir = prefix or f"flux_{asset_type.key}/{pl.batch_output_slug(prompt)}"
    graph = pl.build_workflow(
        asset_type=asset_type,
        prompt=prompt,
        seed=seed,
        guide_image_name=guide_name,
        filename_prefix=batch_dir,
    )
    print(f"[harness] output prefix: {batch_dir}")

    # 3. Submit to the RUNNING server over standard HTTP /prompt.
    prompt_id = str(uuid.uuid4())
    status, resp = _http_json(
        "POST", f"{base_url}/prompt",
        {"prompt": graph, "client_id": "harness", "prompt_id": prompt_id},
    )
    if status != 200:
        raise RuntimeError(f"/prompt failed {status}: {resp}")
    print(f"[harness] submitted prompt_id={prompt_id} seed={seed}")

    # 4. Poll history until the prompt is done.
    deadline = time.time() + 300
    last = None
    while time.time() < deadline:
        time.sleep(5)
        _status, hist = _http_json("GET", f"{base_url}/history/{prompt_id}")
        if prompt_id not in hist:
            continue
        entry = hist[prompt_id]
        if "outputs" in entry:
            last = entry
            break
        if entry.get("status", {}).get("status_str") == "error":
            raise RuntimeError(f"prompt {prompt_id} errored: {entry}")
    if last is None:
        raise TimeoutError(f"prompt {prompt_id} did not finish in 300s")

    print("[harness] done; outputs:")
    for node_id, out in last["outputs"].items():
        for img in out.get("images", []):
            print(f"  {img.get('subfolder', '')}/{img['filename']}")
    return last


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mask", required=True, help="medal/asset mask PNG to guide off")
    ap.add_argument("--prompt", required=True, help="free-text subject (no type suffix)")
    ap.add_argument("--type", dest="asset_key", default=None,
                    help="override asset type key: prop|weapon|armor|background")
    ap.add_argument("--base-url", default=DEFAULT_URL)
    ap.add_argument("--prefix", default=None,
                    help="SaveImage filename prefix (defaults to flux_<type>)")
    args = ap.parse_args()

    run(
        mask_path=args.mask,
        prompt=args.prompt,
        asset_key=args.asset_key,
        base_url=args.base_url,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()