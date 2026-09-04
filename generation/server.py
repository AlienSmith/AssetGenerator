"""HTTP-over-Unix-socket route layer for the generation microservice.

Mirrors the HomeBot contract exactly (see ``HomeBot/dev/mock_generation.py`` and
``HomeBot/bot/gen_client.py``) so the orchestrator works unchanged:

    GET    /health                        -> 200
    POST   /generations  {image, prompt, count} -> 201 {job_id}
    GET    /generations/{id}              -> 200 {status, variant, progress, stage, message} | 404
    DELETE /generations/{id}              -> 204 | 404

Serves over a aiohttp ``web.UnixSite`` at ``GEN_SOCKET_PATH``
(default ``/run/feishu-bot/gen.sock``), gated behind an env flag so normal ComfyUI
launch is untouched.
"""
from __future__ import annotations

import asyncio
import logging
import os

from aiohttp import web

from generation.service import GenerationService, make_generation_service

logger = logging.getLogger(__name__)

DEFAULT_SOCKET_PATH = os.environ.get("GEN_SOCKET_PATH", "/run/feishu-bot/gen.sock")
SOCKET_MODE = int(os.environ.get("GEN_SOCKET_MODE", "0o660"), 8)


def build_app(server, service: GenerationService | None = None) -> web.Application:
    """Build the aiohttp app bound to the HomeBot Unix-socket contract."""
    service = service or make_generation_service(server)
    app = web.Application()

    async def handle_health(request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def handle_start(request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception:  # noqa: BLE001
            return web.json_response({"error": "bad json"}, status=400)

        image = data.get("image")
        prompt = data.get("prompt")
        count = data.get("count")
        if not image or not prompt:
            return web.json_response({"error": "image and prompt required"}, status=400)

        try:
            count_i = int(count) if count is not None else None
        except (TypeError, ValueError):
            return web.json_response({"error": "count must be an integer"}, status=400)

        prompt = str(prompt).strip()
        if not prompt:
            return web.json_response({"error": "prompt required"}, status=400)

        # Everything above is cheap; the heavy decode/queue work happens here.
        try:
            job_id = await asyncio.to_thread(service.start, image, prompt, count_i)
        except Exception as err:  # noqa: BLE001
            logger.exception("generation start failed")
            return web.json_response({"error": str(err)}, status=500)

        return web.json_response({"job_id": job_id}, status=201)

    async def handle_get(request: web.Request) -> web.Response:
        job_id = request.match_info["id"]
        job = service.get(job_id)
        if job is None:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response(job)

    async def handle_cancel(request: web.Request) -> web.Response:
        job_id = request.match_info["id"]
        canceled = service.cancel(job_id)
        if not canceled:
            return web.json_response({"error": "not found"}, status=404)
        return web.Response(status=204)

    app.router.add_get("/health", handle_health)
    app.router.add_post("/generations", handle_start)
    app.router.add_get("/generations/{id}", handle_get)
    app.router.add_delete("/generations/{id}", handle_cancel)

    return app


async def start_unix_site(
    server, *, socket_path: str | None = None, app: web.Application | None = None
) -> web.UnixSite:
    """Start the generation service on a Unix socket; returns the running site."""
    path = socket_path or DEFAULT_SOCKET_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass

    app = app or build_app(server)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.UnixSite(runner, path)
    await site.start()
    try:
        os.chmod(path, SOCKET_MODE)
    except OSError:
        logger.warning("could not chmod socket %s", path)
    logger.info("generation service listening on %s", path)
    return site

