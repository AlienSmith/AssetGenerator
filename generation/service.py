"""Batch job manager for the single-layer static asset microservice.

This is the bridge between the HTTP contract HomeBot already calls and ComfyUI's
own in-process queue. A *generation job* is the unit HomeBot polls; internally it
unpacks to N *variants*, each submitted to ComfyUI's queue as its own `/prompt`
(a freshly minted `prompt_id` per variant). ComfyUI already registers produced
PNGs as assets, so this service only needs to track variant -> prompt_id and map
each variant's live state (queued/running/done/error, sampler progress) into the
HomeBot contract fields (`status`, `variant`, `progress`, `stage`).

Status lifecycle (matches HomeBot, see ``mock_generation.py``):

    pending -> running -> completed | failed | canceled

Model loading is entirely ComfyUI's job (`loader.py` resolves patchers once and
dynamic VRAM handles the ~12 GB Flux unet in 16 GB). We run variants serially
(they share one ComfyUI queue) so weights are reused across a batch.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import threading
import time
import uuid

import execution

from generation import pipeline as pl
from generation.asset_types import AssetType, resolve_asset_type
from generation.guides import make_guide_from_base64, make_guide_with_detail
from generation.loader import Loader, make_loader

logger = logging.getLogger(__name__)

# --- config (env overridable, see plan §6) --------------------------------
DEFAULT_MAX_BATCH = int(os.environ.get("GEN_MAX_BATCH_COUNT", "8"))
DEFAULT_DEFAULT_BATCH = int(os.environ.get("GEN_DEFAULT_BATCH_COUNT", "1"))
MAX_PROMPT_LEN = int(os.environ.get("GEN_MAX_PROMPT_LEN", "500"))


class GenerationService:
    """Owns generation jobs and multiplexes them over the ComfyUI queue."""

    def __init__(
        self,
        server,
        *,
        loader: Loader | None = None,
        max_batch: int = DEFAULT_MAX_BATCH,
        default_batch: int = DEFAULT_DEFAULT_BATCH,
    ) -> None:
        self._server = server
        self._loader = loader or make_loader()
        # `execution.validate_prompt` is a coroutine in current ComfyUI; since
        # `start()` runs in a worker thread (asyncio.to_thread), we capture the
        # main loop here (constructed on it in main.py) to hop validation onto.
        try:
            self._loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        self.max_batch = max_batch
        self.default_batch = default_batch
        # job_id -> JobHandle
        self.jobs: dict[str, "JobHandle"] = {}
        self._lock = threading.RLock()

    # -- contract entry points -------------------------------------------
    def start(self, image_base64: str, prompt: str, count: int | None) -> str:
        """Create a batch job, submit every variant, return the job_id."""
        asset_type = resolve_asset_type(prompt)
        count = self._clamp_count(count)
        job = self._create_job(asset_type, prompt, count)

        # Decode + persist the ControlNet reference once for the whole batch.
        # Background assets have no silhouette (pipeline skips the guide), so a
        # plain resize is enough; everything else gets the material-bearing
        # variant so the model has surface-detail cues, not a flat shape.
        if asset_type.key == "background":
            guide_name = make_guide_from_base64(
                image_base64, width=asset_type.canvas, height=asset_type.canvas
            )
        else:
            guide_name = make_guide_with_detail(
                image_base64, width=asset_type.canvas, height=asset_type.canvas
            )
        job.guide_name = guide_name
        job.stage = "queued"

        # Submit variants (synchronously; validation is the slow async part but
        # each graph is tiny). Serial submission keeps dynamic-VRAM reuse simple.
        for i in range(count):
            if job.canceled:
                break
            variant = job.variants[i]
            try:
                variant.prompt_id = self._submit_variant(asset_type, prompt, i, guide_name)
                variant.state = "queued"
            except Exception as err:  # noqa: BLE001
                logger.exception("failed to submit variant %d of job %s", i, job.job_id)
                variant.state = "error"
                job.error = str(err)

        return job.job_id

    def get(self, job_id: str) -> dict | None:
        """Return the HomeBot GET payload for a job, or None if unknown."""
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            return self._job_state(job)

    def cancel(self, job_id: str) -> bool:
        """Cancel every variant of a job. Returns False if the job is unknown."""
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                return False
            if job.canceled:
                return True
        job.canceled = True
        job.stage = "canceled"
        for variant in job.variants:
            if variant.prompt_id:
                self._cancel_prompt(variant.prompt_id)
        logger.info("generation job %s canceled", job_id)
        return True

    def health(self) -> bool:
        return self._server is not None

    def warm(self) -> None:
        """Optional startup warm-up / VRAM budget check (see PRELOAD_MODELS_ON_START)."""
        self._loader.warm_on_start()

    # -- internals ---------------------------------------------------------
    def _clamp_count(self, count: int | None) -> int:
        if count is None or count <= 0:
            return self.default_batch
        return min(count, self.max_batch)

    def _create_job(self, asset_type: AssetType, prompt: str, count: int) -> "JobHandle":
        job = JobHandle(
            job_id=f"gen_{uuid.uuid4().hex[:12]}",
            asset_type=asset_type,
            prompt=prompt,
            count=count,
        )
        with self._lock:
            self.jobs[job.job_id] = job
        return job

    def _submit_variant(
        self, asset_type: AssetType, prompt: str, index: int, guide_name: str
    ) -> str:
        prompt_id = str(uuid.uuid4())
        seed = int(time.time() * 1000) + index * 7919  # per-variant reproducible-ish seed
        graph = pl.build_workflow(
            asset_type=asset_type,
            prompt=prompt,
            seed=seed,
            guide_image_name=guide_name,
            filename_prefix=f"flux_{asset_type.key}/v{index + 1:02d}",
        )
        return self._queue_prompt(prompt_id, graph)

    def _queue_prompt(self, prompt_id: str, prompt: dict) -> str:
        server = self._server
        # Mirror server.post_prompt: validate then put onto the same in-process
        # queue, using the server's global number so ordering stays consistent
        # with any other (web) submissions.
        number = server.number
        server.number += 1

        valid, _err, outputs_to_execute, _node_errors = self._validate_prompt(prompt_id, prompt)
        if not valid:
            raise ValueError(_err)

        extra_data = {"client_id": "generation-service", "create_time": int(time.time() * 1000)}
        server.prompt_queue.put(
            (number, prompt_id, prompt, extra_data, outputs_to_execute, {})
        )
        return prompt_id

    def _validate_prompt(self, prompt_id: str, prompt: dict):
        """Await ComfyUI's async validate_prompt from this (worker) thread.

        The coroutine must run on the main event loop (it touches loop-bound
        state), so we schedule it there and block this worker thread on the
        result. Falls back to a private loop when none was captured (offline
        tests / direct construction outside asyncio).
        """
        coro = execution.validate_prompt(prompt_id, prompt, None)
        loop = self._loop
        if loop is not None and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result()
        return asyncio.run(coro)

    def _cancel_prompt(self, prompt_id: str) -> None:
        try:
            self._server.prompt_queue.interrupt_if_running(prompt_id)
            self._server.prompt_queue.delete_queue_item(lambda item: item[1] == prompt_id)
        except Exception:  # noqa: BLE001
            logger.exception("error cancelling prompt %s", prompt_id)

    def _job_state(self, job: "JobHandle") -> dict:
        history = self._server.prompt_queue.get_history()
        running_ids, queued = self._server.prompt_queue.get_current_queue_volatile()
        running_ids = {item[1] for item in running_ids}
        queued_ids = {item[1] for item in queued}
        sampler_frac = self._sampler_frac_for(job)

        done = 0
        failed = 0
        frac_sum = 0.0
        active = 0  # queued or running

        for variant in job.variants:
            pid = variant.prompt_id
            pid_hist = history.get(pid)
            if pid_hist is not None:
                status = pid_hist.get("status", {})
                status_str = status.get("status_str")
                if status_str == "success":
                    variant.state = "done"
                    done += 1
                    frac_sum += 1.0
                elif status_str == "error":
                    variant.state = "error"
                    failed += 1
                else:
                    active += 1
            elif pid in running_ids:
                variant.state = "running"
                active += 1
                frac = sampler_frac if sampler_frac is not None else 0.0
                frac_sum += frac
            elif pid in queued_ids:
                variant.state = "queued"
                active += 1
            else:
                variant.state = "pending"

        if job.canceled:
            return {
                "status": "canceled", "variant": f"{done}/{job.count}",
                "progress": done / job.count, "stage": "canceled",
                "message": "batch canceled by user",
            }

        if failed:
            return {
                "status": "failed", "variant": f"{done}/{job.count}",
                "progress": done / job.count, "stage": "failed",
                "message": job.error or "one or more variants failed",
            }

        if done == job.count:
            return {
                "status": "completed", "variant": f"{job.count}/{job.count}",
                "progress": 1.0, "stage": "done",
                "message": f"saved {job.count} image(s) to output dir",
            }

        progress = frac_sum / job.count
        current = done + (1 if active else 0)
        variant_label = f"{current}/{job.count}"
        if progress <= 0.0:
            stage = job.stage or "queued"
        else:
            stage = f"denoising variant {current}"
        return {
            "status": "running", "variant": variant_label,
            "progress": progress, "stage": stage, "message": "",
        }

    def _sampler_frac_for(self, job: "JobHandle") -> float | None:
        """Live sampler progress for the currently running variant of this job.

        ComfyUI's global progress registry tracks only the currently executing
        prompt. If it belongs to this job, we read the furthest node progress
        (the KSampler denoising loop) as a 0..1 fraction for that variant.
        """
        try:
            from comfy_execution.progress import get_progress_state

            reg = get_progress_state()
        except Exception:  # noqa: BLE001
            return None
        if reg is None or not reg.prompt_id:
            return None
        if reg.prompt_id not in {v.prompt_id for v in job.variants}:
            return None
        best = 0.0
        for state in reg.nodes.values():
            if state.get("max"):
                best = max(best, state["value"] / state["max"])
        return best if best < 1.0 else 1.0


class JobHandle:
    """One HomeBot batch: a list of variants, each tracked by a ComfyUI prompt_id."""

    def __init__(self, *, job_id: str, asset_type: AssetType, prompt: str, count: int) -> None:
        self.job_id = job_id
        self.asset_type = asset_type
        self.prompt = prompt
        self.count = count
        self.progress = 0.0
        self.stage = "queued"
        self.canceled = False
        self.error: str | None = None
        self.guide_name: str | None = None
        self.created = time.time()
        # One variant per requested image.
        self.variants: list[VariantHandle] = [VariantHandle() for _ in range(count)]


class VariantHandle:
    """Tracks one submitted workflow."""

    def __init__(self) -> None:
        self.prompt_id: str | None = None
        self.state = "pending"  # pending | queued | running | done | error


def make_generation_service(server, **kwargs) -> GenerationService:
    return GenerationService(server, **kwargs)