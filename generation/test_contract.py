"""Contract test for the generation microservice.

Exercises the exact HTTP-over-Unix-socket contract HomeBot relies on
(``HomeBot/bot/gen_client.py``) against a live ComfyUI process started with
``--generation-socket``, mirroring ``HomeBot/dev/test_contract.py``.

The test talks to an already-running ComfyUI instance; it does NOT start one
(that needs models/VRAM). Point it at the socket with ``GEN_SOCKET_PATH`` or the
first ``argv``. It is intentionally the same shape as HomeBot's dev contract
test so a single runner can verify both the mock and the real service.

Usage:
    python generation/test_contract.py [/run/feishu-bot/gen.sock]
"""
from __future__ import annotations

import asyncio
import base64
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOMEBOT = os.path.dirname(ROOT)  # .../AssetGeneration, HomeBot/ sits next to ComfyUI_t/
sys.path.insert(0, os.path.join(HOMEBOT, "HomeBot"))
from bot.gen_client import GenClient  # noqa: E402

SOCK = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "GEN_SOCKET_PATH", "/run/feishu-bot/gen.sock"
)

# A tiny valid PNG so `guides.decode_image` doesn't blow up on b64 decode.
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "YPhfDwAChQGA60r6kgAAAABJRU5ErkJggg=="
)

PASS = []


def report(name: str) -> None:
    PASS.append(name)
    print(f"PASS {name}")


async def main() -> None:
    gen = GenClient(socket_path=SOCK, poll_interval=0.5)

    try:
        ok = await gen.health()
    except Exception as exc:  # noqa: BLE001
        print(f"SKIP: generation service not reachable on {SOCK}: {exc}")
        return

    assert ok is True, "health endpoint should return ok"
    report("health")

    # start
    jid = await gen.start(_TINY_PNG, "iron broadsword", 2)
    assert jid and jid.startswith("gen_"), jid
    report(f"start -> {jid}")

    # initial get should be pending/running (never 404 for a known id)
    first = await gen.get(jid)
    assert first.status in ("pending", "running", "completed", "failed")
    report(f"get initial -> {first.status} variant={first.variant}")

    # poll to a terminal state (completed or failed); bounded so CI doesn't hang
    last = first
    for _ in range(60):
        last = await gen.get(jid)
        if last.status in ("completed", "failed", "canceled"):
            break
        await asyncio.sleep(1.0)
    print(f"terminal -> status={last.status} variant={last.variant} progress={last.progress}")
    assert last.status in ("completed", "failed"), last.status
    assert last.variant.startswith("/") is False
    report("poll to terminal")

    # cancel a fresh batch mid-run (bounded: we cancel immediately)
    jid2 = await gen.start(_TINY_PNG, "brass key", 4)
    assert await gen.cancel(jid2) is True
    job2 = await gen.get(jid2)
    if job2.status != "canceled":
        # cancel raced ahead of submit; still acceptable if terminal
        assert job2.status in ("canceled", "completed", "failed")
    report(f"cancel -> {job2.status}")

    # 404 handling
    try:
        await gen.get("definitely-not-a-job")
        raise SystemExit("expected GenServiceError for missing job")
    except Exception as exc:  # noqa: BLE001
        report(f"404 -> {type(exc).__name__}")

    await gen.close()
    print(f"CONTRACT TEST OK ({len(PASS)} checks)")


if __name__ == "__main__":
    asyncio.run(main())