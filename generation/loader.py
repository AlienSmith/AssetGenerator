"""Model resolution + optional preload for the generation service.

Because the service lives *inside* the ComfyUI process, the GGUF/Flux/ControlNet
node classes are already registered in `nodes.NODE_CLASS_MAPPINGS` (loaded as
custom nodes). `resolve()` builds each referenced model into a normal ComfyUI
`ModelPatcher` once, keyed by file name, and `preload()` feeds the sample stack
through `comfy.model_management.load_models_gpu()` once.

We deliberately do NOT hand-roll VRAM management: `load_models_gpu()` (dynamic
VRAM mode) decides how much stays resident and what gets offloaded to CPU, which
is what fits the ~12 GB Flux unet into 16 GB. Reusing the same `ModelPatcher`
identity across variants is what avoids re-reading weights from disk on every
batch (see `comfy/model_management.py:LoadedModel`).
"""
from __future__ import annotations

import logging

import comfy.model_management
import nodes

from generation import pipeline as pl

logger = logging.getLogger(__name__)


class Loader:
    """Resolves pipeline model files into in-memory ModelPatchers once."""

    def __init__(self) -> None:
        # name -> ModelPatcher, kept alive so dynamic VRAM can reuse the loaded
        # identity across variants instead of re-reading from disk.
        self._cache: dict[str, object] = {}

    def _resolve(self, class_type: str, **inputs) -> object:
        cls = nodes.NODE_CLASS_MAPPINGS[class_type]
        node = cls()
        function = getattr(node, cls.FUNCTION)
        out = function(**inputs)
        return out[0]

    def unet(self) -> object:
        name = pl.UNET_NAME
        if name not in self._cache:
            self._cache[name] = self._resolve("UnetLoaderGGUF", unet_name=name)
        return self._cache[name]

    def clip(self) -> object:
        name = "clip_l+t5"
        if name not in self._cache:
            self._cache[name] = self._resolve(
                "DualCLIPLoaderGGUF",
                clip_name1=pl.CLIP_L_NAME,
                clip_name2=pl.T5_NAME,
                type="flux",
            )
        return self._cache[name]

    def vae(self) -> object:
        if pl.VAE_NAME not in self._cache:
            self._cache[pl.VAE_NAME] = self._resolve("VAELoader", vae_name=pl.VAE_NAME)
        return self._cache[pl.VAE_NAME]

    def lora_model(self) -> object:
        """Unet patched with the game-asset LoRA (the sampler's model)."""
        key = f"lora:{pl.LORA_NAME}"
        if key not in self._cache:
            model = self.unet()
            self._cache[key] = self._resolve(
                "LoraLoaderModelOnly",
                model=model,
                lora_name=pl.LORA_NAME,
                strength_model=0.8,
            )
        return self._cache[key]

    def controlnet(self) -> object:
        if pl.CONTROLNET_NAME not in self._cache:
            self._cache[pl.CONTROLNET_NAME] = self._resolve(
                "ControlNetLoader", control_net_name=pl.CONTROLNET_NAME
            )
        return self._cache[pl.CONTROLNET_NAME]

    def warm_on_start(self) -> None:
        """Load every model into RAM/VRAM once to validate the budget.

        Drives the whole stack (unet+lora, clip, vae, controlnet) through
        `load_models_gpu` exactly as a real variant would, then lets dynamic
        VRAM decide residency. Runs the first disk read at startup instead of
        on the first user batch. A failure here surfaces OOM / missing files
        early. Off by default (see `PRELOAD_MODELS_ON_START`).
        """
        stack = [
            self.lora_model(),
            self.clip(),
            self.vae(),
            self.controlnet(),
        ]
        logger.info("Preloading generation model stack (warm-up / budget check)")
        comfy.model_management.load_models_gpu(stack)
        comfy.model_management.soft_empty_cache()
        logger.info("Generation model stack loaded OK")


def make_loader() -> Loader:
    return Loader()