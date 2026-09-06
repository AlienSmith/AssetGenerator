"""Offline unit tests for the guide-less (background) txt2img path.

Backgrounds are the first asset type with ``guide_mode="none"``: the payload
image is ignored, the graph runs pure txt2img (no LoadImage / ControlNet), and
the type carries its own style/negative prompt blocks because the shared
"single object on a white background" wording fights a full-canvas scene.

Everything here runs without models, a server, or disk writes (guide
persistence is monkeypatched).
"""
from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from generation import pipeline as pl
from generation import prompts as pr
from generation import service as svc
from generation.asset_types import (
    ASSET_TYPES,
    GUIDE_MODES,
    AssetType,
    resolve_asset_type,
)
from generation.guides import prepare_guide


# --- helpers ----------------------------------------------------------------

def _tiny_png_b64(size: int = 8) -> str:
    """A small real PNG so decode_image() works without fixtures on disk."""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _node_types(graph: dict) -> list[str]:
    return [node["class_type"] for node in graph.values()]


class _FakeQueue:
    def __init__(self) -> None:
        self.items: list = []

    def put(self, item) -> None:
        self.items.append(item)


class _FakeServer:
    def __init__(self) -> None:
        self.number = 0
        self.prompt_queue = _FakeQueue()


# --- registry: guide_mode + prompt overrides --------------------------------

class TestAssetTypeRegistry:
    def test_guide_modes_constant(self):
        assert GUIDE_MODES == ("edge_map", "raw", "none")

    def test_background_is_guide_less_with_overrides(self):
        bg = ASSET_TYPES["background"]
        assert bg.guide_mode == "none"
        assert bg.style is not None
        assert bg.negative is not None

    def test_object_types_default_to_edge_map_and_shared_blocks(self):
        for key in ("weapon", "prop", "armor"):
            at = ASSET_TYPES[key]
            assert at.guide_mode == "edge_map", key
            assert at.style is None, key
            assert at.negative is None, key

    def test_invalid_guide_mode_rejected_at_construction(self):
        with pytest.raises(ValueError, match="guide_mode"):
            AssetType(
                key="bogus", canvas=64, strength=0.5, guidance=3.5, steps=2,
                denoise=1.0, desc="x", guide_mode="teleport",
            )

    def test_background_keyword_resolution(self):
        assert resolve_asset_type("misty forest background").key == "background"
        assert resolve_asset_type("dungeon scene").key == "background"
        # non-background prompts must not be captured by the background keywords
        assert resolve_asset_type("iron sword").key == "weapon"
        assert resolve_asset_type("red potion").key == "prop"


# --- prompts: per-type style/negative blocks --------------------------------

class TestPromptOverrides:
    def test_background_positive_drops_white_background_wording(self):
        bg = ASSET_TYPES["background"]
        pos = pr.build_positive("misty forest at dawn", bg)
        assert pos.startswith("GRPZA")
        assert "misty forest at dawn" in pos
        assert bg.style in pos
        assert "white background" not in pos
        assert "single object" not in pos

    def test_background_negative_drops_colored_background_ban(self):
        neg = pr.build_negative(ASSET_TYPES["background"])
        assert "colored background" not in neg
        assert "solid color background" not in neg
        assert "white background" in neg  # actively pushed in the negative

    def test_prop_still_uses_shared_blocks(self):
        prop = ASSET_TYPES["prop"]
        pos = pr.build_positive("red potion", prop)
        assert pr.STYLE in pos
        neg = pr.build_negative(prop)
        assert neg.startswith(pr.NEGATIVE)
        assert prop.extra_neg in neg


# --- guides: prepare_guide dispatch -----------------------------------------

class TestPrepareGuide:
    def test_none_mode_returns_none_without_disk_io(self, monkeypatch):
        def boom(*a, **k):  # any persistence attempt fails the test
            raise AssertionError("guide-less type must not persist an image")

        monkeypatch.setattr("generation.guides.persist_guide", boom)
        assert prepare_guide(ASSET_TYPES["background"], _tiny_png_b64()) is None

    @pytest.mark.parametrize("mode,fn", [("raw", "make_guide_from_base64"),
                                         ("edge_map", "make_guide_with_detail")])
    def test_image_modes_delegate(self, monkeypatch, mode, fn):
        calls = []
        monkeypatch.setattr("generation.guides.persist_guide",
                            lambda img, **k: calls.append(img) or "gen_guide_fake.png")
        stub = AssetType(
            key="stub", canvas=64, strength=0.5, guidance=3.5, steps=2,
            denoise=1.0, desc="x", guide_mode=mode,
        )
        name = prepare_guide(stub, _tiny_png_b64(), width=64, height=64)
        assert name == "gen_guide_fake.png"
        assert len(calls) == 1

    def test_unknown_mode_raises(self):
        class _Stub:
            guide_mode = "teleport"
            key = "stub"

        with pytest.raises(ValueError, match="guide_mode"):
            prepare_guide(_Stub(), _tiny_png_b64())


# --- pipeline: graph shape ---------------------------------------------------

class TestWorkflowGraphShape:
    def test_background_graph_is_pure_txt2img(self):
        graph = pl.build_workflow(
            asset_type=ASSET_TYPES["background"],
            prompt="misty forest background",
            seed=42,
        )
        types = _node_types(graph)
        assert "LoadImage" not in types
        assert "ControlNetLoader" not in types
        assert "ControlNetApplyAdvanced" not in types
        # loaders + conditioning + latent + sampler + decode + save
        assert types.count("SaveImage") == 1

        # KSampler's positive must come straight from FluxGuidance (no CN hop)
        ksampler = next(n for n in graph.values()
                        if n["class_type"] == "KSampler")
        pos_id = str(ksampler["inputs"]["positive"][0])
        assert graph[pos_id]["class_type"] == "FluxGuidance"
        neg_id = str(ksampler["inputs"]["negative"][0])
        assert graph[neg_id]["class_type"] == "CLIPTextEncode"

    def test_guided_graph_still_has_controlnet(self):
        graph = pl.build_workflow(
            asset_type=ASSET_TYPES["prop"],
            prompt="red potion",
            seed=42,
            guide_image_name="gen_guide_fake.png",
            save_hint=False,
        )
        types = _node_types(graph)
        assert "LoadImage" in types
        assert "ControlNetLoader" in types
        assert "ControlNetApplyAdvanced" in types


# --- service: end-to-end wiring (mocked queue) -------------------------------

class TestServiceWiring:
    def _make_service(self, monkeypatch, guide_name):
        """A GenerationService with the queue/validation machinery faked out."""
        monkeypatch.setattr(svc, "prepare_guide",
                            lambda at, b64, **k: guide_name)
        monkeypatch.setattr(svc.execution, "validate_prompt",
                            lambda *a, **k: _validate_ok())
        s = svc.GenerationService(_FakeServer(), loader=object())
        return s

    def test_background_job_skips_the_guide(self, monkeypatch):
        s = self._make_service(monkeypatch, guide_name=None)
        captured = {}
        real = pl.build_workflow

        def spy(**kwargs):
            captured.update(kwargs)
            return real(**kwargs)

        monkeypatch.setattr(pl, "build_workflow", spy)
        job_id = s.start(_tiny_png_b64(), "misty forest background", 1)
        assert job_id.startswith("gen_")
        assert captured["guide_image_name"] is None
        assert captured["asset_type"].key == "background"
        assert s.jobs[job_id].guide_name is None

    def test_prop_job_still_gets_a_guide(self, monkeypatch):
        s = self._make_service(monkeypatch, guide_name="gen_guide_fake.png")
        captured = {}
        real = pl.build_workflow

        def spy(**kwargs):
            captured.update(kwargs)
            return real(**kwargs)

        monkeypatch.setattr(pl, "build_workflow", spy)
        s.start(_tiny_png_b64(), "red potion", 1)
        assert captured["guide_image_name"] == "gen_guide_fake.png"


def _validate_ok():
    """Return shape matching execution.validate_prompt: (valid, err, outputs, node_errors)."""
    return (True, None, set(), {})
