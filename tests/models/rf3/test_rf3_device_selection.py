"""Tests for RF3Wrapper device selection (issue #37).

Fabric's ``devices`` argument treats an ``int`` as "take this many GPUs, starting
from index 0", which pins every worker to ``cuda:0`` and serialises nominally
parallel jobs. The wrapper must accept an explicit device and bind Fabric to
that specific GPU index via ``devices_per_node=[idx]``.
"""

from pathlib import Path

import pytest
import torch
from sampleworks.utils.imports import require_rf3, RF3_AVAILABLE


pytestmark = pytest.mark.skipif(not RF3_AVAILABLE, reason="RF3 dependencies not installed")

if RF3_AVAILABLE:
    import sampleworks.models.rf3.wrapper as rf3_wrapper_module
    from sampleworks.models.rf3.wrapper import _cuda_index, RF3Wrapper


class TestCudaIndex:
    """Validate CUDA index extraction used to drive Fabric device selection."""

    @pytest.mark.parametrize(
        ("device", "expected"),
        [
            ("cuda:0", 0),
            ("cuda:3", 3),
            (torch.device("cuda", 5), 5),
            ("cuda", 0),
        ],
    )
    def test_returns_index(self, device, expected):
        assert _cuda_index(device) == expected

    @pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
    def test_rejects_non_cuda(self, device):
        with pytest.raises(ValueError, match="CUDA device"):
            _cuda_index(device)


class TestRF3ModelReuse:
    """Verify model reuse preserves the complete initialized RF3 runtime."""

    @pytest.fixture
    def stub_engine(self, monkeypatch):
        """Replace RF3InferenceEngine with a lightweight initialized runtime."""

        class StubFabric:
            """Provide the Fabric attributes RF3Wrapper reads."""

            device = torch.device("cuda:0")

        class StubTrainer:
            """Provide initialized trainer state with a model."""

            def __init__(self):
                self.fabric = StubFabric()
                self.state = {"model": torch.nn.Linear(1, 1)}

        class StubInferenceEngine:
            """Track construction while avoiding checkpoint and GPU work."""

            instances = 0

            def __init__(self, **kwargs):
                type(self).instances += 1
                self.ckpt_path = Path(kwargs["ckpt_path"]).resolve()
                self.trainer = StubTrainer()

            def initialize(self):
                """Match the real inference engine initialization interface."""

        monkeypatch.setattr(rf3_wrapper_module, "RF3InferenceEngine", StubInferenceEngine)
        return StubInferenceEngine

    def test_reuses_engine_trainer_and_model(self, tmp_path, stub_engine):
        """A reused model must not initialize a second RF3 inference engine."""
        checkpoint = tmp_path / "rf3.ckpt"
        original = RF3Wrapper(checkpoint_path=checkpoint, device="cuda:0")

        reused = RF3Wrapper(
            checkpoint_path=checkpoint,
            device="cuda:0",
            model=original.model,
        )

        assert stub_engine.instances == 1
        assert reused.inference_engine is original.inference_engine
        assert reused.inference_engine.trainer is original.inference_engine.trainer
        assert reused.model is original.model

    def test_rejects_model_without_inference_context(self, tmp_path, stub_engine):
        """A bare RF3 module cannot be safely attached to a new Fabric runtime."""
        with pytest.raises(ValueError, match="model created by RF3Wrapper"):
            RF3Wrapper(
                checkpoint_path=tmp_path / "rf3.ckpt",
                device="cuda:0",
                model=torch.nn.Linear(1, 1),
            )

        assert stub_engine.instances == 0

    def test_rejects_reuse_on_different_device(self, tmp_path, stub_engine):
        """An initialized RF3 runtime cannot be silently moved between GPUs."""
        checkpoint = tmp_path / "rf3.ckpt"
        original = RF3Wrapper(checkpoint_path=checkpoint, device="cuda:0")

        with pytest.raises(ValueError, match="not requested device"):
            RF3Wrapper(
                checkpoint_path=checkpoint,
                device="cuda:1",
                model=original.model,
            )

    def test_rejects_model_spoofing_engine_context(self, tmp_path, stub_engine):
        """The supplied model must be the model owned by the attached trainer."""
        checkpoint = tmp_path / "rf3.ckpt"
        original = RF3Wrapper(checkpoint_path=checkpoint, device="cuda:0")
        other_model = torch.nn.Linear(1, 1)
        object.__setattr__(
            other_model,
            rf3_wrapper_module._INFERENCE_ENGINE_ATTR,
            original.inference_engine,
        )

        with pytest.raises(ValueError, match="does not belong"):
            RF3Wrapper(
                checkpoint_path=checkpoint,
                device="cuda:0",
                model=other_model,
            )


@pytest.mark.gpu
@pytest.mark.slow
class TestRF3WrapperDeviceBinding:
    """Regression for issue #37: device must propagate to Fabric.

    Prior behaviour: RF3Wrapper ignored caller-specified device and always
    landed on ``cuda:0`` because Fabric defaults to ``devices=1``.
    """

    @require_rf3()
    def test_wrapper_honors_requested_cuda_index(self, rf3_checkpoint_path):
        if torch.cuda.device_count() < 2:
            pytest.skip("multi-GPU regression test needs >= 2 CUDA devices")

        wrapper = RF3Wrapper(checkpoint_path=rf3_checkpoint_path, device="cuda:1")
        assert wrapper.device == torch.device("cuda:1")
