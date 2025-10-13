import sys
import types
import unittest
import warnings
from unittest.mock import MagicMock, patch

from neuttsair.neutts import NeuTTSAir


class DeviceSelectionTests(unittest.TestCase):
    def test_select_torch_device_auto_prefers_cuda(self):
        with patch("torch.cuda.is_available", return_value=True), patch(
            "neuttsair.neutts.NeuTTSAir._is_mps_available", return_value=False
        ):
            self.assertEqual(NeuTTSAir._select_torch_device("auto"), "cuda")

    def test_select_torch_device_auto_prefers_mps_when_cuda_missing(self):
        with patch("torch.cuda.is_available", return_value=False), patch(
            "neuttsair.neutts.NeuTTSAir._is_mps_available", return_value=True
        ):
            self.assertEqual(NeuTTSAir._select_torch_device("auto"), "mps")

    def test_select_torch_device_auto_cpu_fallback(self):
        with patch("torch.cuda.is_available", return_value=False), patch(
            "neuttsair.neutts.NeuTTSAir._is_mps_available", return_value=False
        ):
            self.assertEqual(NeuTTSAir._select_torch_device("auto"), "cpu")

    def test_select_torch_device_cuda_falls_back_when_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False), patch(
            "neuttsair.neutts.NeuTTSAir._is_mps_available", return_value=False
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = NeuTTSAir._select_torch_device("cuda")
        self.assertEqual(result, "cpu")
        self.assertTrue(any("CUDA" in str(warning.message) for warning in caught))

    def test_select_backbone_device_for_gguf_prefers_gpu(self):
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(
                NeuTTSAir._select_backbone_device("model.gguf", "auto"), "gpu"
            )

    def test_select_backbone_device_for_gguf_cpu_fallback(self):
        with patch("torch.cuda.is_available", return_value=False):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = NeuTTSAir._select_backbone_device("model.gguf", "auto")
        self.assertEqual(result, "cpu")
        self.assertTrue(any("GPU-backed GGUF" in str(warning.message) for warning in caught))

    def test_configure_onnx_codec_prefers_cuda_provider(self):
        dummy_session = MagicMock()
        instance = object.__new__(NeuTTSAir)
        instance.codec = types.SimpleNamespace(session=dummy_session)

        module = types.ModuleType("onnxruntime")
        module.get_available_providers = lambda: [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        with patch.dict(sys.modules, {"onnxruntime": module}):
            instance._configure_onnx_codec_session("cuda")

        dummy_session.set_providers.assert_called_once_with(
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            [{}, {}],
        )

    def test_configure_onnx_codec_falls_back_when_cuda_missing(self):
        dummy_session = MagicMock()
        instance = object.__new__(NeuTTSAir)
        instance.codec = types.SimpleNamespace(session=dummy_session)

        module = types.ModuleType("onnxruntime")
        module.get_available_providers = lambda: ["CPUExecutionProvider"]

        with patch.dict(sys.modules, {"onnxruntime": module}):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                instance._configure_onnx_codec_session("cuda")

        dummy_session.set_providers.assert_called_once_with(
            ["CPUExecutionProvider"],
            [{}],
        )
        self.assertTrue(any("falling back" in str(warning.message).lower() for warning in caught))


if __name__ == "__main__":
    unittest.main()
