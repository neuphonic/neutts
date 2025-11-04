from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from neuttsair.benchmark import (
    BenchmarkResult,
    BenchmarkSample,
    benchmark_codec_device,
    candidate_codec_devices,
    iter_benchmarks,
    summarise_metrics,
    _format_providers,
    _format_system_metadata,
    _render_table,
)


_MB = 1024 * 1024


class _DummyProcess:
    def __init__(self, rss_values: list[int]):
        self._values = rss_values

    def memory_info(self):
        value = self._values.pop(0)
        return SimpleNamespace(rss=value)


class _DummySession:
    def get_providers(self):
        return ["MockProvider", "CPUExecutionProvider"]


class _DummyTTS:
    sample_rate = 24_000

    def __init__(self, *args, **kwargs):
        self.codec = SimpleNamespace(session=_DummySession())

    def infer(self, *_args, **_kwargs):
        return np.zeros(24_000, dtype=np.float32)


class BenchmarkModuleTests(unittest.TestCase):
    def test_candidate_codec_devices_defaults_to_cpu_when_missing(self):
        module = SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
        devices = candidate_codec_devices(ort_module=module)
        self.assertEqual(devices, ["auto", "cpu"])

    def test_candidate_codec_devices_includes_cuda_variants(self):
        module = SimpleNamespace(
            get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        with patch("neuttsair.benchmark.torch.cuda.device_count", return_value=2):
            devices = candidate_codec_devices(ort_module=module)
        self.assertIn("cuda", devices)
        self.assertIn("cuda:0", devices)
        self.assertIn("cuda:1", devices)
        self.assertEqual(devices[-1], "cpu")

    def test_candidate_codec_devices_respects_explicit_list(self):
        devices = candidate_codec_devices(["cuda", "cpu", "cuda"])
        self.assertEqual(devices, ["cuda", "cpu"])

    def test_benchmark_codec_device_collects_metrics(self):
        sample = BenchmarkSample(
            input_text="Hello",
            ref_codes=np.zeros((10,), dtype=np.int32),
            ref_text="Hello",
        )

        rss_values = [100 * _MB, 120 * _MB, 120 * _MB, 120 * _MB]
        process = _DummyProcess(rss_values)

        with patch("neuttsair.benchmark.torch.cuda.is_available", return_value=False), patch(
            "neuttsair.benchmark.time.perf_counter",
            side_effect=[0.0, 0.1, 0.1, 0.4],
        ):
            result = benchmark_codec_device(
                "cpu",
                sample=sample,
                tts_factory=_DummyTTS,
                process=process,
            )

        self.assertAlmostEqual(result.load_s, 0.1, places=6)
        self.assertAlmostEqual(result.inference_s, 0.3, places=6)
        self.assertAlmostEqual(result.total_s, 0.4, places=6)
        self.assertAlmostEqual(result.audio_seconds, 1.0, places=6)
        self.assertAlmostEqual(result.realtime_factor, 0.3, places=6)
        self.assertAlmostEqual(result.ram_mb, 20.0, places=6)
        self.assertIsNone(result.vram_mb)
        self.assertEqual(result.providers, ["MockProvider", "CPUExecutionProvider"])

    def test_summarise_metrics(self):
        results = [
            BenchmarkResult(
                codec_device="cpu",
                providers=None,
                load_s=1.0,
                inference_s=2.0,
                total_s=3.0,
                audio_seconds=1.0,
                realtime_factor=2.0,
                ram_mb=100.0,
                vram_mb=None,
            ),
            BenchmarkResult(
                codec_device="cpu",
                providers=None,
                load_s=3.0,
                inference_s=4.0,
                total_s=7.0,
                audio_seconds=1.0,
                realtime_factor=4.0,
                ram_mb=300.0,
                vram_mb=None,
            ),
        ]
        summary = summarise_metrics(results)
        self.assertAlmostEqual(summary["inference_s_mean"], 3.0)
        self.assertAlmostEqual(summary["inference_s_std"], 1.0)
        self.assertAlmostEqual(summary["ram_mb_mean"], 200.0)
        self.assertIn("total_s_mean", summary)

    def test_format_providers_wraps_long_lists(self):
        short = _format_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(short, "CUDAExecutionProvider, CPUExecutionProvider")

        long = _format_providers([
            "CUDAExecutionProvider",
            "ROCMExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ])
        self.assertTrue(long.startswith("- "))
        self.assertIn("\n- ROCMExecutionProvider", long)

    def test_render_table_handles_multiline_cells(self):
        headers = ["A", "B"]
        rows = [["row1", "line1\nline2"], ["row2", "single"]]
        table = _render_table(headers, rows)
        lines = table.splitlines()
        self.assertIn("A", lines[0])
        self.assertIn("line2", table)

    def test_format_system_metadata_handles_missing_info(self):
        metadata = {
            "os": "TestOS",
            "cpu": "UnitCPU",
            "cpu_counts": "2 physical / 4 logical",
            "ram": None,
            "gpus": [],
        }
        lines = _format_system_metadata(metadata)
        self.assertIn("OS: TestOS", lines[0])
        self.assertTrue(any(line.startswith("CPU:") for line in lines))
        self.assertIn("GPUs: none detected", lines)

    def test_iter_benchmarks_reuses_models(self):
        class CountingTTS(_DummyTTS):
            instances = 0

            def __init__(self, *args, **kwargs):
                CountingTTS.instances += 1
                super().__init__(*args, **kwargs)

        sample = BenchmarkSample(
            input_text="Hello",
            ref_codes=np.zeros((10,), dtype=np.int32),
            ref_text="Hello",
        )

        rss_values = [100 * _MB] * 8
        process = _DummyProcess(rss_values)

        with patch("neuttsair.benchmark.torch.cuda.is_available", return_value=False):
            results = iter_benchmarks(
                ["cpu"],
                sample=sample,
                runs=2,
                warmup_runs=1,
                reuse_models=True,
                tts_factory=CountingTTS,
                process=process,
            )

        self.assertEqual(CountingTTS.instances, 1)
        self.assertEqual(len(results["cpu"]), 2)

    def test_iter_benchmarks_reuses_models(self):
        class CountingTTS(_DummyTTS):
            instances = 0

            def __init__(self, *args, **kwargs):
                CountingTTS.instances += 1
                super().__init__(*args, **kwargs)

        sample = BenchmarkSample(
            input_text="Hello",
            ref_codes=np.zeros((10,), dtype=np.int32),
            ref_text="Hello",
        )

        rss_values = [100 * _MB] * 8
        process = _DummyProcess(rss_values)

        with patch("neuttsair.benchmark.torch.cuda.is_available", return_value=False):
            results = iter_benchmarks(
                ["cpu"],
                sample=sample,
                runs=2,
                warmup_runs=1,
                reuse_models=True,
                tts_factory=CountingTTS,
                process=process,
            )

        self.assertEqual(CountingTTS.instances, 1)
        self.assertEqual(len(results["cpu"]), 2)


if __name__ == "__main__":
    unittest.main()
