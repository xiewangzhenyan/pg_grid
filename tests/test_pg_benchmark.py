"""PG-Benchmark 测试：可复现场景、扰动评估回路与 A/B 对比。"""

import json
from pathlib import Path

import numpy as np


def test_scene_generation_is_reproducible():
    """同一 seed 必须生成逐字节一致的场景；不同 seed 必须不同。"""
    from pg_benchmark import make_scene

    image_a, points_a, values_a = make_scene(grid_size=10, seed=7)
    image_b, points_b, values_b = make_scene(grid_size=10, seed=7)
    image_c, _, _ = make_scene(grid_size=10, seed=8)

    assert np.array_equal(image_a, image_b)
    assert np.array_equal(points_a, points_b)
    assert np.array_equal(values_a, values_b)
    assert not np.array_equal(image_a, image_c)
    assert points_a.shape == (100, 2)
    assert values_a.shape == (100,)


def test_clean_scene_case_localizes_and_quantifies(tmp_path):
    """无扰动场景必须定位准确（<5% pitch）且定量保序（Spearman>0.9）。"""
    from pg_benchmark import make_scene, run_case

    image, points, values = make_scene(grid_size=10, seed=3)
    metrics = run_case(image, points, values, grid_size=10, work_dir=tmp_path)

    assert metrics["loc_mean_pct_pitch"] < 5.0
    assert metrics["quant_spearman_all"] > 0.9
    assert metrics["quant_reliable_ratio"] > 0.9
    assert metrics["lattice_trusted"] is True


def test_perturbation_moves_truth_consistently():
    """几何扰动必须同步变换真值点位，光度扰动不得移动真值。"""
    from pg_benchmark import apply_perturbation, make_scene

    image, points, _ = make_scene(grid_size=10, seed=1)
    rng = np.random.default_rng(1)

    rotated_image, rotated_points = apply_perturbation(image, points, "rotation", 10.0, rng)
    assert rotated_image.shape == image.shape
    # 旋转 10 度后点位必须移动，且点间距保持不变（刚体变换）。
    assert float(np.abs(rotated_points - points).max()) > 10.0
    dist_before = np.linalg.norm(points[1] - points[0])
    dist_after = np.linalg.norm(rotated_points[1] - rotated_points[0])
    assert abs(dist_before - dist_after) < 0.5

    rng2 = np.random.default_rng(1)
    blurred_image, blurred_points = apply_perturbation(image, points, "blur", 3.0, rng2)
    assert np.allclose(blurred_points, points)
    assert not np.array_equal(blurred_image, image)


def test_benchmark_run_writes_report_and_csv(tmp_path):
    """基准运行必须写出 benchmark_report.json 与 csv，且包含完整指标。"""
    from pg_benchmark import BenchmarkConfig, run_benchmark

    config = BenchmarkConfig(grid_size=10, seeds=(0,), families={"rotation": [0.0, 6.0]})
    report = run_benchmark(config, output_dir=tmp_path)

    assert (tmp_path / "benchmark_report.json").exists()
    assert (tmp_path / "benchmark_report.csv").exists()
    assert report["schema"] == "pg-benchmark-v1"
    assert len(report["cases"]) == 2

    required = {
        "family", "level", "seed", "grid_size",
        "loc_mean_px", "loc_p90_px", "loc_mean_pct_pitch",
        "chip_method", "lattice_trusted", "support_ratio",
        "quant_reliable_ratio", "quant_spearman_all", "quality_status",
    }
    for case in report["cases"]:
        assert required.issubset(case.keys())
        assert np.isfinite(float(case["loc_mean_pct_pitch"]))

    # summary 按扰动族/强度聚合。
    assert "rotation" in report["summary"]
    assert len(report["summary"]["rotation"]) == 2

    with (tmp_path / "benchmark_report.json").open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert len(loaded["cases"]) == 2


def test_compare_benchmarks_reports_deltas_and_regressions():
    """A/B 对比必须逐 case 对齐、给出指标差值并识别回归。"""
    from pg_benchmark import compare_benchmarks

    def _case(family, level, loc, spearman):
        return {
            "family": family, "level": level, "seed": 0, "grid_size": 10,
            "loc_mean_pct_pitch": loc, "quant_spearman_all": spearman,
            "quant_reliable_ratio": 1.0,
        }

    report_a = {"cases": [_case("rotation", 0.0, 0.5, 0.99), _case("rotation", 6.0, 1.0, 0.98)]}
    report_b = {"cases": [_case("rotation", 0.0, 0.4, 0.99), _case("rotation", 6.0, 3.0, 0.80)]}

    comparison = compare_benchmarks(report_a, report_b)

    assert comparison["common_case_count"] == 2
    rotation_delta = comparison["per_family"]["rotation"]["loc_mean_pct_pitch_delta"]
    assert rotation_delta > 0.0  # B 整体劣化
    # level 6 的定位与保序双双回归，必须被点名。
    assert any(r["level"] == 6.0 for r in comparison["regressions"])
    # level 0 有改善，不应出现在回归清单。
    assert not any(r["level"] == 0.0 for r in comparison["regressions"])
