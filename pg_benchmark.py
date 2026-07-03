"""PG-Benchmark：合成扰动基准与 A/B 对比。

设计目标（对应设计文档 §8）：
- 场景由 seed 完全决定，任意版本在同一 seed 下评估同一批图像，
  因此跨版本对比只需各自跑一遍基准、再对齐 case 比较指标；
- 七个扰动族（旋转/透视/模糊/光照梯度/遮挡/高光/噪声）各带强度阶梯，
  输出随强度的退化曲线数据，而不是单点指标；
- 每个 case 走完整 process_image 管线，同时评估定位（px 与 %pitch）、
  定量（真值梯度的 Spearman 保序性）、可靠率与兜底遥测；
- 大图不入库：报告只含指标，图像按需再生成。

A/B 工作流：
  git checkout <旧版本> && python pg_benchmark_demo.py --output reports/a
  git checkout <新版本> && python pg_benchmark_demo.py --output reports/b
  python pg_benchmark_demo.py --compare reports/a/benchmark_report.json reports/b/benchmark_report.json
"""

from __future__ import annotations

import csv
import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pg_grid import process_image, write_image_unicode


# 各扰动族的默认强度阶梯。级别 0 恒为"无扰动"，保证每族都有干净基线。
DEFAULT_FAMILIES: dict[str, list[float]] = {
    "rotation": [0.0, 3.0, 6.0, 10.0],        # 整图旋转角（度）
    "perspective": [0.0, 0.02, 0.05, 0.08],   # 角点位移占图像边长比例
    "blur": [0.0, 1.5, 3.0, 5.0],             # 高斯模糊 sigma
    "illumination": [0.0, 0.2, 0.4, 0.6],     # 乘性光照梯度幅度
    "occlusion": [0.0, 3.0, 8.0, 15.0],       # 被异物覆盖的单元数
    "glare": [0.0, 2.0, 5.0],                 # 高光斑数量
    "noise": [0.0, 4.0, 8.0, 14.0],           # 高斯噪声 sigma
}


@dataclass
class BenchmarkConfig:
    """基准配置。families 为 None 时使用 DEFAULT_FAMILIES 全集。"""

    grid_size: int = 10
    seeds: tuple[int, ...] = (0, 1)
    families: dict[str, list[float]] | None = None

    def resolved_families(self) -> dict[str, list[float]]:
        return dict(self.families) if self.families is not None else dict(DEFAULT_FAMILIES)


def make_scene(grid_size: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成可播种的基准场景。

    暗背景 + 亮面板 + 规则阵列；面板位置随 seed 抖动。
    每个单元的光度真值取自线性梯度序列（40..110），用于评估
    定量读数的保序性（Spearman）。
    返回 (图像, 单元中心真值[原图坐标 N×2], 单元光度真值 N)。
    """

    rng = np.random.default_rng(seed)
    size = 1200
    unit_count = grid_size * grid_size
    truth_values = np.linspace(40.0, 110.0, unit_count)

    image = np.full((size, size, 3), 8, dtype=np.uint8)
    plate_side = 280
    plate_x = 460 + int(rng.integers(-20, 21))
    plate_y = 430 + int(rng.integers(-20, 21))

    if grid_size >= 15:
        # 亮点阵列：偏暗面板 + 亮点（120 + delta）。
        plate_gray = 120
        cv2.rectangle(image, (plate_x, plate_y), (plate_x + plate_side, plate_y + plate_side),
                      (plate_gray, plate_gray, plate_gray), -1)
        margin = 30.0
        pitch = (plate_side - 2.0 * margin) / (grid_size - 1)
        centers = []
        for row in range(grid_size):
            for col in range(grid_size):
                cx = int(round(plate_x + margin + col * pitch))
                cy = int(round(plate_y + margin + row * pitch))
                value = int(np.clip(plate_gray + truth_values[row * grid_size + col], 0, 255))
                cv2.circle(image, (cx, cy), 3, (value, value, value), -1)
                centers.append((float(cx), float(cy)))
    else:
        # 暗方块阵列：亮面板 + 暗方块（205 - delta）。
        plate_gray = 205
        cv2.rectangle(image, (plate_x, plate_y), (plate_x + plate_side, plate_y + plate_side),
                      (plate_gray, plate_gray, plate_gray), -1)
        margin = 32.0
        pitch = (plate_side - 2.0 * margin) / (grid_size - 1)
        square = 9
        centers = []
        for row in range(grid_size):
            for col in range(grid_size):
                cx = plate_x + margin + col * pitch
                cy = plate_y + margin + row * pitch
                x0 = int(round(cx - square / 2))
                y0 = int(round(cy - square / 2))
                value = int(np.clip(plate_gray - truth_values[row * grid_size + col], 0, 255))
                image[y0:y0 + square, x0:x0 + square] = value
                centers.append((x0 + (square - 1) / 2.0, y0 + (square - 1) / 2.0))

    return image, np.asarray(centers, dtype=np.float32), truth_values.astype(np.float64)


def apply_perturbation(
    image: np.ndarray,
    truth_points: np.ndarray,
    family: str,
    level: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """对场景施加单一扰动。几何扰动会同步变换真值点位。

    level 为 0 时恒等返回（各族的干净基线）。
    """

    if level <= 0:
        return image.copy(), truth_points.copy()

    height, width = image.shape[:2]
    if family == "rotation":
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), float(level), 1.0)
        warped = cv2.warpAffine(image, matrix, (width, height), borderValue=(8, 8, 8))
        homogeneous = np.column_stack([truth_points, np.ones(truth_points.shape[0])])
        moved = (homogeneous @ matrix.T).astype(np.float32)
        return warped, moved

    if family == "perspective":
        shift = float(level) * min(width, height)
        src = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        dst = src + rng.uniform(-shift, shift, size=(4, 2)).astype(np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, matrix, (width, height), borderValue=(8, 8, 8))
        moved = cv2.perspectiveTransform(truth_points.reshape(-1, 1, 2).astype(np.float32), matrix).reshape(-1, 2)
        return warped, moved

    if family == "blur":
        ksize = int(round(float(level) * 4)) | 1
        return cv2.GaussianBlur(image, (ksize, ksize), float(level)), truth_points.copy()

    if family == "illumination":
        ramp = np.linspace(1.0 - level / 2.0, 1.0 + level / 2.0, width, dtype=np.float64)
        adjusted = np.clip(image.astype(np.float64) * ramp[None, :, None], 0, 255).astype(np.uint8)
        return adjusted, truth_points.copy()

    if family == "occlusion":
        occluded = image.copy()
        count = min(int(level), truth_points.shape[0])
        chosen = rng.choice(truth_points.shape[0], size=count, replace=False)
        # 遮挡半径按单元间距估计，覆盖单元本体及少量邻域。
        pitch = float(np.median(np.abs(np.diff(np.sort(np.unique(truth_points[:, 0]))))))
        radius = max(6, int(round(pitch * 0.7)))
        for index in chosen:
            cx, cy = truth_points[index]
            cv2.circle(occluded, (int(cx), int(cy)), radius, (60, 60, 60), -1)
        return occluded, truth_points.copy()

    if family == "glare":
        glared = image.copy()
        xs = truth_points[:, 0]
        ys = truth_points[:, 1]
        for _ in range(int(level)):
            gx = float(rng.uniform(xs.min(), xs.max()))
            gy = float(rng.uniform(ys.min(), ys.max()))
            cv2.circle(glared, (int(gx), int(gy)), int(rng.integers(14, 26)), (255, 255, 255), -1)
        return glared, truth_points.copy()

    if family == "noise":
        noisy = np.clip(
            image.astype(np.float64) + rng.normal(0.0, float(level), size=image.shape), 0, 255
        ).astype(np.uint8)
        return noisy, truth_points.copy()

    raise ValueError(f"未知扰动族：{family}")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman 秩相关（无 scipy 依赖）。样本退化时返回 0。"""

    if a.size < 3 or b.size < 3 or a.size != b.size:
        return 0.0
    rank_a = np.argsort(np.argsort(a)).astype(np.float64)
    rank_b = np.argsort(np.argsort(b)).astype(np.float64)
    std_a, std_b = rank_a.std(), rank_b.std()
    if std_a < 1e-9 or std_b < 1e-9:
        return 0.0
    return float(np.mean((rank_a - rank_a.mean()) * (rank_b - rank_b.mean())) / (std_a * std_b))


def _pitch_from_truth(truth_rect: np.ndarray, grid_size: int) -> float:
    """从矫正坐标系真值估计间距（行内/列内相邻点欧氏距离中位数）。"""

    lattice = truth_rect.reshape(grid_size, grid_size, 2)
    row_steps = np.linalg.norm(np.diff(lattice, axis=1), axis=2)
    col_steps = np.linalg.norm(np.diff(lattice, axis=0), axis=2)
    return (float(np.median(row_steps)) + float(np.median(col_steps))) / 2.0


def run_case(
    image: np.ndarray,
    truth_points: np.ndarray,
    truth_values: np.ndarray,
    grid_size: int,
    work_dir: str | Path,
) -> dict[str, object]:
    """对单张扰动图跑完整管线并计算全部指标。"""

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    image_path = work_dir / "scene.png"
    write_image_unicode(image_path, image)

    result = process_image(image_path=image_path, grid_size=grid_size, output_dir=work_dir / "out")

    # 真值映射到矫正坐标系（用管线自己的四角单应，保证坐标系一致）。
    region = np.asarray(result["chip_region"]["points"], dtype=np.float32)
    rect_size = int(result["rectified_size"])
    dst = np.array(
        [[0, 0], [rect_size - 1, 0], [rect_size - 1, rect_size - 1], [0, rect_size - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(region, dst)
    truth_rect = cv2.perspectiveTransform(truth_points.reshape(-1, 1, 2).astype(np.float32), matrix).reshape(-1, 2)

    predicted = np.asarray([[p["x"], p["y"]] for p in result["grid_points"]], dtype=np.float64)
    errors = np.linalg.norm(predicted - truth_rect.astype(np.float64), axis=1)
    pitch = max(_pitch_from_truth(truth_rect, grid_size), 1e-6)

    # 定量保序性：10x10 暗单元的信号为负，取反后与真值同向。
    with (work_dir / "out" / "quant_result.json").open("r", encoding="utf-8") as f:
        quant = json.load(f)
    polarity = -1.0 if grid_size < 15 else 1.0
    measured = np.asarray([polarity * float(u["corr_signal_gray"]) for u in quant["units"]], dtype=np.float64)
    reliable_mask = np.asarray([bool(u["quant_reliable"]) for u in quant["units"]], dtype=bool)
    spearman_all = _spearman(truth_values, measured)
    spearman_reliable = (
        _spearman(truth_values[reliable_mask], measured[reliable_mask]) if int(reliable_mask.sum()) >= 3 else 0.0
    )

    lattice = result["lattice_consistency"]
    support = lattice.get("candidate_support_ratio")
    return {
        "grid_size": int(grid_size),
        "loc_mean_px": round(float(errors.mean()), 4),
        "loc_p90_px": round(float(np.percentile(errors, 90)), 4),
        "loc_max_px": round(float(errors.max()), 4),
        "loc_mean_pct_pitch": round(float(errors.mean() / pitch * 100.0), 4),
        "loc_p90_pct_pitch": round(float(np.percentile(errors, 90) / pitch * 100.0), 4),
        "chip_method": str(result["chip_region"]["method"]),
        "lattice_applied": bool(lattice.get("applied", False)),
        "lattice_trusted": bool(lattice.get("trusted", False)),
        "support_ratio": None if support is None else round(float(support), 4),
        "imputed_count": int(lattice.get("outlier_count", 0)),
        "quality_status": str(result["quality"]["status"]),
        "quant_reliable_ratio": round(float(reliable_mask.mean()), 4),
        "quant_spearman_all": round(spearman_all, 4),
        "quant_spearman_reliable": round(spearman_reliable, 4),
    }


def run_benchmark(config: BenchmarkConfig, output_dir: str | Path) -> dict[str, object]:
    """按配置跑完整基准，写出 benchmark_report.json / benchmark_report.csv。"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    families = config.resolved_families()

    cases: list[dict[str, object]] = []
    for family, levels in families.items():
        for level in levels:
            for seed in config.seeds:
                image, truth_points, truth_values = make_scene(config.grid_size, seed)
                # 扰动随机源由 (seed, family, level) 决定，保证逐 case 可复现。
                # 注意必须用确定性哈希：Python 内建 hash() 对字符串跨进程随机化，
                # 会破坏 A/B 工作流中两次独立运行的场景一致性。
                family_code = zlib.crc32(family.encode("utf-8")) & 0xFFFF
                rng = np.random.default_rng([seed, family_code, int(level * 100)])
                perturbed, moved_truth = apply_perturbation(image, truth_points, family, float(level), rng)
                work_dir = output_dir / "work" / f"{family}_{level}_{seed}"
                metrics = run_case(perturbed, moved_truth, truth_values, config.grid_size, work_dir)
                metrics.update({"family": family, "level": float(level), "seed": int(seed)})
                cases.append(metrics)

    # 按 扰动族 -> 强度 聚合（跨 seed 取均值），即退化曲线数据。
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for family, levels in families.items():
        summary[family] = {}
        for level in levels:
            subset = [c for c in cases if c["family"] == family and c["level"] == float(level)]
            summary[family][str(level)] = {
                "loc_mean_pct_pitch": round(float(np.mean([c["loc_mean_pct_pitch"] for c in subset])), 4),
                "loc_p90_pct_pitch": round(float(np.mean([c["loc_p90_pct_pitch"] for c in subset])), 4),
                "quant_spearman_all": round(float(np.mean([c["quant_spearman_all"] for c in subset])), 4),
                "quant_reliable_ratio": round(float(np.mean([c["quant_reliable_ratio"] for c in subset])), 4),
                "trusted_ratio": round(float(np.mean([1.0 if c["lattice_trusted"] else 0.0 for c in subset])), 4),
            }

    report: dict[str, object] = {
        "schema": "pg-benchmark-v1",
        "grid_size": int(config.grid_size),
        "seeds": list(config.seeds),
        "families": {k: [float(v) for v in vs] for k, vs in families.items()},
        "cases": cases,
        "summary": summary,
    }

    with (output_dir / "benchmark_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with (output_dir / "benchmark_report.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(cases[0].keys()))
        writer.writeheader()
        writer.writerows(cases)
    return report


def compare_benchmarks(report_a: dict[str, object], report_b: dict[str, object]) -> dict[str, object]:
    """对齐两份基准报告并给出逐指标差值与回归清单（B 相对 A）。

    回归判定（保守）：定位 %pitch 误差恶化超过 0.5 个百分点且相对恶化
    超过 20%，或 Spearman 保序性下降超过 0.05。
    """

    def _index(report: dict[str, object]) -> dict[tuple, dict[str, object]]:
        return {
            (c["family"], float(c["level"]), int(c["seed"]), int(c["grid_size"])): c
            for c in report["cases"]
        }

    index_a = _index(report_a)
    index_b = _index(report_b)
    common = sorted(set(index_a) & set(index_b))

    metric_keys = ["loc_mean_pct_pitch", "quant_spearman_all", "quant_reliable_ratio"]
    per_family: dict[str, dict[str, float]] = {}
    regressions: list[dict[str, object]] = []

    for key in common:
        case_a, case_b = index_a[key], index_b[key]
        family = str(key[0])
        bucket = per_family.setdefault(family, {f"{m}_delta": 0.0 for m in metric_keys})
        bucket.setdefault("_count", 0.0)
        for metric in metric_keys:
            bucket[f"{metric}_delta"] += float(case_b[metric]) - float(case_a[metric])
        bucket["_count"] += 1.0

        loc_a, loc_b = float(case_a["loc_mean_pct_pitch"]), float(case_b["loc_mean_pct_pitch"])
        sp_a, sp_b = float(case_a["quant_spearman_all"]), float(case_b["quant_spearman_all"])
        loc_regressed = (loc_b - loc_a) > 0.5 and loc_b > loc_a * 1.2
        spearman_regressed = (sp_a - sp_b) > 0.05
        if loc_regressed or spearman_regressed:
            regressions.append(
                {
                    "family": family, "level": float(key[1]), "seed": int(key[2]),
                    "loc_mean_pct_pitch": {"a": loc_a, "b": loc_b},
                    "quant_spearman_all": {"a": sp_a, "b": sp_b},
                }
            )

    for family, bucket in per_family.items():
        count = max(bucket.pop("_count"), 1.0)
        for metric in metric_keys:
            bucket[f"{metric}_delta"] = round(bucket[f"{metric}_delta"] / count, 6)

    return {
        "common_case_count": len(common),
        "per_family": per_family,
        "regressions": regressions,
    }
