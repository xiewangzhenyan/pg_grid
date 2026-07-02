"""PG-Grid 定位标注与误差评价工具函数。

本模块只处理“矫正图坐标系”下的点位评价：
- reference_points 表示人工标注或人工修正后的真值点；
- predicted_points 表示 PG-Grid 自动定位输出点；
- 二者都按 row/col 对齐，不做隐式最近邻匹配。

这样设计是为了让定位评价可复现，避免 10×10 和 15×15 混用、
点序错误或缺点时仍然输出看似合理的错误结果。
"""

from __future__ import annotations

import math
import csv
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from pg_grid import read_image_unicode, write_image_unicode


PointRecord = dict[str, float | int]


def normalize_point_records(points: Iterable[PointRecord]) -> list[PointRecord]:
    """把点位记录规范化为 row、col、x、y 四个字段。

    输入可以来自 result.json、annotation.json 或 values.csv。只要包含这四个字段，
    就会被转成稳定类型并按 row/col 排序，方便后续逐点比较。
    """

    normalized: list[PointRecord] = []
    for point in points:
        normalized.append(
            {
                "row": int(point["row"]),
                "col": int(point["col"]),
                "x": float(point["x"]),
                "y": float(point["y"]),
            }
        )
    normalized.sort(key=lambda item: (int(item["row"]), int(item["col"])))
    return normalized


def _validate_point_sets(reference_points: list[PointRecord], predicted_points: list[PointRecord], grid_size: int) -> None:
    """检查人工标注点和预测点是否可以逐点比较。"""

    expected_count = grid_size * grid_size
    if len(reference_points) != expected_count:
        raise ValueError(f"人工标注点数应为 {expected_count}，实际为 {len(reference_points)}")
    if len(predicted_points) != expected_count:
        raise ValueError(f"预测点数应为 {expected_count}，实际为 {len(predicted_points)}")

    for reference, predicted in zip(reference_points, predicted_points):
        if int(reference["row"]) != int(predicted["row"]) or int(reference["col"]) != int(predicted["col"]):
            raise ValueError(
                "人工标注点和预测点的 row/col 不一致："
                f"reference=({reference['row']},{reference['col']}), "
                f"predicted=({predicted['row']},{predicted['col']})"
            )


def compute_localization_errors(
    reference_points: Iterable[PointRecord],
    predicted_points: Iterable[PointRecord],
    grid_size: int,
) -> tuple[list[dict[str, float | int]], dict[str, float | int]]:
    """计算预测点相对人工标注点的定位误差。

    返回：
    - errors：逐点误差列表，包含 dx、dy、error_px；
    - metrics：整体统计指标，包含平均值、中位数、P90、最大值和阈值通过率。
    """

    reference = normalize_point_records(reference_points)
    predicted = normalize_point_records(predicted_points)
    _validate_point_sets(reference, predicted, grid_size)

    errors: list[dict[str, float | int]] = []
    error_values: list[float] = []
    for ref_point, pred_point in zip(reference, predicted):
        dx = float(pred_point["x"]) - float(ref_point["x"])
        dy = float(pred_point["y"]) - float(ref_point["y"])
        error_px = math.hypot(dx, dy)
        error_values.append(error_px)
        errors.append(
            {
                "row": int(ref_point["row"]),
                "col": int(ref_point["col"]),
                "reference_x": round(float(ref_point["x"]), 4),
                "reference_y": round(float(ref_point["y"]), 4),
                "predicted_x": round(float(pred_point["x"]), 4),
                "predicted_y": round(float(pred_point["y"]), 4),
                "dx": round(dx, 4),
                "dy": round(dy, 4),
                "error_px": round(error_px, 6),
            }
        )

    values = np.asarray(error_values, dtype=np.float32)
    metrics: dict[str, float | int] = {
        "grid_size": int(grid_size),
        "point_count": int(values.size),
        "mean_error_px": round(float(values.mean()), 6),
        "median_error_px": round(float(np.median(values)), 6),
        "p90_error_px": round(float(np.percentile(values, 90)), 6),
        "max_error_px": round(float(values.max()), 6),
        "rmse_error_px": round(float(np.sqrt(np.mean(values * values))), 6),
        "pass_ratio_5px": round(float((values <= 5.0).mean()), 6),
        "pass_ratio_10px": round(float((values <= 10.0).mean()), 6),
    }
    return errors, metrics


def write_annotation_json(
    path: str | Path,
    image_path: str | Path,
    grid_size: int,
    points: Iterable[PointRecord],
) -> dict[str, object]:
    """写出 PG-Grid 人工标注 JSON。

    这里不判断点是否真的来自人工点击；它也可用于把预测点复制成“初始标注”，
    后续再由人工交互工具微调。
    """

    normalized = normalize_point_records(points)
    _validate_annotation_count(normalized, grid_size)
    annotation: dict[str, object] = {
        "schema": "pg-grid-annotation-v1",
        "image_path": str(image_path),
        "grid_size": int(grid_size),
        "coordinate_space": "rectified",
        "points": normalized,
    }

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(annotation, f, ensure_ascii=False, indent=2)
    return annotation


def _validate_annotation_count(points: list[PointRecord], grid_size: int) -> None:
    """检查标注点数量是否匹配阵列规格。"""

    expected_count = grid_size * grid_size
    if len(points) != expected_count:
        raise ValueError(f"点数应为 {expected_count}，实际为 {len(points)}")


def load_annotation_json(path: str | Path) -> dict[str, object]:
    """读取人工标注 JSON，并做最基本的结构检查。"""

    annotation_path = Path(path)
    with annotation_path.open("r", encoding="utf-8") as f:
        annotation = json.load(f)

    if annotation.get("schema") != "pg-grid-annotation-v1":
        raise ValueError(f"不支持的标注 schema：{annotation.get('schema')}")
    grid_size = int(annotation["grid_size"])
    points = normalize_point_records(annotation["points"])
    _validate_annotation_count(points, grid_size)
    annotation["points"] = points
    return annotation


def _load_points_from_csv(path: Path) -> tuple[int | None, list[PointRecord]]:
    """从 values.csv 读取 row、col、x、y 点位。

    CSV 本身不一定包含 grid_size，因此返回 None；调用方可用 annotation 的 grid_size。
    """

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    points = normalize_point_records(rows)
    return None, points


def _load_points_from_json(path: Path) -> tuple[int, list[PointRecord]]:
    """从 result.json 或 annotation.json 读取点位。"""

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if "grid_points" in payload:
        grid_size = int(payload["grid_size"])
        points = normalize_point_records(payload["grid_points"])
        return grid_size, points
    if payload.get("schema") == "pg-grid-annotation-v1":
        grid_size = int(payload["grid_size"])
        points = normalize_point_records(payload["points"])
        return grid_size, points
    raise ValueError(f"JSON 中没有可识别的 grid_points 或 annotation points：{path}")


def load_prediction_points(path: str | Path) -> tuple[int | None, list[PointRecord]]:
    """从 PG-Grid 输出文件读取预测点。

    支持：
    - `result.json`：优先使用 V1.3 新增的 `grid_points`；
    - `values.csv`：兼容旧输出，读取 row、col、x、y。
    """

    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".json":
        return _load_points_from_json(input_path)
    if suffix == ".csv":
        return _load_points_from_csv(input_path)
    raise ValueError(f"不支持的预测点文件类型：{input_path}")


def _write_errors_csv(path: Path, errors: list[dict[str, float | int]]) -> None:
    """写出逐点误差 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not errors:
        raise ValueError("没有误差记录可写出")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(errors[0].keys()))
        writer.writeheader()
        writer.writerows(errors)


def draw_error_overlay(
    image: np.ndarray,
    errors: list[dict[str, float | int]],
) -> np.ndarray:
    """在矫正图上绘制定位误差叠图。

    绿色圆点为人工标注点，红色圆点为算法预测点，黄色线段表示偏移方向和大小。
    """

    overlay = image.copy()
    for item in errors:
        reference = (int(round(float(item["reference_x"]))), int(round(float(item["reference_y"]))))
        predicted = (int(round(float(item["predicted_x"]))), int(round(float(item["predicted_y"]))))
        error_px = float(item["error_px"])

        cv2.line(overlay, reference, predicted, (0, 255, 255), 1)
        cv2.circle(overlay, reference, 4, (0, 180, 0), -1)
        cv2.circle(overlay, predicted, 3, (0, 0, 255), -1)
        if error_px > 10:
            cv2.putText(
                overlay,
                f"{error_px:.1f}",
                (predicted[0] + 4, predicted[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (0, 255, 255),
                1,
            )
    return overlay


def write_evaluation_report(
    annotation_path: str | Path,
    prediction_path: str | Path,
    image_path: str | Path,
    output_dir: str | Path,
) -> dict[str, float | int | str]:
    """生成单张目标平面图的定位误差评价报告。"""

    annotation = load_annotation_json(annotation_path)
    annotation_grid_size = int(annotation["grid_size"])
    prediction_grid_size, predicted_points = load_prediction_points(prediction_path)
    if prediction_grid_size is not None and prediction_grid_size != annotation_grid_size:
        raise ValueError(f"标注 grid_size={annotation_grid_size} 与预测 grid_size={prediction_grid_size} 不一致")

    errors, metrics = compute_localization_errors(
        reference_points=annotation["points"],
        predicted_points=predicted_points,
        grid_size=annotation_grid_size,
    )

    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics_with_paths: dict[str, float | int | str] = dict(metrics)
    metrics_with_paths.update(
        {
            "annotation_path": str(annotation_path),
            "prediction_path": str(prediction_path),
            "image_path": str(image_path),
            "coordinate_space": "rectified",
        }
    )

    with (report_dir / "localization_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_with_paths, f, ensure_ascii=False, indent=2)
    _write_errors_csv(report_dir / "localization_errors.csv", errors)

    image = read_image_unicode(image_path)
    overlay = draw_error_overlay(image, errors)
    write_image_unicode(report_dir / "localization_error_overlay.jpg", overlay)
    return metrics_with_paths
