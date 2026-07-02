from pathlib import Path

import numpy as np
import pytest


def test_compute_localization_metrics_reports_known_offsets():
    """验证定位误差计算：3-4-5 偏移必须得到 5 px 误差。"""
    from pg_grid_eval import compute_localization_errors

    reference = [
        {"row": 0, "col": 0, "x": 0.0, "y": 0.0},
        {"row": 0, "col": 1, "x": 10.0, "y": 0.0},
        {"row": 1, "col": 0, "x": 0.0, "y": 10.0},
        {"row": 1, "col": 1, "x": 10.0, "y": 10.0},
    ]
    predicted = [
        {"row": 0, "col": 0, "x": 3.0, "y": 4.0},
        {"row": 0, "col": 1, "x": 10.0, "y": 0.0},
        {"row": 1, "col": 0, "x": 0.0, "y": 10.0},
        {"row": 1, "col": 1, "x": 10.0, "y": 10.0},
    ]

    errors, metrics = compute_localization_errors(reference, predicted, grid_size=2)

    assert errors[0]["error_px"] == pytest.approx(5.0)
    assert errors[1]["error_px"] == pytest.approx(0.0)
    assert metrics["mean_error_px"] == pytest.approx(1.25)
    assert metrics["median_error_px"] == pytest.approx(0.0)
    assert metrics["max_error_px"] == pytest.approx(5.0)
    assert metrics["pass_ratio_5px"] == pytest.approx(1.0)


def test_write_evaluation_report_creates_json_csv_and_overlay(tmp_path):
    """验证评价报告写出：必须生成指标 JSON、逐点 CSV 和误差叠图。"""
    from pg_grid import write_image_unicode
    from pg_grid_eval import write_annotation_json, write_evaluation_report

    image = np.full((80, 80, 3), 220, dtype=np.uint8)
    image_path = tmp_path / "rectified.jpg"
    write_image_unicode(image_path, image)

    reference = [
        {"row": 0, "col": 0, "x": 20.0, "y": 20.0},
        {"row": 0, "col": 1, "x": 60.0, "y": 20.0},
        {"row": 1, "col": 0, "x": 20.0, "y": 60.0},
        {"row": 1, "col": 1, "x": 60.0, "y": 60.0},
    ]
    predicted = [
        {"row": 0, "col": 0, "x": 23.0, "y": 24.0},
        {"row": 0, "col": 1, "x": 60.0, "y": 20.0},
        {"row": 1, "col": 0, "x": 20.0, "y": 60.0},
        {"row": 1, "col": 1, "x": 60.0, "y": 60.0},
    ]

    annotation_path = tmp_path / "annotation.json"
    prediction_path = tmp_path / "prediction.json"
    report_dir = tmp_path / "report"

    write_annotation_json(annotation_path, image_path=image_path, grid_size=2, points=reference)
    prediction_path.write_text(
        (
            '{'
            '"grid_size": 2,'
            '"point_count": 4,'
            '"grid_points": ['
            '{"row":0,"col":0,"x":23.0,"y":24.0},'
            '{"row":0,"col":1,"x":60.0,"y":20.0},'
            '{"row":1,"col":0,"x":20.0,"y":60.0},'
            '{"row":1,"col":1,"x":60.0,"y":60.0}'
            ']'
            '}'
        ),
        encoding="utf-8",
    )

    metrics = write_evaluation_report(
        annotation_path=annotation_path,
        prediction_path=prediction_path,
        image_path=image_path,
        output_dir=report_dir,
    )

    assert metrics["mean_error_px"] == pytest.approx(1.25)
    assert (report_dir / "localization_metrics.json").exists()
    assert (report_dir / "localization_errors.csv").exists()
    assert (report_dir / "localization_error_overlay.jpg").exists()


def test_create_initial_annotation_from_prediction_json(tmp_path):
    """验证可从 PG-Grid 预测点生成初始标注文件，供人工微调。"""
    from pg_grid_annotator import create_initial_annotation
    from pg_grid_eval import load_annotation_json

    prediction_path = tmp_path / "prediction.json"
    image_path = tmp_path / "rectified.jpg"
    annotation_path = tmp_path / "annotation.json"
    prediction_path.write_text(
        (
            '{'
            '"grid_size": 1,'
            '"point_count": 1,'
            '"grid_points": [{"row":0,"col":0,"x":12.5,"y":18.5}]'
            '}'
        ),
        encoding="utf-8",
    )

    create_initial_annotation(
        image_path=image_path,
        prediction_path=prediction_path,
        grid_size=1,
        output_path=annotation_path,
    )

    annotation = load_annotation_json(annotation_path)
    assert annotation["grid_size"] == 1
    assert annotation["points"][0]["x"] == pytest.approx(12.5)
    assert annotation["points"][0]["y"] == pytest.approx(18.5)
