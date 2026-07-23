"""PG-Unit-Export 测试：逐单元紧致裁切的零背景与零漏裁保证。"""

import json
from pathlib import Path

import numpy as np


def _make_dark_square_scene(grid_size=10, size=800, start=112.0, pitch=64.0,
                            unit_half=12, background=200, unit_value=90):
    """亮面板 + 暗方块合成矫正图，返回 (图像, 中心点, 方块边长)。"""
    image = np.full((size, size, 3), background, dtype=np.uint8)
    points = []
    for row in range(grid_size):
        for col in range(grid_size):
            cx, cy = start + col * pitch, start + row * pitch
            x0, y0 = int(round(cx - unit_half)), int(round(cy - unit_half))
            image[y0:y0 + 2 * unit_half, x0:x0 + 2 * unit_half] = unit_value
            points.append((x0 + unit_half - 0.5, y0 + unit_half - 0.5))
    return image, np.asarray(points, dtype=np.float32), 2 * unit_half


def test_crops_are_tight_and_contain_no_background(tmp_path):
    """导出的每张小图必须恰好是方块本体：尺寸吻合且不含任何背景像素。"""
    from pg_unit_export import export_unit_crops

    image, points, unit_size = _make_dark_square_scene()
    manifest = export_unit_crops(
        image, points, grid_size=10, polarity="dark", output_dir=tmp_path
    )

    assert manifest["unit_count"] == 100
    assert manifest["fallback_count"] == 0
    import cv2

    for unit in manifest["units"]:
        crop_path = tmp_path / unit["file"]
        assert crop_path.exists(), unit["file"]
        crop = cv2.imdecode(np.fromfile(crop_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        # 尺寸必须与真实方块边长一致（±2px 容差）。
        assert abs(crop.shape[0] - unit_size) <= 2, (unit["row"], unit["col"], crop.shape)
        assert abs(crop.shape[1] - unit_size) <= 2
        # 零背景：背景灰度 200，方块 90，crop 内不允许出现任何接近背景的像素。
        assert int(crop.max()) < 150, (unit["row"], unit["col"], int(crop.max()))


def test_every_unit_exported_with_stable_naming(tmp_path):
    """必须恰好导出 grid_size^2 张小图，命名可按行列直接索引（零漏裁）。"""
    from pg_unit_export import export_unit_crops

    image, points, _ = _make_dark_square_scene()
    manifest = export_unit_crops(image, points, grid_size=10, polarity="dark", output_dir=tmp_path)

    files = sorted(p.name for p in tmp_path.glob("unit_r*_c*.png"))
    assert len(files) == 100
    expected = sorted(
        f"unit_r{row:02d}_c{col:02d}.png" for row in range(10) for col in range(10)
    )
    assert files == expected
    assert (tmp_path / "crop_manifest.json").exists()
    assert (tmp_path / "crop_contact_sheet.jpg").exists()

    with (tmp_path / "crop_manifest.json").open("r", encoding="utf-8") as f:
        saved = json.load(f)
    assert len(saved["units"]) == 100


def test_occluded_unit_falls_back_to_median_box(tmp_path):
    """单元被遮挡导致分割失败时，必须用中位尺寸盒兜底并在清单中标明。"""
    from pg_unit_export import export_unit_crops

    image, points, unit_size = _make_dark_square_scene()
    bad = 4 * 10 + 7
    cx, cy = points[bad]
    x0, y0 = int(round(cx - 20)), int(round(cy - 20))
    image[y0:y0 + 40, x0:x0 + 40] = 200  # 抹成背景色

    manifest = export_unit_crops(image, points, grid_size=10, polarity="dark", output_dir=tmp_path)

    assert manifest["unit_count"] == 100
    assert manifest["fallback_count"] >= 1
    bad_unit = manifest["units"][bad]
    assert bad_unit["source"] == "fallback_median_box"
    # 兜底盒尺寸取自成功分割单元的中位数，仍与真实方块边长一致。
    assert abs(bad_unit["width"] - unit_size) <= 2
    assert abs(bad_unit["height"] - unit_size) <= 2
    # 兜底盒中心落在晶格点上。
    assert abs((bad_unit["box"][0] + bad_unit["box"][2]) / 2.0 - float(points[bad][0])) <= 2.0


def test_bright_polarity_units_crop(tmp_path):
    """亮单元极性同样支持：紧致框应吻合亮点外接尺寸。"""
    import cv2
    from pg_unit_export import export_unit_crops

    image = np.full((800, 800, 3), 30, dtype=np.uint8)
    points = []
    radius = 11
    for row in range(10):
        for col in range(10):
            cx, cy = 112 + col * 64, 112 + row * 64
            cv2.circle(image, (cx, cy), radius, (220, 220, 220), -1)
            points.append((float(cx), float(cy)))
    points = np.asarray(points, dtype=np.float32)

    manifest = export_unit_crops(image, points, grid_size=10, polarity="bright", output_dir=tmp_path)

    assert manifest["unit_count"] == 100
    assert manifest["fallback_count"] == 0
    for unit in manifest["units"]:
        assert abs(unit["width"] - (2 * radius + 1)) <= 2
        assert abs(unit["height"] - (2 * radius + 1)) <= 2


def test_real_photo_15x15_exports_225_tight_crops(tmp_path):
    """实拍 15x15 图必须导出 225 张紧致小图，且绝大多数来自真实分割。"""
    from pg_grid import process_image
    from pg_unit_export import export_units_from_result

    photo = Path(__file__).resolve().parent.parent / "examples" / "手机通过短塔正式拍摄-EL板发光(15×15芯片)04.jpg"
    result = process_image(image_path=photo, grid_size=15, output_dir=tmp_path / "pipeline")

    manifest = export_units_from_result(tmp_path / "pipeline" / "result.json", output_dir=tmp_path / "crops")

    assert manifest["unit_count"] == 225
    assert manifest["segmented_count"] >= 200, manifest["fallback_count"]
    files = list((tmp_path / "crops").glob("unit_r*_c*.png"))
    assert len(files) == 225
    # 尺寸合理性：实拍单元在矫正图中的边长应远小于间距且非退化。
    widths = [u["width"] for u in manifest["units"]]
    assert 8 <= float(np.median(widths)) <= 60
