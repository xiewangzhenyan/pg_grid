"""PG-Quant 命令行工具：基于已有定位结果重跑逐单元定量。

示例：
python pg_quant_demo.py --result outputs/sample_10x10/result.json
python pg_quant_demo.py --result outputs/sample_10x10/result.json --image outputs/sample_10x10/rectified_chip.jpg --output reports/quant
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pg_grid import read_image_unicode
from pg_quant import quantify_rectified, write_quant_outputs


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="PG-Quant 逐单元定量")
    parser.add_argument("--result", required=True, help="PG-Grid result.json 路径")
    parser.add_argument("--image", default=None, help="矫正图路径；默认使用 result.json 中记录的 rectified_chip")
    parser.add_argument("--output", default=None, help="输出目录；默认写到 result.json 所在目录")
    return parser.parse_args()


def main() -> None:
    """读取定位结果，执行定量并写出 quant_values.csv / quant_result.json。"""

    args = parse_args()
    result_path = Path(args.result)
    with result_path.open("r", encoding="utf-8") as f:
        result = json.load(f)

    grid_size = int(result["grid_size"])
    points = np.asarray(
        [[float(p["x"]), float(p["y"])] for p in result["grid_points"]],
        dtype=np.float64,
    )

    image_path = args.image or result.get("outputs", {}).get("rectified_chip")
    if not image_path:
        raise ValueError("result.json 中没有 rectified_chip 路径，请用 --image 显式指定矫正图")
    image = read_image_unicode(image_path)

    output_dir = Path(args.output) if args.output else result_path.parent
    records, meta = quantify_rectified(image, points, grid_size)
    outputs = write_quant_outputs(output_dir, records, meta)

    summary = meta["summary"]
    print("PG-Quant 定量完成")
    print(f"阵列规格: {grid_size}x{grid_size}")
    print(f"单元数量: {meta['unit_count']}")
    print(f"可靠单元: {summary['reliable_count']} ({summary['reliable_ratio']:.1%})")
    print(f"光照模型: {meta['illumination_model']}  均匀度: {meta['illumination_uniformity']}")
    print(f"平均|SNR|: {summary['mean_abs_snr']}")
    print(f"标记统计: {summary['flag_counts']}")
    print(f"输出: {outputs['quant_values_csv']}")
    print(f"输出: {outputs['quant_result_json']}")


if __name__ == "__main__":
    main()
