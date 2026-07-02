"""PG-Grid 定位误差评价命令行工具。

示例：
python pg_grid_evaluate.py --annotation annotations\10x10_a.json --prediction outputs\10x10_a\result.json --image outputs\10x10_a\rectified_chip.jpg --output reports\10x10_a
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pg_grid_eval import write_evaluation_report


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="PG-Grid 定位误差评价")
    parser.add_argument("--annotation", required=True, help="人工标注 annotation.json")
    parser.add_argument("--prediction", required=True, help="PG-Grid result.json 或 values.csv")
    parser.add_argument("--image", required=True, help="矫正图 rectified_chip.jpg")
    parser.add_argument("--output", required=True, help="评价报告输出目录")
    return parser.parse_args()


def main() -> None:
    """执行评价并打印关键指标。"""

    args = parse_args()
    metrics = write_evaluation_report(
        annotation_path=args.annotation,
        prediction_path=args.prediction,
        image_path=args.image,
        output_dir=args.output,
    )

    print("PG-Grid 定位评价完成")
    print(f"标注文件: {args.annotation}")
    print(f"预测文件: {args.prediction}")
    print(f"平均误差(px): {metrics['mean_error_px']}")
    print(f"P90误差(px): {metrics['p90_error_px']}")
    print(f"最大误差(px): {metrics['max_error_px']}")
    print(f"5px通过率: {metrics['pass_ratio_5px']}")
    print(f"10px通过率: {metrics['pass_ratio_10px']}")
    print(f"输出目录: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
