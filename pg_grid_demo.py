"""PG-Grid 命令行 demo。

示例：
python pg_grid_demo.py --image "相机通过成像视野正式拍摄-uniform illumination照明板照明(15×15目标平面).jpg" --grid 15 --output outputs\15x15
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pg_grid import process_image


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="PG-Grid 规则阵列目标平面阵列定位 demo")
    parser.add_argument("--image", required=True, help="输入目标平面图像路径，支持中文文件名")
    parser.add_argument("--grid", required=True, type=int, choices=[10, 15], help="阵列规格：10 或 15")
    parser.add_argument("--output", default=None, help="输出目录；默认写入 outputs/<图片名>_<grid>x<grid>")
    return parser.parse_args()


def main() -> None:
    """执行单张图片处理，并打印关键质量指标。"""

    args = parse_args()
    image_path = Path(args.image)
    if args.output is None:
        safe_name = image_path.stem.replace(" ", "_")
        output_dir = Path("outputs") / f"{safe_name}_{args.grid}x{args.grid}"
    else:
        output_dir = Path(args.output)

    result = process_image(image_path=image_path, grid_size=args.grid, output_dir=output_dir)
    quality = result["quality"]

    print("PG-Grid 处理完成")
    print(f"输入图像: {image_path}")
    print(f"阵列规格: {args.grid}x{args.grid}")
    print(f"点位数量: {result['point_count']}")
    print(f"质量状态: {quality['status']}")
    print(f"总体评分: {quality['overall_score']}")
    print(f"过曝比例: {quality['saturation_ratio']}")
    print(f"输出目录: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
