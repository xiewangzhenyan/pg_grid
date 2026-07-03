"""PG-Benchmark 命令行工具：跑扰动基准或对比两份报告。

跑基准：
python pg_benchmark_demo.py --grid 10 --seeds 2 --output reports/bench_10
python pg_benchmark_demo.py --grid 10 --families rotation,blur --seeds 1 --output reports/quick

A/B 对比（先后在两个版本上各跑一次，seed 相同则场景相同）：
python pg_benchmark_demo.py --compare reports/a/benchmark_report.json reports/b/benchmark_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pg_benchmark import DEFAULT_FAMILIES, BenchmarkConfig, compare_benchmarks, run_benchmark


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="PG-Benchmark 扰动基准与 A/B 对比")
    parser.add_argument("--grid", type=int, default=10, choices=[10, 15], help="阵列规格")
    parser.add_argument("--seeds", type=int, default=2, help="每个 (扰动族, 强度) 的随机场景数")
    parser.add_argument("--families", default=None,
                        help="逗号分隔的扰动族子集（默认全部）：" + ",".join(DEFAULT_FAMILIES))
    parser.add_argument("--output", default=None, help="基准输出目录")
    parser.add_argument("--compare", nargs=2, metavar=("REPORT_A", "REPORT_B"),
                        help="对比两份 benchmark_report.json（B 相对 A）")
    return parser.parse_args()


def _print_summary(report: dict) -> None:
    """打印退化曲线摘要表。"""

    print(f"{'扰动族':<14}{'强度':>8}  {'定位%pitch':>10}  {'P90%pitch':>10}  {'Spearman':>9}  {'可靠率':>7}  {'可信率':>7}")
    for family, levels in report["summary"].items():
        for level, row in levels.items():
            print(
                f"{family:<14}{level:>8}  {row['loc_mean_pct_pitch']:>10.3f}  "
                f"{row['loc_p90_pct_pitch']:>10.3f}  {row['quant_spearman_all']:>9.4f}  "
                f"{row['quant_reliable_ratio']:>7.2f}  {row['trusted_ratio']:>7.2f}"
            )


def main() -> None:
    """执行基准或对比。"""

    args = parse_args()

    if args.compare:
        with Path(args.compare[0]).open("r", encoding="utf-8") as f:
            report_a = json.load(f)
        with Path(args.compare[1]).open("r", encoding="utf-8") as f:
            report_b = json.load(f)
        comparison = compare_benchmarks(report_a, report_b)
        print(f"对齐 case 数: {comparison['common_case_count']}")
        print("逐扰动族平均差值 (B - A，定位越小越好，其余越大越好):")
        for family, deltas in comparison["per_family"].items():
            print(f"  {family:<14} 定位%pitch {deltas['loc_mean_pct_pitch_delta']:+.4f}  "
                  f"Spearman {deltas['quant_spearman_all_delta']:+.4f}  "
                  f"可靠率 {deltas['quant_reliable_ratio_delta']:+.4f}")
        if comparison["regressions"]:
            print(f"回归 case ({len(comparison['regressions'])} 个):")
            for reg in comparison["regressions"]:
                print(f"  {reg['family']} level={reg['level']} seed={reg['seed']}: "
                      f"定位 {reg['loc_mean_pct_pitch']['a']:.3f}->{reg['loc_mean_pct_pitch']['b']:.3f} %pitch, "
                      f"Spearman {reg['quant_spearman_all']['a']:.3f}->{reg['quant_spearman_all']['b']:.3f}")
        else:
            print("未发现回归。")
        return

    if args.output is None:
        raise SystemExit("跑基准需要 --output；对比模式请用 --compare")

    families = None
    if args.families:
        names = [name.strip() for name in args.families.split(",") if name.strip()]
        unknown = [n for n in names if n not in DEFAULT_FAMILIES]
        if unknown:
            raise SystemExit(f"未知扰动族：{unknown}，可选：{list(DEFAULT_FAMILIES)}")
        families = {name: DEFAULT_FAMILIES[name] for name in names}

    config = BenchmarkConfig(grid_size=args.grid, seeds=tuple(range(args.seeds)), families=families)
    report = run_benchmark(config, output_dir=args.output)
    print(f"PG-Benchmark 完成：{len(report['cases'])} 个 case")
    _print_summary(report)
    print(f"报告: {Path(args.output).resolve() / 'benchmark_report.json'}")


if __name__ == "__main__":
    main()
