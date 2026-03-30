import argparse
import csv
import datetime as dt
import os
import pathlib
import re
import statistics
import subprocess
import sys
from typing import Dict, List, Tuple


RESULT_PATTERN = re.compile(r"^BENCHMARK_RESULT\s+(.*)$")
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run astar_bidirectional.exe against every .map/.scen pair in data/maps."
    )
    parser.add_argument(
        "--maps-dir",
        default="data/maps",
        help="Directory containing MovingAI .map and .map.scen files.",
    )
    parser.add_argument(
        "--binary",
        default="bin/astar_bidirectional.exe",
        help="Path to astar_bidirectional executable.",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Directory where logs and summaries will be written.",
    )
    parser.add_argument(
        "--limit-scenarios",
        type=int,
        default=10,
        help="Cap on scenarios per .scen file. Defaults to 10.",
    )
    parser.add_argument(
        "--map-filter",
        default=None,
        help="Only run maps whose filename contains this substring.",
    )
    return parser.parse_args()


def discover_pairs(maps_dir: pathlib.Path, map_filter: str = None) -> List[Tuple[pathlib.Path, pathlib.Path]]:
    pairs: List[Tuple[pathlib.Path, pathlib.Path]] = []
    for map_path in sorted(maps_dir.glob("*.map")):
        if map_filter and map_filter not in map_path.name:
            continue
        scen_path = maps_dir / f"{map_path.name}.scen"
        if scen_path.exists():
            pairs.append((map_path, scen_path))
    return pairs


def read_scenarios(scen_path: pathlib.Path, limit: int = None) -> List[Dict[str, object]]:
    scenarios: List[Dict[str, object]] = []
    with scen_path.open("r", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("version"):
                continue
            parts = line.split()
            if len(parts) < 9:
                raise ValueError(f"Malformed scenario line {line_number} in {scen_path}")
            bucket, map_name, width, height, start_x, start_y, goal_x, goal_y, optimal_length = parts[:9]
            scenarios.append(
                {
                    "line_number": line_number,
                    "bucket": int(bucket),
                    "map_name": map_name,
                    "width": int(width),
                    "height": int(height),
                    "start_x": int(start_x),
                    "start_y": int(start_y),
                    "goal_x": int(goal_x),
                    "goal_y": int(goal_y),
                    "optimal_length": float(optimal_length),
                }
            )
            if limit is not None and len(scenarios) >= limit:
                break
    return scenarios


def parse_benchmark_result(stdout: str) -> Dict[str, str]:
    for line in stdout.splitlines():
        clean_line = ANSI_PATTERN.sub("", line).strip()
        match = RESULT_PATTERN.match(clean_line)
        if not match:
            continue
        result: Dict[str, str] = {}
        for token in match.group(1).split():
            key, value = token.split("=", 1)
            result[key] = value
        return result
    raise ValueError("Benchmark result marker not found in process output.")


def run_single(binary: pathlib.Path, map_path: pathlib.Path, scenario: Dict[str, object]) -> Tuple[subprocess.CompletedProcess, Dict[str, str], float]:
    command = [
        str(binary),
        "--map",
        str(map_path),
        "--start-x",
        str(scenario["start_x"]),
        "--start-y",
        str(scenario["start_y"]),
        "--goal-x",
        str(scenario["goal_x"]),
        "--goal-y",
        str(scenario["goal_y"]),
        "--no-image",
    ]
    wall_start = dt.datetime.now(dt.timezone.utc)
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    wall_seconds = (dt.datetime.now(dt.timezone.utc) - wall_start).total_seconds()
    result = parse_benchmark_result(completed.stdout)
    return completed, result, wall_seconds


def main() -> int:
    args = parse_args()
    repo_root = pathlib.Path.cwd()
    maps_dir = (repo_root / args.maps_dir).resolve()
    binary = (repo_root / args.binary).resolve()
    output_dir = (repo_root / args.output_dir).resolve()

    if not binary.exists():
        print(f"Binary not found: {binary}", file=sys.stderr)
        return 1
    if not maps_dir.exists():
        print(f"Maps directory not found: {maps_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    detail_csv = output_dir / "benchmark_details.csv"
    summary_txt = output_dir / "benchmark_summary.txt"

    pairs = discover_pairs(maps_dir, args.map_filter)
    if not pairs:
        print("No map/scenario pairs found.", file=sys.stderr)
        return 1

    rows: List[Dict[str, object]] = []
    timestamp = dt.datetime.now().isoformat(timespec="seconds")

    for map_path, scen_path in pairs:
        scenarios = read_scenarios(scen_path, args.limit_scenarios)
        for index, scenario in enumerate(scenarios):
            completed, result, wall_seconds = run_single(binary, map_path, scenario)
            log_name = f"{map_path.stem}__line{scenario['line_number']}.log"
            log_path = logs_dir / log_name
            with log_path.open("w", encoding="utf-8", newline="") as handle:
                handle.write("COMMAND:\n")
                handle.write(
                    " ".join(
                        [
                            str(binary),
                            "--map",
                            str(map_path),
                            "--start-x",
                            str(scenario["start_x"]),
                            "--start-y",
                            str(scenario["start_y"]),
                            "--goal-x",
                            str(scenario["goal_x"]),
                            "--goal-y",
                            str(scenario["goal_y"]),
                            "--no-image",
                        ]
                    )
                )
                handle.write("\n\nSTDOUT:\n")
                handle.write(completed.stdout)
                handle.write("\nSTDERR:\n")
                handle.write(completed.stderr)

            rows.append(
                {
                    "timestamp": timestamp,
                    "map_file": map_path.name,
                    "scenario_file": scen_path.name,
                    "scenario_index": index,
                    "scenario_line": scenario["line_number"],
                    "bucket": scenario["bucket"],
                    "map_width": scenario["width"],
                    "map_height": scenario["height"],
                    "start_x": scenario["start_x"],
                    "start_y": scenario["start_y"],
                    "goal_x": scenario["goal_x"],
                    "goal_y": scenario["goal_y"],
                    "optimal_length": scenario["optimal_length"],
                    "status": result["status"],
                    "kernel_runtime_seconds": float(result["runtime_seconds"]),
                    "wall_runtime_seconds": wall_seconds,
                    "expanded_nodes": int(result["expanded_nodes"]),
                    "path_cost": float(result["path_cost"]),
                    "exit_code": completed.returncode,
                    "log_file": str(log_path.relative_to(output_dir)),
                }
            )
            print(
                f"[{len(rows)}] {map_path.name} line {scenario['line_number']}: "
                f"status={result['status']} kernel={result['runtime_seconds']}s wall={wall_seconds:.6f}s"
            )

    fieldnames = [
        "timestamp",
        "map_file",
        "scenario_file",
        "scenario_index",
        "scenario_line",
        "bucket",
        "map_width",
        "map_height",
        "start_x",
        "start_y",
        "goal_x",
        "goal_y",
        "optimal_length",
        "status",
        "kernel_runtime_seconds",
        "wall_runtime_seconds",
        "expanded_nodes",
        "path_cost",
        "exit_code",
        "log_file",
    ]
    with detail_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    kernel_times = [row["kernel_runtime_seconds"] for row in rows]
    wall_times = [row["wall_runtime_seconds"] for row in rows]
    found_count = sum(1 for row in rows if row["status"] == "found")
    not_found_count = sum(1 for row in rows if row["status"] != "found")
    map_names = sorted({row["map_file"] for row in rows})

    with summary_txt.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"Benchmark run: {timestamp}\n")
        handle.write(f"Binary: {binary}\n")
        handle.write(f"Maps directory: {maps_dir}\n")
        handle.write(f"Maps processed: {len(map_names)}\n")
        handle.write(f"Scenarios processed: {len(rows)}\n")
        handle.write(f"Successful paths: {found_count}\n")
        handle.write(f"Unsuccessful paths: {not_found_count}\n")
        handle.write(f"Average kernel runtime (s): {statistics.fmean(kernel_times):.9f}\n")
        handle.write(f"Average wall runtime (s): {statistics.fmean(wall_times):.9f}\n")
        handle.write(f"Min kernel runtime (s): {min(kernel_times):.9f}\n")
        handle.write(f"Max kernel runtime (s): {max(kernel_times):.9f}\n")
        handle.write(f"Detailed CSV: {detail_csv.name}\n")
        handle.write("Per-map averages:\n")
        for map_name in map_names:
            map_rows = [row for row in rows if row["map_file"] == map_name]
            handle.write(
                f"  {map_name}: avg_kernel={statistics.fmean(row['kernel_runtime_seconds'] for row in map_rows):.9f}s "
                f"avg_wall={statistics.fmean(row['wall_runtime_seconds'] for row in map_rows):.9f}s "
                f"scenarios={len(map_rows)}\n"
            )

    print(f"\nWrote details to {detail_csv}")
    print(f"Wrote summary to {summary_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
