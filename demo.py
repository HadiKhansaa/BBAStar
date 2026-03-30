import argparse
import pathlib
import re
import subprocess
import sys
from typing import Dict, List


ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
RESULT_PATTERN = re.compile(r"^BENCHMARK_RESULT\s+(.*)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal end-to-end demo for the CUDA bidirectional A* executable."
    )
    parser.add_argument(
        "--binary",
        default="bin/astar_bidirectional.exe",
        help="Path to the CUDA executable relative to the repository root.",
    )
    parser.add_argument(
        "--map",
        default="data/maps/arena.map",
        help="Path to the demo map relative to the repository root.",
    )
    parser.add_argument(
        "--scen",
        default="data/maps/arena.map.scen",
        help="Path to the scenario file relative to the repository root.",
    )
    parser.add_argument(
        "--scenario-index",
        type=int,
        default=0,
        help="Zero-based scenario index inside the .scen file.",
    )
    parser.add_argument(
        "--no-image",
        action="store_true",
        help="Disable PNG generation.",
    )
    return parser.parse_args()


def read_scenarios(scen_path: pathlib.Path) -> List[Dict[str, int]]:
    scenarios: List[Dict[str, int]] = []
    with scen_path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("version"):
                continue
            parts = line.split()
            if len(parts) < 9:
                raise ValueError(f"Malformed scenario line {line_number} in {scen_path}")
            scenarios.append(
                {
                    "line_number": line_number,
                    "start_x": int(parts[4]),
                    "start_y": int(parts[5]),
                    "goal_x": int(parts[6]),
                    "goal_y": int(parts[7]),
                }
            )
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
    raise ValueError("BENCHMARK_RESULT line not found in process output.")


def main() -> int:
    args = parse_args()
    repo_root = pathlib.Path(__file__).resolve().parent
    binary = (repo_root / args.binary).resolve()
    map_path = (repo_root / args.map).resolve()
    scen_path = (repo_root / args.scen).resolve()

    if not binary.exists():
        print(f"Binary not found: {binary}", file=sys.stderr)
        print("Build it first with `make` from the repository root.", file=sys.stderr)
        return 1
    if not map_path.exists():
        print(f"Map not found: {map_path}", file=sys.stderr)
        return 1
    if not scen_path.exists():
        print(f"Scenario file not found: {scen_path}", file=sys.stderr)
        return 1

    scenarios = read_scenarios(scen_path)
    if not scenarios:
        print(f"No scenarios found in {scen_path}", file=sys.stderr)
        return 1
    if args.scenario_index < 0 or args.scenario_index >= len(scenarios):
        print(
            f"Scenario index {args.scenario_index} is out of range for {scen_path}. "
            f"Available scenarios: 0-{len(scenarios) - 1}",
            file=sys.stderr,
        )
        return 1

    scenario = scenarios[args.scenario_index]
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
    ]
    if args.no_image:
        command.append("--no-image")

    print("Running demo command:")
    print("  " + subprocess.list2cmdline(command))

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.stdout:
        print("\nProcess output:")
        print(completed.stdout.strip())
    if completed.stderr:
        print("\nProcess stderr:", file=sys.stderr)
        print(completed.stderr.strip(), file=sys.stderr)

    if completed.returncode != 0:
        print(f"\nDemo failed with exit code {completed.returncode}.", file=sys.stderr)
        return completed.returncode

    result = parse_benchmark_result(completed.stdout)

    print("\nSummary:")
    print(f"  status: {result['status']}")
    print(f"  kernel runtime (s): {result['runtime_seconds']}")
    print(f"  expanded nodes: {result['expanded_nodes']}")
    print(f"  path cost: {result['path_cost']}")
    print(
        "  scenario: "
        f"({scenario['start_x']}, {scenario['start_y']}) -> "
        f"({scenario['goal_x']}, {scenario['goal_y']})"
    )
    if not args.no_image:
        print(f"  image: {(repo_root / 'data' / 'AstarPath.png').resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
