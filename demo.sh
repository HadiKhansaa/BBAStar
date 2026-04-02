#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CASES="256:random,256:maze,256:rectangle,256:zigzag,512:random,512:maze,512:rectangle,512:zigzag,1024:random,1024:rectangle"

GREEN=$'\033[32m'
YELLOW=$'\033[33m'
RED=$'\033[31m'
BLUE=$'\033[34m'
RESET=$'\033[0m'
PURPLE=$'\033[35m'

BINARY="bin/astar_bidirectional"
OBSTACLE_RATE=20
REPEATS=1
SCENARIO_INDEX=0
RUN_MAP_CASE=1
KEEP_IMAGE=0
OUTPUT_PATH=""
CASES="$DEFAULT_CASES"
MAP_PATH="data/maps/arena.map"
SCEN_PATH="data/maps/arena.map.scen"

usage() {
    cat <<'EOF'
Usage: ./demo.sh [options]

Run a small end-to-end experiment suite and summarize the results.

Options:
  --binary PATH          CUDA executable to run. Default: bin/astar_bidirectional
  --obstacle-rate N      Obstacle rate for procedural grids. Default: 20
  --repeats N            Repeats per procedural case. Default: 1
  --cases LIST           Comma-separated size:type cases.
                         Default: 256:random,256:maze,256:rectangle,256:zigzag,
                                  512:random,512:maze,512:rectangle,512:zigzag,
                                  1024:random,1024:rectangle
  --map PATH             MovingAI map used for the reference map case.
  --scen PATH            Scenario file paired with --map.
  --scenario-index N     Zero-based scenario index inside the .scen file. Default: 0
  --skip-map             Skip the bundled MovingAI reference case.
  --keep-image           Keep the generated data/AstarPath.png image.
  --output PATH          Write raw TSV results to this path.
  --help                 Show this message.
EOF
}

normalize_rel_path() {
    local path="$1"
    printf '%s' "${path//\\//}"
}

strip_ansi() {
    sed -E 's/\x1B\[[0-9;]*[[:alpha:]]//g'
}

color_runtime_text() {
    sed -E "s/([0-9]+(\.[0-9]+)?)( seconds?)/${GREEN}\1${RESET}\3/g; s/(runtime_seconds=)([0-9]+(\.[0-9]+)?)/\1${GREEN}\2${RESET}/g"
}

status_color() {
    case "$1" in
        found|completed)
            printf '%s' "$GREEN"
            ;;
        not_found)
            printf '%s' "$YELLOW"
            ;;
        error)
            printf '%s' "$RED"
            ;;
        *)
            printf '%s' "$RESET"
            ;;
    esac
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --binary)
                BINARY="$2"
                shift 2
                ;;
            --obstacle-rate)
                OBSTACLE_RATE="$2"
                shift 2
                ;;
            --repeats)
                REPEATS="$2"
                shift 2
                ;;
            --cases)
                CASES="$2"
                shift 2
                ;;
            --map)
                MAP_PATH="$2"
                shift 2
                ;;
            --scen)
                SCEN_PATH="$2"
                shift 2
                ;;
            --scenario-index)
                SCENARIO_INDEX="$2"
                shift 2
                ;;
            --skip-map)
                RUN_MAP_CASE=0
                shift
                ;;
            --keep-image)
                KEEP_IMAGE=1
                shift
                ;;
            --output)
                OUTPUT_PATH="$2"
                shift 2
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                printf '%sUnknown option:%s %s\n' "$RED" "$RESET" "$1" >&2
                usage >&2
                exit 1
                ;;
        esac
    done
}

assert_exists() {
    local label="$1"
    local path="$2"
    if [[ ! -f "$path" ]]; then
        printf '%s%s not found:%s %s\n' "$RED" "$label" "$RESET" "$path" >&2
        exit 1
    fi
}

get_scenario_coords() {
    local scen_file="$1"
    local index="$2"
    awk -v target="$index" '
        /^version/ { next }
        NF {
            if (seen == target) {
                print $5, $6, $7, $8
                exit
            }
            seen++
        }
        END {
            if (seen <= target) {
                exit 1
            }
        }
    ' "$scen_file"
}

append_result() {
    local suite="$1"
    local case_label="$2"
    local size_label="$3"
    local grid_type="$4"
    local repeat_index="$5"
    local status="$6"
    local runtime="$7"
    local expanded="$8"
    local path_cost="$9"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$suite" "$case_label" "$size_label" "$grid_type" "$repeat_index" \
        "$status" "$runtime" "$expanded" "$path_cost" >>"$RESULTS_FILE"
}

parse_benchmark_tokens() {
    local line="$1"
    local token
    for token in $line; do
        case "$token" in
            status=*) PARSED_STATUS="${token#status=}" ;;
            runtime_seconds=*) PARSED_RUNTIME="${token#runtime_seconds=}" ;;
            expanded_nodes=*) PARSED_EXPANDED="${token#expanded_nodes=}" ;;
            path_cost=*) PARSED_PATH_COST="${token#path_cost=}" ;;
        esac
    done
}

run_and_record() {
    local suite="$1"
    local case_label="$2"
    local size_label="$3"
    local grid_type="$4"
    local repeat_index="$5"
    shift 5

    local output
    local exit_code=0
    printf '\n%s[%s]%s %s%s%s (repeat %s)\n' "$PURPLE" "$suite" "$RESET" "$BLUE" "$case_label" "$RESET" "$repeat_index"
    printf '  %sCommand:%s' "$BLUE" "$RESET"
    printf ' %q' "$@"
    printf '\n'

    set +e
    output="$("$@" 2>&1)"
    exit_code=$?
    set -e

    local clean_output
    clean_output="$(printf '%s\n' "$output" | strip_ansi)"
    printf '%s\n' "$clean_output" | color_runtime_text

    if [[ "$KEEP_IMAGE" -eq 0 && -f "$SCRIPT_DIR/data/AstarPath.png" ]]; then
        rm -f "$SCRIPT_DIR/data/AstarPath.png"
    fi

    PARSED_STATUS="error"
    PARSED_RUNTIME="NA"
    PARSED_EXPANDED="NA"
    PARSED_PATH_COST="NA"

    local benchmark_line
    benchmark_line="$(printf '%s\n' "$clean_output" | awk '/^BENCHMARK_RESULT / { sub(/^BENCHMARK_RESULT /, ""); print; exit }')"
    if [[ -n "$benchmark_line" ]]; then
        parse_benchmark_tokens "$benchmark_line"
    else
        PARSED_RUNTIME="$(printf '%s\n' "$clean_output" | awk '/Execution time/ { print $(NF-1); exit }')"
        PARSED_EXPANDED="$(printf '%s\n' "$clean_output" | awk -F': ' '/Total number of expanded nodes/ { print $2; exit }')"
        PARSED_PATH_COST="$(printf '%s\n' "$clean_output" | awk -F': ' '/Path found with cost/ { print $2; exit }')"
        if [[ "$exit_code" -ne 0 ]]; then
            PARSED_STATUS="error"
        elif printf '%s\n' "$clean_output" | grep -q 'Path found with cost'; then
            PARSED_STATUS="found"
        elif printf '%s\n' "$clean_output" | grep -qi 'no path'; then
            PARSED_STATUS="not_found"
        else
            PARSED_STATUS="completed"
        fi
    fi

    if [[ "$exit_code" -ne 0 ]]; then
        printf '  %sResult:%s command failed with exit code %s\n' "$RED" "$RESET" "$exit_code" >&2
    fi

    append_result \
        "$suite" "$case_label" "$size_label" "$grid_type" "$repeat_index" \
        "$PARSED_STATUS" "$PARSED_RUNTIME" "$PARSED_EXPANDED" "$PARSED_PATH_COST"

    printf '  %sStatus:%s %s%s%s\n' "$BLUE" "$RESET" "$(status_color "$PARSED_STATUS")" "$PARSED_STATUS" "$RESET"
}

print_detailed_results() {
    printf '\n%sDetailed results%s\n' "$PURPLE" "$RESET"
    printf '%s%-12s %-18s %-10s %-12s %-8s %-12s %-16s %-12s%s\n' \
        "$BLUE" \
        "suite" "case" "size" "type" "status" "runtime (s)" "expanded_nodes" "path_cost" \
        "$RESET"
    awk -F'\t' -v green="$GREEN" -v reset="$RESET" 'NR > 1 {
        runtime = ($7 != "" && $7 != "NA") ? green $7 reset : $7
        printf "%-12s %-18s %-10s %-12s %-8s %-21s %-16s %-12s\n",
            $1, $2, $3, $4, $6, runtime, $8, $9
    }' "$RESULTS_FILE"
}

print_summary_by_field() {
    local field_index="$1"
    local title="$2"
    printf '\n%s%s%s\n' "$PURPLE" "$title" "$RESET"
    printf '%s%-12s %-6s %-10s %-14s %-16s %-14s %-14s%s\n' \
        "$BLUE" \
        "group" "runs" "successes" "avg_runtime_s" "avg_expanded" "avg_path_cost" "max_runtime_s" \
        "$RESET"
    awk -F'\t' -v key_index="$field_index" -v green="$GREEN" -v reset="$RESET" '
        NR == 1 { next }
        {
            key = $key_index
            runs[key]++
            if ($6 == "found" || $6 == "completed") {
                successes[key]++
            }
            if ($7 != "NA" && $7 != "") {
                runtime_sum[key] += $7
                runtime_count[key]++
                if ($7 > runtime_max[key]) {
                    runtime_max[key] = $7
                }
            }
            if ($8 != "NA" && $8 != "") {
                expanded_sum[key] += $8
                expanded_count[key]++
            }
            if ($9 != "NA" && $9 != "") {
                cost_sum[key] += $9
                cost_count[key]++
            }
        }
        END {
            for (key in runs) {
                avg_runtime = runtime_count[key] ? runtime_sum[key] / runtime_count[key] : 0
                avg_expanded = expanded_count[key] ? expanded_sum[key] / expanded_count[key] : 0
                avg_cost = cost_count[key] ? cost_sum[key] / cost_count[key] : 0
                printf "%-12s %-6d %-10d %s%-14.6f%s %-16.2f %-14.2f %s%-14.6f%s\n",
                    key, runs[key], successes[key] + 0,
                    green, avg_runtime, reset,
                    avg_expanded, avg_cost,
                    green, runtime_max[key] + 0, reset
            }
        }
    ' "$RESULTS_FILE" | sort
}

print_overall_summary() {
    printf '\n%sOverall summary%s\n' "$PURPLE" "$RESET"
    awk -F'\t' '
        NR == 1 { next }
        {
            runs++
            if ($6 == "found" || $6 == "completed") {
                successes++
            }
            if ($7 != "NA" && $7 != "") {
                runtime_sum += $7
                runtime_count++
                if ($7 > runtime_max) {
                    runtime_max = $7
                }
            }
            if ($8 != "NA" && $8 != "") {
                expanded_sum += $8
                expanded_count++
            }
            if ($9 != "NA" && $9 != "") {
                cost_sum += $9
                cost_count++
            }
        }
        END {
            printf "  runs: %d\n", runs
            printf "  successes: %d\n", successes
            if (runtime_count > 0) {
                printf "  avg runtime (s): %.6f\n", runtime_sum / runtime_count
                printf "  max runtime (s): %.6f\n", runtime_max
            }
            if (expanded_count > 0) {
                printf "  avg expanded nodes: %.2f\n", expanded_sum / expanded_count
            }
            if (cost_count > 0) {
                printf "  avg path cost: %.2f\n", cost_sum / cost_count
            }
        }
    ' "$RESULTS_FILE" | sed -E "s/(avg runtime \(s\): )([0-9]+\.[0-9]+)/\1${GREEN}\2${RESET}/; s/(max runtime \(s\): )([0-9]+\.[0-9]+)/\1${GREEN}\2${RESET}/"
}

parse_args "$@"

BINARY_REL="$(normalize_rel_path "$BINARY")"
MAP_REL="$(normalize_rel_path "$MAP_PATH")"
SCEN_REL="$(normalize_rel_path "$SCEN_PATH")"

BINARY_PATH="$SCRIPT_DIR/$BINARY_REL"
MAP_FILE="$SCRIPT_DIR/$MAP_REL"
SCEN_FILE="$SCRIPT_DIR/$SCEN_REL"

assert_exists "Binary" "$BINARY_PATH"
if [[ "$RUN_MAP_CASE" -eq 1 ]]; then
    assert_exists "Map" "$MAP_FILE"
    assert_exists "Scenario file" "$SCEN_FILE"
fi

mkdir -p "$SCRIPT_DIR/benchmark_results"
if [[ -z "$OUTPUT_PATH" ]]; then
    OUTPUT_PATH="benchmark_results/demo_results_$(date +%Y%m%d_%H%M%S)_$$.tsv"
fi
OUTPUT_REL="$(normalize_rel_path "$OUTPUT_PATH")"
RESULTS_FILE="$SCRIPT_DIR/$OUTPUT_REL"
RESULTS_DIR="$(dirname "$RESULTS_FILE")"
mkdir -p "$RESULTS_DIR"

printf 'suite\tcase\tsize\tgrid_type\trepeat\tstatus\truntime_seconds\texpanded_nodes\tpath_cost\n' >"$RESULTS_FILE"

printf '%sDemo configuration%s\n' "$PURPLE" "$RESET"
printf '  %sbinary:%s %s\n' "$BLUE" "$RESET" "$BINARY_REL"
printf '  %sobstacle rate:%s %s\n' "$BLUE" "$RESET" "$OBSTACLE_RATE"
printf '  %srepeats:%s %s\n' "$BLUE" "$RESET" "$REPEATS"
printf '  %scases:%s %s\n' "$BLUE" "$RESET" "$CASES"
printf '  %sresults file:%s %s\n' "$BLUE" "$RESET" "$OUTPUT_REL"

if [[ "$RUN_MAP_CASE" -eq 1 ]]; then
    read -r START_X START_Y GOAL_X GOAL_Y < <(get_scenario_coords "$SCEN_FILE" "$SCENARIO_INDEX")
    run_and_record \
        "movingai" "arena_scen_${SCENARIO_INDEX}" "map" "movingai" "1" \
        "$BINARY_PATH" \
        --map "$MAP_FILE" \
        --start-x "$START_X" \
        --start-y "$START_Y" \
        --goal-x "$GOAL_X" \
        --goal-y "$GOAL_Y" \
        --no-image
fi

IFS=',' read -r -a CASE_LIST <<<"$CASES"
for case_spec in "${CASE_LIST[@]}"; do
    size="${case_spec%%:*}"
    grid_type="${case_spec#*:}"
    if [[ -z "$size" || -z "$grid_type" || "$size" == "$grid_type" ]]; then
        printf '%sInvalid case specification:%s %s\n' "$RED" "$RESET" "$case_spec" >&2
        exit 1
    fi
    for ((repeat_index = 1; repeat_index <= REPEATS; repeat_index++)); do
        run_and_record \
            "procedural" "${size}_${grid_type}" "$size" "$grid_type" "$repeat_index" \
            "$BINARY_PATH" "$size" "$OBSTACLE_RATE" "$grid_type"
    done
done

print_detailed_results
print_summary_by_field 4 "Summary by grid type"
print_summary_by_field 3 "Summary by size"
print_overall_summary

printf '\n%sSaved raw results to%s %s\n' "$GREEN" "$RESET" "$RESULTS_FILE"
