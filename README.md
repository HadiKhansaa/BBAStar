# Bucket_Astar

Implementation accompanying the paper **Parallel Bidirectional A* Search for GPU-Accelerated Pathfinding**. The repository contains CUDA implementations of both bidirectional and unidirectional bucket-based A* for grid pathfinding, plus a shell-based demo that runs a small experiment suite and summarizes the results.

## Paper

- Title: *Parallel Bidirectional A* Search for GPU-Accelerated Pathfinding*
- Authors: Hadi Al Khansa, Juan Gomez-Luna, Amer E. Mouawad, Izzat El Hajj
- DOI: <https://doi.org/10.1145/3797905.3805620>

## Repository Layout

- `src/`: CUDA implementations and executable entrypoints.
- `include/`: CUDA headers, shared constants, and implementation-specific types.
- `CPU/`: reference CPU baselines kept for comparison.
- `scripts/`: benchmark utilities and legacy exploratory helpers.
- `data/maps/`: bundled sample MovingAI map and scenario used by the demo.
- `data/generated/`: bundled compressed-grid sample for procedural/generated-grid testing.

## Requirements

- Windows with an NVIDIA GPU that supports cooperative kernel launch
- CUDA Toolkit with `nvcc` in `PATH`
- MSVC with `cl.exe` in `PATH`
- GNU Make-compatible `make`
- Git Bash or another Bash-compatible shell for `demo.sh`

The Makefile currently targets `sm_89` by default. If your GPU uses a different architecture, override it when building:

```powershell
make CUDA_ARCH=sm_86
```

## Build

From the repository root:

```powershell
make
```

This produces:

```text
bin/astar_bidirectional.exe
```

To build the unidirectional CUDA implementation:

```powershell
make unidirectional
```

This produces:

```text
bin/astar_unidirectional.exe
```

Other supported targets:

```powershell
make debug
make unidirectional_debug
make clean
make help
```

## Run The Demo

The recommended entrypoint is the top-level shell demo. It runs:

- one bundled MovingAI `.map` reference case
- multiple larger procedural grid cases across several grid types
- a summary table with runtime, expanded nodes, path cost, and aggregate statistics

1. Build the CUDA binary.

   ```powershell
   make
   ```

2. Run the demo.

   From Git Bash:

   ```bash
   ./demo.sh
   ```

   From PowerShell with Git Bash installed in the default location:

   ```powershell
   & "C:\Program Files\Git\bin\bash.exe" ./demo.sh
   ```

   To run the shell demo with the unidirectional implementation, start with a smaller procedural case set:

   ```bash
   ./demo.sh --binary bin/astar_unidirectional.exe --cases 64:rectangle,64:zigzag,128:rectangle
   ```

3. Optional: increase repetitions to stabilize the summary statistics.

   ```bash
   ./demo.sh --repeats 3
   ```

4. Optional: customize the procedural cases.

   ```bash
   ./demo.sh --cases 512:random,512:maze,1024:rectangle --repeats 2
   ```

On success the script prints:

- every command it ran
- a detailed per-run results table
- summary statistics grouped by grid type
- summary statistics grouped by grid size
- an overall summary across the whole experiment suite

The raw results are also saved as a TSV file under `benchmark_results/`. By default the script removes `data/AstarPath.png` after each run to keep the repository clean; use `--keep-image` if you want to inspect the last rendered path.

The default larger procedural suite is tuned for the bidirectional CUDA binary. The extracted unidirectional kernel is still useful for comparison, but in practice it is more reliable on smaller procedural cases.

## Direct CLI Usage

The executable also supports direct invocation.

Bundled sample map:

```powershell
bin\astar_bidirectional.exe --map data\maps\arena.map --start-x 19 --start-y 26 --goal-x 19 --goal-y 29
```

The unidirectional binary supports the same `.map` benchmark interface:

```powershell
bin\astar_unidirectional.exe --map data\maps\arena.map --start-x 19 --start-y 26 --goal-x 19 --goal-y 29 --no-image
```

Procedural grids:

```powershell
bin\astar_bidirectional.exe [size [obstacle_rate [grid_type [compressed_grid_path]]]]
bin\astar_unidirectional.exe [size [obstacle_rate [grid_type [compressed_grid_path]]]]
```

Supported `grid_type` values:

- `random`
- `maze`
- `blockCenter`
- `zigzag`
- `rectangle`

Bundled generated-grid sample with the bidirectional binary:

```powershell
bin\astar_bidirectional.exe 64 20 rectangle data\generated\demo_rectangle_64.bin
```

The same bundled sample also works with the unidirectional binary:

```powershell
bin\astar_unidirectional.exe 64 20 rectangle data\generated\demo_rectangle_64.bin
```

Both binaries accept the same positional compressed-grid path, so you can compare them directly on the same generated sample.

## Benchmarks

The repository intentionally ships only minimal bundled samples. To run larger `.map` experiments, place additional MovingAI `.map` and `.map.scen` files under `data/maps/` and use:

```powershell
python scripts\run_maps_benchmark.py --binary bin\astar_bidirectional.exe --maps-dir data\maps --limit-scenarios 10
```

To benchmark the unidirectional binary instead:

```powershell
python scripts\run_maps_benchmark.py --binary bin\astar_unidirectional.exe --maps-dir data\maps --limit-scenarios 10
```

Results are written to `benchmark_results/`.

## Notes

- The CUDA implementation relies on cooperative groups and grid-wide synchronization.
- The CPU sources are provided as reference baselines, but the primary supported workflows in this release are the two CUDA binaries above.
- `demo.sh` targets Bash. On Windows, Git Bash works well and can still launch the compiled `.exe` binaries directly.
