# Bucket_Astar

Implementation accompanying the paper **Parallel Bidirectional A* Search for GPU-Accelerated Pathfinding**. The repository contains a CUDA implementation of bidirectional A* for grid pathfinding using bucketed open lists, plus a small bundled sample so researchers can run the code immediately.

## Paper

- Title: *Parallel Bidirectional A* Search for GPU-Accelerated Pathfinding*
- Authors: Hadi Al Khansa, Juan Gomez-Luna, Amer E. Mouawad, Izzat El Hajj
- DOI: <https://doi.org/10.1145/3797905.3805620>

## Repository Layout

- `src/`: CUDA implementation and the main executable entrypoint.
- `include/`: CUDA headers and shared constants.
- `CPU/`: reference CPU baselines kept for comparison.
- `scripts/`: benchmark utilities and legacy exploratory helpers.
- `data/maps/`: bundled sample MovingAI map and scenario used by the demo.

## Requirements

- Windows with an NVIDIA GPU that supports cooperative kernel launch
- CUDA Toolkit with `nvcc` in `PATH`
- MSVC with `cl.exe` in `PATH`
- GNU Make-compatible `make`
- Python 3 for `demo.py`

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

Other supported targets:

```powershell
make debug
make clean
make help
```

## Run The Demo

The recommended entrypoint is the top-level demo script. It uses the bundled `arena` sample and the first scenario from `arena.map.scen`.

1. Build the CUDA binary.

   ```powershell
   make
   ```

2. Run the demo.

   ```powershell
   python demo.py
   ```

3. Optional: skip image generation for a faster smoke test.

   ```powershell
   python demo.py --no-image
   ```

On success the script prints:

- the exact executable command it ran
- pathfinding status
- kernel runtime
- expanded node count
- path cost

If image generation is enabled, the path visualization is written to `data/AstarPath.png`.

## Direct CLI Usage

The executable also supports direct invocation.

Bundled sample map:

```powershell
bin\astar_bidirectional.exe --map data\maps\arena.map --start-x 19 --start-y 26 --goal-x 19 --goal-y 29
```

Procedural grids:

```powershell
bin\astar_bidirectional.exe [size [obstacle_rate [grid_type [compressed_grid_path]]]]
```

Supported `grid_type` values:

- `random`
- `maze`
- `blockCenter`
- `zigzag`
- `rectangle`

## Benchmarks

The repository intentionally ships only a minimal sample dataset for the demo path. To run larger experiments, place additional MovingAI `.map` and `.map.scen` files under `data/maps/` and use:

```powershell
python scripts\run_maps_benchmark.py --binary bin\astar_bidirectional.exe --maps-dir data\maps --limit-scenarios 10
```

Results are written to `benchmark_results/`.

## Notes

- The CUDA implementation relies on cooperative groups and grid-wide synchronization.
- The CPU sources are provided as reference baselines, but the primary supported workflow in this release is the CUDA path above.
