#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <string>

#include "constants.cuh"
#include "grid_generation.cuh"
#include "unidirectional_astar.cuh"
#include "utils.cuh"

struct BenchmarkOptions {
    bool enabled = false;
    bool saveImage = true;
    std::string mapPath;
    int startX = 0;
    int startY = 0;
    int goalX = -1;
    int goalY = -1;
};

enum class CliParseResult {
    kOk,
    kHelp,
    kError,
};

static void printUsage() {
    std::cout
        << "Usage:\n"
        << "  astar_unidirectional [size [obstacle_rate [grid_type [compressed_grid_path]]]]\n"
        << "  astar_unidirectional --map <map_path> --start-x <x> --start-y <y> "
           "--goal-x <x> --goal-y <y> [--no-image]\n";
}

static bool parseIntArg(const char *value, int &result) {
    char *end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        return false;
    }
    result = static_cast<int>(parsed);
    return true;
}

static CliParseResult parseBenchmarkOptions(int argc, char **argv, BenchmarkOptions &options) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--map" || arg == "--start-x" || arg == "--start-y" ||
            arg == "--goal-x" || arg == "--goal-y") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << std::endl;
                return CliParseResult::kError;
            }

            const char *value = argv[++i];
            options.enabled = true;
            if (arg == "--map") {
                options.mapPath = value;
            } else if (arg == "--start-x") {
                if (!parseIntArg(value, options.startX)) {
                    std::cerr << "Invalid integer for --start-x: " << value << std::endl;
                    return CliParseResult::kError;
                }
            } else if (arg == "--start-y") {
                if (!parseIntArg(value, options.startY)) {
                    std::cerr << "Invalid integer for --start-y: " << value << std::endl;
                    return CliParseResult::kError;
                }
            } else if (arg == "--goal-x") {
                if (!parseIntArg(value, options.goalX)) {
                    std::cerr << "Invalid integer for --goal-x: " << value << std::endl;
                    return CliParseResult::kError;
                }
            } else if (arg == "--goal-y") {
                if (!parseIntArg(value, options.goalY)) {
                    std::cerr << "Invalid integer for --goal-y: " << value << std::endl;
                    return CliParseResult::kError;
                }
            }
        } else if (arg == "--no-image") {
            options.enabled = true;
            options.saveImage = false;
        } else if (arg == "--help" || arg == "-h") {
            printUsage();
            return CliParseResult::kHelp;
        }
    }

    if (!options.enabled) {
        return CliParseResult::kOk;
    }

    if (options.mapPath.empty() || options.goalX < 0 || options.goalY < 0) {
        std::cerr << "Benchmark mode requires --map, --start-x, --start-y, --goal-x, and --goal-y."
                  << std::endl;
        return CliParseResult::kError;
    }

    return CliParseResult::kOk;
}

static void parseProceduralArguments(
    int argc,
    char **argv,
    int &width,
    int &height,
    float &obstacleRate,
    std::string &gridType,
    std::string &gridPath) {
    if (argc >= 2) {
        height = std::atoi(argv[1]);
        width = std::atoi(argv[1]);
    }
    if (argc >= 3) {
        obstacleRate = std::atoi(argv[2]) / 100.0f;
    }
    if (argc >= 4) {
        gridType = argv[3];
    }
    if (argc >= 5) {
        gridPath = argv[4];
    }
}

static float computePathCost(const int *path, int pathLength, int width) {
    float totalCost = 0.0f;
    for (int i = pathLength - 1; i > 0; --i) {
        int currentNodeId = path[i];
        int nextNodeId = path[i - 1];
        int xCurrent = currentNodeId % width;
        int yCurrent = currentNodeId / width;
        int xNext = nextNodeId % width;
        int yNext = nextNodeId / width;
        int dx = std::abs(xNext - xCurrent);
        int dy = std::abs(yNext - yCurrent);
        bool isDiagonal = (dx + dy == 2);
        float movementCost = isDiagonal ? std::sqrt(2.0f) : 1.0f;
        totalCost += movementCost;
    }
    return totalCost;
}

int main(int argc, char **argv) {
    int width = 1001;
    int height = 1001;
    float obstacleRate = 0.2f;
    std::string gridType;
    std::string gridPath;
    BenchmarkOptions benchmarkOptions;

    CliParseResult parseResult = parseBenchmarkOptions(argc, argv, benchmarkOptions);
    if (parseResult == CliParseResult::kHelp) {
        return EXIT_SUCCESS;
    }
    if (parseResult == CliParseResult::kError) {
        return EXIT_FAILURE;
    }

    if (!benchmarkOptions.enabled) {
        parseProceduralArguments(argc, argv, width, height, obstacleRate, gridType, gridPath);
    }

    int startNodeId = 0;
    int goalNodeId = width * height - 1;
    int *h_grid = nullptr;
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    if (benchmarkOptions.enabled) {
        if (!loadMovingAiMapFromFile(h_grid, width, height, benchmarkOptions.mapPath)) {
            return EXIT_FAILURE;
        }
        if (benchmarkOptions.startX < 0 || benchmarkOptions.startX >= width ||
            benchmarkOptions.goalX < 0 || benchmarkOptions.goalX >= width ||
            benchmarkOptions.startY < 0 || benchmarkOptions.startY >= height ||
            benchmarkOptions.goalY < 0 || benchmarkOptions.goalY >= height) {
            std::cerr << "Scenario coordinates are out of bounds for map: "
                      << benchmarkOptions.mapPath << std::endl;
            std::free(h_grid);
            return EXIT_FAILURE;
        }
        startNodeId = benchmarkOptions.startY * width + benchmarkOptions.startX;
        goalNodeId = benchmarkOptions.goalY * width + benchmarkOptions.goalX;
    } else {
        int gridSize = width * height;
        h_grid = static_cast<int *>(std::malloc(gridSize * sizeof(int)));
        if (!h_grid) {
            std::fprintf(stderr, "Failed to allocate host memory for grid\n");
            return EXIT_FAILURE;
        }
    }

    if (!benchmarkOptions.enabled && !gridPath.empty()) {
        std::free(h_grid);
        h_grid = nullptr;
        if (!loadCompressedGridFromFile(h_grid, width, height, gridPath)) {
            return EXIT_FAILURE;
        }
    } else if (!benchmarkOptions.enabled && gridType == "random") {
        applyRandomObstacles(h_grid, width, height, obstacleRate);
    } else if (!benchmarkOptions.enabled && gridType == "maze") {
        createMaze(h_grid, height);
    } else if (!benchmarkOptions.enabled && gridType == "blockCenter") {
        createConcentratedObstacles(h_grid, height);
    } else if (!benchmarkOptions.enabled && gridType == "zigzag") {
        createZigzagPattern(h_grid, width, height);
    } else if (!benchmarkOptions.enabled && gridType == "rectangle") {
        applyRandomRectangleObstacles(h_grid, width, height, obstacleRate);
    } else if (!benchmarkOptions.enabled) {
        applyRandomObstacles(h_grid, width, height, obstacleRate);
    }

    int gridSize = width * height;
    h_grid[startNodeId] = PASSABLE;
    h_grid[goalNodeId] = PASSABLE;

    int *d_grid = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_grid), gridSize * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_grid, h_grid, gridSize * sizeof(int), cudaMemcpyHostToDevice));

    Node *d_nodes = nullptr;
    int *d_path = nullptr;
    int *d_pathLength = nullptr;
    bool *d_found = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_nodes), gridSize * sizeof(Node)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_path), gridSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_pathLength), sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_found), sizeof(bool)));

    dim3 threadsPerBlockInit(256);
    dim3 blocksPerGridInit((gridSize + threadsPerBlockInit.x - 1) / threadsPerBlockInit.x);
    initializeNodes<<<blocksPerGridInit, threadsPerBlockInit>>>(d_nodes, width, height);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(d_path, -1, gridSize * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_pathLength, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_found, 0, sizeof(bool)));

    int *d_openListBins = nullptr;
    int *d_binCounts = nullptr;
    unsigned long long *d_binBitMask = nullptr;
    int *d_totalExpandedNodes = nullptr;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_openListBins),
        UNIDIRECTIONAL_MAX_BINS * UNIDIRECTIONAL_MAX_BIN_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_binCounts),
        UNIDIRECTIONAL_MAX_BINS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_binBitMask),
        ((UNIDIRECTIONAL_MAX_BINS + 63) / 64) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_totalExpandedNodes), sizeof(int)));

    CUDA_CHECK(cudaMemset(
        d_openListBins,
        -1,
        UNIDIRECTIONAL_MAX_BINS * UNIDIRECTIONAL_MAX_BIN_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_binCounts, 0, UNIDIRECTIONAL_MAX_BINS * sizeof(int)));
    CUDA_CHECK(cudaMemset(
        d_binBitMask,
        0,
        ((UNIDIRECTIONAL_MAX_BINS + 63) / 64) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_totalExpandedNodes, 0, sizeof(int)));

    Node h_startNode{};
    h_startNode.id = startNodeId;
    h_startNode.g = 0;
    h_startNode.h = static_cast<int>(unidirectionalHeuristic(startNodeId, goalNodeId, width));
    h_startNode.f = h_startNode.g + h_startNode.h;
    h_startNode.parent = -1;
    CUDA_CHECK(cudaMemcpy(&d_nodes[startNodeId], &h_startNode, sizeof(Node), cudaMemcpyHostToDevice));

    int startBin = 0;
    int one = 1;
    unsigned long long startMask = 1ULL;
    CUDA_CHECK(cudaMemcpy(
        &d_openListBins[startBin * UNIDIRECTIONAL_MAX_BIN_SIZE],
        &startNodeId,
        sizeof(int),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&d_binCounts[startBin], &one, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&d_binBitMask[0], &startMask, sizeof(unsigned long long), cudaMemcpyHostToDevice));

    unsigned int minFValue = unidirectionalHeuristic(startNodeId, goalNodeId, width);

    auto startTime = std::chrono::high_resolution_clock::now();

    int threadsPerBlock = 256;
    int numBlocks = (UNIDIRECTIONAL_TOTAL_THREADS + threadsPerBlock - 1) / threadsPerBlock;
    dim3 gridDim(numBlocks);
    dim3 blockDim(threadsPerBlock);
    int frontierSize = UNIDIRECTIONAL_FRONTIER_SIZE;

    void *kernelArgs[] = {
        static_cast<void *>(&d_grid),
        static_cast<void *>(&width),
        static_cast<void *>(&height),
        static_cast<void *>(&goalNodeId),
        static_cast<void *>(&minFValue),
        static_cast<void *>(&d_nodes),
        static_cast<void *>(&d_openListBins),
        static_cast<void *>(&d_binCounts),
        static_cast<void *>(&d_binBitMask),
        static_cast<void *>(&d_found),
        static_cast<void *>(&d_path),
        static_cast<void *>(&d_pathLength),
        static_cast<void *>(&frontierSize),
        static_cast<void *>(&d_totalExpandedNodes),
    };

    CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void *>(aStarMultipleBucketsShared),
        gridDim,
        blockDim,
        kernelArgs,
        0));
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedSeconds = endTime - startTime;

    int h_pathLength = 0;
    int *h_path = static_cast<int *>(std::malloc(gridSize * sizeof(int)));
    if (!h_path) {
        std::fprintf(stderr, "Failed to allocate host memory for the path\n");
        return EXIT_FAILURE;
    }

    bool h_found = false;
    int h_totalExpandedNodes = 0;
    CUDA_CHECK(cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        &h_totalExpandedNodes,
        d_totalExpandedNodes,
        sizeof(int),
        cudaMemcpyDeviceToHost));

    if (h_found) {
        CUDA_CHECK(cudaMemcpy(&h_pathLength, d_pathLength, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_path, d_path, h_pathLength * sizeof(int), cudaMemcpyDeviceToHost));
        float totalCost = computePathCost(h_path, h_pathLength, width);

        std::cout << GREEN << "Execution time (Unidirectional A* kernel): "
                  << elapsedSeconds.count() << " seconds" << RESET << std::endl;

        if (benchmarkOptions.saveImage) {
            visualizeAStarPathOnGrid(
                h_grid,
                width,
                height,
                h_path,
                h_pathLength,
                nullptr,
                0,
                "./data/AstarPath.png");
        }

        std::cout << BLUE << "Total number of expanded nodes: "
                  << h_totalExpandedNodes << RESET << std::endl;
        if (benchmarkOptions.enabled) {
            std::cout << "BENCHMARK_RESULT status=found"
                      << " runtime_seconds=" << elapsedSeconds.count()
                      << " expanded_nodes=" << h_totalExpandedNodes
                      << " path_cost=" << totalCost
                      << " width=" << width
                      << " height=" << height
                      << " start_x=" << benchmarkOptions.startX
                      << " start_y=" << benchmarkOptions.startY
                      << " goal_x=" << benchmarkOptions.goalX
                      << " goal_y=" << benchmarkOptions.goalY
                      << std::endl;
        }
    } else {
        std::cout << "Path not found." << std::endl;
        std::cout << GREEN << "Execution time (Unidirectional A* kernel): "
                  << elapsedSeconds.count() << " seconds" << RESET << std::endl;
        std::cout << BLUE << "Total number of expanded nodes: "
                  << h_totalExpandedNodes << RESET << std::endl;
        if (benchmarkOptions.enabled) {
            std::cout << "BENCHMARK_RESULT status=not_found"
                      << " runtime_seconds=" << elapsedSeconds.count()
                      << " expanded_nodes=" << h_totalExpandedNodes
                      << " path_cost=-1"
                      << " width=" << width
                      << " height=" << height
                      << " start_x=" << benchmarkOptions.startX
                      << " start_y=" << benchmarkOptions.startY
                      << " goal_x=" << benchmarkOptions.goalX
                      << " goal_y=" << benchmarkOptions.goalY
                      << std::endl;
        }
    }

    CUDA_CHECK(cudaFree(d_nodes));
    CUDA_CHECK(cudaFree(d_path));
    CUDA_CHECK(cudaFree(d_pathLength));
    CUDA_CHECK(cudaFree(d_found));
    CUDA_CHECK(cudaFree(d_grid));
    CUDA_CHECK(cudaFree(d_openListBins));
    CUDA_CHECK(cudaFree(d_binCounts));
    CUDA_CHECK(cudaFree(d_binBitMask));
    CUDA_CHECK(cudaFree(d_totalExpandedNodes));

    std::free(h_grid);
    std::free(h_path);

    return 0;
}
