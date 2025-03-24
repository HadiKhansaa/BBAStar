#pragma once
#include "grid_generation.hpp"
#include "bidirectional_astar.cuh"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <iostream>
#include <cmath>

#define CUDA_CHECK(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n",         \
                    cudaGetErrorString(err), err, __FILE__, __LINE__);        \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

#define CUDA_KERNEL_CHECK()                                                   \
    {                                                                         \
        cudaError_t err = cudaGetLastError();                                 \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Kernel Error: %s (err_num=%d) at %s:%d\n",    \
                    cudaGetErrorString(err), err, __FILE__, __LINE__);        \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

// Main function for bidirectional A* using the bidirectional kernel
int main(int argc, char** argv) {
    // --- Grid initialization ---
    int width = 1001;  
    int height = 1001;
    float obstacleRate = 0.2; // default 20%
    std::string gridType = "";
    std::string gridPath = "";

    if(argc == 2) {
        height = atoi(argv[1]);
        width  = atoi(argv[1]);
    } else if(argc == 3) {
        height = atoi(argv[1]);
        width  = atoi(argv[1]);
        obstacleRate = atoi(argv[2]) / 100.0;
    } else if(argc == 4) {
        height = atoi(argv[1]);
        width  = atoi(argv[1]);
        obstacleRate = atoi(argv[2]) / 100.0;
        gridType = argv[3];
    } else if(argc == 5) {
        height = atoi(argv[1]);
        width  = atoi(argv[1]);
        obstacleRate = atoi(argv[2]) / 100.0;
        gridType = argv[3];
        gridPath = argv[4];
    }
    int gridSize = width * height;

    // Define start (forward) and goal (backward) nodes.
    int startNodeId = 0;                 // Top-left corner (forward start)
    int goalNodeId  = width * height - 1;  // Bottom-right corner (backward start)

    // Allocate and initialize the grid on host.
    int *h_grid = (int *)malloc(gridSize * sizeof(int));
    if (!h_grid) {
        fprintf(stderr, "Failed to allocate host memory for grid\n");
        exit(EXIT_FAILURE);
    }
    srand(time(NULL));
    if(gridPath != "")
        loadCompressedGridFromFile(h_grid, width, height, gridPath);
    else if(gridType == "random")
        applyRandomObstacles(h_grid, width, height, obstacleRate);
    else if(gridType == "maze")
        createMaze(h_grid, height);
    else if(gridType == "blockCenter")
        createConcentratedObstacles(h_grid, height);
    else if(gridType == "zigzag")
        createZigzagPattern(h_grid, width, height);
    else if(gridType == "rectangle")
        applyRandomRectangleObstacles(h_grid, width, height, obstacleRate);
    else
        applyRandomObstacles(h_grid, width, height, obstacleRate);

    // Ensure start and goal are free.
    h_grid[startNodeId] = 0;
    h_grid[goalNodeId] = 0;

    auto startCopyTime = std::chrono::high_resolution_clock::now();

    // --- Device memory allocations ---
    int *d_grid;
    CUDA_CHECK(cudaMalloc((void **)&d_grid, gridSize * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_grid, h_grid, gridSize * sizeof(int), cudaMemcpyHostToDevice));

    // Use BiNode instead of Node for bidirectional search.
    BiNode *d_nodes;
    CUDA_CHECK(cudaMalloc((void **)&d_nodes, gridSize * sizeof(BiNode)));

    int *d_path;
    int *d_pathLength;
    bool *d_found;
    CUDA_CHECK(cudaMalloc((void **)&d_path, gridSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_pathLength, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_found, sizeof(bool)));

    auto endCopyTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> copyTime = endCopyTime - startCopyTime;

    // --- Initialize nodes (set costs to INF, etc.) ---
    dim3 threadsPerBlockInit(256);
    dim3 blocksPerGridInit((gridSize + threadsPerBlockInit.x - 1) / threadsPerBlockInit.x);
    // Assume initializeBiNodes kernel sets forward and backward costs appropriately.
    initializeBiNodes<<<blocksPerGridInit, threadsPerBlockInit>>>(d_nodes, width, height);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(d_path, -1, gridSize * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_pathLength, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_found, 0, sizeof(bool)));

    // --- Allocate open list bins and related arrays for forward search ---
    int *d_forward_openListBins, *d_forward_binCounts;
    unsigned long long *d_forward_binBitMask;
    int *d_forward_expansionBuffers, *d_forward_expansionCounts;
    int binBitMaskSize = (MAX_BINS + 63) / 64;
    CUDA_CHECK(cudaMalloc((void **)&d_forward_openListBins, MAX_BINS * MAX_BIN_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_forward_binCounts, MAX_BINS * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_forward_binBitMask, binBitMaskSize * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc((void **)&d_forward_expansionBuffers, MAX_BINS * MAX_BIN_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_forward_expansionCounts, MAX_BINS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_forward_openListBins, -1, MAX_BINS * MAX_BIN_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_forward_binCounts, 0, MAX_BINS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_forward_binBitMask, 0, binBitMaskSize * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_forward_expansionCounts, 0, MAX_BINS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_forward_expansionBuffers, -1, MAX_BINS * MAX_BIN_SIZE * sizeof(int)));

    // --- Allocate open list bins for backward search ---
    int *d_backward_openListBins, *d_backward_binCounts;
    unsigned long long *d_backward_binBitMask;
    int *d_backward_expansionBuffers, *d_backward_expansionCounts;
    CUDA_CHECK(cudaMalloc((void **)&d_backward_openListBins, MAX_BINS * MAX_BIN_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_backward_binCounts, MAX_BINS * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_backward_binBitMask, binBitMaskSize * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc((void **)&d_backward_expansionBuffers, MAX_BINS * MAX_BIN_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_backward_expansionCounts, MAX_BINS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_backward_openListBins, -1, MAX_BINS * MAX_BIN_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_backward_binCounts, 0, MAX_BINS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_backward_binBitMask, 0, binBitMaskSize * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_backward_expansionCounts, 0, MAX_BINS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_backward_expansionBuffers, -1, MAX_BINS * MAX_BIN_SIZE * sizeof(int)));

    // --- Allocate first/last non-empty mask for each direction ---
    int *d_forward_firstNonEmptyMask, *d_forward_lastNonEmptyMask;
    int *d_backward_firstNonEmptyMask, *d_backward_lastNonEmptyMask;
    CUDA_CHECK(cudaMalloc((void**)&d_forward_firstNonEmptyMask, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_forward_lastNonEmptyMask, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_backward_firstNonEmptyMask, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_backward_lastNonEmptyMask, sizeof(int)));

    // --- Copy BIN_SIZE constant to device ---
    float minFValue = height * 1414;
    // float BIN_SIZE = 3000; // chosen for testing
    // CUDA_CHECK(cudaMemcpyToSymbol(BIN_SIZE_DEVICE, &BIN_SIZE, sizeof(float)));

    // --- Initialize forward search start node ---
    BiNode h_startNode;
    h_startNode.id = startNodeId;
    h_startNode.g_forward = 0;
    h_startNode.h_forward = heuristic(startNodeId, goalNodeId, width);
    h_startNode.f_forward = h_startNode.g_forward + h_startNode.h_forward;
    h_startNode.parent_forward = -1;
    // Set backward fields to a high value.
    h_startNode.g_backward = INT_MAX;
    h_startNode.h_backward = 0;
    h_startNode.f_backward = INT_MAX;
    h_startNode.parent_backward = -1;
    CUDA_CHECK(cudaMemcpy(&d_nodes[startNodeId], &h_startNode, sizeof(BiNode), cudaMemcpyHostToDevice));

    int startBin = (int)((h_startNode.f_forward - minFValue) / BUCKET_F_RANGE);
    startBin = std::min(startBin, MAX_BINS - 1);
    startBin = std::max(startBin, 0); // for safety

    int *h_forward_binCounts = (int*)malloc(MAX_BINS * sizeof(int));
    memset(h_forward_binCounts, 0, MAX_BINS * sizeof(int));
    int *h_forward_openListBins = (int*)malloc(MAX_BINS * MAX_BIN_SIZE * sizeof(int));
    memset(h_forward_openListBins, -1, MAX_BINS * MAX_BIN_SIZE * sizeof(int));
    h_forward_binCounts[startBin] = 1;
    h_forward_openListBins[startBin * MAX_BIN_SIZE] = startNodeId;

    unsigned long long *h_forward_binBitMask = (unsigned long long*)malloc(binBitMaskSize * sizeof(unsigned long long));
    memset(h_forward_binBitMask, 0, binBitMaskSize * sizeof(unsigned long long));
    int forwardMaskIndex = startBin / 64;
    unsigned long long forwardMask = 1ULL << (startBin % 64);
    h_forward_binBitMask[forwardMaskIndex] |= forwardMask;

    CUDA_CHECK(cudaMemcpy(d_forward_openListBins, h_forward_openListBins, MAX_BINS * MAX_BIN_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_forward_binCounts, h_forward_binCounts, MAX_BINS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_forward_binBitMask, h_forward_binBitMask, binBitMaskSize * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_forward_firstNonEmptyMask, &forwardMaskIndex, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_forward_lastNonEmptyMask, &forwardMaskIndex, sizeof(int), cudaMemcpyHostToDevice));

    // --- Initialize backward search goal node ---
    BiNode h_goalNode;
    h_goalNode.id = goalNodeId;
    h_goalNode.g_backward = 0;
    h_goalNode.h_backward = heuristic(goalNodeId, startNodeId, width);
    h_goalNode.f_backward = h_goalNode.g_backward + h_goalNode.h_backward;
    h_goalNode.parent_backward = -1;
    // Set forward fields to INF.
    h_goalNode.g_forward = INT_MAX;
    h_goalNode.h_forward = 0;
    h_goalNode.f_forward = INT_MAX;
    h_goalNode.parent_forward = -1;
    CUDA_CHECK(cudaMemcpy(&d_nodes[goalNodeId], &h_goalNode, sizeof(BiNode), cudaMemcpyHostToDevice));

    int goalBin = (int)((h_goalNode.f_backward - minFValue) / BUCKET_F_RANGE);
    goalBin = std::min(goalBin, MAX_BINS - 1);
    goalBin = std::max(goalBin, 0); // for safety

    int *h_backward_binCounts = (int*)malloc(MAX_BINS * sizeof(int));
    memset(h_backward_binCounts, 0, MAX_BINS * sizeof(int));
    int *h_backward_openListBins = (int*)malloc(MAX_BINS * MAX_BIN_SIZE * sizeof(int));
    memset(h_backward_openListBins, -1, MAX_BINS * MAX_BIN_SIZE * sizeof(int));
    h_backward_binCounts[goalBin] = 1;
    h_backward_openListBins[goalBin * MAX_BIN_SIZE] = goalNodeId;

    unsigned long long *h_backward_binBitMask = (unsigned long long*)malloc(binBitMaskSize * sizeof(unsigned long long));
    memset(h_backward_binBitMask, 0, binBitMaskSize * sizeof(unsigned long long));
    int backwardMaskIndex = goalBin / 64;
    unsigned long long backwardMask = 1ULL << (goalBin % 64);
    h_backward_binBitMask[backwardMaskIndex] |= backwardMask;

    CUDA_CHECK(cudaMemcpy(d_backward_openListBins, h_backward_openListBins, MAX_BINS * MAX_BIN_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_backward_binCounts, h_backward_binCounts, MAX_BINS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_backward_binBitMask, h_backward_binBitMask, binBitMaskSize * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_backward_firstNonEmptyMask, &backwardMaskIndex, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_backward_lastNonEmptyMask, &backwardMaskIndex, sizeof(int), cudaMemcpyHostToDevice));

    // --- Allocate variable to track total expanded nodes ---
    int *d_totalExpandedNodes;
    CUDA_CHECK(cudaMalloc((void**)&d_totalExpandedNodes, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_totalExpandedNodes, 0, sizeof(int)));

    // --- Initialize global control variables (assumed to be device symbols) ---
    int zeroInt = 0;
    bool falseBool = false;
    // CUDA_CHECK(cudaMemcpyToSymbol(d_currentBin, &zeroInt, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_done_forward, &falseBool, sizeof(bool)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_done_backward, &falseBool, sizeof(bool)));
    // CUDA_CHECK(cudaMemcpyToSymbol(d_localFound, &falseBool, sizeof(bool)));

    // --- Initialize global best cost for bidirectional search ---
    int h_globalBestCost = INT_MAX; // an arbitrarily high value
    CUDA_CHECK(cudaMemcpyToSymbol(globalBestCost, &h_globalBestCost, sizeof(int)));

    // --- Initialize global best node for bidirectional search ---
    BiNode h_globalBestNode;
    h_globalBestNode.id = -1;
    h_globalBestNode.g_forward = INT_MAX;
    h_globalBestNode.h_forward = 0;
    h_globalBestNode.f_forward = INT_MAX;
    h_globalBestNode.parent_forward = -1;
    h_globalBestNode.g_backward = INT_MAX;
    h_globalBestNode.h_backward = 0;
    h_globalBestNode.f_backward = INT_MAX;
    h_globalBestNode.parent_backward = -1;
    CUDA_CHECK(cudaMemcpyToSymbol(globalBestNode, &h_globalBestNode, sizeof(BiNode)));

    // --- Timing the bidirectional A* execution ---
    auto startTime = std::chrono::high_resolution_clock::now();

    // --- Set grid/block dimensions for kernel launch ---
    int frontierSize = 256;
    int threadsPerBlock = 256;
    int totalThreadsKernel = frontierSize * 16;
    // int totalThreadsKernel = 30000;
    int numBlocks = (totalThreadsKernel + threadsPerBlock - 1) / threadsPerBlock;
    dim3 gridDim(numBlocks);
    dim3 blockDim(threadsPerBlock);

    // --- Create streams for concurrent kernel execution ---
    cudaStream_t forwardStream, backwardStream;
    CUDA_CHECK(cudaStreamCreate(&forwardStream));
    CUDA_CHECK(cudaStreamCreate(&backwardStream));

    bool trueVariable = true;
    bool falseVariable = false;

    // --- Prepare kernel arguments for the bidirectional kernel ---
    // For the forward search, pass forward = true and target = goalNodeId.
    void *kernelArgsForward[] = {
        // (void *)&trueVariable,                        // forward flag
        (void *)&d_grid,
        (void *)&width,
        (void *)&height,
        (void *)&startNodeId,                 // target node for bacward search
        (void *)&goalNodeId,                  // target node for forward search
        (void *)&d_nodes,
        (void *)&d_forward_openListBins,
        (void *)&d_forward_binCounts,
        (void *)&d_forward_binBitMask,
        (void *)&d_forward_expansionBuffers,
        (void *)&d_forward_expansionCounts,
        (void *)&d_backward_openListBins,
        (void *)&d_backward_binCounts,
        (void *)&d_backward_binBitMask,
        (void *)&d_backward_expansionBuffers,
        (void *)&d_backward_expansionCounts,
        (void *)&d_found,
        (void *)&d_path,
        (void *)&d_pathLength,
        (void *)&binBitMaskSize,
        (void *)&frontierSize,
        (void *)&d_totalExpandedNodes,
        (void *)&d_forward_firstNonEmptyMask,
        (void *)&d_forward_lastNonEmptyMask
    };

    // --- Launch the forward and backward kernels concurrently ---
    CUDA_CHECK(cudaLaunchCooperativeKernel((void *)biAStarMultipleBucketsSingleKernel, gridDim, blockDim, kernelArgsForward, 0, forwardStream));
    CUDA_KERNEL_CHECK();

    // Synchronize both streams
    CUDA_CHECK(cudaStreamSynchronize(forwardStream));
    // CUDA_CHECK(cudaStreamSynchronize(backwardStream));
    CUDA_CHECK(cudaStreamDestroy(forwardStream));
    // CUDA_CHECK(cudaStreamDestroy(backwardStream));

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedSeconds = endTime - startTime;

    // --- Retrieve results ---
    int h_pathLength;
    int *h_path = (int *)malloc(gridSize * sizeof(int));
    if (!h_path) {
        fprintf(stderr, "Failed to allocate host memory for h_path\n");
        exit(EXIT_FAILURE);
    }
    bool h_found;
    CUDA_CHECK(cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost));

    if (h_found) {
        CUDA_CHECK(cudaMemcpy(&h_pathLength, d_pathLength, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_path, d_path, h_pathLength * sizeof(int), cudaMemcpyDeviceToHost));
        float totalCost = 0.0f;
        for (int i = h_pathLength - 1; i > 0; i--) {
            int currentNodeId = h_path[i];
            int nextNodeId = h_path[i - 1];
            int xCurrent = currentNodeId % width;
            int yCurrent = currentNodeId / width;
            int xNext = nextNodeId % width;
            int yNext = nextNodeId / width;
            int dx = abs(xNext - xCurrent);
            int dy = abs(yNext - yCurrent);
            bool isDiagonal = (dx + dy == 2);
            float movementCost = isDiagonal ? sqrtf(2.0f) : 1.0f;
            totalCost += movementCost;
        }
        std::cout << "Path found with length " << h_pathLength 
                  << " and total cost " << totalCost << std::endl;
        std::cout << "Execution time (Bidirectional A* kernel): " 
                  << elapsedSeconds.count() << " seconds" << std::endl;

        int h_totalExpandedNodes;
        cudaMemcpy(&h_totalExpandedNodes, d_totalExpandedNodes, sizeof(int), cudaMemcpyDeviceToHost);

        std::cout << "Total number of expanded nodes: " << h_totalExpandedNodes << std::endl;
    } else {
        std::cout << "Path not found." << std::endl;
        std::cout << "Execution time (Bidirectional A* kernel): " 
                  << elapsedSeconds.count() << " seconds" << std::endl;

        int h_totalExpandedNodes;
        cudaMemcpy(&h_totalExpandedNodes, d_totalExpandedNodes, sizeof(int), cudaMemcpyDeviceToHost);

        std::cout << "Total number of expanded nodes: " << h_totalExpandedNodes << std::endl;
    }

    // --- Cleanup device memory ---
    CUDA_CHECK(cudaFree(d_nodes));
    CUDA_CHECK(cudaFree(d_path));
    CUDA_CHECK(cudaFree(d_pathLength));
    CUDA_CHECK(cudaFree(d_found));
    CUDA_CHECK(cudaFree(d_grid));
    CUDA_CHECK(cudaFree(d_forward_openListBins));
    CUDA_CHECK(cudaFree(d_forward_binCounts));
    CUDA_CHECK(cudaFree(d_forward_binBitMask));
    CUDA_CHECK(cudaFree(d_forward_expansionBuffers));
    CUDA_CHECK(cudaFree(d_forward_expansionCounts));
    CUDA_CHECK(cudaFree(d_backward_openListBins));
    CUDA_CHECK(cudaFree(d_backward_binCounts));
    CUDA_CHECK(cudaFree(d_backward_binBitMask));
    CUDA_CHECK(cudaFree(d_backward_expansionBuffers));
    CUDA_CHECK(cudaFree(d_backward_expansionCounts));
    CUDA_CHECK(cudaFree(d_forward_firstNonEmptyMask));
    CUDA_CHECK(cudaFree(d_forward_lastNonEmptyMask));
    CUDA_CHECK(cudaFree(d_backward_firstNonEmptyMask));
    CUDA_CHECK(cudaFree(d_backward_lastNonEmptyMask));
    CUDA_CHECK(cudaFree(d_totalExpandedNodes));

    // --- Cleanup host memory ---
    free(h_grid);
    free(h_path);
    free(h_forward_binCounts);
    free(h_forward_openListBins);
    free(h_forward_binBitMask);
    free(h_backward_binCounts);
    free(h_backward_openListBins);
    free(h_backward_binBitMask);

    return 0;
}
