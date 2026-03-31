#pragma once

#include <cuda_runtime.h>

#include "constants.cuh"

struct Node {
    int id;
    int g;
    int h;
    int f;
    int parent;
};

static constexpr int UNIDIRECTIONAL_BUCKET_RANGE = 3000;
static constexpr int UNIDIRECTIONAL_MAX_BINS = 3000;
static constexpr int UNIDIRECTIONAL_MAX_BIN_SIZE = 30000;
static constexpr int UNIDIRECTIONAL_SHARED_BUCKET_RANGE = 15;
static constexpr int UNIDIRECTIONAL_SHARED_MAX_BIN_SIZE = 100;
static constexpr int UNIDIRECTIONAL_FRONTIER_SIZE = 512;
static constexpr int UNIDIRECTIONAL_TOTAL_THREADS = 30000;

__host__ __device__ inline unsigned int unidirectionalHeuristic(
    int currentNodeId,
    int goalNodeId,
    int width) {
    int xCurrent = currentNodeId % width;
    int yCurrent = currentNodeId / width;
    int xGoal = goalNodeId % width;
    int yGoal = goalNodeId / width;

    int dx = abs(xCurrent - xGoal);
    int dy = abs(yCurrent - yGoal);
    return DIAGONAL_COST * min(dx, dy) + SCALE_FACTOR * abs(dx - dy);
}

__global__ void initializeNodes(Node *nodes, int width, int height);

__global__ void aStarMultipleBucketsShared(
    int *grid,
    int width,
    int height,
    int goalNodeId,
    unsigned int minFValue,
    Node *nodes,
    int *openListBins,
    int *binCounts,
    unsigned long long *binBitMask,
    bool *found,
    int *path,
    int *pathLength,
    int frontierSize,
    int *totalExpandedNodes);
