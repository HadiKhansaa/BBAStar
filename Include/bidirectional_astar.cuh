
#pragma once
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <string.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

#include "astar_helper.cuh"

namespace cg = cooperative_groups;

enum ThreadAssignment {
    FORWARD = 1,
    BACKWARD = 0,
    UNASSIGNNED = -1
};

// Global stucture for inter-block communication
struct BidirectionalState {
    bool d_done_forward = false;
    bool d_done_backward = false;

    unsigned int globalBestCost = INT_MAX;
    BiNode globalBestNode = {-1, INT_MAX, 0, INT_MAX, INT_MAX, 0, INT_MAX, -1, -1 };

    // Logical (unwrapped) bucket numbers for iteration and early stopping
    int global_forward_logicalBucketStart = -1;
    int global_forward_bucketCount = 0;  // number of buckets in range
    int global_forward_totalElementsInRange = 0;

    int global_backward_logicalBucketStart = -1;
    int global_backward_bucketCount = 0;  // number of buckets in range
    int global_backward_totalElementsInRange = 0;
};

__global__ void initializeBiNodes(BiNode* nodes, int width, int height);

__global__ void biAStarMultipleBucketsSingleKernel(
    int *grid, int width, int height,    // grid dimensions and obstacle grid
    int startNodeId, int targetNodeId,                   // for forward search, this is the goal; for backward, the start
    BiNode *nodes,                      // array of BiNodes (both forward and backward fields integrated)
    // Open list arrays for forward search
    int *forward_openListBins, int *forward_binCounts,
    int *forward_expansionBuffers, int *forward_expansionCounts,
    // Open list arrays for backward search
    int *backward_openListBins, int *backward_binCounts,
    int *backward_expansionBuffers, int *backward_expansionCounts,
    bool *found, int *path, int *pathLength,
    int frontierSize, 
    int *totalExpandedNodes, int* expandedNodes,
    BidirectionalState* state);