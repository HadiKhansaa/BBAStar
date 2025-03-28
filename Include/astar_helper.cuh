#pragma once
#include <iostream>

#include "constants.cuh"

// Node structure
struct Node {
    int id;
    int g;
    int h;
    int f;
    int parent;
};

// bidirectional Node
struct BiNode {
    int id;
    unsigned int g_forward;
    unsigned int h_forward;
    unsigned int f_forward;
    unsigned int g_backward;
    unsigned int h_backward;
    unsigned int f_backward;
    int parent_forward;
    int parent_backward;
};

__device__ inline void wait(int cycles) {
    clock_t start = clock();
    clock_t now;
    for (;;) {
        now = clock();
        clock_t cyclesPassed = now > start ? now - start : now + (0xffffffff - start);
        if (cyclesPassed >= cycles) {
            break;
        }
    }
}



// Heuristic function (Euclidean distance)
__host__ __device__ inline unsigned int heuristic(int currentNodeId, int goalNodeId, int width) {
    int xCurrent = currentNodeId % width;
    int yCurrent = currentNodeId / width;
    int xGoal = goalNodeId % width;
    int yGoal = goalNodeId / width;

    unsigned int dx = abs(xCurrent - xGoal);
    unsigned int dy = abs(yCurrent - yGoal);

    // return (dx + dy) * SCALE_FACTOR;
    return sqrtf((float)(dx * dx + dy * dy)) * SCALE_FACTOR;
}

// This function reconstructs a bidirectional path.
// Two threads call this function: threadIdx.x==0 for the forward pass,
// threadIdx.x==1 for the backward pass.
__device__ void constractBidirectionalPath(int startNodeId, int endNodeId, BiNode& meetingNode, int* path, int* pathLength, BiNode* g_nodes);




/////////////////////////////////////////////////////////////////////////////////
// Naive single-thread prefix sum over [start..end], inclusive.
/////////////////////////////////////////////////////////////////////////////////
__device__ int sumRangeInclusive(const int* __restrict__ bucketCount, int start, int end);

/////////////////////////////////////////////////////////////////////////////////
// Naive single-block inclusive scan of shPrefix[0..(n-1)] in-place.
// (Replace with a parallel scan in real code.)
/////////////////////////////////////////////////////////////////////////////////
__device__ void inclusiveScanInBlock(int* arr, int n);

/////////////////////////////////////////////////////////////////////////////////
// loadBucketsToSharedInclusive
// Copies *all* elements from buckets in the range [bucketRangeStart .. bucketRangeEnd]
// into 'output' contiguously in shared memory.
//
// Assumptions:
//  - One block does the entire copy for that subrange.
//  - bucketData is laid out so that bucket 0’s elements come first, then bucket 1’s, etc.
//  - bucketCount[i] is the size of bucket i.
//  - outputSize = sum of bucketCount[bucketRangeStart..bucketRangeEnd] (inclusive).
/////////////////////////////////////////////////////////////////////////////////
__device__ void loadBucketsToShared(int* __restrict__ bucketData,
                                             int* __restrict__ bucketCount,
                                             int bucketRangeStart,
                                             int bucketRangeEnd,
                                             int outputSize,  // sum of all bucket sizes in [start..end]
                                             int* __restrict__ output);  // in shared memory
