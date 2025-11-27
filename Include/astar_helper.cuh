#pragma once
#include <iostream>

#include "constants.cuh"

// Node structure
// struct Node {
//     int id;
//     int g;
//     int h;
//     int f;
//     int parent;
// };

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
    int openListAddress_forward;
    int openListAddress_backward;
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

    int dx = abs(xCurrent - xGoal);
    int dy = abs(yCurrent - yGoal);

    // return (dx + dy) * SCALE_FACTOR;
    // return (unsigned int) dx * SCALE_FACTOR;
    return DIAGONAL_COST * min(dx, dy) + SCALE_FACTOR * abs(dx - dy);
    // return SCALE_FACTOR * ((dx + dy) + (DIAGONAL_COST - 2 * SCALE_FACTOR) * min(dx, dy));
    // return (unsigned int) sqrtf((float)(dx * dx + dy * dy)) * SCALE_FACTOR;
}

__host__ __device__ inline unsigned int binForNode(unsigned int fValue, int width) {
    unsigned int minFValue = DIAGONAL_COST * (width-1);
    unsigned int adjustedF =  fValue - minFValue; 
    return (adjustedF / BUCKET_F_RANGE);
}

// This function reconstructs a bidirectional path.
// Two threads call this function: threadIdx.x==0 for the forward pass,
// threadIdx.x==1 for the backward pass.
__device__ inline void constractBidirectionalPath(int startNodeId, int endNodeId, BiNode& meetingNode, int* path, int* pathLength, BiNode* g_nodes) {

    // Use a couple of shared integers to communicate chain lengths.
    __shared__ volatile int backwardCount; // number of nodes in backward chain (includes meeting & start)
    __shared__ volatile int forwardCount;  // number of nodes in forward chain (excluding the meeting node)

    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    // printf("tid: %d\n", tid);

    // __syncthreads();
    
    // --- Backward Pass (thread 1) ---
    if (tid == 1) {
        backwardCount = -1;
        int count = 0;
        int nodeId = meetingNode.id;
        // First pass: count backward chain length (from meeting node to startNodeId)
        // (Assume each parent's pointer is a valid index in g_nodes.)
        while (nodeId != -1) {
            count++;
            nodeId = g_nodes[nodeId].parent_backward;
        }
        backwardCount = count;
        *pathLength += backwardCount;  // -1 to exclude the meeting node
        
        while(forwardCount == -1) { }  // wait for forwardCount to be set
        
        // Second pass: iterate again and write the chain in reverse order so that
        // path[0] is the start node and path[backwardCount-1] is the meeting node.
        count = 0;
        nodeId = meetingNode.id;
        while (nodeId != -1) {
            path[backwardCount - count - 1] = nodeId;
            nodeId = g_nodes[nodeId].parent_backward;
            count++;
        }
    }
    
    // --- Forward Pass (thread 0) ---
    else if (tid == 0) {
        forwardCount = -1;
        int count = 0;
        int nodeId = meetingNode.id;
        // First pass: count forward chain length, but do not count the meeting node
        while (nodeId != -1) {
            count++;
            nodeId = g_nodes[nodeId].parent_forward;
        }
        forwardCount = count;
        *pathLength += forwardCount - 1;
        
        while(backwardCount == -1) { }  // wait for backwardCount to be set
        
        // Second pass: fill in the forward chain (skip meeting node to avoid duplication)
        count = 0;
        // start with the node after meeting node
        nodeId = meetingNode.parent_forward;
        while (nodeId != -1) {
            path[backwardCount + count] = nodeId;
            nodeId = g_nodes[nodeId].parent_forward;
            count++;
        }
    }
}




/////////////////////////////////////////////////////////////////////////////////
// Naive single-thread prefix sum over [start..end], inclusive.
/////////////////////////////////////////////////////////////////////////////////
// __device__ int sumRangeInclusive(const int* __restrict__ bucketCount, int start, int end);

// /////////////////////////////////////////////////////////////////////////////////
// // Naive single-block inclusive scan of shPrefix[0..(n-1)] in-place.
// // (Replace with a parallel scan in real code.)
// /////////////////////////////////////////////////////////////////////////////////
// __device__ void inclusiveScanInBlock(int* arr, int n);

// /////////////////////////////////////////////////////////////////////////////////
// // loadBucketsToSharedInclusive
// // Copies *all* elements from buckets in the range [bucketRangeStart .. bucketRangeEnd]
// // into 'output' contiguously in shared memory.
// //
// // Assumptions:
// //  - One block does the entire copy for that subrange.
// //  - bucketData is laid out so that bucket 0’s elements come first, then bucket 1’s, etc.
// //  - bucketCount[i] is the size of bucket i.
// //  - outputSize = sum of bucketCount[bucketRangeStart..bucketRangeEnd] (inclusive).
// /////////////////////////////////////////////////////////////////////////////////
// __device__ void loadBucketsToShared(int* __restrict__ bucketData,
//                                              int* __restrict__ bucketCount,
//                                              int bucketRangeStart,
//                                              int bucketRangeEnd,
//                                              int outputSize,  // sum of all bucket sizes in [start..end]
//                                              int* __restrict__ output);  // in shared memory
