#pragma once

#include <iostream>

#include "constants.cuh"

// Per-cell state shared by the forward and backward searches.
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

// Octile distance encoded as an integer so both searches can safely use atomicMin.
__host__ __device__ inline unsigned int heuristic(int currentNodeId, int goalNodeId, int width) {
    int xCurrent = currentNodeId % width;
    int yCurrent = currentNodeId / width;
    int xGoal = goalNodeId % width;
    int yGoal = goalNodeId / width;

    int dx = abs(xCurrent - xGoal);
    int dy = abs(yCurrent - yGoal);

    return DIAGONAL_COST * min(dx, dy) + SCALE_FACTOR * abs(dx - dy);
}

__host__ __device__ inline unsigned int binForNode(unsigned int fValue, unsigned int minFValue) {
    unsigned int adjustedF = fValue - minFValue;
    return (adjustedF / BUCKET_F_RANGE) % MAX_BINS;
}

// Thread 1 writes the start->meeting prefix and thread 0 appends the meeting->goal suffix.
__device__ inline void constractBidirectionalPath(
    int startNodeId,
    int endNodeId,
    int meetingNodeId,
    int *path,
    int *pathLength,
    BiNode *g_nodes) {
    (void)startNodeId;
    (void)endNodeId;

    __shared__ volatile int backwardCount;
    __shared__ volatile int forwardCount;

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid == 1) {
        backwardCount = -1;
        int count = 0;
        int nodeId = meetingNodeId;
        while (nodeId != -1) {
            count++;
            nodeId = g_nodes[nodeId].parent_backward;
        }
        backwardCount = count;
        *pathLength += backwardCount;

        while (forwardCount == -1) {
        }

        count = 0;
        nodeId = meetingNodeId;
        while (nodeId != -1) {
            path[backwardCount - count - 1] = nodeId;
            nodeId = g_nodes[nodeId].parent_backward;
            count++;
        }
    } else if (tid == 0) {
        forwardCount = -1;
        int count = 0;
        int nodeId = meetingNodeId;
        while (nodeId != -1) {
            count++;
            nodeId = g_nodes[nodeId].parent_forward;
        }
        forwardCount = count;
        *pathLength += forwardCount - 1;

        while (backwardCount == -1) {
        }

        count = 0;
        nodeId = g_nodes[meetingNodeId].parent_forward;
        while (nodeId != -1) {
            path[backwardCount + count] = nodeId;
            nodeId = g_nodes[nodeId].parent_forward;
            count++;
        }
    }
}
