
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
#include "constants.hpp"

#define MAX_NEIGHBORS 8 // 8-directional movement

#define BUCKET_F_RANGE 3000
#define MAX_BINS 3000         // Maximum number of bins (adjust as needed)
#define MAX_BIN_SIZE 2000     // Maximum number of nodes per bin (adjust as needed)
#define SCALE_FACTOR 1000     // Scale factor for cost

// #define SH_MAX_RANGE 15
// #define MAX_SHARED_BIN_SIZE 100
// #define TILE_WIDTH  16
// #define TILE_HEIGHT 16

#define FRONTIER_SIZE 500

#define MAX_PATH_LENGTH 10000000

namespace cg = cooperative_groups;

// Define BIN_SIZE based on the grid dimensions
// __device__ __constant__ float BIN_SIZE_DEVICE;

// Global variables for inter-block communication
// __device__ int d_currentBin;
__device__ bool d_done_forward;
__device__ bool d_done_backward;
// __device__ bool d_localFound;

// Shared or block-scope variables for each while iteration
// __device__ int s_bucketRangeStart;
// __device__ int s_bucketRangeEnd;
// __device__ int s_totalElementsInRange;  // total nodes (NOT including neighbors)

__device__ int globalBestCost;
__device__ BiNode globalBestNode;


// global variables for the active bucket range
__device__ int global_forward_bucketRangeStart;
__device__ int global_forward_bucketRangeEnd;
__device__ int global_forward_totalElementsInRange;

__device__ int global_backward_bucketRangeStart;
__device__ int global_backward_bucketRangeEnd;
__device__ int global_backward_totalElementsInRange;

__global__ void initializeBiNodes(BiNode* nodes, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        nodes[idx].id = idx;
        nodes[idx].g_forward = INT_MAX;
        nodes[idx].h_forward = INT_MAX;
        nodes[idx].f_forward = INT_MAX;
        nodes[idx].parent_forward = -1;
        nodes[idx].g_backward = INT_MAX;
        nodes[idx].h_backward = INT_MAX;
        nodes[idx].f_backward = INT_MAX;
        nodes[idx].parent_backward = -1;
    }
}

__global__ void biAStarMultipleBucketsSingleKernel(
    int *grid, int width, int height,    // grid dimensions and obstacle grid
    int startNodeId, int targetNodeId,                   // for forward search, this is the goal; for backward, the start
    BiNode *nodes,                      // array of BiNodes (both forward and backward fields integrated)
    // Open list arrays for forward search
    int *forward_openListBins, int *forward_binCounts, unsigned long long *forward_binBitMask,
    int *forward_expansionBuffers, int *forward_expansionCounts,
    // Open list arrays for backward search
    int *backward_openListBins, int *backward_binCounts, unsigned long long *backward_binBitMask,
    int *backward_expansionBuffers, int *backward_expansionCounts,
    bool *found, int *path, int *pathLength,
    int binBitMaskSize, int K, 
    int *totalExpandedNodes, int* firstNonEmptyMask, int* lastNonEmptyMask)
{
    // Determine search direction based on block index.
    // Odd-indexed blocks: forward search (forward = true)
    // Even-indexed blocks: backward search (forward = false)
    bool forward = (blockIdx.x % 2 == 1);

    // Select the proper set of open list arrays based on the search direction.
    int *openListBinsPtr       = forward ? forward_openListBins      : backward_openListBins;
    int *binCountsPtr          = forward ? forward_binCounts         : backward_binCounts;
    unsigned long long *binBitMaskPtr = forward ? forward_binBitMask  : backward_binBitMask;
    int *expansionBuffersPtr   = forward ? forward_expansionBuffers  : backward_expansionBuffers;
    int *expansionCountsPtr    = forward ? forward_expansionCounts   : backward_expansionCounts;
    
    // Cooperative groups for grid-wide synchronization.
    cg::grid_group gridGroup = cg::this_grid();

    // Linear thread ID across the entire grid.
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int totalThreads = gridGroup.size();

    // Declare shared bucket range variables per block.
    __shared__ int sh_bucketRangeStart;
    __shared__ int sh_bucketRangeEnd;
    __shared__ int sh_totalElementsInRange;    

    // One-time init on the first thread of the first block (global flags d_done and d_localFound assumed).
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_done_forward       = false;
        d_done_backward      = false;
        globalBestCost       = INT_MAX;
        globalBestNode.id    = -1;
        // d_localFound = false;
    }
    gridGroup.sync();

    // Main bidirectional A* loop.
    while (!d_done_forward || !d_done_backward)
    {
        // printf("forward: %d\n", forward);
        // Thread 0 of block 0 computes the active bucket range forward.
        // Thread 1 of block 0 computes the active bucket range backward.
        if (threadIdx.x == 0 && (blockIdx.x == 0 || blockIdx.x == 1)) {
            int bucketRangeStart = -1;
            int bucketRangeEnd   = -1;
            // sh_totalElementsInRange = 0; // still global, if needed.

            __volatile int elementsAccumulated = 0;
            bool done = false;
            for (int i = 0; i <= binBitMaskSize && !done; ++i) {
                unsigned long long tmpMask = binBitMaskPtr[i];
                while (tmpMask != 0 && !done) {
                    int firstSetBit = __ffsll(tmpMask) - 1;
                    tmpMask &= ~(1ULL << firstSetBit);
                    int bucket = i * 64 + firstSetBit;
                    if (bucket >= MAX_BINS) {
                        done = true;
                        break;
                    }
                    if (bucketRangeStart == -1)
                        bucketRangeStart = bucket;
                    int countHere = binCountsPtr[bucket];
                    if (elementsAccumulated + countHere < K) {
                        elementsAccumulated += countHere;
                        bucketRangeEnd = bucket;
                    } else {
                        bucketRangeEnd = bucket;
                        done = true;
                    }
                }
            }
            if (bucketRangeStart == bucketRangeEnd && elementsAccumulated == 0 &&
                bucketRangeStart + 1 < MAX_BINS && binCountsPtr[bucketRangeStart + 1] > 0)
                sh_totalElementsInRange = elementsAccumulated;
            // else if (elementsAccumulated == 0)
            //     d_done = true;
            // else
            //     sh_totalElementsInRange = elementsAccumulated;
            
            // 2 flags for forward pass and backward pass
            if(elementsAccumulated == 0 && blockIdx.x == 1)
            {
                d_done_forward = true;
            }

            if(elementsAccumulated == 0 && blockIdx.x == 0)
            {
                d_done_backward = true;
            }
            
            // broadcast to all other blocks
            if (blockIdx.x == 1 && elementsAccumulated > 0)
            {
                global_forward_bucketRangeStart = bucketRangeStart;
                global_forward_bucketRangeEnd = bucketRangeEnd;
                global_forward_totalElementsInRange = elementsAccumulated;
            }
            else if (blockIdx.x == 0 && elementsAccumulated > 0)
            {
                global_backward_bucketRangeStart = bucketRangeStart;
                global_backward_bucketRangeEnd = bucketRangeEnd;
                global_backward_totalElementsInRange = elementsAccumulated;
            }

            // active elements forward
            if(forward)
            {
                printf("Active elements forward: %d\n", elementsAccumulated);
                printf("Bucket range forward: %d - %d\n", sh_bucketRangeStart, sh_bucketRangeEnd);

            }
            else
            {
                printf("Active elements backward: %d\n", elementsAccumulated);
                printf("Bucket range backward: %d - %d\n", sh_bucketRangeStart, sh_bucketRangeEnd);
            }

            wait(10000000000);
        }

        gridGroup.sync(); // sync all blocks
        // all blocks should copy the shared bucket range variables into shared memory
        if(forward && threadIdx.x == 0)
        {
            sh_bucketRangeStart = global_forward_bucketRangeStart;
            sh_bucketRangeEnd = global_forward_bucketRangeEnd;
            sh_totalElementsInRange = global_forward_totalElementsInRange;
        }

        if (!forward && threadIdx.x == 0)
        {
            sh_bucketRangeStart = global_backward_bucketRangeStart;
            sh_bucketRangeEnd = global_backward_bucketRangeEnd;
            sh_totalElementsInRange = global_backward_totalElementsInRange;
        }

        gridGroup.sync(); // sync all blocks
        if (d_done_forward && d_done_backward)
            break;

        // Map this thread to a (bucket, node, neighbor) tuple using the shared bucket range.
        int assignedBucket = -1;
        int threadPosition = idx; // linear index among (node, neighbor) pairs.
        for (int b = global_backward_bucketRangeStart; b <= global_forward_bucketRangeEnd; ++b) {
            int bucketSize = binCountsPtr[b] * MAX_NEIGHBORS;
            if (threadPosition < bucketSize) {
                assignedBucket = b;
                break;
            }
            threadPosition -= bucketSize;
        }
        
        if (assignedBucket != -1) {
            // Count this expansion.
            atomicAdd(totalExpandedNodes, 1);

            int nodeIndex     = threadPosition / MAX_NEIGHBORS;  
            int neighborIndex = threadPosition % MAX_NEIGHBORS;  

            int currentNodeId = openListBinsPtr[assignedBucket * MAX_BIN_SIZE + nodeIndex];
            BiNode currentNode = nodes[currentNodeId];

            // Early pruning: skip if the current node’s f-value is not promising.
            int currentF = forward ? currentNode.f_forward : currentNode.f_backward;
            if (currentF < globalBestCost)
            {
                // 8-direction neighbor offsets.
                int neighborOffsets[8][2] = {
                    {  0, -1}, { 1, -1}, { 1,  0}, { 1,  1},
                    {  0,  1}, {-1,  1}, {-1,  0}, {-1, -1}
                };

                int xCurrent = currentNodeId % width;
                int yCurrent = currentNodeId / width;
                int dx = neighborOffsets[neighborIndex][0];
                int dy = neighborOffsets[neighborIndex][1];
                int xNeighbor = xCurrent + dx;
                int yNeighbor = yCurrent + dy;

                if (xNeighbor >= 0 && xNeighbor < width && yNeighbor >= 0 && yNeighbor < height) {
                    int neighborId = yNeighbor * width + xNeighbor;
                    if (grid[neighborId] == 0) {  // if passable
                        bool isDiagonal = (abs(dx) + abs(dy) == 2);
                        int moveCost = isDiagonal ? 1414 : 1000;
                        int tentativeG = (forward ? currentNode.g_forward : currentNode.g_backward) + moveCost;

                        // Atomically update the neighbor's cost using the appropriate field.
                        int oldG;
                        if (forward) {
                            oldG = atomicMin(&nodes[neighborId].g_forward, tentativeG);
                        } else {
                            oldG = atomicMin(&nodes[neighborId].g_backward, tentativeG);
                        }
                        if (tentativeG < oldG) {
                            // Update neighbor's fields accordingly.
                            if (forward) {
                                nodes[neighborId].id = neighborId;
                                nodes[neighborId].parent_forward = currentNodeId;
                                nodes[neighborId].h_forward = heuristic(neighborId, targetNodeId, width);
                                nodes[neighborId].f_forward = tentativeG + nodes[neighborId].h_forward;
                                nodes[neighborId].g_forward = tentativeG;
                            } else {
                                nodes[neighborId].id = neighborId;
                                nodes[neighborId].parent_backward = currentNodeId;
                                nodes[neighborId].h_backward = heuristic(neighborId, startNodeId, width);
                                nodes[neighborId].f_backward = tentativeG + nodes[neighborId].h_backward;
                                nodes[neighborId].g_backward = tentativeG;
                            }

                            // Check if this neighbor has already been reached from the opposite search.
                            if (forward) {
                                if (nodes[neighborId].g_backward < INT_MAX) {
                                    int candidateCost = nodes[neighborId].g_forward + nodes[neighborId].g_backward;
                                    int oldCost = atomicMin(&globalBestCost, candidateCost);
                                    if (oldCost > candidateCost) {
                                        globalBestNode = nodes[neighborId];
                                    }
                                }
                                else if (neighborId == targetNodeId) {
                                    int candidateCost = nodes[neighborId].f_forward;
                                    int oldCost = atomicMin(&globalBestCost, candidateCost);
                                    if (oldCost > candidateCost) {
                                        globalBestNode = nodes[neighborId];
                                    }
                                }

                            } else {
                                if (nodes[neighborId].g_forward < INT_MAX) {
                                    int candidateCost = nodes[neighborId].g_forward + nodes[neighborId].g_backward;
                                    int oldCost = atomicMin(&globalBestCost, candidateCost);
                                    if (oldCost > candidateCost) {
                                        globalBestNode = nodes[neighborId];
                                    }
                                } else if (neighborId == startNodeId) {
                                    int candidateCost = nodes[neighborId].f_backward;
                                    int oldCost = atomicMin(&globalBestCost, candidateCost);
                                    if (oldCost > candidateCost) {
                                        globalBestNode = nodes[neighborId];
                                    }
                                }
                            }

                            // Compute the bin for the neighbor based on its updated f-value.
                            int newF = forward ? nodes[neighborId].f_forward : nodes[neighborId].f_backward;
                            
                            printf("New f-value: %d\n", newF);
                            int minFValue = 1414 * width;
                            int adjustedF = newF - minFValue;
                            int binForNghbr = (int)(adjustedF / BUCKET_F_RANGE);
                            binForNghbr = max(0, min(binForNghbr, MAX_BINS - 1));

                            // Insert neighbor into the expansion buffer or open list bins.
                            if (binForNghbr >= global_backward_bucketRangeStart && binForNghbr <= global_forward_bucketRangeEnd) {
                                int pos = atomicAdd(&expansionCountsPtr[binForNghbr], 1);
                                int offset = binForNghbr * MAX_BIN_SIZE + pos;
                                expansionBuffersPtr[offset] = neighborId;
                                // printf("Adding neighbor %d to expansion buffer %d\n", neighborId, binForNghbr);
                            } else {
                                int pos = atomicAdd(&binCountsPtr[binForNghbr], 1);
                                openListBinsPtr[binForNghbr * MAX_BIN_SIZE + pos] = neighborId;
                                if (pos == 0) {
                                    int maskIndex = binForNghbr / 64;
                                    unsigned long long m = 1ULL << (binForNghbr % 64);
                                    // if (!binBitMaskPtr[maskIndex]) {
                                    //     atomicMin(firstNonEmptyMask, maskIndex);
                                    //     atomicMax(lastNonEmptyMask, maskIndex);
                                    // }
                                    atomicOr(&binBitMaskPtr[maskIndex], m);
                                    // printf("Adding neighbor %d to open list bin %d\n", neighborId, binForNghbr);
                                }
                            }
                        }
                    }
                }
            }
        }
        gridGroup.sync();

        // Merge expansion buffers into the open list.
        if (blockIdx.x == 0 || blockIdx.x == 1) {
            int start = forward ? global_forward_bucketRangeStart : global_backward_bucketRangeStart;
            int end = forward ? global_forward_bucketRangeEnd : global_backward_bucketRangeEnd;
            for (int bucket = start; bucket <= end; ++bucket) {
                int eCount = expansionCountsPtr[bucket];
                if (eCount > 0) {
                    binCountsPtr[bucket] = eCount;
                    for (int i = threadIdx.x; i < eCount; i += blockDim.x) {
                        int offset = bucket * MAX_BIN_SIZE + i;
                        openListBinsPtr[offset] = expansionBuffersPtr[offset];
                    }
                } else {
                    if (threadIdx.x == 0) {
                        binCountsPtr[bucket] = 0;
                        int maskIndex = bucket / 64;
                        unsigned long long m = ~(1ULL << (bucket % 64));
                        atomicAnd(&binBitMaskPtr[maskIndex], m);

                        // firstNonEmptyMask and lastNonEmptyMask are not updated here.
                        // if (binBitMaskPtr[maskIndex] == 0 && maskIndex == *firstNonEmptyMask) {
                        //     int newIndex = maskIndex + 1;
                        //     while (newIndex <= *lastNonEmptyMask && binBitMaskPtr[newIndex] == 0ULL)
                        //         newIndex++;
                        //     atomicMax(firstNonEmptyMask, newIndex);
                        // }
                        // if (binBitMaskPtr[maskIndex] == 0 && maskIndex == *lastNonEmptyMask) {
                        //     int newIndex = maskIndex - 1;
                        //     while (newIndex >= *firstNonEmptyMask && binBitMaskPtr[newIndex] == 0ULL)
                        //         newIndex--;
                        //     atomicMin(lastNonEmptyMask, newIndex);
                        // }
                    }
                }
                __syncthreads();
                expansionCountsPtr[bucket] = 0;
            }
        }


        // Merge expansion buffers into the open list for the designated block:
        // For forward search: if (forward && blockIdx.x == 0)
        // For backward search: if (!forward && blockIdx.x == 1)
        // if (blockIdx.x == 0 || blockIdx.x == 1) {
        //     for (int bucket = global_backward_bucketRangeStart; bucket <= global_forward_bucketRangeEnd; ++bucket) {
        //         // Read the expansion count for this bucket.
        //         int eCount = expansionCountsPtr[bucket];
        //         // Update the open list bin count.
        //         binCountsPtr[bucket] = eCount;
                
        //         // Parallel copy: each thread copies a portion of the expansion buffer.
        //         for (int i = threadIdx.x; i < eCount; i += blockDim.x) {
        //             int offset = bucket * MAX_BIN_SIZE + i;
        //             openListBinsPtr[offset] = expansionBuffersPtr[offset];
        //         }
        //         __syncthreads();
                
        //         // If no elements were added, clear the corresponding bit in the bin bitmask.
        //         if (eCount == 0 && threadIdx.x == 0) {
        //             int maskIndex = bucket / 64;
        //             unsigned long long clearMask = ~(1ULL << (bucket % 64));
        //             atomicAnd(&binBitMaskPtr[maskIndex], clearMask);
        //         }
        //         __syncthreads();
                
        //         // Reset the expansion count for this bucket.
        //         if (threadIdx.x == 0) {
        //             expansionCountsPtr[bucket] = 0;
        //         }
        //         __syncthreads();
        //     }
        // }

        __threadfence();
        gridGroup.sync();

        // gridGroup.sync();
    } // end while(!d_done)

    // When done, check best path.
    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     *pathLength = 0;
    //     if (globalBestCost == INT_MAX) {
    //         *found = false;
    //     } else {
    //         *found = true;
    //         int currentId = globalBestNode.id;
    //         while (currentId != -1) {
    //             path[atomicAdd(pathLength, 1)] = currentId;
    //             if (forward) {
    //                 currentId = nodes[currentId].parent_forward;
    //             } else {
    //                 currentId = nodes[currentId].parent_backward;
    //             }
    //         }
    //     }
    // }



    // When done, reconstruct the complete path.
    // The meeting node (globalBestNode.id) lies somewhere along the path.
    // We first follow the forward parent pointers from the meeting node back to the start,
    // then follow the backward parent pointers from the meeting node toward the goal.
    // Finally, we reverse the forward portion to get the correct order from start to meeting,
    // reverse the backward portion to get meeting to goal in proper order,
    // and then concatenate them (omitting the duplicate meeting node).

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        if (globalBestCost == INT_MAX) {
            *found = false;
            *pathLength = 0;
        } else {
            *found = true;
            
            // // Define a maximum path length (adjust as needed).
            // // int MAX_PATH_LENGTH = 10000000;
            // int forwardBuffer[MAX_PATH_LENGTH];
            // int backwardBuffer[MAX_PATH_LENGTH];
            // int fCount = 0, bCount = 0;
            
            // // Reconstruct forward chain: from meeting node back to start.
            // // (Follow parent's forward pointers.)
            // int cur = globalBestNode.id;
            // while (cur != -1 && fCount < MAX_PATH_LENGTH) {
            //     forwardBuffer[fCount++] = cur;
            //     cur = nodes[cur].parent_forward;
            // }
            // // Now reverse forwardBuffer so that it goes from start to meeting.
            // int forwardPath[MAX_PATH_LENGTH];
            // for (int i = 0; i < fCount; i++) {
            //     forwardPath[i] = forwardBuffer[fCount - 1 - i];
            // }
            
            // // Reconstruct backward chain: from meeting node toward goal.
            // // (Follow parent's backward pointers.)
            // // We skip the meeting node here (to avoid duplicate) and start with its backward parent.
            // cur = nodes[globalBestNode.id].parent_backward;
            // while (cur != -1 && bCount < MAX_PATH_LENGTH) {
            //     backwardBuffer[bCount++] = cur;
            //     cur = nodes[cur].parent_backward;
            // }
            // // Reverse backwardBuffer so that it goes from meeting to goal.
            // int backwardPath[MAX_PATH_LENGTH];
            // for (int i = 0; i < bCount; i++) {
            //     backwardPath[i] = backwardBuffer[bCount - 1 - i];
            // }
            
            // // Merge the two parts into the final path:
            // int totalPathLength = fCount + bCount;
            // *pathLength = totalPathLength;
            // // Write forward part: from start to meeting.
            // for (int i = 0; i < fCount; i++) {
            //     path[i] = forwardPath[i];
            // }
            // // Write backward part: from meeting+1 to goal.
            // for (int i = 0; i < bCount; i++) {
            //     path[fCount + i] = backwardPath[i];
            // }
        }
    }

}
