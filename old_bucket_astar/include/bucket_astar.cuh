
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

// #define INF_FLT 1e20f  // A large float value representing infinity
#define MAX_NEIGHBORS 8        // 8-directional movement

#define MAX_BINS 3000         // Maximum number of bins (adjust as needed)
#define MAX_BIN_SIZE 2000    // Maximum number of nodes per bin (adjust as needed)
#define SCALE_FACTOR 1000   // Khansa is based (via the universal Axiom of Consistant-Basedness)

#define SH_MAX_RANGE 15
#define MAX_SHARED_BIN_SIZE 100

#define TILE_WIDTH  16
#define TILE_HEIGHT 16

#define FRONTIER_SIZE 512

namespace cg = cooperative_groups;


// Define BIN_SIZE based on the grid dimensions
__device__ __constant__ float BIN_SIZE_DEVICE;

// Kernel to initialize nodes
__global__ void initializeNodes(Node* nodes, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        nodes[idx].id = -1;
        nodes[idx].g = INT_MAX;
        nodes[idx].h = 0.0f;
        nodes[idx].f = INT_MAX;
        nodes[idx].parent = -1;
    }
}

// Global variables for inter-block communication
__device__ int d_currentBin;
__device__ bool d_done;
__device__ bool d_localFound;

// A* algorithm kernel with cooperative groups and parallel neighbor expansion
__global__ void aStarKernel(int *grid, int width, int height, int goalNodeId,
                            Node *nodes,
                            int *openListBins, int *binCounts,
                            unsigned int *binBitMask,
                            int *binExpansionBuffer, int *binExpansionCount,
                            bool *found, int *path, int *pathLength,
                            int binBitMaskSize, int *totalExpandedNodes) {

    cg::grid_group gridGroup = cg::this_grid();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridGroup.size();

    // Initialize local copies if necessary
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_done = false;
        d_localFound = false;
    }
    gridGroup.sync();

    while (!d_done) {
        // Thread 0 of block 0 finds the next non-empty bin
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            d_currentBin = -1;
            for (int i = 0; i < binBitMaskSize; ++i) {
                unsigned int mask = binBitMask[i];
                if (mask != 0) {
                    int firstSetBit = __ffs(mask) - 1; // __ffs returns 1-based index
                    d_currentBin = i * 32 + firstSetBit;
                    break;
                }
            }

            if (d_currentBin == -1 || d_currentBin >= MAX_BINS) {
                d_done = true;
            }
        }

        // Ensure all threads have updated currentBin and done
        gridGroup.sync();

        if (d_done) {
            break;
        }

        // All threads read the currentBin
        int currentBin = d_currentBin;
        int binSize = binCounts[currentBin];

        // logging
        // if(threadIdx.x == 0 && blockIdx.x == 0)
        // {
        //     printf("current Bin: %d, Bin Size: %d\n", currentBin, binSize);
        // }

        // Process neighbors of nodes in the current bin
        for (int i = idx; i < binSize * MAX_NEIGHBORS; i += totalThreads) {

            if (*found) {
                // If the goal is found, exit the loop
                break;
            }

            int nodeIndex = i / MAX_NEIGHBORS;
            int neighborIndex = i % MAX_NEIGHBORS;

            int currentNodeId = openListBins[currentBin * MAX_BIN_SIZE + nodeIndex];
            Node currentNode = nodes[currentNodeId];

            // For neighborIndex == 0, perform per-node tasks
            if (neighborIndex == 0) {
                atomicAdd(totalExpandedNodes, 1);

                if (currentNodeId == goalNodeId) {
                    // Use atomic operation to set 'found' flag
                    *found = true;
                    // Set 'localFound' to true
                    d_localFound = true;

                    // Reconstruct the path
                    int tempId = goalNodeId;
                    int count = 0;
                    while (tempId != -1 && count < width * height) {
                        path[count++] = tempId;
                        tempId = nodes[tempId].parent;
                    }
                    *pathLength = count;

                    // Break out of the loop
                    break;
                }
            }

            int xCurrent = currentNodeId % width;
            int yCurrent = currentNodeId / width;

            // Neighbor offsets for 8-directional movement
            int neighborOffsets[8][2] = {
                {0, -1},   // Up
                {1, -1},   // Up-Right
                {1, 0},    // Right
                {1, 1},    // Down-Right
                {0, 1},    // Down
                {-1, 1},   // Down-Left
                {-1, 0},   // Left
                {-1, -1}   // Up-Left
            };

            // Get neighbor offsets
            int dx = neighborOffsets[neighborIndex][0];
            int dy = neighborOffsets[neighborIndex][1];
            int xNeighbor = xCurrent + dx;
            int yNeighbor = yCurrent + dy;

            if (xNeighbor >= 0 && xNeighbor < width && yNeighbor >= 0 && yNeighbor < height) {
                int neighborId = yNeighbor * width + xNeighbor;

                // Check if neighbor is blocked
                if (grid[neighborId] == 0) {  // 0 indicates free cell
                    // Determine movement cost
                    bool isDiagonal = (abs(dx) + abs(dy) == 2);
                    float movementCost = isDiagonal ? 1414 : 1000;

                    int tentativeG = (currentNode.g + movementCost);

                    // Atomically update g value if a better path is found
                    int oldG = atomicMin(&nodes[neighborId].g, tentativeG);

                    if (tentativeG < oldG) {
                        // Update node information
                        nodes[neighborId].id = neighborId;
                        nodes[neighborId].parent = currentNodeId;
                        nodes[neighborId].h = heuristic(neighborId, goalNodeId, width);
                        nodes[neighborId].f = tentativeG + nodes[neighborId].h;
                        nodes[neighborId].g = tentativeG;

                        float minFValue = 1414 * height;

                        // Determine the bin for the neighbor
                        int binForNeighbor = (int)((nodes[neighborId].f - minFValue) / BIN_SIZE_DEVICE);
                        binForNeighbor = min(binForNeighbor, MAX_BINS - 1);

                        if (binForNeighbor == currentBin) {
                            // Collect nodes to the current bin expansion buffer
                            int position = atomicAdd(binExpansionCount, 1);
                            if (position < MAX_BIN_SIZE) {
                                binExpansionBuffer[position] = neighborId;
                            } else {
                                printf("Expansion buffer overflow for bin %d\n", currentBin);
                            }
                        } else {
                            // Atomically add neighbor to the appropriate bin
                            int position = atomicAdd(&binCounts[binForNeighbor], 1);
                            if (position < MAX_BIN_SIZE) {
                                openListBins[binForNeighbor * MAX_BIN_SIZE + position] = neighborId;

                                // If position was 0, the bin was previously empty
                                if (position == 0) {
                                    // Set the corresponding bit in binBitMask
                                    int maskIndex = binForNeighbor / 32;
                                    unsigned int mask = 1U << (binForNeighbor % 32);
                                    atomicOr(&binBitMask[maskIndex], mask);
                                }
                            } else {
                                printf("Bin overflow at bin %d\n", binForNeighbor);
                            }
                        }
                    }
                }
            }
        }

        // Synchronize before proceeding to the next bin
        gridGroup.sync();

        // Check if any thread found the goal
        if (d_localFound) {
            break;
        }

        // Update the current bin (only thread 0 of block 0)
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            int expansionCount = *binExpansionCount;

            if (expansionCount > 0) {
                binCounts[currentBin] = expansionCount;

                // Copy elements from binExpansionBuffer to openListBins
                for (int i = 0; i < expansionCount; ++i) {
                    openListBins[currentBin * MAX_BIN_SIZE + i] = binExpansionBuffer[i];
                }
            } else {
                binCounts[currentBin] = 0;

                // Clear the bit in binBitMask
                int maskIndex = currentBin / 32;
                unsigned int mask = ~(1U << (currentBin % 32));
                atomicAnd(&binBitMask[maskIndex], mask);
            }

            // Reset binExpansionCount for next iteration
            *binExpansionCount = 0;
        }

        // Ensure updates are visible
        __threadfence();

        // Synchronize before next iteration
        gridGroup.sync();
    }
}

// Shared or block-scope variables for each while iteration
__device__ int s_bucketRangeStart;
__device__ int s_bucketRangeEnd;
__device__ int s_totalElementsInRange;  // total nodes (NOT including neighbors)

__global__ void aStarMultipleBuckets(
    int *grid, int width, int height, int goalNodeId, Node *nodes,
    int *openListBins, int *binCounts, unsigned long long *binBitMask,
    int *expansionBuffers, int *expansionCounts, bool *found,
    int *path, int *pathLength, int binBitMaskSize, int K, 
    int *totalExpandedNodes, int* firstNonEmptyMask, int* lastNonEmptyMask)
{
    // Cooperative groups for grid-wide sync
    cg::grid_group gridGroup = cg::this_grid();

    // Linear thread ID across the entire grid
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int totalThreads = gridGroup.size();

    // One-time init on the first thread of the first block
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_done       = false;
        d_localFound = false;
    }
    gridGroup.sync();

    // Main A* loop
    while (!d_done)
    {
        // Thread 0 of block 0 decides the range [startBucket..endBucket]
        if (threadIdx.x == 0 && blockIdx.x == 0) {

            if(K < 500)
                K+=10;

            s_bucketRangeStart       = -1;
            s_bucketRangeEnd         = -1;
            s_totalElementsInRange   = 0;

            int elementsAccumulated = 0;
            bool done = false;  // used to break out of both loops
            for (int i = *firstNonEmptyMask; i <= *lastNonEmptyMask && !done; ++i) {
                unsigned long long tmpMask = binBitMask[i];
                // Process *all* set bits in tmpMask
                while (tmpMask != 0 && !done) {
                    int firstSetBit = __ffsll(tmpMask) - 1; // __ffs() = 1-based index
                    tmpMask &= ~(1ULL << firstSetBit);      // turn off that bit

                    int bucket = i * 64 + firstSetBit;
                    if (bucket >= MAX_BINS) {
                        // we've exceeded valid bucket range
                        done = true;
                        break;
                    }

                    // Set start if we haven't yet
                    if (s_bucketRangeStart == -1) {
                        s_bucketRangeStart = bucket;
                    }

                    // Accumulate node counts
                    int countHere = binCounts[bucket];
                    if (elementsAccumulated < K) {
                        elementsAccumulated += countHere;
                        s_bucketRangeEnd = bucket;
                    }
                    else {
                        // Reached or exceeded our K threshold
                        s_bucketRangeEnd = bucket;
                        done = true;
                    }
                } // end while(tmpMask != 0 && !done)
            } // end for (int i = 0; ...)

            // If no elements found, we're done
            if (elementsAccumulated == 0) {
                d_done = true;
            } else {
                s_totalElementsInRange = elementsAccumulated;
            }

            // printf("Active elements = %d (buckets %d..%d)\n",
            //     elementsAccumulated,
            //     s_bucketRangeStart,
            //     s_bucketRangeEnd);
            
            // printf("first %d, last %d\n", *firstNonEmptyMask, *lastNonEmptyMask);

            // wait for a bit
            // clock_t start = clock();
            // clock_t now;
            // for (;;) {
            //     now = clock();
            //     clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
            //     if (cycles >= 100000000) {
            //         break;
            //     }
            // }         
            
        }


        // Synchronize so all blocks see the updated start/end
        gridGroup.sync();

        // Break if done
        if (d_done) {
            break;
        }

        // 1) Compute how many threads are needed (all nodes * 8 neighbors)
        // int totalThreadsNeeded = s_totalElementsInRange * MAX_NEIGHBORS;

        // // 2) If our thread ID >= that number, skip. We'll still do the final sync though
        // if (idx >= totalThreadsNeeded) {
        //     gridGroup.sync();
        //     // Check if found was set by other threads
        //     if (d_localFound) {
        //         break;
        //     }
        //     continue;
        // }

        // 3) Map this thread to a (bucket, node, neighbor)
        int assignedBucket = -1;
        int threadPosition = idx;  // This is the linear index among all (node, neighbor) combos

        for (int b = s_bucketRangeStart; b <= s_bucketRangeEnd; ++b) {
            int bucketSize = binCounts[b] * MAX_NEIGHBORS;
            if (threadPosition < bucketSize) {
                assignedBucket = b;
                break;
            }
            threadPosition -= bucketSize;
        }
        
        if (assignedBucket != -1) {
            // printf("Thread %d assigned to bucket %d\n", idx, assignedBucket);

            atomicAdd(totalExpandedNodes, 1);

            int nodeIndex     = threadPosition / MAX_NEIGHBORS;  
            int neighborIndex = threadPosition % MAX_NEIGHBORS;  

            int currentNodeId = openListBins[assignedBucket * MAX_BIN_SIZE + nodeIndex];
            Node currentNode  = nodes[currentNodeId];

            // 8-direction neighbor offsets
            int neighborOffsets[8][2] = {
                {  0, -1}, { 1, -1}, { 1,  0}, { 1,  1},
                {  0,  1}, {-1,  1}, {-1,  0}, {-1, -1}
            };

            // (x,y) for current node, and offsets for neighbor
            int xCurrent  = currentNodeId % width;
            int yCurrent  = currentNodeId / width;
            int dx        = neighborOffsets[neighborIndex][0];
            int dy        = neighborOffsets[neighborIndex][1];
            int xNeighbor = xCurrent + dx;
            int yNeighbor = yCurrent + dy;

            // Check bounds
            if (xNeighbor >= 0 && xNeighbor < width && yNeighbor >= 0 && yNeighbor < height) {
                int neighborId = yNeighbor * width + xNeighbor;

                // If passable
                if (grid[neighborId] == 0) {
                    bool isDiagonal  = (abs(dx) + abs(dy) == 2);
                    int moveCost     = isDiagonal ? 1414 : 1000;
                    int tentativeG   = currentNode.g + moveCost;

                    // Try to improve neighbor's G
                    int oldG = atomicMin(&nodes[neighborId].g, tentativeG);
                    if (tentativeG < oldG) {
                        // Update neighbor node
                        nodes[neighborId].id     = neighborId;
                        nodes[neighborId].parent = currentNodeId;
                        nodes[neighborId].h      = heuristic(neighborId, goalNodeId, width);
                        nodes[neighborId].f      = tentativeG + nodes[neighborId].h;
                        nodes[neighborId].g      = tentativeG;

                        // If neighborId == goalNodeId, we found the goal
                        if (neighborId == goalNodeId) {
                            // Mark that we found the goal
                            d_localFound = true;
                        }

                        // Compute bin
                        int minFValue   = 1414 * height;
                        int adjustedF   = nodes[neighborId].f - minFValue;
                        int binForNghbr = (int)(adjustedF / BIN_SIZE_DEVICE);
                        // binForNghbr     = max(0, min(binForNghbr, MAX_BINS - 1));
                        binForNghbr = min(binForNghbr, MAX_BINS - 1);

                        // If new bin is still in our active range, put in expansion buffer
                        if (binForNghbr >= s_bucketRangeStart && binForNghbr <= s_bucketRangeEnd) {
                            int pos = atomicAdd(&expansionCounts[binForNghbr], 1);
                            int offset = binForNghbr * MAX_BIN_SIZE + pos;
                            expansionBuffers[offset] = neighborId;
                        } else {
                            // Insert back into openListBins
                            int pos = atomicAdd(&binCounts[binForNghbr], 1);
                            openListBins[binForNghbr * MAX_BIN_SIZE + pos] = neighborId;
                            // Set bit if first in bin
                            if (pos == 0) {
                                int maskIndex  = binForNghbr / 64;
                                unsigned long long m = 1ULL << (binForNghbr % 64);

                                // if mask was empty
                                if(!binBitMask[maskIndex])
                                {
                                    // Update first/last non-empty mask
                                    atomicMin(firstNonEmptyMask, maskIndex);
                                    atomicMax(lastNonEmptyMask, maskIndex);
                                }

                                atomicOr(&binBitMask[maskIndex], m);
                            }
                        }
                    } // end if (tentativeG < oldG)
                } // end if grid[neighborId]==0
            } // end if valid neighbor
        } // end if (assignedBucket != -1)

        // 6) Sync before updating expansions
        gridGroup.sync();

        // If goal was found by any thread, break
        // if (d_localFound) {
        //     // Optionally do more: gather which thread found it, etc.
        //     // We'll break out of the while loop below, but first let
        //     // every thread see we found the goal.
        //     gridGroup.sync();
        // }

        // 7) Merge expansions buffers
        // Let *all threads* in block 0 share the load:
        if (blockIdx.x == 0) {
            for (int bucket = s_bucketRangeStart; bucket <= s_bucketRangeEnd; ++bucket) {
                int eCount = expansionCounts[bucket];
                if (eCount > 0) {
                    binCounts[bucket] = eCount;

                    // Distribute copy across all threads in block 0:
                    for (int i = threadIdx.x; i < eCount; i += blockDim.x) {
                        int offset = bucket * MAX_BIN_SIZE + i;
                        openListBins[offset] = expansionBuffers[offset];
                    }
                } else {

                    if(threadIdx.x == 0)
                    {
                        binCounts[bucket] = 0;
                        // Clear bit:
                        int maskIndex  = bucket / 64;
                        unsigned long long m = ~(1ULL << (bucket % 64));
                        atomicAnd(&binBitMask[maskIndex], m);

                        // update first/last non-empty mask
                        if(binBitMask[maskIndex] == 0 && maskIndex == *firstNonEmptyMask)
                        {
                            int newIndex = maskIndex + 1;
                            while (newIndex <= *lastNonEmptyMask && binBitMask[newIndex] == 0ULL) {
                                newIndex++;
                            }
                            // *firstNonEmptyMask = newIndex;
                            atomicMax(firstNonEmptyMask, newIndex);
                        }

                        if(binBitMask[maskIndex] == 0 && maskIndex == *lastNonEmptyMask)
                        {
                            int newIndex = maskIndex - 1;
                            while (newIndex >= *firstNonEmptyMask && binBitMask[newIndex] == 0ULL) {
                                newIndex--;
                            }
                            // *lastNonEmptyMask = newIndex;
                            atomicMin(lastNonEmptyMask, newIndex);
                        }
                    }
                }

                // Wait for all threads in block 0 to finish this bucket
                __syncthreads();

                expansionCounts[bucket] = 0;
            }
        }


        __threadfence();
        gridGroup.sync();

        // Finally, if any thread found the goal, do path reconstruction in block 0
        // then set d_done = true to end the loop
        if (d_localFound) {
            // let block 0 do the path build
            if (blockIdx.x == 0 && threadIdx.x == 0) {
                printf("Goal found! Reconstructing path...\n");
               

                // Reconstruct the path
                int tempId = goalNodeId;
                int count = 0;
                while (tempId != -1 && count < width * height) {
                    path[count++] = tempId;
                    tempId = nodes[tempId].parent;
                }
                *pathLength = count;

                // Mark global done so we exit the while loop
                d_done = true;

                // Also set the host-visible found[] if needed
                *found = true;
            }
        }

        gridGroup.sync(); // ensure all see d_done or proceed
    } // end while(!d_done)
}

// We'll assume a small range for demonstration, e.g. up to 8 buckets in range
// If your range can be bigger, adjust SH_MAX_RANGE accordingly.

__global__ void aStarMultipleBucketsShared(
    int *grid, int width, int height, int goalNodeId, Node *nodes,
    int *openListBins, int *binCounts, unsigned long long *binBitMask,
    int *expansionBuffers, int *expansionCounts, bool *found,
    int *path, int *pathLength, 
    int binBitMaskSize, 
    int K, 
    int *totalExpandedNodes,
    int* firstNonEmptyMask,
    int* lastNonEmptyMask)
{
    cg::grid_group gridGroup = cg::this_grid();
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int totalThreads = gridGroup.size();

    // Use dynamic or static shared arrays:
    __shared__ int blockExpansionBuffer[SH_MAX_RANGE * MAX_SHARED_BIN_SIZE];
    __shared__ int blockExpansionCounts[SH_MAX_RANGE];

    // One-time init on the first thread of the first block
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_done       = false;
        d_localFound = false;
    }
    gridGroup.sync();

    while (!d_done)
    {
        // =============================================================
        // (1) The first block decides [startBucket..endBucket]
        // =============================================================
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            if(K < 500) 
                K += 10;

            s_bucketRangeStart     = -1;
            s_bucketRangeEnd       = -1;
            s_totalElementsInRange = 0;

            int elementsAccumulated = 0;
            bool done = false;

            for (int i = 0; i < binBitMaskSize && !done; ++i) {
                unsigned long long tmpMask = binBitMask[i];
                while (tmpMask != 0ULL && !done) {
                    int firstSetBit = __ffsll(tmpMask) - 1; 
                    tmpMask &= ~(1ULL << firstSetBit); 

                    int bucket = i * 64 + firstSetBit;
                    if (bucket >= MAX_BINS) {
                        done = true;
                        break;
                    }

                    // Set start if not set
                    if (s_bucketRangeStart == -1) {
                        s_bucketRangeStart = bucket;
                    }

                    // if we have SH_MAX_RANGE buckets, we're done
                    if(s_bucketRangeEnd - s_bucketRangeStart >= SH_MAX_RANGE)
                    {
                        done = true;
                        break;
                    }

                    if (elementsAccumulated + binCounts[bucket] < K) {
                        elementsAccumulated += binCounts[bucket];
                        s_bucketRangeEnd = bucket;
                    } else {
                        // Reached or exceeded K
                        s_bucketRangeEnd = bucket;
                        done = true;
                    }
                }
            }
            if (elementsAccumulated == 0) {
                d_done = true;
            } else {
                s_totalElementsInRange = elementsAccumulated;
            }

            // debugging
            // printf("Active elements = %d (buckets %d..%d)\n",
            //     elementsAccumulated,
            //     s_bucketRangeStart,
            //     s_bucketRangeEnd);

            // // wait a bit
            // clock_t start = clock();
            // clock_t now;
            // for (;;) {
            //     now = clock();
            //     clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
            //     if (cycles >= 10000000) {
            //         break;
            //     }
            // }

            // printf("first %d, last %d\n", *firstNonEmptyMask, *lastNonEmptyMask);
        }
        gridGroup.sync();

        if (d_done) {
            break;
        }

        // =============================================================
        // (2) Figure out which bucket this thread is assigned
        // =============================================================
        int assignedBucket = -1;
        int threadPosition = idx; 
        for (int b = s_bucketRangeStart; b <= s_bucketRangeEnd; ++b) {
            int bucketSize = binCounts[b] * 8; // 8 neighbors
            if (threadPosition < bucketSize) {
                assignedBucket = b;
                break;
            }
            threadPosition -= bucketSize;
        }

        // =============================================================
        // (3) Zero out the shared arrays at start of iteration
        //     or at least once per iteration BEFORE expansions
        // =============================================================
        // We'll do a small loop in each block:
        // (only need up to s_bucketRangeEnd - s_bucketRangeStart + 1 indices)
        if (threadIdx.x < SH_MAX_RANGE) {
            blockExpansionCounts[threadIdx.x] = 0;  
        }
        __syncthreads();
        // (No need to clear blockExpansionBuffer fully; we'll write only the needed positions)

        // =============================================================
        // (4) Expand the assigned bucket, storing expansions in shared memory
        // =============================================================
        if (assignedBucket != -1) {
            atomicAdd(totalExpandedNodes, 1);

            int nodeIndex     = threadPosition / 8;  
            int neighborIndex = threadPosition % 8;  

            int currentNodeId = openListBins[assignedBucket * MAX_BIN_SIZE + nodeIndex];
            Node currentNode  = nodes[currentNodeId];

            int neighborOffsets[8][2] = {
                {  0, -1}, {  1, -1}, { 1, 0}, { 1,  1},
                {  0,  1}, { -1,  1}, {-1, 0}, {-1, -1}
            };

            int xCurrent  = currentNodeId % width;
            int yCurrent  = currentNodeId / width;
            int dx        = neighborOffsets[neighborIndex][0];
            int dy        = neighborOffsets[neighborIndex][1];
            int xNeighbor = xCurrent + dx;
            int yNeighbor = yCurrent + dy;

            if (xNeighbor >= 0 && xNeighbor < width && yNeighbor >= 0 && yNeighbor < height) {
                int neighborId = yNeighbor * width + xNeighbor;
                if (grid[neighborId] == 0) {
                    bool isDiagonal = ((abs(dx) + abs(dy)) == 2);
                    int moveCost    = isDiagonal ? 1414 : 1000;
                    int tentativeG  = currentNode.g + moveCost;

                    int oldG = atomicMin(&nodes[neighborId].g, tentativeG);
                    if (tentativeG < oldG) {
                        // We improved the path
                        nodes[neighborId].id     = neighborId;
                        nodes[neighborId].parent = currentNodeId;
                        nodes[neighborId].h      = heuristic(neighborId, goalNodeId, width);
                        nodes[neighborId].f      = tentativeG + nodes[neighborId].h;
                        nodes[neighborId].g      = tentativeG;

                        if (neighborId == goalNodeId) {
                            d_localFound = true;
                        }

                        // Decide which bin this neighbor goes to
                        int minFValue   = 1414 * height;
                        int adjustedF   = nodes[neighborId].f - minFValue;
                        int binForNghbr = (int)(adjustedF / BIN_SIZE_DEVICE);
                        binForNghbr     = min(binForNghbr, MAX_BINS - 1);

                        // If we want to store expansions in shared memory, we convert:
                        //   s_bin = binForNghbr - s_bucketRangeStart
                        // Make sure it's within [0..SH_MAX_RANGE-1]
                        int s_bin = binForNghbr - s_bucketRangeStart;
                        if (s_bin >= 0 && s_bin < SH_MAX_RANGE) {
                            // put into shared memory

                            int pos = atomicAdd(&blockExpansionCounts[s_bin], 1);
                            
                            // printf("Shared bin: %d, pos: %d, neighborId: %d\n", s_bin, pos, neighborId);

                            if (pos < MAX_SHARED_BIN_SIZE) {
                                int s_offset = s_bin * MAX_SHARED_BIN_SIZE + pos;
                                blockExpansionBuffer[s_offset] = neighborId;
                            }
                            else
                            {
                                printf("overflow in shared memory\n");
                            }

                            // printf("Shared bin: %d, pos: %d, neighborId: %d\n", s_bin, pos, neighborId);
                            // else: handle overflow if needed
                        } 
                        // else if (binForNghbr >= s_bucketRangeStart && binForNghbr <= s_bucketRangeEnd) {
                        //     // If neighbor's bin is outside [start..end], we can do your old logic
                        //     // e.g. write to global openListBins or expansionBuffers with an atomic
                        //     // int pos = atomicAdd(&binCounts[binForNghbr], 1);
                        //     // openListBins[binForNghbr * MAX_BIN_SIZE + pos] = neighborId;
                        //     // if (pos == 0) {
                        //     //     int maskIndex  = binForNghbr / 64;
                        //     //     unsigned long long m = 1ULL << (binForNghbr % 64);
                        //     //     atomicOr(&binBitMask[maskIndex], m);
                        //     // }

                        //     // add to global expansion buffer instead
                        //     int pos = atomicAdd(&expansionCounts[binForNghbr], 1);
                        //     int offset = binForNghbr * MAX_BIN_SIZE + pos;
                        //     expansionBuffers[offset] = neighborId;
                        // }
                        else
                        {
                            // Insert back into openListBins
                            // printf("Inserting back into openListBins in %d\n", s_bin);
                            int pos = atomicAdd(&binCounts[binForNghbr], 1);
                            openListBins[binForNghbr * MAX_BIN_SIZE + pos] = neighborId;
                            // Set bit if first in bin
                            if (pos == 0) {
                                int maskIndex  = binForNghbr / 64;
                                unsigned long long m = 1ULL << (binForNghbr % 64);
                                atomicOr(&binBitMask[maskIndex], m);
                            }
                        }
                    } // end if (tentativeG < oldG)
                }
            }
        }

        // =============================================================
        // (5) Synchronize the block so expansions are done
        // =============================================================
        // __syncthreads();
        gridGroup.sync(); // maybe can remove

        if(blockIdx.x == 0)
        {
            // empty binCounts
            if(threadIdx.x < SH_MAX_RANGE)
            {
                binCounts[s_bucketRangeStart + threadIdx.x] = 0;

                int maskIndex  = (s_bucketRangeStart + threadIdx.x) / 64;
                unsigned long long m = ~(1ULL << ((s_bucketRangeStart + threadIdx.x) % 64));
                atomicAnd(&binBitMask[maskIndex], m);
            }
            // for (int b = s_bucketRangeStart; b < s_bucketRangeStart + SH_MAX_RANGE; ++b)
            // {
            //     binCounts[b] = 0;
            //     int maskIndex  = b / 64;
            //     unsigned long long m = ~(1ULL << (b % 64));
            //     atomicAnd(&binBitMask[maskIndex], m);
            // }
        }

        __threadfence();
        gridGroup.sync();

        // =============================================================
        // (6) Copy shared expansions to global openListBins
        // =============================================================
        for (int b = s_bucketRangeStart; b < s_bucketRangeStart + SH_MAX_RANGE; ++b)
        {
            int localIndex = b - s_bucketRangeStart;
            if (localIndex < 0) continue;  // safety
            // How many expansions we have for this bucket in shared memory
            int count = blockExpansionCounts[localIndex];

            if (count > 0) {
                // Thread 0 does an atomicAdd to find the global offset
                int offset = 0;
                if (threadIdx.x == 0) {
                    offset = atomicAdd(&binCounts[b], count);

                    int maskIndex = b / 64;
                    unsigned long long m = 1ULL << (b % 64);
                    atomicOr(&binBitMask[maskIndex], m);
                }

                __syncthreads();

                // Broadcast the 'offset' to all threads in the block
                offset = __shfl_sync(0xffffffff, offset, 0);

                // Now each thread copies its slice of expansions to openListBins
                for (int i = threadIdx.x; i < count; i += blockDim.x) {
                    int sOffset   = localIndex * MAX_SHARED_BIN_SIZE + i; 
                    int neighborId = blockExpansionBuffer[sOffset];

                    int globalPos = offset + i;  // position in the bucket
                    openListBins[b * MAX_BIN_SIZE + globalPos] = neighborId;
                    // if (globalPos == 0) {
                    //     int maskIndex = b / 64;
                    //     unsigned long long m = 1ULL << (b % 64);
                    //     atomicOr(&binBitMask[maskIndex], m);
                    // }
                }
            }

            __syncthreads();  // ensure we finish copying before next bucket
        }

        // __threadfence();
        // gridGroup.sync();

        // 7) Merge expansions buffers
        // Let *all threads* in block 0 share the load:
        // if (blockIdx.x == 10) {
        //     for (int bucket = s_bucketRangeStart+SH_MAX_RANGE; bucket <= s_bucketRangeEnd; ++bucket) {
        //         int eCount = expansionCounts[bucket];
        //         if (eCount > 0) {
        //             binCounts[bucket] = eCount;

        //             // Distribute copy across all threads in block 0:
        //             for (int i = threadIdx.x; i < eCount; i += blockDim.x) {
        //                 int offset = bucket * MAX_BIN_SIZE + i;
        //                 openListBins[offset] = expansionBuffers[offset];
        //             }
        //         } else {

        //             if(threadIdx.x == 0)
        //             {
        //                 binCounts[bucket] = 0;
        //                 // Clear bit:
        //                 int maskIndex  = bucket / 64;
        //                 unsigned long long m = ~(1ULL << (bucket % 64));
        //                 atomicAnd(&binBitMask[maskIndex], m);
        //             }
        //         }

        //         // Wait for all threads in block 0 to finish this bucket
        //         __syncthreads();

        //         expansionCounts[bucket] = 0;
        //     }
        // }

        // =============================================================
        // (7) If goal was found, do path reconstruction
        // =============================================================
        if (d_localFound) {
            if (blockIdx.x == 0 && threadIdx.x == 0) {
                printf("Goal found! Reconstructing path...\n");
                int tempId = goalNodeId;
                int c = 0;
                while (tempId != -1 && c < width * height) {
                    path[c++] = tempId;
                    tempId = nodes[tempId].parent;
                }
                *pathLength = c;
                *found      = true;
                d_done      = true;
            }
        }
        gridGroup.sync();
    } // end while(!d_done)
}

// __device__ int iterationCount = 0;

// =====================================================================
// Optimized A* kernel
__global__ void aStarMultipleBucketsSharedGrid(
    int *grid, int width, int height, int goalNodeId, Node *nodes,
    int *openListBins, int *binCounts, unsigned long long *binBitMask,
    int *expansionBuffers, int *expansionCounts, bool *found,
    int *path, int *pathLength, int binBitMaskSize, int K,
    int *totalExpandedNodes, int* firstNonEmptyMask, int* lastNonEmptyMask)
{
    // Use cooperative groups for grid–wide sync (if supported)
    cg::grid_group gridGroup = cg::this_grid();
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int totalThreads = gridGroup.size();

    // ------------------------------
    // Shared memory arrays for the bin expansions:
    __shared__ int blockExpansionBuffer[SH_MAX_RANGE * MAX_SHARED_BIN_SIZE];
    __shared__ int blockExpansionCounts[SH_MAX_RANGE];


    // each block copies start and end into shared variable in constant memory
    __shared__ int sh_bucketRangeStart;
    __shared__ int sh_bucketRangeEnd;
    __shared__ int sh_totalElementsInRange;

    // Shared memory arrays fot the openList
    __shared__ int sh_frontier[FRONTIER_SIZE];
    // __shared__ int sh_blockBinCounts[SH_MAX_RANGE];

    // ------------------------------
    // NEW: Shared memory tile for nodes.
    // We “tile” a portion of the nodes array (think of it as a 2D region of the grid)
    __shared__ Node sharedNodes[TILE_WIDTH * TILE_HEIGHT];

    // initialize ids of shared mem Nodes to -1
    for (int i = threadIdx.x; i < TILE_WIDTH * TILE_HEIGHT; i += blockDim.x) {
        sharedNodes[i].id = -1;
    }

    // initialize the shared mem frontier to -1
    for (int i = threadIdx.x; i < FRONTIER_SIZE; i += blockDim.x) {
        sh_frontier[i] = -1;
    }
    
    __syncthreads();

    // decide on bounding box
    // int minX = 0;
    // int minY = 0;
    // int maxX = width;
    // int maxY = height;

    __shared__ int tileOffsetRow;
    __shared__ int tileOffsetCol;

    // __shared__ int tileWidth;
    // __shared__ int tileHeight;
    // if(threadIdx.x == 0)
    // {
    //     tileOffsetRow = 100000001;
    //     tileOffsetCol = 100000001;
    // }
    // __syncthreads();

    // tileWidth = 0;
    // tileHeight = 0;

    // One–time initialization on the first thread of the first block.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_done       = false;
        d_localFound = false;
    }
    gridGroup.sync();

    // Main loop of the search.
    while (!d_done)
    {
        // initialize tile offset
        if(threadIdx.x == 0)
        {
            tileOffsetRow = height * width + 1;
            tileOffsetCol = height * width + 1;
        }
        __syncthreads();

        // ---------------------------------------------------------------------
        // (A) Load the tile from global memory into shared memory.
        // Each thread loads one or more tile cells.

        // only do this every 10 iterations
        // if(iterationCount % 10 == 0)
        // {
            // int tileSize = TILE_WIDTH * TILE_HEIGHT;
            // for (int i = threadIdx.x; i < tileSize; i += blockDim.x) {
            //     int r = i / TILE_WIDTH;
            //     int c = i % TILE_WIDTH;
            //     int globalId = (tileOffsetRow + r) * width + (tileOffsetCol + c);
            //     if ((tileOffsetRow + r) < height && (tileOffsetCol + c) < width)
            //         sharedNodes[i] = nodes[globalId];
            // }
            // __syncthreads();
        // }
        // ---------------------------------------------------------------------

        // =============================================================
        // (1) Determine the bucket range (frontier selection).
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            // if(K < 500) 
            //     K += 10;

            s_bucketRangeStart     = -1;
            s_bucketRangeEnd       = -1;
            s_totalElementsInRange = 0;

            int elementsAccumulated = 0;
            bool done = false;

            for (int i = 0; i < binBitMaskSize && !done; ++i) {
                unsigned long long tmpMask = binBitMask[i];
                while (tmpMask != 0ULL && !done) { // while there are still non-empty buckets
                    int firstSetBit = __ffsll(tmpMask) - 1; // first non empty bucket
                    tmpMask &= ~(1ULL << firstSetBit); 

                    int bucket = i * 64 + firstSetBit;
                    if (bucket >= MAX_BINS) {
                        done = true;
                        break;
                    }

                    if (s_bucketRangeStart == -1) { // determine the start of the range
                        s_bucketRangeStart = bucket;
                    }
                    
                    // I want at max SH_MAX_RANGE buckets to be in the fronteir
                    if(s_bucketRangeStart != -1 && ((bucket - s_bucketRangeStart) >= SH_MAX_RANGE))
                    {
                        s_bucketRangeEnd = bucket;
                        done = true;
                        break;
                    }
                    
                    // elements in the frontier are always <= K
                    if (elementsAccumulated + binCounts[bucket] < K) {
                        elementsAccumulated += binCounts[bucket];
                        s_bucketRangeEnd = bucket;
                    } else {
                        s_bucketRangeEnd = bucket;
                        done = true;
                    }
                }
            }
            // if first bucket has more than K elements just take it alone (special case: only time where we have > K elements in the frontier)
            if(s_bucketRangeStart == s_bucketRangeEnd && elementsAccumulated == 0 && s_bucketRangeStart + 1 < MAX_BINS && binCounts[s_bucketRangeStart + 1] > 0)
            {
                s_totalElementsInRange = elementsAccumulated;
            }
            else if (elementsAccumulated == 0) {
                d_done = true;
            } else {
                s_totalElementsInRange = elementsAccumulated;
            }


            // debugging
            printf("Active elements = %d (buckets %d..%d)\n",
                elementsAccumulated,
                s_bucketRangeStart,
                s_bucketRangeEnd);

            // wait a bit
            wait(10000);
        }

        gridGroup.sync();        

        // place variables in shared mem
        if(threadIdx.x == 0)
        {
            sh_bucketRangeStart = s_bucketRangeStart;
            sh_bucketRangeEnd = s_bucketRangeEnd;
            sh_totalElementsInRange = s_totalElementsInRange;
        }

        __syncthreads();

        // each block loads the fronteir into its shared memory
        // if(blockIdx.x == 0)
        if(blockIdx.x < sh_totalElementsInRange/blockDim.x + 1)
        {
            loadBucketsToShared(openListBins, binCounts, s_bucketRangeStart, s_bucketRangeEnd, sh_totalElementsInRange, sh_frontier);
            __syncthreads();
        }
        gridGroup.sync();
        
        //debug print the elements in shared frontier
        // if(threadIdx.x == 0 && blockIdx.x == 0)
        // {
        //     printf("Block: %d, sh_frontier: ", blockIdx.x);
        //     for(int i = 0; i < sh_totalElementsInRange; i++)
        //     {
        //         printf("%d ", sh_frontier[i]);
        //     }
        //     printf("\n");
        // }

        if (d_done) {
            break;
        }

        // =============================================================
        // (2) Assign each thread a bucket (work assignment)
        int assignedBucket = -1;
        int threadPosition = idx; 
        // for (int b = sh_bucketRangeStart; b <= sh_bucketRangeEnd; ++b) {
        //     int bucketSize = binCounts[b] * 8; // 8 neighbors per node
        //     if (threadPosition < bucketSize) {
        //         assignedBucket = b;
        //         break;
        //     }
        //     threadPosition -= bucketSize;
        // }

        // I want to assign the threads to the sh_blockOpenListBins array such that each block 
        // processes 32 ids, 8 threads for each one, this way we have 32*8 = 256 threads = blockSize
        if(threadPosition < sh_totalElementsInRange * 8 && sh_frontier[threadPosition / 8] != -1) // safty
            assignedBucket = sh_bucketRangeStart; // this is temporary, we will assign the correct bucket later
        

        // =============================================================
        // (3) Zero out the shared arrays (for bin expansions)
        if (threadIdx.x < SH_MAX_RANGE) {
            blockExpansionCounts[threadIdx.x] = 0;  
        }
        // __syncthreads();
        
        // =============================================================
        // (4) Expand
        int currentRow, currentCol, nodeIndex, neighborIndex, currentNodeId, tileSize;

        // determine bounding box for grid for each block
        if (assignedBucket != -1) {
            atomicAdd(totalExpandedNodes, 1);

            nodeIndex     = threadPosition / 8;  
            neighborIndex = threadPosition % 8;
            // currentNodeId = openListBins[assignedBucket * MAX_BIN_SIZE + nodeIndex];
            
            // wokr assignment, threads will process consecutive nodes in the fronteir
            currentNodeId = sh_frontier[nodeIndex];

            // determine the bounding box for the grid
            currentRow = currentNodeId / width;
            currentCol = currentNodeId % width;
            atomicMin(&tileOffsetRow, currentRow);
            atomicMin(&tileOffsetCol, currentCol);

            // maybe use later for dynamic bounding box
            // atomicMax(&tileWidth, currentCol);
            // atomicMax(&tileHeight, currentRow);
        }
        // __syncthreads();

        // debug print tile offset
        // if(threadIdx.x == 0 && blockIdx.x == 0)
        // {
        //     printf("Block: %d, tileOffsetRow: %d, tileOffsetCol: %d\n", blockIdx.x, tileOffsetRow, tileOffsetCol);
        //     wait(10000000);
        // }

        // initialize ids in shared mem nodes to -1
        for (int i = threadIdx.x; i < TILE_WIDTH * TILE_HEIGHT; i += blockDim.x) {
            sharedNodes[i].id = -1;
        }
        __syncthreads();
    
        
        // load bounding box of grid from global memory to shared memory based on bounding box
        tileSize = TILE_WIDTH * TILE_HEIGHT;
        for (int i = threadIdx.x; i < tileSize; i += blockDim.x) {
            int r = i / TILE_WIDTH;
            int c = i % TILE_WIDTH;
            int globalId = (tileOffsetRow + r) * width + (tileOffsetCol + c);

            // printf("r: %d, c: %d\n", r, c);
            // printf("globalId: %d\n", globalId);

            if ((tileOffsetRow + r) < height && (tileOffsetCol + c) < width)
                sharedNodes[i] = nodes[globalId];
        }
        __syncthreads();
        
        if (assignedBucket != -1) {

            Node currentNode;
            // --- Check if current node falls inside our shared tile.
            if ( currentRow >= tileOffsetRow && currentRow < tileOffsetRow + TILE_HEIGHT &&
                 currentCol >= tileOffsetCol && currentCol < tileOffsetCol + TILE_WIDTH ) {
                int localIndex = (currentRow - tileOffsetRow) * TILE_WIDTH + (currentCol - tileOffsetCol);
                currentNode = sharedNodes[localIndex];
                // printf("Current node falls inside shared tile\n");
            } else {
                currentNode = nodes[currentNodeId];
                // printf("Current node falls outside shared tile\n");
            }

            int neighborOffsets[8][2] = {
                {  0, -1}, {  1, -1}, { 1, 0}, { 1,  1},
                {  0,  1}, { -1,  1}, {-1, 0}, {-1, -1}
            };

            int xCurrent  = currentNodeId % width;
            int yCurrent  = currentNodeId / width;
            int dx        = neighborOffsets[neighborIndex][0];
            int dy        = neighborOffsets[neighborIndex][1];
            int xNeighbor = xCurrent + dx;
            int yNeighbor = yCurrent + dy;

            // if neighbor is within the grid
            if (xNeighbor >= 0 && xNeighbor < width && yNeighbor >= 0 && yNeighbor < height) {
                int neighborId = yNeighbor * width + xNeighbor;
                if (grid[neighborId] == 0) { // neighbor is not an obstacle

                    // calculate weight of node neighbor edge
                    bool isDiagonal = ((abs(dx) + abs(dy)) == 2);
                    int moveCost    = isDiagonal ? 1414 : 1000;
                    int tentativeG  = currentNode.g + moveCost;

                    int localIndex = (yNeighbor - tileOffsetRow) * TILE_WIDTH + (xNeighbor - tileOffsetCol);

                    // Check if neighbor falls in our shared tile.
                    if (yNeighbor >= tileOffsetRow && yNeighbor < tileOffsetRow + TILE_HEIGHT &&
                         xNeighbor >= tileOffsetCol && xNeighbor < tileOffsetCol + TILE_WIDTH 
                         && sharedNodes[localIndex].id != -1) { // the node is not empty

                        // printf("Neighbor falls inside shared tile\n");

                        int oldG = atomicMin(&sharedNodes[localIndex].g, tentativeG);
                        if (tentativeG < oldG) { // we found a better path
                            // update the node locally in shared memory
                            sharedNodes[localIndex].id     = neighborId;
                            sharedNodes[localIndex].parent = currentNodeId;
                            sharedNodes[localIndex].h      = heuristic(neighborId, goalNodeId, width);
                            sharedNodes[localIndex].f      = tentativeG + sharedNodes[localIndex].h;
                            sharedNodes[localIndex].g      = tentativeG;
                            if (neighborId == goalNodeId) { // if we found the goal we are done
                                d_localFound = true;
                            }
                            // calculating the bin for neighbor (formula: bin = (f - minFValue) / BIN_SIZE_DEVICE)
                            int minFValue   = 1414 * height;
                            int adjustedF   = sharedNodes[localIndex].f - minFValue;
                            int binForNghbr = (int)(adjustedF / BIN_SIZE_DEVICE);
                            binForNghbr     = min(binForNghbr, MAX_BINS - 1);

                            assert(binForNghbr >= 0 && binForNghbr < MAX_BINS);
                            // printf("binForNghbr: %d\n", binForNghbr);

                            // In shared bins we “shift” the index.
                            int s_bin = max(0, binForNghbr - sh_bucketRangeStart);
                            // if bin falls in the shared range for the shared mem expansion buffer
                            if (s_bin >= 0 && s_bin < SH_MAX_RANGE) {
                                int pos = atomicAdd(&blockExpansionCounts[s_bin], 1);
                                if (pos < MAX_SHARED_BIN_SIZE) {
                                    int s_offset = s_bin * MAX_SHARED_BIN_SIZE + pos;
                                    blockExpansionBuffer[s_offset] = neighborId;
                                }
                                else {
                                    printf("overflow in shared memory\n");
                                }
                            } else {
                                // printf("expanding outside shared memory\n");
                                printf("bin Counts: %d\n", binCounts[binForNghbr]);
                                if(binCounts[binForNghbr] < MAX_BIN_SIZE)
                                {
                                    int pos = atomicAdd(&binCounts[binForNghbr], 1);
                                    openListBins[binForNghbr * MAX_BIN_SIZE + pos] = neighborId;
                                    if (pos == 0) {
                                        int maskIndex  = binForNghbr / 64;
                                        unsigned long long m = 1ULL << (binForNghbr % 64);
                                        atomicOr(&binBitMask[maskIndex], m);
                                    }
                                }
                                else
                                {
                                    printf("bin overflow in global memory\n");
                                }
                            }
                        }
                    }
                    else {
                        // printf("Neighbor falls outside shared tile\n");
                        // --- Neighbor is outside our tile so use global memory.
                        int oldG = atomicMin(&nodes[neighborId].g, tentativeG);
                        if (tentativeG < oldG) {
                            // updating node with better path in global memory
                            nodes[neighborId].id     = neighborId;
                            nodes[neighborId].parent = currentNodeId;
                            nodes[neighborId].h      = heuristic(neighborId, goalNodeId, width);
                            nodes[neighborId].f      = tentativeG + nodes[neighborId].h;
                            nodes[neighborId].g      = tentativeG;
                            if (neighborId == goalNodeId) { // goal is found
                                d_localFound = true;
                            }
                            // calculating the bin for neighbor (formula: bin = (f - minFValue) / BIN_SIZE_DEVICE)
                            int minFValue   = 1414 * height;
                            int adjustedF   = nodes[neighborId].f - minFValue;
                            int binForNghbr = (int)(adjustedF / BIN_SIZE_DEVICE);
                            binForNghbr     = max(0, min(binForNghbr, MAX_BINS - 1));

                            // In shared bins we “shift” the index.
                            int s_bin = max(0, binForNghbr - sh_bucketRangeStart);
                            // if bin falls in the shared range for the shared mem expansion buffer
                            if (s_bin >= 0 && s_bin < SH_MAX_RANGE) {
                                int pos = atomicAdd(&blockExpansionCounts[s_bin], 1);
                                if (pos < MAX_SHARED_BIN_SIZE) {
                                    int s_offset = s_bin * MAX_SHARED_BIN_SIZE + pos;
                                    blockExpansionBuffer[s_offset] = neighborId;
                                }
                                else {
                                    printf("overflow in shared memory\n");
                                }
                            }
                            else {
                                // printf("expanding outside shared memory\n");
                                if(binCounts[binForNghbr] < MAX_BIN_SIZE)
                                {
                                    int pos = atomicAdd(&binCounts[binForNghbr], 1);
                                    openListBins[binForNghbr * MAX_BIN_SIZE + pos] = neighborId;
                                    if (pos == 0) {
                                        int maskIndex  = binForNghbr / 64;
                                        unsigned long long m = 1ULL << (binForNghbr % 64);
                                        atomicOr(&binBitMask[maskIndex], m);
                                    }
                                }
                                else
                                {
                                    printf("bin overflow in global memory\n");
                                }
                            }
                        }
                    }
                }
            }
        }
        // =============================================================
        // Synchronize: all expansions in shared memory are complete.
        // (6) Clear binCounts for the buckets that we used.
        if(blockIdx.x == 0)
        {
            if(threadIdx.x < SH_MAX_RANGE && sh_bucketRangeStart + threadIdx.x < MAX_BINS)
            // if(threadIdx.x <= sh_bucketRangeEnd && sh_bucketRangeStart + threadIdx.x < MAX_BINS)
            {
                binCounts[sh_bucketRangeStart + threadIdx.x] = 0;
                int maskIndex  = (sh_bucketRangeStart + threadIdx.x) / 64;
                unsigned long long m = ~(1ULL << ((sh_bucketRangeStart + threadIdx.x) % 64));
                atomicAnd(&binBitMask[maskIndex], m);
            }
        }
        // __threadfence();
        gridGroup.sync(); // this needs to be done here so all threads syncronize after expansion

        // ---------------------------------------------------------------------
        //  Write back updated tile from shared memory to global memory.
        // (If nodes in the tile were updated via shared memory, write them back.)
        for (int i = threadIdx.x; i < tileSize; i += blockDim.x) {
            int r = i / TILE_WIDTH;
            int c = i % TILE_WIDTH;
            int globalId = (tileOffsetRow + r) * width + (tileOffsetCol + c);
            if (sharedNodes[i].id != -1 && (tileOffsetRow + r) < height && (tileOffsetCol + c) < width)
            {
                nodes[globalId] = sharedNodes[i];
                // printf("Writing back to global memory\n");
            }
        }
        __syncthreads();

        // ---------------------------------------------------------------------
        // __syncthreads();

        // =============================================================
        // (7) Copy shared expansions back to global openListBins.
        for (int b = sh_bucketRangeStart; b < sh_bucketRangeStart + SH_MAX_RANGE; ++b)
        // for (int b = sh_bucketRangeStart; b <= sh_bucketRangeEnd; ++b)
        {
            int localIndex = b - sh_bucketRangeStart;
            if (localIndex < 0) continue;
            int count = blockExpansionCounts[localIndex];

            if (count > 0) {
                int offset = 0;
                if (threadIdx.x == 0) {
                    offset = atomicAdd(&binCounts[b], count);
                    int maskIndex = b / 64;
                    unsigned long long m = 1ULL << (b % 64);
                    atomicOr(&binBitMask[maskIndex], m);
                }

                __syncthreads();

                // Broadcast offset to all threads
                offset = __shfl_sync(0xffffffff, offset, 0);

                for (int i = threadIdx.x; i < count; i += blockDim.x) {
                    int sOffset   = localIndex * MAX_SHARED_BIN_SIZE + i; 
                    int neighborId = blockExpansionBuffer[sOffset];
                    int globalPos = offset + i;
                    openListBins[b * MAX_BIN_SIZE + globalPos] = neighborId;
                }
            }
            __syncthreads();
        }

        // gridGroup.sync(); // removed

        // =============================================================
        // (8) If the goal is found, reconstruct the path.
        if (d_localFound) {
            if (blockIdx.x == 0 && threadIdx.x == 0) {
                printf("Goal found! Reconstructing path...\n");
                int tempId = goalNodeId;
                int c = 0;
                while (tempId != -1 && c < width * height) {
                    path[c++] = tempId;
                    tempId = nodes[tempId].parent;
                }
                *pathLength = c;
                *found      = true;
                d_done      = true;
            }
        }
        gridGroup.sync();

        // if(threadIdx.x == 0 && blockIdx.x == 0)
        // {
        //     iterationCount++;
        //     // printf("Iteration %d\n", iterationCount);
        // }
    } // end while(!d_done)
}
