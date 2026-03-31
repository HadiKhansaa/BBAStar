#include <assert.h>

#include <cooperative_groups.h>

#include "unidirectional_astar.cuh"

namespace cg = cooperative_groups;

__constant__ int2 UNIDIRECTIONAL_NEIGHBOR_OFFSETS[8] = {
    {0, -1}, {1, -1}, {1, 0}, {1, 1},
    {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}
};

__device__ int d_unidirectional_bucketRangeStart;
__device__ int d_unidirectional_bucketRangeEnd;
__device__ int d_unidirectional_totalElementsInRange;
__device__ bool d_unidirectional_done;
__device__ bool d_unidirectional_localFound;

__global__ void initializeNodes(Node *nodes, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        nodes[idx].id = -1;
        nodes[idx].g = INT_MAX;
        nodes[idx].h = 0;
        nodes[idx].f = INT_MAX;
        nodes[idx].parent = -1;
    }
}

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
    int *totalExpandedNodes) {
    cg::grid_group gridGroup = cg::this_grid();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int blockExpansionBuffer[
        UNIDIRECTIONAL_SHARED_BUCKET_RANGE * UNIDIRECTIONAL_SHARED_MAX_BIN_SIZE];
    __shared__ int blockExpansionCounts[UNIDIRECTIONAL_SHARED_BUCKET_RANGE];
    __shared__ int blockWriteOffset;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_unidirectional_done = false;
        d_unidirectional_localFound = false;
    }
    gridGroup.sync();

    while (!d_unidirectional_done) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            d_unidirectional_bucketRangeStart = -1;
            d_unidirectional_bucketRangeEnd = -1;
            d_unidirectional_totalElementsInRange = 0;

            int elementsAccumulated = 0;
            bool rangeComplete = false;
            int bitMaskSize = (UNIDIRECTIONAL_MAX_BINS + 63) / 64;

            for (int maskIndex = 0; maskIndex < bitMaskSize && !rangeComplete; ++maskIndex) {
                unsigned long long tmpMask = binBitMask[maskIndex];
                while (tmpMask != 0ULL && !rangeComplete) {
                    int firstSetBit = __ffsll(tmpMask) - 1;
                    tmpMask &= ~(1ULL << firstSetBit);

                    int bucket = maskIndex * 64 + firstSetBit;
                    if (bucket >= UNIDIRECTIONAL_MAX_BINS) {
                        rangeComplete = true;
                        break;
                    }

                    if (d_unidirectional_bucketRangeStart == -1) {
                        d_unidirectional_bucketRangeStart = bucket;
                    }

                    elementsAccumulated += binCounts[bucket];
                    d_unidirectional_bucketRangeEnd = bucket;

                    int bucketSpan =
                        d_unidirectional_bucketRangeEnd - d_unidirectional_bucketRangeStart + 1;
                    if (bucketSpan >= UNIDIRECTIONAL_SHARED_BUCKET_RANGE ||
                        elementsAccumulated >= frontierSize) {
                        rangeComplete = true;
                    }
                }
            }

            if (d_unidirectional_bucketRangeStart == -1 || elementsAccumulated == 0) {
                d_unidirectional_done = true;
            } else {
                d_unidirectional_totalElementsInRange = elementsAccumulated;
            }
        }
        gridGroup.sync();

        if (d_unidirectional_done) {
            break;
        }

        int activeBucketCount =
            d_unidirectional_bucketRangeEnd - d_unidirectional_bucketRangeStart + 1;

        if (threadIdx.x < UNIDIRECTIONAL_SHARED_BUCKET_RANGE) {
            blockExpansionCounts[threadIdx.x] = 0;
        }
        __syncthreads();

        int assignedBucket = -1;
        int threadPosition = idx;
        for (int bucket = d_unidirectional_bucketRangeStart;
             bucket <= d_unidirectional_bucketRangeEnd;
             ++bucket) {
            int bucketSize = binCounts[bucket] * MAX_NEIGHBORS;
            if (threadPosition < bucketSize) {
                assignedBucket = bucket;
                break;
            }
            threadPosition -= bucketSize;
        }

        if (assignedBucket != -1) {
            int nodeIndex = threadPosition / MAX_NEIGHBORS;
            int neighborIndex = threadPosition % MAX_NEIGHBORS;
            if (neighborIndex == 0) {
                atomicAdd(totalExpandedNodes, 1);
            }

            int currentNodeId =
                openListBins[assignedBucket * UNIDIRECTIONAL_MAX_BIN_SIZE + nodeIndex];
            if (currentNodeId >= 0) {
                Node currentNode = nodes[currentNodeId];

                int2 offset = UNIDIRECTIONAL_NEIGHBOR_OFFSETS[neighborIndex];
                int xCurrent = currentNodeId % width;
                int yCurrent = currentNodeId / width;
                int xNeighbor = xCurrent + offset.x;
                int yNeighbor = yCurrent + offset.y;

                if (xNeighbor >= 0 && xNeighbor < width &&
                    yNeighbor >= 0 && yNeighbor < height) {
                    int neighborId = yNeighbor * width + xNeighbor;
                    if (grid[neighborId] == PASSABLE) {
                        bool isDiagonal = (abs(offset.x) + abs(offset.y) == 2);
                        int moveCost = isDiagonal ? DIAGONAL_COST : SCALE_FACTOR;
                        int tentativeG = currentNode.g + moveCost;

                        int oldG = atomicMin(&nodes[neighborId].g, tentativeG);
                        if (tentativeG < oldG) {
                            nodes[neighborId].id = neighborId;
                            nodes[neighborId].parent = currentNodeId;
                            nodes[neighborId].h =
                                static_cast<int>(unidirectionalHeuristic(neighborId, goalNodeId, width));
                            nodes[neighborId].f = tentativeG + nodes[neighborId].h;
                            nodes[neighborId].g = tentativeG;

                            if (neighborId == goalNodeId) {
                                d_unidirectional_localFound = true;
                            }

                            int adjustedF = nodes[neighborId].f - static_cast<int>(minFValue);
                            int binForNeighbor = adjustedF / UNIDIRECTIONAL_BUCKET_RANGE;
                            binForNeighbor = max(0, min(binForNeighbor, UNIDIRECTIONAL_MAX_BINS - 1));

                            int localBucket = binForNeighbor - d_unidirectional_bucketRangeStart;
                            if (localBucket >= 0 && localBucket < activeBucketCount) {
                                int position = atomicAdd(&blockExpansionCounts[localBucket], 1);
                                if (position < UNIDIRECTIONAL_SHARED_MAX_BIN_SIZE) {
                                    int sharedOffset =
                                        localBucket * UNIDIRECTIONAL_SHARED_MAX_BIN_SIZE + position;
                                    blockExpansionBuffer[sharedOffset] = neighborId;
                                } else {
                                    printf(
                                        "Shared expansion overflow at bucket %d\n",
                                        binForNeighbor);
                                }
                            } else {
                                int position = atomicAdd(&binCounts[binForNeighbor], 1);
                                if (position < UNIDIRECTIONAL_MAX_BIN_SIZE) {
                                    openListBins[
                                        binForNeighbor * UNIDIRECTIONAL_MAX_BIN_SIZE + position] =
                                        neighborId;
                                    if (position == 0) {
                                        int maskIndex = binForNeighbor / 64;
                                        unsigned long long mask =
                                            1ULL << (binForNeighbor % 64);
                                        atomicOr(&binBitMask[maskIndex], mask);
                                    }
                                } else {
                                    printf("Open-list overflow at bucket %d\n", binForNeighbor);
                                }
                            }
                        }
                    }
                }
            }
        }

        gridGroup.sync();

        if (blockIdx.x == 0 && threadIdx.x < activeBucketCount) {
            int bucket = d_unidirectional_bucketRangeStart + threadIdx.x;
            binCounts[bucket] = 0;
            int maskIndex = bucket / 64;
            unsigned long long clearMask = ~(1ULL << (bucket % 64));
            atomicAnd(&binBitMask[maskIndex], clearMask);
        }

        __threadfence();
        gridGroup.sync();

        for (int bucket = d_unidirectional_bucketRangeStart;
             bucket <= d_unidirectional_bucketRangeEnd;
             ++bucket) {
            int localBucket = bucket - d_unidirectional_bucketRangeStart;
            int count = blockExpansionCounts[localBucket];

            if (count > 0) {
                if (threadIdx.x == 0) {
                    blockWriteOffset = atomicAdd(&binCounts[bucket], count);
                    int maskIndex = bucket / 64;
                    unsigned long long mask = 1ULL << (bucket % 64);
                    atomicOr(&binBitMask[maskIndex], mask);
                }
                __syncthreads();

                int offset = blockWriteOffset;
                for (int i = threadIdx.x; i < count; i += blockDim.x) {
                    int sharedOffset =
                        localBucket * UNIDIRECTIONAL_SHARED_MAX_BIN_SIZE + i;
                    int globalOffset = bucket * UNIDIRECTIONAL_MAX_BIN_SIZE + offset + i;
                    if (offset + i < UNIDIRECTIONAL_MAX_BIN_SIZE) {
                        openListBins[globalOffset] = blockExpansionBuffer[sharedOffset];
                    }
                }
            }
            __syncthreads();
        }

        if (d_unidirectional_localFound) {
            if (blockIdx.x == 0 && threadIdx.x == 0) {
                int tempId = goalNodeId;
                int count = 0;
                while (tempId != -1 && count < width * height) {
                    path[count++] = tempId;
                    tempId = nodes[tempId].parent;
                }
                *pathLength = count;
                *found = true;
                d_unidirectional_done = true;
            }
        }
        gridGroup.sync();
    }
}
