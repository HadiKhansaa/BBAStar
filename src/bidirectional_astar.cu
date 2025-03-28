#include "astar_helper.cuh"
#include "bidirectional_astar.cuh"
#include "constants.cuh"

__global__ void initializeBiNodes(BiNode* nodes, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        nodes[idx].id = idx;
        nodes[idx].g_forward = INT_MAX;
        nodes[idx].h_forward = 0;
        nodes[idx].f_forward = INT_MAX;
        nodes[idx].parent_forward = -1;
        nodes[idx].g_backward = INT_MAX;
        nodes[idx].h_backward = 0;
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
    int binBitMaskSize, int frontierSize, 
    int *totalExpandedNodes, int* expandedNodes,
    int* firstNonEmptyMask, int* lastNonEmptyMask
    , BidirectionalState* state)
{
    // thread local variables for direction of search
    ThreadAssignment threadAssignment = UNASSIGNNED; // to be determined for each thread
    int bucketRangeStart; // to be determined for each thread
    int bucketRangeEnd;   // to be determined for each thread
    int totalElementsInRange; // to be determined for each thread
    int *binCountsPtr; // to be determined for each thread
    unsigned long long *binBitMaskPtr; // to be determined for each thread
    int *openListBinsPtr; // to be determined for each thread
    int *expansionBuffersPtr; // to be determined for each thread
    int *expansionCountsPtr; // to be determined for each thread
    int gridSize = width * height; // total number of nodes in the grid

    // Cooperative groups for grid-wide synchronization.
    cg::grid_group gridGroup = cg::this_grid();

    // Linear thread ID across the entire grid.
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int totalThreads = gridGroup.size();   

    // Main bidirectional A* loop.
    while (!state->d_done_forward || !state->d_done_backward)
    {
        // printf("forward: %d\n", forward);
        // Thread 0 of block 0 computes the active bucket range forward.
        // Thread 1 of block 0 computes the active bucket range backward.
        if (blockIdx.x == 0 && (threadIdx.x == 0 || threadIdx.x == 1)) {
            // initialization for each thread
            bucketRangeStart = -1;
            bucketRangeEnd   = -1;
            totalElementsInRange = 0;

            // temporarly assigned pointers will change later for each thread
            binCountsPtr = threadIdx.x == 0 ? forward_binCounts : backward_binCounts;
            binBitMaskPtr = threadIdx.x == 0 ? forward_binBitMask : backward_binBitMask;

            bool done = false;
            for (int i = 0; i < binBitMaskSize - 1 && !done; ++i) {
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
                    if (totalElementsInRange + countHere < frontierSize) {
                        totalElementsInRange += countHere;
                        bucketRangeEnd = bucket;
                    } else {
                        bucketRangeEnd = bucket;
                        done = true;
                    }
                }
            }
            // edge case: first bucket is > frontierSize
            if (bucketRangeStart == bucketRangeEnd && totalElementsInRange == 0 &&
                bucketRangeStart + 1 < MAX_BINS && binCountsPtr[bucketRangeStart + 1] > 0)
                totalElementsInRange = binCountsPtr[bucketRangeStart + 1];
            
            // 2 flags for forward pass and backward pass
            else if(totalElementsInRange == 0 && threadIdx.x == 0)
                state->d_done_forward = true;
            
            else if(totalElementsInRange == 0 && threadIdx.x == 1)
                state->d_done_backward = true;
            
            // broadcast to all other blocks
            if (threadIdx.x == 0 && totalElementsInRange > 0)
            {
                state->global_forward_bucketRangeStart = bucketRangeStart;
                state->global_forward_bucketRangeEnd = bucketRangeEnd;
                state->global_forward_totalElementsInRange = totalElementsInRange;
            }
            else if (threadIdx.x == 1 && totalElementsInRange > 0)
            {
                state->global_backward_bucketRangeStart = bucketRangeStart;
                state->global_backward_bucketRangeEnd = bucketRangeEnd;
                state->global_backward_totalElementsInRange = totalElementsInRange;
            }

            // active elements forward
// #ifdef DEBUG
//             if(threadIdx.x == 0)
//             {
//                 printf("Active elements forward: %d\n", totalElementsInRange);
//                 printf("Bucket range forward: %d - %d\n", state->global_forward_bucketRangeStart, state->global_forward_bucketRangeEnd);

//             }
//             else
//             {
//                 printf("Active elements backward: %d\n", totalElementsInRange);
//                 printf("Bucket range backward: %d - %d\n", state->global_backward_bucketRangeStart, state->global_backward_bucketRangeEnd);
//             }

//             // wait(100000);
// #endif
        }

        gridGroup.sync(); // sync all blocks
        if (state->d_done_forward && state->d_done_backward)
            break;

        // Assignment of thread-specific variables
        // first 8 * totalElementsInRange threads are responsible for forward search
        if(idx < state->global_forward_totalElementsInRange * MAX_NEIGHBORS)
            threadAssignment = FORWARD;
        // second 8 * totalElementsInRange threads are responsible for backward search
        else if (idx >= state->global_forward_totalElementsInRange * MAX_NEIGHBORS
        && idx < state->global_forward_totalElementsInRange * MAX_NEIGHBORS + state->global_backward_totalElementsInRange * MAX_NEIGHBORS)
            threadAssignment = BACKWARD;

        // decide thread <-> direction
        bucketRangeStart = threadAssignment == FORWARD ? state->global_forward_bucketRangeStart : state->global_backward_bucketRangeStart;
        bucketRangeEnd = threadAssignment == FORWARD ? state->global_forward_bucketRangeEnd : state->global_backward_bucketRangeEnd;

        openListBinsPtr       = threadAssignment == FORWARD ? forward_openListBins      : backward_openListBins;
        expansionBuffersPtr   = threadAssignment == FORWARD ? forward_expansionBuffers  : backward_expansionBuffers;
        expansionCountsPtr    = threadAssignment == FORWARD ? forward_expansionCounts   : backward_expansionCounts;
        binCountsPtr          = threadAssignment == FORWARD ? forward_binCounts         : backward_binCounts;
        binBitMaskPtr         = threadAssignment == FORWARD ? forward_binBitMask        : backward_binBitMask;

        // Work Assignment: each 8 threads are responsible for one node in the open list cocnsecutively
        int assignedBucket = -1;
        int threadPosition = idx; // linear index among (node, neighbor) pairs.
        int assignmentOffset = 0;
        if(threadAssignment == BACKWARD)
            assignmentOffset = state->global_forward_totalElementsInRange * 8;
        if(threadAssignment != UNASSIGNNED)
        {
            assignedBucket = -1;
            threadPosition = idx - assignmentOffset; // linear index among (node, neighbor) pairs.
            for (int b = bucketRangeStart; b <= bucketRangeEnd; ++b) {
                int bucketSize = binCountsPtr[b] * MAX_NEIGHBORS;
                if (threadPosition < bucketSize) {
                    assignedBucket = b;
                    break;
                }
                threadPosition -= bucketSize;
            }
        }
        
        if (assignedBucket != -1 && threadAssignment != UNASSIGNNED) {
            // Count this expansion.

            int nodeIndex     = threadPosition / MAX_NEIGHBORS;  
            int neighborIndex = threadPosition % MAX_NEIGHBORS;  

            expandedNodes[atomicAdd(totalExpandedNodes, 1)] = nodeIndex;
            
            int currentNodeId = openListBinsPtr[assignedBucket * MAX_BIN_SIZE + nodeIndex];
            BiNode currentNode = nodes[currentNodeId];

            // Early pruning: skip if the current node’s f-value is not promising.
            int currentF = threadAssignment == FORWARD ? currentNode.f_forward : currentNode.f_backward;
            if (state->globalBestCost!=INT_MAX || currentF < state->globalBestCost)
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
                        int moveCost = isDiagonal ? DIAGONAL_COST : SCALE_FACTOR;
                        int tentativeG = (threadAssignment == FORWARD ? currentNode.g_forward : currentNode.g_backward) + moveCost;

                        // Atomically update the neighbor's cost using the appropriate field.
                        int oldG;
                        if (threadAssignment == FORWARD) {
                            oldG = atomicMin(&nodes[neighborId].g_forward, tentativeG);
                        } else {
                            oldG = atomicMin(&nodes[neighborId].g_backward, tentativeG);
                        }
                        if (tentativeG < oldG) {
                            // Update neighbor's fields accordingly.
                            if (threadAssignment == FORWARD) {
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
                            if (threadAssignment == FORWARD) {
                                if (neighborId == targetNodeId) {
                                    unsigned int candidateCost = nodes[neighborId].f_forward;
                                    int oldCost = atomicMin(&state->globalBestCost, candidateCost);
                                }
                                else if (nodes[neighborId].g_backward < gridSize + 1) {
                                    unsigned int candidateCost = nodes[neighborId].g_forward + nodes[neighborId].g_backward;
                                    int oldCost = atomicMin(&state->globalBestCost, candidateCost);
                                }
                            } else {
                                if (neighborId == startNodeId) {
                                    unsigned int candidateCost = nodes[neighborId].f_backward;
                                    int oldCost = atomicMin(&state->globalBestCost, candidateCost);
                                }
                                else if (nodes[neighborId].g_forward < gridSize + 1) {
                                    unsigned int candidateCost = nodes[neighborId].g_forward + nodes[neighborId].g_backward;
                                    int oldCost = atomicMin(&state->globalBestCost, candidateCost);
                                } 
                            }

                            // EXPAND
                            // Compute the bin for the neighbor based on its updated f-value.
                            int newF = threadAssignment == FORWARD ? nodes[neighborId].f_forward : nodes[neighborId].f_backward;
// #ifdef DEBUG
//                             // printf("f-value of expanded node: %d\n", newF);
// #endif
                            int minFValue = DIAGONAL_COST * width;
                            int adjustedF = newF - minFValue;
                            int binForNghbr = (int)(adjustedF / BUCKET_F_RANGE);
                            binForNghbr = max(0, min(binForNghbr, MAX_BINS - 1));

                            // Insert neighbor into the expansion buffer or open list bins.
                            if (binForNghbr >= bucketRangeStart && binForNghbr <= bucketRangeEnd) {
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

        // first forward_totalNbElementsExpansionBuffer threads are responsible for forward copy
        // if(threadIdx.x < forward_totalNbElementsExpansionBuffer)
        if(blockIdx.x == 0) // block 0 is responsible for copying forward pass
        {
            expansionCountsPtr = forward_expansionCounts;
            openListBinsPtr = forward_openListBins;
            binCountsPtr = forward_binCounts;
            binBitMaskPtr = forward_binBitMask;
            expansionBuffersPtr = forward_expansionBuffers;
            bucketRangeStart = state->global_forward_bucketRangeStart;
            bucketRangeEnd = state->global_forward_bucketRangeEnd;

            for (int bucket = bucketRangeStart; bucket <= bucketRangeEnd; ++bucket) {
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
                    }
                }
                __syncthreads();
                expansionCountsPtr[bucket] = 0;
            }
        }
        // else if (threadIdx.x < forward_totalNbElementsExpansionBuffer + backward_totalNbElementsExpansionBuffer)
        else if(blockIdx.x == 1) // block 1 is responsible for copying backward pass
        {
            expansionCountsPtr = backward_expansionCounts;
            openListBinsPtr = backward_openListBins;
            binCountsPtr = backward_binCounts;
            binBitMaskPtr = backward_binBitMask;
            expansionBuffersPtr = backward_expansionBuffers;
            bucketRangeStart = state->global_backward_bucketRangeStart;
            bucketRangeEnd = state->global_backward_bucketRangeEnd;

            for (int bucket = bucketRangeStart; bucket <= bucketRangeEnd; ++bucket) {
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
                    }
                }
                __syncthreads();
                expansionCountsPtr[bucket] = 0;
            }
        }
        __threadfence();
        gridGroup.sync();

        // gridGroup.sync();
    } // end while(!d_done)
    gridGroup.sync();

    // When done, reconstruct the complete path.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        if (state->globalBestCost == INT_MAX) {
            *found = false;
            *pathLength = 0;
            printf("Path not found\n");
        } else {
            *found = true;
            printf("Path found\n");
            // printf("Meeting node: (%d, %d)\n", globalBestNode.id/width, globalBestNode.id%width);
            printf("cost: %d\n", state->globalBestCost/SCALE_FACTOR);
        }
    }

    // found reconstruct path
    // if(*found)
    // if(*found && blockIdx.x == 0 && (threadIdx.x == 0 || threadIdx.x == 1))
    //     constractBidirectionalPath(startNodeId, targetNodeId, globalBestNode, path, pathLength, nodes);
    // end of kernel
}
