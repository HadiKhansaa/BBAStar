#include "astar_helper.cuh"
#include "bidirectional_astar.cuh"
#include "constants.cuh"

__global__ void initializeBiNodes(BiNode* nodes, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        nodes[idx].id = idx;
        nodes[idx].g_forward = INT_MAX; // set to max value
        nodes[idx].h_forward = 0;
        nodes[idx].f_forward = INT_MAX; // set to max value
        nodes[idx].parent_forward = -1;
        nodes[idx].g_backward = INT_MAX; // set to max value
        nodes[idx].h_backward = 0;
        nodes[idx].f_backward = INT_MAX; // set to max value
        nodes[idx].parent_backward = -1;
        nodes[idx].openListAddress_forward = -1;
        nodes[idx].openListAddress_backward = -1;
    }
}

__global__ void biAStarMultipleBucketsSingleKernel(
    int *grid, int width, int height,    // grid dimensions and obstacle grid
    int startNodeId, int targetNodeId,                   // for forward search, this is the goal; for backward, the start
    BiNode *nodes,                      // array of BiNodes (both forward and backward fields integrated)
    // Open list arrays for forward search
    int *forward_openListBins, int *forward_binCounts, uint64_t *forward_binBitMask,
    int *forward_expansionBuffers, int *forward_expansionCounts,
    // Open list arrays for backward search
    int *backward_openListBins, int *backward_binCounts, uint64_t *backward_binBitMask,
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
    uint64_t *binBitMaskPtr; // to be determined for each thread
    int *openListBinsPtr; // to be determined for each thread
    int *expansionBuffersPtr; // to be determined for each thread
    int *expansionCountsPtr; // to be determined for each thread
    // int gridSize = width * height; // total number of nodes in the grid

    // Cooperative groups for grid-wide synchronization.
    cg::grid_group gridGroup = cg::this_grid();

    // Linear thread ID across the entire grid.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridGroup.size();   
    // Main bidirectional A* loop.
    while (!state->d_done_forward && !state->d_done_backward)
    {
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
                uint64_t tmpMask = binBitMaskPtr[i];
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
                    if (totalElementsInRange < frontierSize) {
                        totalElementsInRange += countHere;
                        bucketRangeEnd = bucket;
                    } else {
                        // bucketRangeEnd = bucket;
                        done = true;
                    }
                }
            }

            // 2 flags for forward pass and backward pass
            if(totalElementsInRange == 0 && threadIdx.x == 0)
            {
                state->d_done_forward = true;
                state->global_forward_bucketRangeStart = -1;
                state->global_forward_bucketRangeEnd = -1;
                state->global_forward_totalElementsInRange = 0;
            }
            
            if(totalElementsInRange == 0 && threadIdx.x == 1)
            {
                state->d_done_backward = true;
                state->global_forward_bucketRangeStart = -1;
                state->global_forward_bucketRangeEnd = -1;
                state->global_forward_totalElementsInRange = 0;
            }
            
            // broadcast to all other blocks
            if (threadIdx.x == 0 && totalElementsInRange > 0)
            {
                state->global_forward_bucketRangeStart = bucketRangeStart;
                state->global_forward_bucketRangeEnd = bucketRangeEnd;
                state->global_forward_totalElementsInRange = totalElementsInRange;
            }
            if (threadIdx.x == 1 && totalElementsInRange > 0)
            {
                state->global_backward_bucketRangeStart = bucketRangeStart;
                state->global_backward_bucketRangeEnd = bucketRangeEnd;
                state->global_backward_totalElementsInRange = totalElementsInRange;
            }

            // if first bucket is greater than the best cost, then we are done in this direction
            if(threadIdx.x == 0 && state->global_forward_bucketRangeStart * BUCKET_F_RANGE + (DIAGONAL_COST * (width - 1)) >= state->globalBestCost)
            {
                state->d_done_forward = true;
            }
            if(threadIdx.x == 1 && state->global_backward_bucketRangeStart * BUCKET_F_RANGE  + (DIAGONAL_COST * (width - 1)) >= state->globalBestCost)
            {
                state->d_done_backward = true;
            }

            // debugging
// #ifdef DEBUG
            // if(threadIdx.x == 0)
            // {
            //     printf("current best cost: %d\n", state->globalBestCost);

            //     printf("Active elements forward: %d\n", totalElementsInRange);
            //     printf("Bucket range forward: %d - %d\n\n", state->global_forward_bucketRangeStart, state->global_forward_bucketRangeEnd);

            // }
            // if(threadIdx.x == 1)
            // {
            //     printf("Active elements backward: %d\n", totalElementsInRange);
            //     printf("Bucket range backward: %d - %d\n\n", state->global_backward_bucketRangeStart, state->global_backward_bucketRangeEnd);
            // }

            // wait(10000000);
// #endif
        }

        // clear expansion count buffers
        if(blockIdx.x == 0)
        {
            for(int i = threadIdx.x; i < MAX_BINS ; i += blockDim.x) {
            forward_expansionCounts[i] = 0;
            backward_expansionCounts[i] = 0;
            }
        }

        gridGroup.sync(); // sync all blocks

        // check minForward and minFBackward, if their sum is greater than the global best cost, then we are done
        // if(blockIdx.x == 0 && threadIdx.x == 0) {
        //     if (state->global_forward_bucketRangeStart != -1 && state->global_backward_bucketRangeStart != -1) {
        //         unsigned int minForward = state->global_forward_bucketRangeStart * BUCKET_F_RANGE + (DIAGONAL_COST * (width - 1));
        //         unsigned int minBackward = state->global_backward_bucketRangeStart * BUCKET_F_RANGE + (DIAGONAL_COST * (width - 1));
        //         if (minForward + minBackward >= state->globalBestCost) {
        //             state->d_done_forward = true;
        //             state->d_done_backward = true;
        //         }
        //     }
        // }

        // gridGroup.sync(); // sync all blocks

        if (state->d_done_forward && state->d_done_backward)
            break;

        // Assignment of thread-specific variables
        // first 8 * totalElementsInRange threads are responsible for forward search
        if(idx < state->global_forward_totalElementsInRange * MAX_NEIGHBORS)
            threadAssignment = FORWARD;
        // second 8 * totalElementsInRange threads are responsible for backward search
        else if (idx >= state->global_forward_totalElementsInRange * MAX_NEIGHBORS
        && idx < state->global_forward_totalElementsInRange * MAX_NEIGHBORS +
        state->global_backward_totalElementsInRange * MAX_NEIGHBORS)
            threadAssignment = BACKWARD;

        
        // if both searches are done, break
        if(threadAssignment == FORWARD && state->d_done_forward)
        {
            threadAssignment = UNASSIGNNED;
            totalElementsInRange = 0;
            state->global_forward_totalElementsInRange = 0;
        }

        if(threadAssignment == BACKWARD && state->d_done_backward)
        {
            threadAssignment = UNASSIGNNED;
            totalElementsInRange = 0;
            state->global_backward_totalElementsInRange = 0;
        }
        
        // assert(state->global_forward_totalElementsInRange * MAX_NEIGHBORS +
        //     state->global_backward_totalElementsInRange * MAX_NEIGHBORS <= TOTAL_THREADS);

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

            // atomicAdd(totalExpandedNodes, 1);
            
            int currentNodeId = openListBinsPtr[assignedBucket * MAX_BIN_SIZE + nodeIndex];
            BiNode& currentNode = nodes[currentNodeId];

            // check if the current node is valid i.e. in the correct bucket
            unsigned int minFValue = DIAGONAL_COST * (width - 1);
            unsigned int appropriateBucket = threadAssignment == FORWARD ? (currentNode.f_forward - minFValue)/BUCKET_F_RANGE
             : (currentNode.f_backward - minFValue)/BUCKET_F_RANGE;


            // Early pruning: skip if the current node’s f-value is not promising.
            unsigned int currentF = threadAssignment == FORWARD ? currentNode.f_forward : currentNode.f_backward;

            if ( 
               ((threadAssignment == FORWARD && (nodes[currentNodeId].openListAddress_forward == -1 || (assignedBucket * MAX_BIN_SIZE + nodeIndex) == nodes[currentNodeId].openListAddress_forward)) 
            || (threadAssignment == BACKWARD && (nodes[currentNodeId].openListAddress_backward == -1 || (assignedBucket * MAX_BIN_SIZE + nodeIndex) == nodes[currentNodeId].openListAddress_backward))
            )
            
            && appropriateBucket == assignedBucket && currentF < state->globalBestCost)
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
                    
                    if (grid[neighborId] == PASSABLE &&
                        !(threadAssignment == FORWARD && neighborId == currentNode.parent_forward) &&
                        !(threadAssignment == BACKWARD && neighborId == currentNode.parent_backward)
                    ) {  // if passable or not the parent node                       
                        bool isDiagonal = (abs(dx) + abs(dy) == 2);
                        unsigned int moveCost = isDiagonal ? DIAGONAL_COST : SCALE_FACTOR;
                        unsigned int tentativeG = (threadAssignment == FORWARD ? currentNode.g_forward : currentNode.g_backward) + moveCost;

                        // Atomically update the neighbor's cost using the appropriate field.
                        unsigned int oldG;
                        if (threadAssignment == FORWARD) {
                            oldG = atomicMin(&nodes[neighborId].g_forward, tentativeG);
                        } else {
                            oldG = atomicMin(&nodes[neighborId].g_backward, tentativeG);
                        }
                        if (tentativeG < oldG) {
                            __threadfence(); // Ensure the update is visible to all threads before proceeding.

                            expandedNodes[atomicAdd(totalExpandedNodes, 1)] = neighborId;

                            // Update neighbor's fields accordingly.
                            if (threadAssignment == FORWARD) {
                                if(tentativeG == nodes[neighborId].g_forward) {
                                    nodes[neighborId].id = neighborId;
                                    nodes[neighborId].parent_forward = currentNodeId;
                                    nodes[neighborId].h_forward = heuristic(neighborId, targetNodeId, width);
                                    nodes[neighborId].f_forward = tentativeG + nodes[neighborId].h_forward;
                                    nodes[neighborId].g_forward = tentativeG;
                                }
                            } else {
                                if(tentativeG == nodes[neighborId].g_backward) {
                                    nodes[neighborId].id = neighborId;
                                    nodes[neighborId].parent_backward = currentNodeId;
                                    nodes[neighborId].h_backward = heuristic(neighborId, startNodeId, width);
                                    nodes[neighborId].f_backward = tentativeG + nodes[neighborId].h_backward;
                                    nodes[neighborId].g_backward = tentativeG;
                                }
                            }

                            // bool localPathFound = false;
                            // Check if this neighbor has already been reached from the opposite search.
                            if (threadAssignment == FORWARD && nodes[neighborId].g_backward != INT_MAX) {
                                unsigned int candidateCost = nodes[neighborId].g_forward + nodes[neighborId].g_backward;
                                unsigned int oldCost = atomicMin(&state->globalBestCost, candidateCost);
                                if(candidateCost < oldCost)
                                    state->globalBestNode = nodes[neighborId]; // needs changing
                                
                                // if(neighborId == targetNodeId)
                                    // localPathFound = true;
                                // continue;
                            } 
                            
                            if (threadAssignment == BACKWARD && nodes[neighborId].g_forward != INT_MAX) {
                                unsigned int candidateCost = nodes[neighborId].g_backward + nodes[neighborId].g_forward;
                                unsigned int oldCost = atomicMin(&state->globalBestCost, candidateCost);
                                if(candidateCost < oldCost)    
                                    state->globalBestNode = nodes[neighborId]; // needs changing
                                
                                // if(neighborId == startNodeId) // continue expanding otherwise
                                    // localPathFound = true;
                                // continue;

                            }

                            // don't expand if we found a local path
                            // don't expand the node of the opposite search
                            if((threadAssignment == FORWARD && tentativeG == nodes[neighborId].g_forward) || (threadAssignment == BACKWARD && tentativeG == nodes[neighborId].g_backward))
                            {
                                // EXPAND
                                // Compute the bin for the neighbor based on its updated f-value.
                                unsigned int newF = threadAssignment == FORWARD ? nodes[neighborId].f_forward : nodes[neighborId].f_backward;
// #ifdef DEBUG
                                // if(FORWARD == threadAssignment)
                                //     printf("forward f-value of expanded node: %d\n", newF);
                                // else
                                //     printf("backward f-value of expanded node: %d\n", newF);
// #endif
                                unsigned int binForNghbr = binForNode(newF, width);
                                assert(binForNghbr < MAX_BINS);
                                binForNghbr = max(0, min(binForNghbr, MAX_BINS - 1));
                                
                                if(newF < state->globalBestCost) // only expand if the f-value is less than the global best cost
                                {
                                    
                                    if (binForNghbr >= bucketRangeStart && binForNghbr <= bucketRangeEnd) {
                                        unsigned int pos = atomicAdd(&expansionCountsPtr[binForNghbr], 1);

                                        assert(pos < MAX_BIN_SIZE);

                                        unsigned int offset = binForNghbr * MAX_BIN_SIZE + pos;
                                        expansionBuffersPtr[offset] = neighborId;
                                        threadAssignment == FORWARD ? nodes[neighborId].openListAddress_forward = offset : nodes[neighborId].openListAddress_backward = offset;

                                        // printf("Adding neighbor %d to expansion buffer %d\n", neighborId, binForNghbr);
                                    } else {
                                        unsigned int pos = atomicAdd(&binCountsPtr[binForNghbr], 1);
                                        assert(pos < MAX_BIN_SIZE);
                                        unsigned int offset = binForNghbr * MAX_BIN_SIZE + pos;
                                        openListBinsPtr[offset] = neighborId;
                                        threadAssignment == FORWARD ? nodes[neighborId].openListAddress_forward = offset : nodes[neighborId].openListAddress_backward = offset;


                                        if (pos == 0) {
                                            int maskIndex = binForNghbr / 64;
                                            uint64_t m = 1ULL << (binForNghbr % 64);
                                            atomicOr(&binBitMaskPtr[maskIndex], m);
                                            // printf("Adding neighbor %d to open list bin %d\n", neighborId, binForNghbr);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        gridGroup.sync();

        if(state->d_done_forward && state->d_done_backward)
            break;

        // int totalElementsProcessed =  state->global_forward_totalElementsInRange * MAX_NEIGHBORS + 
        // state->global_backward_totalElementsInRange * MAX_NEIGHBORS;

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
                    int remainder = 0;
                    if(state->global_forward_totalElementsInRange * MAX_NEIGHBORS > TOTAL_THREADS)
                    {
                        remainder = (TOTAL_THREADS - state->global_forward_totalElementsInRange * MAX_NEIGHBORS)/8;
                    }
                    binCountsPtr[bucket] = eCount + remainder; // + remainder 
                    for (int i = threadIdx.x; i < eCount; i += blockDim.x) {
                        int offset = bucket * MAX_BIN_SIZE + i + remainder; // + remainder
                        openListBinsPtr[offset] = expansionBuffersPtr[offset];
                        if(i == 0) // first thread does the atomic or
                        {
                            int maskIndex = bucket / 64;
                            uint64_t m = 1ULL << (bucket % 64);
                            atomicOr(&binBitMaskPtr[maskIndex], m);
                        }
                    }
                } else {
                    if (threadIdx.x == 0) {
                        expansionCountsPtr[bucket] = 0;
                        binCountsPtr[bucket] = 0;
                        int maskIndex = bucket / 64;
                        uint64_t m = ~(1ULL << (bucket % 64));
                        atomicAnd(&binBitMaskPtr[maskIndex], m);
                    }
                }
                __syncthreads();
            }
        }
        // else if (threadIdx.x < forward_totalNbElementsExpansionBuffer + backward_totalNbElementsExpansionBuffer)
        if(blockIdx.x == 1) // block 1 is responsible for copying backward pass
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
                    int remainder = 0;
                    if(state->global_forward_totalElementsInRange * MAX_NEIGHBORS + 
                        state->global_backward_totalElementsInRange * MAX_NEIGHBORS > TOTAL_THREADS)
                    {
                        remainder = (TOTAL_THREADS - state->global_forward_totalElementsInRange * MAX_NEIGHBORS + 
                        state->global_backward_totalElementsInRange * MAX_NEIGHBORS) / 8;
                    }
                    if(remainder < 0) break; // we didn't process anything

                    binCountsPtr[bucket] = eCount + remainder;
                    for (int i = threadIdx.x; i < eCount; i += blockDim.x) {
                        int offset = bucket * MAX_BIN_SIZE + i + remainder;
                        openListBinsPtr[offset] = expansionBuffersPtr[offset];
                        if(i==0)
                        {
                            int maskIndex = bucket / 64;
                            uint64_t m = 1ULL << (bucket % 64);
                            atomicOr(&binBitMaskPtr[maskIndex], m);
                        }
                    }
                } else {
                    if (threadIdx.x == 0) {
                        expansionCountsPtr[bucket] = 0;
                        binCountsPtr[bucket] = 0;
                        int maskIndex = bucket / 64;
                        uint64_t m = ~(1ULL << (bucket % 64));
                        atomicAnd(&binBitMaskPtr[maskIndex], m);
                    }
                }
                __syncthreads();
            }
        }
        // __threadfence();
        gridGroup.sync();

        // gridGroup.sync();
    } // end while(!d_done)
    gridGroup.sync();

    // When done, reconstruct the complete path.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        if (state->globalBestCost == INT_MAX) {
            *found = false;
            *pathLength = 0;
            // printf("Path not found\n");
        } else {
            *found = true;
            printf("Path found with cost: %d\n", state->globalBestCost/SCALE_FACTOR);
            printf("Meeting node: (%d, %d)\n", state->globalBestNode.id/width, state->globalBestNode.id%width);
        }
    }

    // found reconstruct path
    if(*found && blockIdx.x == 0 && (threadIdx.x == 0 || threadIdx.x == 1))
        constractBidirectionalPath(startNodeId, targetNodeId, state->globalBestNode, path, pathLength, nodes);
    // end of kernel
}
