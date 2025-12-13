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

 // 8-direction neighbor offsets.
__constant__ int2 NEIGHBOR_OFFSETS[8] = {
    {0,-1}, {1,-1}, {1,0}, {1,1},
    {0,1}, {-1,1}, {-1,0}, {-1,-1}
};

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
    bool *found, int *path, int *pathLength, int frontierSize, 
    int *totalExpandedNodes, int* expandedNodes
    , BidirectionalState* state)
{
    // thread local variables for direction of search
    ThreadAssignment threadAssignment = UNASSIGNNED; // to be determined for each thread
    int bucketRangeStart; // to be determined for each thread
    int bucketRangeEnd;   // to be determined for each thread
    int *binCountsPtr; // to be determined for each thread
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
    while (!state->d_done_forward || !state->d_done_backward)
    {
        // Thread 0 of block 0 computes the active bucket range forward.
        // Thread 1 of block 0 computes the active bucket range backward.
        if (blockIdx.x == 0 && ((threadIdx.x == 0 && !state->d_done_forward) || (threadIdx.x == 1 && !state->d_done_backward))) {
            const bool isForward = (threadIdx.x == 0);

            // choose the right arrays
            binCountsPtr = isForward ? forward_binCounts : backward_binCounts;

            // get previous bucket start for this direction
            int prevStart = isForward
                ? state->global_forward_bucketRangeStart
                : state->global_backward_bucketRangeStart;

            // on the first iteration or invalid value, start from 0
            if (prevStart < 0 || prevStart >= MAX_BINS)
                prevStart = 0;

            int localStart = -1;
            int localEnd   = -1;
            int localTotal = 0;

            // 1) find first non-empty bucket at or after prevStart
            int b = prevStart;
            while (b < MAX_BINS && binCountsPtr[b] == 0) {
                ++b;
            }

            if (b == MAX_BINS) {
                // no work in this direction
                if (isForward) {
                    state->d_done_forward = true;
                    state->global_forward_bucketRangeStart        = -1;
                    state->global_forward_bucketRangeEnd          = -1;
                    state->global_forward_totalElementsInRange    = 0;
                } else {
                    state->d_done_backward = true;
                    state->global_backward_bucketRangeStart       = -1;
                    state->global_backward_bucketRangeEnd         = -1;
                    state->global_backward_totalElementsInRange   = 0;
                }
            } else {
                localStart = b;

                // 2) extend bucket range to the right until we reach frontierSize
                for (int i = b; i < MAX_BINS && localTotal < frontierSize; ++i) {
                    int c = binCountsPtr[i];
                    if (c > 0) {
                        localEnd = i;
                        localTotal += c;
                    }
                }

                if (isForward) {
                    state->global_forward_bucketRangeStart        = localStart;
                    state->global_forward_bucketRangeEnd          = localEnd;
                    state->global_forward_totalElementsInRange    = localTotal;
                } else {
                    state->global_backward_bucketRangeStart       = localStart;
                    state->global_backward_bucketRangeEnd         = localEnd;
                    state->global_backward_totalElementsInRange   = localTotal;
                }
            }

            // early stopping when the min f in this direction cannot beat globalBestCost
            unsigned int minFBase = DIAGONAL_COST * (width - 1);

            if (isForward &&
                state->global_forward_bucketRangeStart != -1 &&
                state->global_forward_bucketRangeStart * BUCKET_F_RANGE + minFBase >= state->globalBestCost)
            {
                state->d_done_forward = true;
            }

            if (!isForward &&
                state->global_backward_bucketRangeStart != -1 &&
                state->global_backward_bucketRangeStart * BUCKET_F_RANGE + minFBase >= state->globalBestCost)
            {
                state->d_done_backward = true;
            }



            // debugging
// #ifdef DEBUG
            if(threadIdx.x == 0)
            {
                printf("current best cost: %d\n", state->globalBestCost);

                printf("Active elements forward: %d\n", localTotal);
                printf("Bucket range forward: %d - %d\n\n", state->global_forward_bucketRangeStart, state->global_forward_bucketRangeEnd);

            }
            if(threadIdx.x == 1)
            {
                printf("Active elements backward: %d\n", localTotal);
                printf("Bucket range backward: %d - %d\n\n", state->global_backward_bucketRangeStart, state->global_backward_bucketRangeEnd);
            }

            wait(10000000);
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

        // if (blockIdx.x == 0 && threadIdx.x == 0) {
        //     if (state->global_forward_bucketRangeStart != -1 && state->global_backward_bucketRangeStart != -1 && state->globalBestCost != INT_MAX) {
        //         unsigned int minFBase = DIAGONAL_COST * (width - 1);
        //         unsigned int minForward = state->global_forward_bucketRangeStart * BUCKET_F_RANGE + minFBase;
        //         unsigned int minBackward = state->global_backward_bucketRangeStart * BUCKET_F_RANGE + minFBase;
        //         if ((minForward + minBackward) >= state->globalBestCost * 1.9) {
        //             state->d_done_forward = true;
        //             state->d_done_backward = true;
        //         }
        //     }
        // }

        gridGroup.sync(); // sync all blocks

        // if(state->d_done_forward || state->d_done_backward)
        //     printf("%d %d\n", state->d_done_forward, state->d_done_backward);

        if (state->d_done_forward && state->d_done_backward)
            break;

        // ---------------------------
        // Per-iteration capacities
        // ---------------------------
        int maxNodesForward  = min(state->global_forward_totalElementsInRange,
                                TOTAL_THREADS_FORWARD   / MAX_NEIGHBORS);
        int maxNodesBackward = min(state->global_backward_totalElementsInRange,
                                TOTAL_THREADS_BACKWARDS / MAX_NEIGHBORS);

        // Reserve [0 .. TOTAL_THREADS_FORWARD-1] for FORWARD
        // and [TOTAL_THREADS_FORWARD .. TOTAL_THREADS_FORWARD + TOTAL_THREADS_BACKWARDS - 1] for BACKWARD.
        threadAssignment = UNASSIGNNED;
        int threadPosition = -1;  // local position among (node, neighbor) pairs for this direction

        // Forward segment
        if (idx < TOTAL_THREADS_FORWARD) {
            int local = idx; // 0 .. TOTAL_THREADS_FORWARD-1
            int maxForwardPairs = maxNodesForward * MAX_NEIGHBORS;

            if (local < maxForwardPairs) {
                threadAssignment = FORWARD;
                threadPosition   = local; // 0 .. maxForwardPairs-1
            }
        }
        // Backward segment
        else if (idx < TOTAL_THREADS_FORWARD + TOTAL_THREADS_BACKWARDS) {
            int local = idx - TOTAL_THREADS_FORWARD; // 0 .. TOTAL_THREADS_BACKWARDS-1
            int maxBackwardPairs = maxNodesBackward * MAX_NEIGHBORS;

            if (local < maxBackwardPairs) {
                threadAssignment = BACKWARD;
                threadPosition   = local; // 0 .. maxBackwardPairs-1
            }
        }

        // Respect done flags
        if (threadAssignment == FORWARD && state->d_done_forward) {
            threadAssignment = UNASSIGNNED;
        }
        if (threadAssignment == BACKWARD && state->d_done_backward) {
            threadAssignment = UNASSIGNNED;
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

        // Work Assignment: each 8 threads are responsible for one node consecutively
        int assignedBucket = -1;
        if (threadAssignment != UNASSIGNNED && threadPosition >= 0) {
            int localPos = threadPosition; // 0 .. (#pairs for this direction - 1)

            for (int b = bucketRangeStart; b <= bucketRangeEnd; ++b) {
                int bucketSize = binCountsPtr[b] * MAX_NEIGHBORS;
                if (localPos < bucketSize) {
                    assignedBucket = b;
                    threadPosition = localPos;  // keep it as "position within range"
                    break;
                }
                localPos -= bucketSize;
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
                int xCurrent = currentNodeId % width;
                int yCurrent = currentNodeId / width;

                int2 off = NEIGHBOR_OFFSETS[neighborIndex];
                int xNeighbor = xCurrent + off.x;
                int yNeighbor = yCurrent + off.y;


                if (xNeighbor >= 0 && xNeighbor < width && yNeighbor >= 0 && yNeighbor < height) {
                    int neighborId = yNeighbor * width + xNeighbor;
                    
                    if (grid[neighborId] == PASSABLE &&
                        !(threadAssignment == FORWARD && neighborId == currentNode.parent_forward) &&
                        !(threadAssignment == BACKWARD && neighborId == currentNode.parent_backward)
                    ) {  // if passable or not the parent node                       
                        bool isDiagonal = (abs(off.x) + abs(off.y) == 2);
                        unsigned int moveCost = isDiagonal ? DIAGONAL_COST : SCALE_FACTOR;
                        unsigned int tentativeG = (threadAssignment == FORWARD ? currentNode.g_forward : currentNode.g_backward) + moveCost;
                        
                        // skip atomic if possible
                        unsigned int oldG_local = (threadAssignment == FORWARD) ?
                        nodes[neighborId].g_forward : nodes[neighborId].g_backward;
                        unsigned int oldG = UINT32_MAX;
                        if (tentativeG < oldG_local) // skip atomic if possible
                        {
                            // Atomically update the neighbor's cost using the appropriate field.
                            if (threadAssignment == FORWARD) {
                                oldG = atomicMin(&nodes[neighborId].g_forward, tentativeG);
                            } else {
                                oldG = atomicMin(&nodes[neighborId].g_backward, tentativeG);
                            }
                        }
                        if ((oldG!=UINT32_MAX) && (tentativeG < oldG)) {
                            // __threadfence(); // Ensure the update is visible to all threads before proceeding.

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

                            // Check if this neighbor has already been reached from the opposite search.
                            if (threadAssignment == FORWARD && nodes[neighborId].g_backward != INT_MAX) {
                                unsigned int candidateCost = nodes[neighborId].g_forward + nodes[neighborId].g_backward;
                                unsigned int oldCost = atomicMin(&state->globalBestCost, candidateCost);
                                if(candidateCost < oldCost)
                                    state->globalBestNode = nodes[neighborId]; // needs changing
                            } 
                            
                            if (threadAssignment == BACKWARD && nodes[neighborId].g_forward != INT_MAX) {
                                unsigned int candidateCost = nodes[neighborId].g_backward + nodes[neighborId].g_forward;
                                unsigned int oldCost = atomicMin(&state->globalBestCost, candidateCost);
                                if(candidateCost < oldCost)    
                                    state->globalBestNode = nodes[neighborId]; // needs changing
                            }

                            // don't expand if we found a local path
                            // don't expand the node of the opposite search
                            if((threadAssignment == FORWARD && tentativeG == nodes[neighborId].g_forward) || (threadAssignment == BACKWARD && tentativeG == nodes[neighborId].g_backward))
                            {
                                // EXPAND
                                // Compute the bin for the neighbor based on its updated f-value.
                                unsigned int newF = threadAssignment == FORWARD ? nodes[neighborId].f_forward : nodes[neighborId].f_backward;
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

        // ------------------------------------------------------------------
        // Copy FORWARD expansion buffer back into forward_openListBins
        // while keeping unexpanded nodes (remainder) for next iteration.
        // ------------------------------------------------------------------
        if (blockIdx.x == 0)  // block 0 is responsible for forward pass
        {
            expansionCountsPtr = forward_expansionCounts;
            openListBinsPtr    = forward_openListBins;
            binCountsPtr       = forward_binCounts;
            expansionBuffersPtr= forward_expansionBuffers;
            bucketRangeStart   = state->global_forward_bucketRangeStart;
            bucketRangeEnd     = state->global_forward_bucketRangeEnd;

            // Max forward nodes we could actually expand this iteration
            const int maxForwardNodesPerIter = TOTAL_THREADS_FORWARD / MAX_NEIGHBORS;

            int nodesForwardInRange   = state->global_forward_totalElementsInRange;      // sum of binCounts in [start..end]
            int nodesExpandedForward  = min(nodesForwardInRange, maxForwardNodesPerIter);
            int remainingExpanded     = nodesExpandedForward;  // how many nodes we really expanded across buckets

            for (int bucket = bucketRangeStart; bucket <= bucketRangeEnd; ++bucket) {
                int oldCount = binCountsPtr[bucket];           // nodes in open list before expansion
                int eCount   = expansionCountsPtr[bucket];     // neighbors generated this iteration

                if (oldCount == 0 && eCount == 0) {
                    // nothing here
                    if (threadIdx.x == 0) {
                        binCountsPtr[bucket]       = 0;
                        expansionCountsPtr[bucket] = 0;
                    }
                    __syncthreads();
                    continue;
                }

                // How many nodes from this bucket were ACTUALLY expanded by threads?
                // We consume buckets in order until we run out of "expanded" budget.
                int usedHere = 0;
                if (remainingExpanded > 0) {
                    usedHere = min(oldCount, remainingExpanded);
                    remainingExpanded -= usedHere;
                }
                // Nodes that were NOT expanded this iteration (the remainder we keep)
                int leftover = oldCount - usedHere;  // may be 0 or more

                // 1) If we expanded *some* nodes here and there is leftover,
                //    move leftover nodes to the front [0 .. leftover-1]
                if (leftover > 0 && usedHere > 0) {
                    for (int i = threadIdx.x; i < leftover; i += blockDim.x) {
                        int src = bucket * MAX_BIN_SIZE + (usedHere + i);
                        int dst = bucket * MAX_BIN_SIZE + i;
                        int id = openListBinsPtr[src];
                        openListBinsPtr[dst] = openListBinsPtr[src];
                        nodes[id].openListAddress_forward = dst;   // or _forward in the forward section
                    }
                }
                __syncthreads();

                // 2) Append neighbors from expansionBuffers after the leftover
                if (eCount > 0) {
                    for (int i = threadIdx.x; i < eCount; i += blockDim.x) {
                        int dst = bucket * MAX_BIN_SIZE + leftover + i;
                        int src = bucket * MAX_BIN_SIZE + i;   // expansion buffer is always 0..eCount-1
                        int id = expansionBuffersPtr[src];
                        openListBinsPtr[dst] = id;
                        nodes[id].openListAddress_forward = dst;   // or _forward
                    }
                }

                if (threadIdx.x == 0) {
                    binCountsPtr[bucket]       = leftover + eCount;  // new size of bucket
                    expansionCountsPtr[bucket] = 0;                  // reset expansion count
                }
                __syncthreads();
            }
        }

        // ------------------------------------------------------------------
        // Copy BACKWARD expansion buffer back into backward_openListBins
        // with the same remainder logic for the backward direction.
        // ------------------------------------------------------------------
        if (blockIdx.x == 1)  // block 1 is responsible for backward pass
        {
            expansionCountsPtr = backward_expansionCounts;
            openListBinsPtr    = backward_openListBins;
            binCountsPtr       = backward_binCounts;
            expansionBuffersPtr= backward_expansionBuffers;
            bucketRangeStart   = state->global_backward_bucketRangeStart;
            bucketRangeEnd     = state->global_backward_bucketRangeEnd;

            // Max backward nodes we could actually expand this iteration
            const int maxBackwardNodesPerIter = TOTAL_THREADS_BACKWARDS / MAX_NEIGHBORS;

            int nodesBackwardInRange   = state->global_backward_totalElementsInRange;
            int nodesExpandedBackward  = min(nodesBackwardInRange, maxBackwardNodesPerIter);
            int remainingExpanded      = nodesExpandedBackward;

            for (int bucket = bucketRangeStart; bucket <= bucketRangeEnd; ++bucket) {
                int oldCount = binCountsPtr[bucket];
                int eCount   = expansionCountsPtr[bucket];

                if (oldCount == 0 && eCount == 0) {
                    if (threadIdx.x == 0) {
                        binCountsPtr[bucket]       = 0;
                        expansionCountsPtr[bucket] = 0;
                    }
                    __syncthreads();
                    continue;
                }

                int usedHere = 0;
                if (remainingExpanded > 0) {
                    usedHere = min(oldCount, remainingExpanded);
                    remainingExpanded -= usedHere;
                }
                int leftover = oldCount - usedHere;

                if (leftover > 0 && usedHere > 0) {
                    for (int i = threadIdx.x; i < leftover; i += blockDim.x) {
                        int src = bucket * MAX_BIN_SIZE + (usedHere + i);
                        int dst = bucket * MAX_BIN_SIZE + i;
                        int id = openListBinsPtr[src];
                        openListBinsPtr[dst] = openListBinsPtr[src];
                        nodes[id].openListAddress_backward = dst;   // or _forward in the forward section
                    }
                }
                __syncthreads();

                if (eCount > 0) {
                    for (int i = threadIdx.x; i < eCount; i += blockDim.x) {
                        int dst = bucket * MAX_BIN_SIZE + leftover + i;
                        int src = bucket * MAX_BIN_SIZE + i;
                        int id = expansionBuffersPtr[src];
                        openListBinsPtr[dst] = expansionBuffersPtr[src];
                        nodes[id].openListAddress_backward = dst;   // or _forward
                    }
                }

                if (threadIdx.x == 0) {
                    binCountsPtr[bucket]       = leftover + eCount;
                    expansionCountsPtr[bucket] = 0;
                }
                __syncthreads();
            }
        }
        // __threadfence();
        gridGroup.sync();
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
