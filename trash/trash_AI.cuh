// __global__ void aStarMultipleBucketsSharedAI(
//     int *grid, int width, int height, int goalNodeId, Node *nodes,
//     int *openListBins, int *binCounts, unsigned long long *binBitMask,
//     int *expansionBuffers, int *expansionCounts, bool *found,
//     int *path, int *pathLength, 
//     int binBitMaskSize, 
//     int K, 
//     int *totalExpandedNodes,
//     int* firstNonEmptyMask,
//     int* lastNonEmptyMask)
// {
//     cg::grid_group gridGroup = cg::this_grid();
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int totalThreads = gridGroup.size();

//     // Shared memory declarations
//     __shared__ int s_nodeIds[MAX_BIN_SIZE];
//     __shared__ Node s_nodes[MAX_BIN_SIZE];
//     __shared__ int s_gridTile[16][16];  // 32x32 grid tile
//     __shared__ int s_minX, s_maxX, s_minY, s_maxY;
//     __shared__ int s_tileWidth, s_tileHeight;
    
//     // Existing shared memory buffers
//     __shared__ int blockExpansionBuffer[SH_MAX_RANGE * MAX_BIN_SIZE];
//     __shared__ int blockExpansionCounts[SH_MAX_RANGE];

//     // Initialization
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
//         d_done = false;
//         d_localFound = false;
//     }
//     gridGroup.sync();

//     // Neighbor offsets constant
//     __shared__ int s_neighborOffsets[8][2];

//     if (threadIdx.x < 8) {
//         int offsets[8][2] = {{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1}};
//         s_neighborOffsets[threadIdx.x][0] = offsets[threadIdx.x][0];
//         s_neighborOffsets[threadIdx.x][1] = offsets[threadIdx.x][1];
//     }
//     __syncthreads();

//     while (!d_done) {
//         // Existing bucket selection logic remains unchanged
//         if (threadIdx.x == 0 && blockIdx.x == 0) {
//             if (K < 1000) K += 10;
//             s_bucketRangeStart = -1;
//             s_bucketRangeEnd = -1;
//             s_totalElementsInRange = 0;
            
//             int elementsAccumulated = 0;
//             bool done = false;
//             for (int i = 0; i < binBitMaskSize && !done; ++i) {
//                 unsigned long long tmpMask = binBitMask[i];
//                 while (tmpMask != 0ULL && !done) {
//                     int firstSetBit = __ffsll(tmpMask) - 1;
//                     tmpMask &= ~(1ULL << firstSetBit);
//                     int bucket = i * 64 + firstSetBit;
//                     if (bucket >= MAX_BINS) {
//                         done = true;
//                         break;
//                     }
//                     if (s_bucketRangeStart == -1) s_bucketRangeStart = bucket;
//                     if (s_bucketRangeEnd - s_bucketRangeStart >= SH_MAX_RANGE) {
//                         done = true;
//                         break;
//                     }
//                     if (elementsAccumulated + binCounts[bucket] < K) {
//                         elementsAccumulated += binCounts[bucket];
//                         s_bucketRangeEnd = bucket;
//                     } else {
//                         s_bucketRangeEnd = bucket;
//                         done = true;
//                     }
//                 }
//             }
//             if (elementsAccumulated == 0) d_done = true;
//             else s_totalElementsInRange = elementsAccumulated;



//             // debugging
//             printf("Active elements = %d (buckets %d..%d)\n",
//                 elementsAccumulated,
//                 s_bucketRangeStart,
//                 s_bucketRangeEnd);

//             // wait a bit
//             clock_t start = clock();
//             clock_t now;
//             for (;;) {
//                 now = clock();
//                 clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
//                 if (cycles >= 10000000) {
//                     break;
//                 }
//             }

//             printf("first %d, last %d\n", *firstNonEmptyMask, *lastNonEmptyMask);
        

//         }
//         gridGroup.sync();

//         if (d_done) break;

//         // Get assigned bucket
//         int assignedBucket = -1;
//         int threadPosition = idx;
//         for (int b = s_bucketRangeStart; b <= s_bucketRangeEnd; ++b) {
//             int bucketSize = binCounts[b] * 8;
//             if (threadPosition < bucketSize) {
//                 assignedBucket = b;
//                 break;
//             }
//             threadPosition -= bucketSize;
//         }

//         // Initialize shared buffers
//         if (threadIdx.x < SH_MAX_RANGE) blockExpansionCounts[threadIdx.x] = 0;
//         __syncthreads();

//         if (assignedBucket != -1) {
//             atomicAdd(totalExpandedNodes, 1);

//             // Load current bucket's nodes into shared memory
//             int numNodes = binCounts[assignedBucket];
//             for (int i = threadIdx.x; i < numNodes; i += blockDim.x) {
//                 s_nodeIds[i] = openListBins[assignedBucket * MAX_BIN_SIZE + i];
//             }
//             __syncthreads();

//             // Load node data into shared memory
//             for (int i = threadIdx.x; i < numNodes; i += blockDim.x) {
//                 int nodeId = s_nodeIds[i];
//                 s_nodes[i] = nodes[nodeId];
//             }
//             __syncthreads();

//             // Calculate grid tile bounds
//             if (threadIdx.x == 0) {
//                 s_minX = width; s_maxX = 0;
//                 s_minY = height; s_maxY = 0;
//             }
//             __syncthreads();

//             for (int i = threadIdx.x; i < numNodes; i += blockDim.x) {
//                 int nodeId = s_nodeIds[i];
//                 int x = nodeId % width;
//                 int y = nodeId / width;
//                 atomicMin(&s_minX, x);
//                 atomicMax(&s_maxX, x);
//                 atomicMin(&s_minY, y);
//                 atomicMax(&s_maxY, y);
//             }
//             __syncthreads();

//             // Expand bounds by 1 for neighbors
//             s_minX = max(s_minX - 1, 0);
//             s_maxX = min(s_maxX + 1, width - 1);
//             s_minY = max(s_minY - 1, 0);
//             s_maxY = min(s_maxY + 1, height - 1);
//             s_tileWidth = s_maxX - s_minX + 1;
//             s_tileHeight = s_maxY - s_minY + 1;

//             // Load grid tile into shared memory
//             for (int y = threadIdx.y; y < s_tileHeight; y += blockDim.y) {
//                 for (int x = threadIdx.x; x < s_tileWidth; x += blockDim.x) {
//                     int gx = s_minX + x;
//                     int gy = s_minY + y;
//                     if (gx < width && gy < height) {
//                         s_gridTile[y][x] = grid[gy * width + gx];
//                     }
//                 }
//             }
//             __syncthreads();

//             // Process neighbors using shared memory
//             int nodeIndex = threadPosition / 8;
//             int neighborIndex = threadPosition % 8;

//             if (nodeIndex < numNodes) {
//                 Node currentNode = s_nodes[nodeIndex];
//                 int currentNodeId = s_nodeIds[nodeIndex];
//                 int xCurrent = currentNodeId % width;
//                 int yCurrent = currentNodeId / width;

//                 int dx = s_neighborOffsets[neighborIndex][0];
//                 int dy = s_neighborOffsets[neighborIndex][1];
//                 int xNeighbor = xCurrent + dx;
//                 int yNeighbor = yCurrent + dy;

//                 bool valid = false;
//                 if (xNeighbor >= s_minX && xNeighbor <= s_maxX &&
//                     yNeighbor >= s_minY && yNeighbor <= s_maxY) {
//                     // Check using shared memory tile
//                     int tx = xNeighbor - s_minX;
//                     int ty = yNeighbor - s_minY;
//                     valid = (s_gridTile[ty][tx] == 0);
//                 } else {
//                     // Fallback to global memory check
//                     if (xNeighbor >= 0 && xNeighbor < width &&
//                         yNeighbor >= 0 && yNeighbor < height) {
//                         valid = (grid[yNeighbor * width + xNeighbor] == 0);
//                     }
//                 }

//                 if (valid) {
//                     bool isDiagonal = (abs(dx) + abs(dy)) == 2;
//                     int moveCost = isDiagonal ? 1414 : 1000;
//                     int tentativeG = currentNode.g + moveCost;

//                     int neighborId = yNeighbor * width + xNeighbor;
//                     int oldG = atomicMin(&nodes[neighborId].g, tentativeG);
                    
//                     if (tentativeG < oldG) {
//                         nodes[neighborId].id = neighborId;
//                         nodes[neighborId].parent = currentNodeId;
//                         nodes[neighborId].h = heuristic(neighborId, goalNodeId, width);
//                         nodes[neighborId].f = tentativeG + nodes[neighborId].h;
//                         nodes[neighborId].g = tentativeG;

//                         if (neighborId == goalNodeId) d_localFound = true;

//                         // Bin calculation
//                         int minFValue = 1414 * height;
//                         int adjustedF = nodes[neighborId].f - minFValue;
//                         int binForNghbr = (int)(adjustedF / BIN_SIZE_DEVICE);
//                         binForNghbr = min(binForNghbr, MAX_BINS - 1);

//                         // Store in shared expansion buffer
//                         int s_bin = binForNghbr - s_bucketRangeStart;
//                         if (s_bin >= 0 && s_bin < SH_MAX_RANGE) {
//                             int pos = atomicAdd(&blockExpansionCounts[s_bin], 1);
//                             if (pos < MAX_BIN_SIZE) {
//                                 blockExpansionBuffer[s_bin * MAX_BIN_SIZE + pos] = neighborId;
//                             }
//                         } else {
//                             // Fallback to global memory
//                             int pos = atomicAdd(&binCounts[binForNghbr], 1);
//                             openListBins[binForNghbr * MAX_BIN_SIZE + pos] = neighborId;
//                             if (pos == 0) {
//                                 int maskIndex = binForNghbr / 64;
//                                 atomicOr(&binBitMask[maskIndex], 1ULL << (binForNghbr % 64));
//                             }
//                         }
//                     }
//                 }
//             }
//         }

//         // Synchronize before moving to bin updates
//         __threadfence();
//         gridGroup.sync();

//         // Existing bin update logic remains similar
//         if (blockIdx.x == 0 && threadIdx.x < SH_MAX_RANGE) {
//             binCounts[s_bucketRangeStart + threadIdx.x] = 0;
//             int maskIndex = (s_bucketRangeStart + threadIdx.x) / 64;
//             atomicAnd(&binBitMask[maskIndex], ~(1ULL << ((s_bucketRangeStart + threadIdx.x) % 64)));
//         }
//         __syncthreads();

//         // Copy shared expansions to global memory
//         for (int b = s_bucketRangeStart; b < s_bucketRangeStart + SH_MAX_RANGE; ++b) {
//             int localIndex = b - s_bucketRangeStart;
//             int count = blockExpansionCounts[localIndex];
            
//             if (count > 0) {
//                 int offset = 0;
//                 if (threadIdx.x == 0) {
//                     offset = atomicAdd(&binCounts[b], count);
//                     int maskIndex = b / 64;
//                     atomicOr(&binBitMask[maskIndex], 1ULL << (b % 64));
//                 }
//                 __syncthreads();
                
//                 offset = __shfl_sync(0xffffffff, offset, 0);
//                 for (int i = threadIdx.x; i < count; i += blockDim.x) {
//                     openListBins[b * MAX_BIN_SIZE + offset + i] = 
//                         blockExpansionBuffer[localIndex * MAX_BIN_SIZE + i];
//                 }
//             }
//             __syncthreads();
//         }

//         // Goal check and path reconstruction
//         if (d_localFound) {
//             if (blockIdx.x == 0 && threadIdx.x == 0) {
//                 int tempId = goalNodeId;
//                 int c = 0;
//                 while (tempId != -1 && c < width * height) {
//                     path[c++] = tempId;
//                     tempId = nodes[tempId].parent;
//                 }
//                 *pathLength = c;
//                 *found = true;
//                 d_done = true;
//             }
//         }
//         gridGroup.sync();
//     }
// }