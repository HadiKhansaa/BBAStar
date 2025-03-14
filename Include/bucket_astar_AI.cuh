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

namespace cg = cooperative_groups;

#define INF_FLT 1e20f  // A large float value representing infinity
#define MAX_NEIGHBORS 8        // 8-directional movement

#define MAX_BINS 3000         // Maximum number of bins (adjust as needed)
#define MAX_BIN_SIZE 1500    // Maximum number of nodes per bin (adjust as needed)
#define SCALE_FACTOR 1000
#define SH_MAX_RANGE 5

// AI GENERATED 1

static const int WARPS_PER_BLOCK = 32;
static const int BIN_MASK_BITS = 64;

// Define BIN_SIZE based on the grid dimensions
__device__ __constant__ float BIN_SIZE_DEVICE;

// Global variables for inter-block communication
__device__ int d_currentBin;
__device__ bool d_done;
__device__ bool d_localFound;

// Atomic Min for float
__device__ float atomicMinFloat(float* address, float val) {
    float old = *address, assumed;
    do {
        assumed = old;
        old = __int_as_float(atomicCAS((int*)address,
                                       __float_as_int(assumed),
                                       __float_as_int(fminf(val, assumed))));
    } while (__float_as_int(assumed) != __float_as_int(old));
    return old;
}

// Node structure
struct Node {
    int id;
    int g;
    int h;
    int f;
    int parent;
};

// Heuristic function (Euclidean distance)
__device__ __host__ int heuristic(int currentNodeId, int goalNodeId, int width) {
    int xCurrent = currentNodeId % width;
    int yCurrent = currentNodeId / width;
    int xGoal = goalNodeId % width;
    int yGoal = goalNodeId / width;

    int dx = abs(xCurrent - xGoal);
    int dy = abs(yCurrent - yGoal);

    // return (dx + dy) * SCALE_FACTOR;
    return sqrtf((float)(dx * dx + dy * dy))*SCALE_FACTOR;
}

// Kernel to initialize nodes
__global__ void initializeNodes(Node* nodes, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        nodes[idx].g = INT_MAX;
        nodes[idx].h = 0.0f;
        nodes[idx].f = INT_MAX;
        nodes[idx].parent = -1;
    }
}

// __constant__ int d_heuristic[MAP_SIZE_MAX]; // Precomputed heuristic values
__global__ void aStarOptimizedBuckets(
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
    // Cooperative groups setup
    cg::grid_group gridGroup = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Shared memory allocations
    __shared__ struct {
        int currentRangeStart;
        int currentRangeEnd;
        int totalElements;
        unsigned long long warpMasks[WARPS_PER_BLOCK][2];
        Node nodeCache[1024];
        int expansionBuffer[SH_MAX_RANGE][1024];
        int expansionCounts[SH_MAX_RANGE];
        int nextBucket;
    } smem;

    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;
    const int globalTid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    if(threadIdx.x < WARPS_PER_BLOCK) {
        smem.warpMasks[warpId][0] = 0;
        smem.warpMasks[warpId][1] = 0;
    }
    for(int i=threadIdx.x; i<SH_MAX_RANGE; i+=blockDim.x) {
        smem.expansionCounts[i] = 0;
    }
    __syncthreads();

    // Main processing loop
    while(!d_done) {
        // 1. Dynamic Bucket Range Selection -----------------------------------
        if(blockIdx.x == 0 && threadIdx.x == 0) {
            smem.currentRangeStart = atomicAdd(firstNonEmptyMask, 0);
            smem.currentRangeEnd = atomicAdd(lastNonEmptyMask, 0);
            smem.totalElements = 0;
            
            // Find first non-empty bucket range
            for(int i=smem.currentRangeStart; i<=smem.currentRangeEnd && smem.totalElements<K; i++) {
                int elements = binCounts[i];
                if(smem.totalElements + elements <= K) {
                    smem.totalElements += elements;
                    smem.currentRangeEnd = i;
                } else {
                    break;
                }
            }
            
            if(smem.totalElements == 0) d_done = true;
        }
        gridGroup.sync();
        if(d_done) break;

        // 2. Warp-Centric Work Assignment -------------------------------------
        int assignedBucket = -1;
        int elementsProcessed = 0;
        for(int b=smem.currentRangeStart; b<=smem.currentRangeEnd; b++) {
            int bucketSize = binCounts[b];
            if(globalTid - elementsProcessed < bucketSize) {
                assignedBucket = b;
                break;
            }
            elementsProcessed += bucketSize;
        }

        // 3. Coalesced Node Loading ------------------------------------------
        Node currentNode;
        if(assignedBucket != -1) {
            int nodeIdx = globalTid - elementsProcessed;
            if(nodeIdx < binCounts[assignedBucket]) {
                int nodeId = openListBins[assignedBucket * MAX_BIN_SIZE + nodeIdx];
                currentNode = nodes[nodeId];
                
                // Cache node in shared memory with coalesced access
                smem.nodeCache[threadIdx.x] = currentNode;
            }
        }
        __syncthreads();

        // 4. Parallel Node Expansion -----------------------------------------
        if(assignedBucket != -1 && threadIdx.x < binCounts[assignedBucket]) {
            currentNode = smem.nodeCache[threadIdx.x];
            atomicAdd(totalExpandedNodes, 1);

            // Generate neighbors in branchless pattern
            const int offsets[8][2] = {{-1,0}, {1,0}, {0,-1}, {0,1}, {-1,-1}, {-1,1}, {1,-1}, {1,1}};
            for(int dir=0; dir<8; dir++) {
                int dx = offsets[dir][0];
                int dy = offsets[dir][1];
                int x = (currentNode.id % width) + dx;
                int y = (currentNode.id / width) + dy;
                
                // Branchless boundary check
                bool valid = (x >= 0) & (x < width) & (y >= 0) & (y < height);
                int neighborId = valid ? (y * width + x) : -1;

                if(valid && grid[neighborId] == 0) {
                    int cost = (abs(dx)+abs(dy)) == 2 ? 1414 : 1000;
                    int tentativeG = currentNode.g + cost;

                    // AtomicMin with early exit optimization
                    int *g_ptr = &nodes[neighborId].g;
                    int oldG = atomicMin(g_ptr, tentativeG);
                    bool improved = tentativeG < oldG;

                    if(improved) {
                        // Update node properties
                        nodes[neighborId].parent = currentNode.id;
                        nodes[neighborId].h = heuristic(neighborId, goalNodeId, width);;
                        nodes[neighborId].f = tentativeG + nodes[neighborId].h;

                        // Check termination
                        if(neighborId == goalNodeId) {
                            *found = true;
                            d_done = true;
                        }

                        // Bin calculation
                        int minF = width * 1414;
                        int bin = (nodes[neighborId].f - minF) / BIN_SIZE_DEVICE;
                        bin = min(max(bin, 0), MAX_BINS-1);

                        // Warp-level aggregation
                        unsigned mask = __ballot_sync(0xffffffff, improved);
                        if(mask) {
                            int leader = __ffs(mask) - 1;
                            if(warp.thread_rank() == leader) {
                                int count = __popc(mask);
                                int pos = atomicAdd(&binCounts[bin], count);
                                
                                // Write to global memory with coalescing
                                for(int i=0; i<count; i++) {
                                    int src_lane = __ffs(mask >> (i+1)) - 1;
                                    int nghbrId = warp.shfl(neighborId, src_lane);
                                    openListBins[bin * MAX_BIN_SIZE + pos + i] = nghbrId;
                                }

                                // Update warp-local bitmask
                                int mask_idx = bin / BIN_MASK_BITS;
                                unsigned long long bit = 1ull << (bin % BIN_MASK_BITS);
                                atomicOr(&smem.warpMasks[warpId][mask_idx], bit);
                            }
                        }
                    }
                }
            }
        }

        // 5. Merge Warp Masks to Global Bitmask ------------------------------
        __syncthreads();
        for(int w=0; w<WARPS_PER_BLOCK; w++) {
            for(int i=threadIdx.x; i<2; i+=blockDim.x) {
                unsigned long long mask = smem.warpMasks[w][i];
                if(mask) {
                    atomicOr(&binBitMask[i], mask);
                    smem.warpMasks[w][i] = 0;
                }
            }
        }

        // 6. Progress Tracking -----------------------------------------------
        if(blockIdx.x == 0 && threadIdx.x == 0) {
            for(int b=smem.currentRangeStart; b<=smem.currentRangeEnd; b++) {
                binCounts[b] = 0;
                int mask_idx = b / BIN_MASK_BITS;
                unsigned long long mask = ~(1ull << (b % BIN_MASK_BITS));
                atomicAnd(&binBitMask[mask_idx], mask);
            }
        }
        gridGroup.sync();

        // 7. Path Reconstruction ---------------------------------------------
        if(*found && blockIdx.x == 0 && threadIdx.x == 0) {
            int currentId = goalNodeId;
            int pathIndex = 0;
            while(currentId != -1 && pathIndex < 1000000000) {
                path[pathIndex++] = currentId;
                currentId = nodes[currentId].parent;
            }
            *pathLength = pathIndex;
        }
        gridGroup.sync();
    }
}


// AI GENERATED 2
#include <cooperative_groups.h>

// #define MAP_SIZE 10000
#define BLOCK_SIZE 256
#define BIN_MASK_SIZE 32
#define MIN_F 14140000

// Precomputed heuristic in constant memory (assuming MAP_SIZE is defined)
// __constant__ int d_h_values[MAP_SIZE];
// #define HEURISTIC(id) __ldg(&d_h_values[id])

// Warp-centric macros
#define WARP_SIZE 32
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#define BIN_MASK_BITS 32

__global__ void aStarOptimized(
    int *grid, int width, int height, int goalNodeId, Node *nodes,
    int *openListBins, int *binCounts, unsigned long long *binBitMask,
    bool *found, int *path, int *pathLength, 
    int K, int *totalExpandedNodes)
{
    namespace cg = cooperative_groups;
    cg::grid_group gridGroup = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    // Add debug counters
    __shared__ int s_iterCount;
    __shared__ int s_nodesProcessed;
    
    // Init debug variables
    if(threadIdx.x == 0) {
        s_iterCount = 0;
        s_nodesProcessed = 0;
    }
    __syncthreads();
    
    // Shared memory allocations
    __shared__ struct {
        // Coalesced node cache (pad to avoid bank conflicts)
        Node nodeCache[BLOCK_SIZE][2];  // Double buffer
        // Warp-local bin masks
        uint32_t warpMasks[WARPS_PER_BLOCK][BIN_MASK_SIZE];
        // Dynamic scheduling
        int nextBucket;
        // Expansion buffers (padded)
        int expansions[SH_MAX_RANGE][MAX_BIN_SIZE + 32];
        int expansionCounts[SH_MAX_RANGE];
    } smem;

    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;

    // Initialize warp masks
    if (threadIdx.x < WARPS_PER_BLOCK * BIN_MASK_SIZE) {
        smem.warpMasks[warpId][laneId] = 0;
    }
    block.sync();

    // Dynamic bucket scheduling
    __shared__ int currentBucket;
    if (threadIdx.x == 0) {
        smem.nextBucket = atomicAdd(totalExpandedNodes, 0); // Dummy init
    }
    block.sync();

    while (!*found) {

        if(blockIdx.x == 0 && threadIdx.x == 0) {
            atomicAdd(&s_iterCount, 1);
            if(s_iterCount % 100 == 0) {
                printf("[Iter %d] Global state: done=%d, found=%d\n", 
                      s_iterCount, d_done, *found);
            }
        }
        
        // ---------------------------------------------------------------
        // 1. Dynamic bucket assignment (warp-grained)
        // ---------------------------------------------------------------
        uint32_t activeWarps = __ballot_sync(0xffffffff, !*found);
        if (!activeWarps) break;

        if (laneId == 0) {
            atomicAdd(&smem.nextBucket, 1);
            currentBucket = smem.nextBucket;
        }
        currentBucket = __shfl_sync(activeWarps, currentBucket, 0);

        // ---------------------------------------------------------------
        // 2. Coalesced node loading with double buffering
        // ---------------------------------------------------------------
        int loadIdx = threadIdx.x;
        Node loadedNode;
        bool validNode = false;
        
        while (loadIdx < binCounts[currentBucket]) {
            // Coalesced global read
            int nodeId = openListBins[currentBucket * MAX_BIN_SIZE + loadIdx];
            loadedNode = nodes[nodeId];
            
            // Store in shared memory with double buffering
            smem.nodeCache[threadIdx.x][loadIdx % 2] = loadedNode;
            validNode = true;
            
            loadIdx += block.size();
        }
        block.sync();

        // ---------------------------------------------------------------
        // 3. Parallel expansion with warp-level aggregation
        // ---------------------------------------------------------------
        int storeBuffer = 0;
        for (int buffer = 0; buffer < 2; ++buffer) {
            if (validNode) {
                Node currentNode = smem.nodeCache[threadIdx.x][buffer];
                
                // Generate neighbors (branchless)
                int dx = (laneId % 3) - 1;
                int dy = (laneId / 3) - 1;
                int x = (currentNode.id % width) + dx;
                int y = (currentNode.id / width) + dy;
                int valid = (x >= 0) & (x < width) & (y >= 0) & (y < height);
                int neighborId = valid ? (y * width + x) : -1;

                if (valid && grid[neighborId] == 0) {
                    int cost = (abs(dx) + abs(dy) == 2) ? 1414 : 1000;
                    int tentativeG = currentNode.g + cost;

                    // AtomicMin with early exit
                    int *g_ptr = &nodes[neighborId].g;
                    int oldG = atomicMin(g_ptr, tentativeG);
                    int improved = (tentativeG < oldG);

                    if (improved) {
                        // Update node data
                        nodes[neighborId].parent = currentNode.id;
                        nodes[neighborId].f = tentativeG + heuristic(neighborId, goalNodeId, width);

                        // Bin calculation
                        int bin = (nodes[neighborId].f - MIN_F) / MAX_BIN_SIZE;
                        bin = min(max(bin, 0), MAX_BINS-1);

                        // Warp-level aggregation
                        uint32_t bin_mask = __ballot_sync(0xffffffff, improved);
                        uint32_t bin_leader = __ffs(bin_mask) - 1;
                        
                        if (laneId == bin_leader) {
                            int count = __popc(bin_mask);
                            int pos = atomicAdd(&binCounts[bin], count);
                            
                            // Write to global memory with coalescing
                            int idx = pos + __popc(bin_mask & ((1 << laneId) - 1));
                            openListBins[bin * MAX_BIN_SIZE + idx] = neighborId;
                            
                            // Update warp-local bitmask
                            int mask_idx = bin / BIN_MASK_BITS;
                            uint32_t mask_bit = 1u << (bin % BIN_MASK_BITS);
                            atomicOr(&smem.warpMasks[warpId][mask_idx], mask_bit);
                        }

                        // Check termination
                        if (neighborId == goalNodeId) {
                            *found = true;
                            // ... path reconstruction ...
                        }
                    }
                }
            }
            
            // Switch buffers
            storeBuffer = 1 - storeBuffer;
            validNode = (loadIdx + block.size() * buffer) < binCounts[currentBucket];
            block.sync();
        }

        // ---------------------------------------------------------------
        // 4. Merge warp-local masks to global bitmask
        // ---------------------------------------------------------------
        for (int i = threadIdx.x; i < BIN_MASK_SIZE * WARPS_PER_BLOCK; i += block.size()) {
            int warp = i / BIN_MASK_SIZE;
            int mask_idx = i % BIN_MASK_SIZE;
            unsigned long long mask = smem.warpMasks[warp][mask_idx];
            if (mask) {
                atomicOr(&binBitMask[mask_idx], mask);
            }
        }
        
        gridGroup.sync();
    }
}
