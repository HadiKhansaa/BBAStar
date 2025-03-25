#pragma once

#define INF_FLT 1e20f  // A large float value representing infinity
#define MAX_NEIGHBORS 8        // 8-directional movement

#define MAX_BINS 3000         // Maximum number of bins (adjust as needed)
#define MAX_BIN_SIZE 2000    // Maximum number of nodes per bin (adjust as needed)
#define SCALE_FACTOR 1000   // Khansa is based (via the universal Axiom of Consistant-Basedness)

#define SH_MAX_RANGE 15
#define MAX_SHARED_BIN_SIZE 100

#define TILE_WIDTH  16
#define TILE_HEIGHT 16

#define FRONTIER_SIZE 500

__device__ void wait(int cycles) {
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

// Node structure
struct Node {
    int id;
    int g;
    int h;
    int f;
    int parent;
};

// bidirectional Node
struct BiNode {
    int id;
    int g_forward;
    int h_forward;
    int f_forward;
    int g_backward;
    int h_backward;
    int f_backward;
    int parent_forward;
    int parent_backward;
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
    return sqrtf((float)(dx * dx + dy * dy)) * SCALE_FACTOR;
}


// This function reconstructs a bidirectional path.
// Two threads call this function: threadIdx.x==0 for the forward pass,
// threadIdx.x==1 for the backward pass.


__device__ void constractBidirectionalPath(int startNodeId, int endNodeId, BiNode& meetingNode, int* path, int* pathLength, BiNode* g_nodes) {

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
        *pathLength += backwardCount;
        
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
        *pathLength += forwardCount;
        
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
__device__ int sumRangeInclusive(const int* __restrict__ bucketCount, int start, int end)
{
    int s = 0;
    for (int i = start; i <= end; i++) {
        s += bucketCount[i];
    }
    return s;
}

/////////////////////////////////////////////////////////////////////////////////
// Naive single-block inclusive scan of shPrefix[0..(n-1)] in-place.
// (Replace with a parallel scan in real code.)
/////////////////////////////////////////////////////////////////////////////////
__device__ void inclusiveScanInBlock(int* arr, int n)
{
    // Single-thread does the scan for simplicity
    int tid = threadIdx.x;
    if (tid == 0) {
        for (int i = 1; i < n; i++) {
            arr[i] += arr[i - 1];
        }
    }
    __syncthreads();
}

/////////////////////////////////////////////////////////////////////////////////
// loadBucketsToSharedInclusive
// Copies *all* elements from buckets in the range [bucketRangeStart .. bucketRangeEnd]
// into 'output' contiguously in shared memory.
//
// Assumptions:
//  - One block does the entire copy for that subrange.
//  - bucketData is laid out so that bucket 0’s elements come first, then bucket 1’s, etc.
//  - bucketCount[i] is the size of bucket i.
//  - outputSize = sum of bucketCount[bucketRangeStart..bucketRangeEnd] (inclusive).
/////////////////////////////////////////////////////////////////////////////////
__device__ void loadBucketsToShared(int* __restrict__ bucketData,
                                             int* __restrict__ bucketCount,
                                             int bucketRangeStart,
                                             int bucketRangeEnd,
                                             int outputSize,  // sum of all bucket sizes in [start..end]
                                             int* __restrict__ output)  // in shared memory
{
    // Return if invalid range (e.g. end < start).
    if (bucketRangeEnd < bucketRangeStart) {
        return;  // nothing to do
    }

    // Number of buckets in our subrange (inclusive)
    int numBuckets = bucketRangeEnd - bucketRangeStart + 1;
    int tid = threadIdx.x;

    // We'll store partial sums for these subrange buckets in shared memory
    extern __shared__ int shPrefix[];  // at least numBuckets in size

    // 1) Load the relevant bucket counts into shPrefix
    if (tid < numBuckets) {
        // bucketRangeStart + 0, +1, +2, ..., + (numBuckets - 1) = bucketRangeEnd
        shPrefix[tid] = bucketCount[bucketRangeStart + tid];
    }
    __syncthreads();

    // 2) Do an in-block inclusive prefix sum on shPrefix
    inclusiveScanInBlock(shPrefix, numBuckets);
    __syncthreads();

    // After this, shPrefix[i] = sum of bucketCount[bucketRangeStart .. bucketRangeStart + i],
    // i.e., an inclusive partial sum for i=0..(numBuckets-1).

    // 3) Calculate the global base offset for the *first* bucket in our subrange
    //    i.e., sum of all bucketCounts up to (bucketRangeStart - 1).
    __shared__ int baseOffset;
    if (tid == 0) {
        // sum of all buckets from 0..(bucketRangeStart-1), so the subrange starts at baseOffset in bucketData
        if (bucketRangeStart > 0)
            baseOffset = sumRangeInclusive(bucketCount, 0, bucketRangeStart - 1);
        else
            baseOffset = 0;
    }
    __syncthreads();

    // 4) Copy elements from each bucket in [bucketRangeStart..bucketRangeEnd] into 'output' contiguously.
    for (int b = 0; b < numBuckets; b++)
    {
        int bucketSize = bucketCount[bucketRangeStart + b];
        if (bucketSize == 0) continue;

        // The output start offset for bucket b (in our subrange)
        // is "prefix of all buckets up to b-1" => shPrefix[b-1] if b > 0, else 0
        int outStart = (b == 0) ? 0 : shPrefix[b - 1];

        // Copy each element from bucketData into output
        for (int elemIdx = tid; elemIdx < bucketSize; elemIdx += blockDim.x)
        {
            // The position in bucketData where this bucket’s elements start
            int readPos = baseOffset + outStart + elemIdx;

            // The position in output
            int writePos = outStart + elemIdx;

            // Copy
            output[writePos] = bucketData[readPos];
        }
        __syncthreads();
    }

    // Optionally verify that shPrefix[numBuckets - 1] == outputSize
    // if (tid == 0 && shPrefix[numBuckets - 1] != outputSize) { /* debug or assert */ }

    //debug print

    if(tid == 0){
        printf("outputSize: %d\n", outputSize);
        for(int i = 0; i < outputSize; i++){
            printf("%d ", output[i]);
        }
        printf("\n");
    }

}
