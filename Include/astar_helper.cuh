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
