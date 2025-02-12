#include "CPU/grid_generation.hpp"
#include "bucket_astar.cuh"

#define CUDA_CHECK(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n",         \
                    cudaGetErrorString(err), err, __FILE__, __LINE__);        \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

#define CUDA_KERNEL_CHECK()                                                   \
    {                                                                         \
        cudaError_t err = cudaGetLastError();                                 \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Kernel Error: %s (err_num=%d) at %s:%d\n",  \
                    cudaGetErrorString(err), err, __FILE__, __LINE__);        \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }


// Main function
int main(int argc, char** argv) {
    // Grid dimensions and initialization code (same as before)
    int width = 1001;  // Adjusted for demonstration
    int height = 1001;
    float obstacleRate = 0.2; // Default obstacle rate (percentage)
    std::string gridType = "";
    std::string gridPath = "";

    if(argc == 2)
    {
        height = atoi(argv[1]);
        width = atoi(argv[1]);
    } else if (argc == 3)
    {
        height = atoi(argv[1]);
        width = atoi(argv[1]);
        obstacleRate = atoi(argv[2])/100.0;
    }
    else if (argc == 4)
    {
        height = atoi(argv[1]);
        width = atoi(argv[1]);
        obstacleRate = atoi(argv[2])/100.0;
        gridType = argv[3];
    }
    else if (argc == 5)
    {
        height = atoi(argv[1]);
        width = atoi(argv[1]);
        obstacleRate = atoi(argv[2])/100.0;
        gridType = argv[3];
        gridPath = argv[4];
    }
    int gridSize = width * height;

    // Start and goal nodes
    int startNodeId = 0;                 // Top-left corner
    int goalNodeId = width * height - 1; // Bottom-right corner

    // Allocate and initialize grid on host
    int *h_grid = (int *)malloc(gridSize * sizeof(int));
    if (h_grid == NULL) {
        fprintf(stderr, "Failed to allocate host memory for h_grid\n");
        exit(EXIT_FAILURE);
    }

    // Seed the random number generator
    srand(time(NULL));

    if(gridPath!="")
        loadCompressedGridFromFile(h_grid, width, height, gridPath);  // load grid from file

    // Apply obstacles
    else if(gridType == "random")
        applyRandomObstacles(h_grid, width, height, obstacleRate);
    else if(gridType == "maze")
        createMaze(h_grid, height);
    else if(gridType == "blockCenter")
        createConcentratedObstacles(h_grid, height);
    else if(gridType == "zigzag")
        createZigzagPattern(h_grid, width, height);
    else if (gridType == "rectangle")
        applyRandomRectangleObstacles(h_grid, width, height, obstacleRate);
    else
        applyRandomObstacles(h_grid, width, height, obstacleRate);

    // Ensure start and goal nodes are free
    h_grid[startNodeId] = 0;
    h_grid[goalNodeId] = 0;

    // Timing the data copy to GPU
    auto startCopyTime = std::chrono::high_resolution_clock::now();

    // Allocate and copy grid to device
    int *d_grid;
    CUDA_CHECK(cudaMalloc((void **)&d_grid, gridSize * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_grid, h_grid, gridSize * sizeof(int), cudaMemcpyHostToDevice));

    // Device variables
    Node *d_nodes;
    int *d_path;
    int *d_pathLength;
    bool *d_found;

    CUDA_CHECK(cudaMalloc((void **)&d_nodes, gridSize * sizeof(Node)));
    CUDA_CHECK(cudaMalloc((void **)&d_path, gridSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_pathLength, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_found, sizeof(bool)));

    auto endCopyTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> copyTime = endCopyTime - startCopyTime;

    // Initialize nodes[].g to INF_FLT
    dim3 threadsPerBlockInit(256);
    dim3 blocksPerGridInit((gridSize + threadsPerBlockInit.x - 1) / threadsPerBlockInit.x);

    initializeNodes<<<blocksPerGridInit, threadsPerBlockInit>>>(d_nodes, width, height);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Initialize other device variables
    CUDA_CHECK(cudaMemset(d_path, -1, gridSize * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_pathLength, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_found, 0, sizeof(bool)));

    // Initialize open list bins
    int *d_openListBins;
    int *d_binCounts;
    CUDA_CHECK(cudaMalloc((void **)&d_openListBins, MAX_BINS * MAX_BIN_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_binCounts, MAX_BINS * sizeof(int)));

    CUDA_CHECK(cudaMemset(d_openListBins, -1, MAX_BINS * MAX_BIN_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_binCounts, 0, MAX_BINS * sizeof(int)));

    // Initialize binBitMask
    unsigned long long* d_binBitMask;
    int binBitMaskSize = (MAX_BINS + 63) / 64;
    CUDA_CHECK(cudaMalloc((void **)&d_binBitMask, binBitMaskSize * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_binBitMask, 0, binBitMaskSize * sizeof(unsigned long long)));

    // Allocate device memory for bin expansion buffer and count
    int *d_binExpansionBuffer;
    int *d_binExpansionCount;
    CUDA_CHECK(cudaMalloc((void**)&d_binExpansionBuffer, MAX_BIN_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_binExpansionCount, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_binExpansionCount, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_binExpansionBuffer, -1, MAX_BIN_SIZE * sizeof(int)));

    // Initialize the start node and add it to the open list
    Node h_startNode;
    h_startNode.id = startNodeId;
    h_startNode.g = 0;
    h_startNode.h = heuristic(startNodeId, goalNodeId, width);
    h_startNode.f = h_startNode.g + h_startNode.h;
    h_startNode.parent = -1;

    CUDA_CHECK(cudaMemcpy(&d_nodes[startNodeId], &h_startNode, sizeof(Node), cudaMemcpyHostToDevice));

    // Calculate BIN_SIZE
    float minFValue = height*1414;

    float BIN_SIZE = 3000; // added for testing

    // Copy BIN_SIZE to device constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(BIN_SIZE_DEVICE, &BIN_SIZE, sizeof(float)));

    // Determine the bin for the start node
    int startBin = (int)((h_startNode.f - minFValue) / BIN_SIZE);
    startBin = std::min(startBin, MAX_BINS - 1);

    // Add start node to the bins
    int* h_binCounts = (int*)malloc(MAX_BINS * sizeof(int));
    memset(h_binCounts, 0, MAX_BINS * sizeof(int));

    int* h_openListBins = (int*)malloc(MAX_BINS * MAX_BIN_SIZE * sizeof(int));
    memset(h_openListBins, -1, MAX_BINS * MAX_BIN_SIZE * sizeof(int));

    h_binCounts[startBin] = 1;
    h_openListBins[startBin * MAX_BIN_SIZE] = startNodeId;

    // Initialize binBitMask on host
    unsigned long long *h_binBitMask = (unsigned long long*)malloc(binBitMaskSize * sizeof(unsigned long long));
    memset(h_binBitMask, 0, binBitMaskSize * sizeof(unsigned long long));

    // Set the bit for the start bin
    int maskIndex = startBin / 64;
    unsigned long long mask = 1ULL << (startBin % 64);
    h_binBitMask[maskIndex] |= mask;

    // allocate firstNonEmptyMask and lastNonEmptyMask
    int* d_firstNonEmptyMask;
    int* d_lastNonEmptyMask;

    CUDA_CHECK(cudaMalloc((void**)&d_firstNonEmptyMask, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_lastNonEmptyMask, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_firstNonEmptyMask, &maskIndex, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lastNonEmptyMask, &maskIndex,  sizeof(int), cudaMemcpyHostToDevice));

    // Copy bins and binBitMask to device
    CUDA_CHECK(cudaMemcpy(d_openListBins, h_openListBins, MAX_BINS * MAX_BIN_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_binCounts, h_binCounts, MAX_BINS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_binBitMask, h_binBitMask, binBitMaskSize * sizeof(unsigned long long), cudaMemcpyHostToDevice));

    // Variable to track the number of expanded nodes
    int* d_totalExpandedNodes;
    CUDA_CHECK(cudaMalloc((void**)&d_totalExpandedNodes, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_totalExpandedNodes, 0, sizeof(int)));

    // Initialize global variables
    int zeroInt = 0;
    bool falseBool = false;

    CUDA_CHECK(cudaMemcpyToSymbol(d_currentBin, &zeroInt, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_done, &falseBool, sizeof(bool)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_localFound, &falseBool, sizeof(bool)));

    // Timing the execution of the A* algorithm
    auto startTime = std::chrono::high_resolution_clock::now();

    // Define the grid and block dimensions
    // int totalThreads = 50000; // Adjust based on your GPU capability
    // int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;


    // multible buckets

    // Device buffers for bucket expansion
    // We remove the "int **expansionBuffers" and replace it with this:
    int *d_expansionBuffers; 
    int *d_expansionCounts;

    // Allocate memory for expansion buffers and counts
    cudaMalloc(&d_expansionCounts, MAX_BINS * sizeof(int));
    cudaMalloc(&d_expansionBuffers, sizeof(int) * MAX_BINS * MAX_BIN_SIZE);

    cudaMemset(d_expansionCounts, 0, MAX_BINS * sizeof(int));

    // Determine grid and block sizes
    int frontierSize = 500;
    int threadsPerBlock = 256;
    int totalThreads = frontierSize * 8 * 2; // One thread per neighbor for frontierSize nodes
    // totalThreads = 20000;
    int numBlocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    dim3 gridDim(numBlocks);
    dim3 blockDim(threadsPerBlock);

    // Launch kernel using cooperative groups

    // void *kernelOptimizedArgs[] = {
    //     (void *)&d_grid, (void *)&width, (void *)&height, (void *)&goalNodeId, (void *)&d_nodes,
    //     (void *)&d_openListBins, (void *)&d_binCounts, (void *)&d_binBitMask, (void *)&d_found,
    //     (void *)&d_path, (void *)&d_pathLength, (void *)&frontierSize, (void *)&d_totalExpandedNodes
    // };

    // CUDA_CHECK(cudaLaunchCooperativeKernel(
    //     (void *)aStarOptimized, gridDim, blockDim, kernelOptimizedArgs
    // ));
    // CUDA_KERNEL_CHECK();
    // CUDA_CHECK(cudaDeviceSynchronize());

    void *kernelArgsMultipleBuckets[] = {
        (void *)&d_grid, (void *)&width, (void *)&height, (void *)&goalNodeId, (void *)&d_nodes,
        (void *)&d_openListBins, (void *)&d_binCounts, (void *)&d_binBitMask,
        (void *)&d_expansionBuffers, (void *)&d_expansionCounts, (void *)&d_found,
        (void *)&d_path, (void *)&d_pathLength, (void *)&binBitMaskSize, (void *)&frontierSize, (void *)&d_totalExpandedNodes,
        (void *)&d_firstNonEmptyMask, &d_lastNonEmptyMask 
    
    };

    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void *)aStarMultipleBucketsSharedGrid, gridDim, blockDim, kernelArgsMultipleBuckets
    ));
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free expansion buffers
    for (int i = 0; i < MAX_BINS; ++i) {
        int *buffer;
        cudaMemcpy(&buffer, &d_expansionBuffers[i], sizeof(int *), cudaMemcpyDeviceToHost);
        cudaFree(buffer);
    }
    cudaFree(d_expansionBuffers);
    cudaFree(d_expansionCounts);

    // end multiple buckets

    // Prepare for cooperative launch
    void *kernelArgs[] = {
        (void *)&d_grid,
        (void *)&width,
        (void *)&height,
        (void *)&goalNodeId,
        (void *)&d_nodes,
        (void *)&d_openListBins,
        (void *)&d_binCounts,
        (void *)&d_binBitMask,
        (void *)&d_binExpansionBuffer,
        (void *)&d_binExpansionCount,
        (void *)&d_found,
        (void *)&d_path,
        (void *)&d_pathLength,
        (void *)&binBitMaskSize,
        (void *)&d_totalExpandedNodes
    };

    // Launch the kernel with cooperative groups
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));

    std::cout << "Device: " << deviceProp.name << std::endl;    
    printf("Maximum blocks per cooperative launch: %d\n", deviceProp.maxBlocksPerMultiProcessor * deviceProp.multiProcessorCount);

    if (!deviceProp.cooperativeLaunch) {
        fprintf(stderr, "Device does not support cooperative launch\n");
        exit(EXIT_FAILURE);
    }

    std::cout << "Max shared memory per block: " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;




    int device;
    cudaGetDevice(&device);

    int sharedMemPerSM;
    cudaDeviceGetAttribute(&sharedMemPerSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);

    std::cout << "Shared Memory per SM: " << sharedMemPerSM / 1024.0 << " KB" << std::endl;

    // CUDA_CHECK(cudaLaunchCooperativeKernel((void*)aStarKernel, blocksPerGrid, threadsPerBlock, kernelArgs));
    // CUDA_KERNEL_CHECK();
    // CUDA_CHECK(cudaDeviceSynchronize());

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedSeconds = endTime - startTime;

    // Retrieve results
    int h_pathLength;
    int *h_path = (int *)malloc(gridSize * sizeof(int));
    if (h_path == NULL) {
        fprintf(stderr, "Failed to allocate host memory for h_path\n");
        exit(EXIT_FAILURE);
    }

    bool h_found;
    CUDA_CHECK(cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost));

    if (h_found) {
        CUDA_CHECK(cudaMemcpy(&h_pathLength, d_pathLength, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_path, d_path, h_pathLength * sizeof(int), cudaMemcpyDeviceToHost));

        // Calculate total cost
        float totalCost = 0.0f;
        for (int i = h_pathLength - 1; i > 0; i--) {
            int currentNodeId = h_path[i];
            int nextNodeId = h_path[i - 1];

            int xCurrent = currentNodeId % width;
            int yCurrent = currentNodeId / width;
            int xNext = nextNodeId % width;
            int yNext = nextNodeId / width;

            int dx = abs(xNext - xCurrent);
            int dy = abs(yNext - yCurrent);
            bool isDiagonal = (dx + dy == 2);
            float movementCost = isDiagonal ? sqrtf(2.0f) : 1.0f;

            totalCost += movementCost;
        }

        // Print the results
        std::cout << "Path found with length " << h_pathLength << " and total cost " << totalCost << std::endl;
        std::cout << "Execution time (A* kernel): " << elapsedSeconds.count() << " seconds" << std::endl;
        std::cout << "Data copy time to GPU: " << copyTime.count() << " seconds" << std::endl;

        int h_totalExpandedNodes;
        cudaMemcpy(&h_totalExpandedNodes, d_totalExpandedNodes, sizeof(int), cudaMemcpyDeviceToHost);

        std::cout << "Total number of expanded nodes: " << h_totalExpandedNodes << std::endl;

        // Optionally, generate the image
        // std::string filename = "grid_path_visualization.ppm";
        // generatePPMImage(h_grid, width, height, h_path, h_pathLength, filename);
        // std::cout << "Image generated " << std::endl;

    } else {
        std::cout << "Path not found." << std::endl;
        std::cout << "Execution time (A* kernel): " << elapsedSeconds.count() << " seconds" << std::endl;
        std::cout << "Data copy time to GPU: " << copyTime.count() << " seconds" << std::endl;

        int h_totalExpandedNodes;
        cudaMemcpy(&h_totalExpandedNodes, d_totalExpandedNodes, sizeof(int), cudaMemcpyDeviceToHost);

        std::cout << "Total number of expanded nodes: " << h_totalExpandedNodes << std::endl;

        // Optionally, generate an image without the path
        // std::string filename = "grid_no_path.ppm";
        // generatePPMImage(h_grid, width, height, nullptr, 0, filename);
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_nodes));
    CUDA_CHECK(cudaFree(d_path));
    CUDA_CHECK(cudaFree(d_pathLength));
    CUDA_CHECK(cudaFree(d_found));
    CUDA_CHECK(cudaFree(d_openListBins));
    CUDA_CHECK(cudaFree(d_binCounts));
    CUDA_CHECK(cudaFree(d_binBitMask));
    CUDA_CHECK(cudaFree(d_binExpansionBuffer));
    CUDA_CHECK(cudaFree(d_binExpansionCount));
    CUDA_CHECK(cudaFree(d_grid));
    CUDA_CHECK(cudaFree(d_totalExpandedNodes));

    // Free host memory
    free(h_grid);
    free(h_path);
    free(h_binCounts);
    free(h_openListBins);
    free(h_binBitMask);

    return 0;
}
