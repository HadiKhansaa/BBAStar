// a_star_gpu_grid_8dir_parallel_no_closed_list_modified.cu

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <chrono>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define INF_FLT 1e20f  // A large float value representing infinity
#define MAX_NEIGHBORS 8        // 8-directional movement
#define GRID_WIDTH 10001       // Adjusted for demonstration
#define GRID_HEIGHT 10001      // Adjusted for demonstration

#define BIN_SIZE 1.0f          // Size of each bin in the bucket queue
#define MAX_BINS 10            // Maximum number of bins (adjust as needed)
#define MAX_BIN_SIZE 100000    // Maximum number of nodes per bin (adjust as needed)

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
    float g;
    float h;
    float f;
    int parent;
};

// Heuristic function (Manhattan distance)
__device__ __host__ float heuristic(int currentNodeId, int goalNodeId, int width) {
    int xCurrent = currentNodeId % width;
    int yCurrent = currentNodeId / width;
    int xGoal = goalNodeId % width;
    int yGoal = goalNodeId / width;

    int dx = abs(xCurrent - xGoal);
    int dy = abs(yCurrent - yGoal);
    return dx + dy;
}

// Kernel to initialize nodes
__global__ void initializeNodes(Node* nodes, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        nodes[idx].g = INF_FLT;
        nodes[idx].h = 0.0f;
        nodes[idx].f = INF_FLT;
        nodes[idx].parent = -1;
    }
}

// A* algorithm kernel without closed list check
__global__ void aStarKernel(int *grid, int width, int height, int goalNodeId,
                            Node *nodes,
                            int *currentOpenListBins, int *currentBinCounts,
                            unsigned int *currentBinBitMask,
                            int *currentBinExpansionBuffer, int *currentBinExpansionCount,
                            bool *found, int *path, int *pathLength, int currentBin, int *totalExpandedNodes) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Get the number of nodes in the current bin
    int binSize = currentBinCounts[currentBin];

    // Debug
    // if (idx == 0) {
    //     printf("Bin Size: %d\n", binSize);
    //     printf("Current Bin: %d\n\n", currentBin);
    // }

    if (idx >= binSize) return;

    // Each thread processes one node from the current bin
    int currentNodeId = currentOpenListBins[currentBin * MAX_BIN_SIZE + idx];

    atomicAdd(totalExpandedNodes, 1);

    // Check if the goal has been reached
    if (currentNodeId == goalNodeId) {
        *found = true;

        // Reconstruct the path
        int tempId = goalNodeId;
        int count = 0;
        while (tempId != -1 && count < width * height) {
            path[count++] = tempId;
            tempId = nodes[tempId].parent;
        }
        *pathLength = count;

        return;
    }

    Node currentNode = nodes[currentNodeId];

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

    // Expand neighbors
    for (int i = 0; i < MAX_NEIGHBORS; ++i) {
        int dx = neighborOffsets[i][0];
        int dy = neighborOffsets[i][1];
        int xNeighbor = xCurrent + dx;
        int yNeighbor = yCurrent + dy;

        if (xNeighbor >= 0 && xNeighbor < width && yNeighbor >= 0 && yNeighbor < height) {
            int neighborId = yNeighbor * width + xNeighbor;

            // Check if neighbor is blocked
            if (grid[neighborId] == 0) {  // 0 indicates free cell
                // Determine movement cost
                bool isDiagonal = (abs(dx) + abs(dy) == 2);
                float movementCost = isDiagonal ? sqrtf(2.0f) : 1.0f;

                float tentativeG = currentNode.g + movementCost;

                // Atomically update g value if a better path is found
                float oldG = atomicMinFloat(&nodes[neighborId].g, tentativeG);

                if (tentativeG < oldG) {
                    // Update node information
                    nodes[neighborId].id = neighborId;
                    nodes[neighborId].parent = currentNodeId;
                    nodes[neighborId].h = heuristic(neighborId, goalNodeId, width);
                    nodes[neighborId].f = tentativeG + nodes[neighborId].h;
                    nodes[neighborId].g = tentativeG;

                    // Determine the bin for the neighbor
                    float maxFValue = 2 * width;  // Adjusted max value
                    float minFValue = sqrtf((float)(width * width + height * height));

                    float normalizedF = ((nodes[neighborId].f - minFValue) / (maxFValue - minFValue)) * (MAX_BINS - 1);

                    int binForNeighbor = (int)normalizedF;
                    binForNeighbor = min(binForNeighbor, MAX_BINS - 1);

                    if (binForNeighbor == currentBin) {
                        // Collect nodes to the current bin expansion buffer
                        int position = atomicAdd(currentBinExpansionCount, 1);
                        if (position < MAX_BIN_SIZE) {
                            currentBinExpansionBuffer[position] = neighborId;
                        } else {
                            printf("Expansion buffer overflow for bin %d\n", currentBin);
                        }
                    } else {
                        // Atomically add neighbor to the appropriate bin
                        int position = atomicAdd(&currentBinCounts[binForNeighbor], 1);
                        if (position < MAX_BIN_SIZE) {
                            currentOpenListBins[binForNeighbor * MAX_BIN_SIZE + position] = neighborId;

                            // If position was 0, the bin was previously empty
                            if (position == 0) {
                                // Set the corresponding bit in currentBinBitMask
                                int maskIndex = binForNeighbor / 32;
                                unsigned int mask = 1U << (binForNeighbor % 32);
                                atomicOr(&currentBinBitMask[maskIndex], mask);
                            }
                        } else {
                            printf("Bin overflow at bin %d\n", binForNeighbor);
                        }
                    }
                }
            }
        }
    }
}

// Function to create a zigzag pattern in the grid
void createZigzagPattern(int *grid, int width, int height) {
    // Clear the grid
    for (int i = 0; i < width * height; i++) {
        grid[i] = 0;  // 0 indicates free cell
    }

    // Create the zigzag pattern
    int row = 0;
    while (row < height) {
        if (row % 4 == 1) {
            // Block cells except the last column
            for (int col = 0; col < width - 1; col++) {
                grid[row * width + col] = 1;  // 1 indicates blocked cell
            }
        } else if (row % 4 == 3) {
            // Block cells except the first column
            for (int col = 1; col < width; col++) {
                grid[row * width + col] = 1;  // 1 indicates blocked cell
            }
        }
        row++;
    }
}

#include <vector>
#include <cstdlib>

// Function to apply random obstacles and ensure a valid path exists
void applyRandomObstacles(int *grid, int width, int height, float obstacleRate) {
    // Seed the random number generator (if needed)
    // You can seed it once in your main function
    srand(time(NULL));

    // Initialize all cells to free
    for (int i = 0; i < width * height; i++) {
        grid[i] = 0; // 0 indicates free cell
    }

    // Place random obstacles
    for (int i = 0; i < width * height; i++) {
        float randValue = static_cast<float>(rand()) / RAND_MAX;
        if (randValue < obstacleRate) {
            grid[i] = 1; // 1 indicates an obstacle
        }
    }

    for(int i=0; i<width; ++i) // clear top row
        grid[i] = 0;

    for(int i = 0; i<height; ++i) // clear right column
        grid[height * i + width - 1] = 0;

    // Now, create a random path from start to goal and clear obstacles along the way
    // int x = 0;
    // int y = 0;
    // grid[y * width + x] = 0; // Ensure the start cell is free

    // while (x != width - 1 || y != height - 1) {
    //     // Decide randomly whether to move right or down (or stay if at the edge)
    //     bool canMoveRight = (x < width - 1);
    //     bool canMoveDown = (y < height - 1);

    //     if (canMoveRight && canMoveDown) {
    //         // Randomly choose to move right or down
    //         if (rand() % 2 == 0) {
    //             x++; // Move right
    //         } else {
    //             y++; // Move down
    //         }
    //     } else if (canMoveRight) {
    //         x++; // Move right
    //     } else if (canMoveDown) {
    //         y++; // Move down
    //     }

    //     // Clear any obstacles along the path
    //     grid[y * width + x] = 0;
    // }

    // Ensure the goal cell is free
    grid[(height - 1) * width + (width - 1)] = 0;
}

// Helper function to convert 2D index to 1D index
int index(int x, int y, int n) {
    return y * n + x;
}

// Function to divide and create walls recursively
void divide(int* grid, int startX, int startY, int width, int height, bool horizontal, int n) {
    if (width <= 2 || height <= 2) return;

    bool divideHorizontally = horizontal;

    if (divideHorizontally) {
        // Create a horizontal wall
        int wallY = startY + rand() % (height - 2) + 1;
        for (int x = startX; x < startX + width; ++x) {
            grid[index(x, wallY, n)] = 1;
        }

        // Create a passage in the wall
        int passageX = startX + rand() % width;
        grid[index(passageX, wallY, n)] = 0;

        // Recursively divide the top and bottom areas
        divide(grid, startX, startY, width, wallY - startY, !horizontal, n);  // Top half
        divide(grid, startX, wallY + 1, width, startY + height - wallY - 1, !horizontal, n);  // Bottom half
    } else {
        // Create a vertical wall
        int wallX = startX + rand() % (width - 2) + 1;
        for (int y = startY; y < startY + height; ++y) {
            grid[index(wallX, y, n)] = 1;
        }

        // Create a passage in the wall
        int passageY = startY + rand() % height;
        grid[index(wallX, passageY, n)] = 0;

        // Recursively divide the left and right areas
        divide(grid, startX, startY, wallX - startX, height, !horizontal, n);  // Left half
        divide(grid, wallX + 1, startY, startX + width - wallX - 1, height, !horizontal, n);  // Right half
    }
}

// Function to initialize the grid and generate the maze
void createMaze(int* grid, int n) {

    srand(time(0));
    
    // Initialize the grid with all 1s (open cells)
    for (int i = 0; i < n * n; ++i) {
        grid[i] = 0;
    }

    // Start recursive division with full grid
    divide(grid, 0, 0, n, n, rand() % 2 == 0, n);
}

// Function to print the maze
void printMaze(int* grid, int n) {
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            std::cout << (grid[index(x, y, n)] == 0 ? "  " : "[]");
        }
        std::cout << std::endl;
    }
}

// Function to calculate distance from the center of the grid
double distanceFromCenter(int x, int y, int n) {
    int centerX = n / 2;
    int centerY = n / 2;
    return sqrt(pow(x - centerX, 2) + pow(y - centerY, 2));
}

// Function to create a grid with obstacles concentrated near the center
void createConcentratedObstacles(int* grid, int n) {
    // Seed for random generation
    srand(time(0));

    // Maximum possible distance from the center
    double maxDistance = sqrt(2) * (n / 2.0);

    // Populate the grid
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            // Calculate distance from the center
            double distance = distanceFromCenter(x, y, n);

            // Probability of obstacle increases as we move towards the center
            // Here, the closer to the center, the higher the chance of obstacle
            double probability = 1.0 - (distance / maxDistance);

            // Use the probability to decide if the cell should be an obstacle (0) or open (1)
            if (static_cast<double>(rand()) / RAND_MAX < probability) {
                grid[index(x, y, n)] = 1;  // Blocked cell (obstacle)
            } else {
                grid[index(x, y, n)] = 0;  // Open cell
            }
        }
    }
}

// Function to generate a PPM image from the grid and path
void generatePPMImage(const int *grid, int width, int height, const int *path, int pathLength, const std::string &filename) {
    // Open the output file
    std::ofstream ofs(filename, std::ios::out | std::ios::binary);
    if (!ofs) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    // Write the PPM header
    ofs << "P6\n" << width << " " << height << "\n255\n";

    // Create a vector to mark the path cells
    std::vector<bool> isInPath(width * height, false);
    for (int i = 0; i < pathLength; ++i) {
        int nodeId = path[i];
        if (nodeId >= 0 && nodeId < width * height) {
            isInPath[nodeId] = true;
        }
    }

    // Write pixel data
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;

            unsigned char r, g, b;

            if (isInPath[idx]) {
                // Path cell - Red color
                r = 255;
                g = 0;
                b = 0;
            } else if (grid[idx] == 1) {
                // Obstacle - Black color
                r = g = b = 0;
            } else {
                // Free cell - White color
                r = g = b = 255;
            }

            ofs.write(reinterpret_cast<char*>(&r), 1);
            ofs.write(reinterpret_cast<char*>(&g), 1);
            ofs.write(reinterpret_cast<char*>(&b), 1);
        }
    }

    ofs.close();
    std::cout << "Image saved to " << filename << std::endl;
}

// Main function
int main(int argc, char** argv) {
    // Grid dimensions and initialization code (same as before)
    int width = GRID_WIDTH;
    int height = GRID_HEIGHT;
    float obstacleRate = 0.2; // Default obstacle rate (percentage)
    std::string gridType = "";

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

    // Apply obstacles
    if(gridType == "random")
        applyRandomObstacles(h_grid, width, height, obstacleRate);
    else if(gridType == "maze")
        createMaze(h_grid, height);
    else if(gridType == "blockCenter")
        createConcentratedObstacles(h_grid, height);
    else if(gridType == "zigzag")
        createZigzagPattern(h_grid, width, height);
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
    int *d_currentOpenListBins;
    int *d_currentBinCounts;
    CUDA_CHECK(cudaMalloc((void **)&d_currentOpenListBins, MAX_BINS * MAX_BIN_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_currentBinCounts, MAX_BINS * sizeof(int)));

    CUDA_CHECK(cudaMemset(d_currentOpenListBins, -1, MAX_BINS * MAX_BIN_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_currentBinCounts, 0, MAX_BINS * sizeof(int)));

    // Initialize binBitMask
    unsigned int *d_currentBinBitMask;
    int binBitMaskSize = (MAX_BINS + 31) / 32;
    CUDA_CHECK(cudaMalloc((void **)&d_currentBinBitMask, binBitMaskSize * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_currentBinBitMask, 0, binBitMaskSize * sizeof(unsigned int)));

    // Allocate device memory for current bin expansion buffer and count
    int *d_currentBinExpansionBuffer;
    int *d_currentBinExpansionCount;
    CUDA_CHECK(cudaMalloc((void**)&d_currentBinExpansionBuffer, MAX_BIN_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_currentBinExpansionCount, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_currentBinExpansionCount, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_currentBinExpansionBuffer, -1, MAX_BIN_SIZE * sizeof(int)));

    // Initialize the start node and add it to the open list
    Node h_startNode;
    h_startNode.id = startNodeId;
    h_startNode.g = 0.0f;
    h_startNode.h = heuristic(startNodeId, goalNodeId, width);
    h_startNode.f = h_startNode.g + h_startNode.h;
    h_startNode.parent = -1;

    CUDA_CHECK(cudaMemcpy(&d_nodes[startNodeId], &h_startNode, sizeof(Node), cudaMemcpyHostToDevice));

    // Determine the bin for the start node
    // int startBin = (int)(h_startNode.f / BIN_SIZE);
    // startBin = std::min(startBin, MAX_BINS - 1);

    // Determine the bin for the neighbor
    float maxFValue = 2 * width;  // Adjusted max value
    float minFValue = sqrtf((float)(width * width + height * height));

    float normalizedF = ((h_startNode.f - minFValue) / (maxFValue - minFValue)) * (MAX_BINS - 1);

    int startBin = (int)normalizedF;
    startBin = min(startBin, MAX_BINS - 1);

    // Add start node to the current bins
    int* h_binCounts = (int*)malloc(MAX_BINS * sizeof(int));
    if (h_binCounts == NULL) {
        fprintf(stderr, "Failed to allocate memory for h_binCounts\n");
        exit(EXIT_FAILURE);
    }
    memset(h_binCounts, 0, MAX_BINS * sizeof(int));

    int* h_openListBins = (int*)malloc(MAX_BINS * MAX_BIN_SIZE * sizeof(int));
    if (h_openListBins == NULL) {
        fprintf(stderr, "Failed to allocate memory for h_openListBins\n");
        exit(EXIT_FAILURE);
    }
    memset(h_openListBins, -1, MAX_BINS * MAX_BIN_SIZE * sizeof(int));

    h_binCounts[startBin] = 1;
    h_openListBins[startBin * MAX_BIN_SIZE] = startNodeId;

    // Initialize binBitMask on host
    unsigned int *h_binBitMask = (unsigned int*)malloc(binBitMaskSize * sizeof(unsigned int));
    memset(h_binBitMask, 0, binBitMaskSize * sizeof(unsigned int));

    // Set the bit for the start bin
    int maskIndex = startBin / 32;
    unsigned int mask = 1U << (startBin % 32);
    h_binBitMask[maskIndex] |= mask;

    // Copy bins and binBitMask to device
    CUDA_CHECK(cudaMemcpy(d_currentOpenListBins, h_openListBins, MAX_BINS * MAX_BIN_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_currentBinCounts, h_binCounts, MAX_BINS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_currentBinBitMask, h_binBitMask, binBitMaskSize * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // variable to track the number of expanded nodes
    int* d_totalExpandedNodes;
    cudaMalloc((void**)&d_totalExpandedNodes, sizeof(int));
    cudaMemset(d_totalExpandedNodes, 0, sizeof(int));

    // Host variable to check if the goal was found
    bool h_found = false;

    // Timing the execution of the A* algorithm
    auto startTime = std::chrono::high_resolution_clock::now();

    int nbKernels = 0;

    while (!h_found) {
        // Copy binBitMask back to host to find the active bin
        CUDA_CHECK(cudaMemcpy(h_binBitMask, d_currentBinBitMask, binBitMaskSize * sizeof(unsigned int), cudaMemcpyDeviceToHost));

        int currentBin = -1;
        for (int i = 0; i < binBitMaskSize; ++i) {
            if (h_binBitMask[i] != 0) {
                int firstSetBit = __builtin_ffs(h_binBitMask[i]) - 1; // ffs returns 1-based index
                currentBin = i * 32 + firstSetBit;
                break;
            }
        }

        if (currentBin == -1 || currentBin >= MAX_BINS) {
            // Open list is empty, no path found
            break;
        }

        // Get the bin size
        int binSize;
        CUDA_CHECK(cudaMemcpy(&binSize, &d_currentBinCounts[currentBin], sizeof(int), cudaMemcpyDeviceToHost));

        // Process nodes in the current bin
        if (binSize > 0) {
            int threadsPerBlock = 256;
            int blocksPerGrid = (binSize + threadsPerBlock - 1) / threadsPerBlock;

            // Reset currentBinExpansionCount and buffer
            CUDA_CHECK(cudaMemset(d_currentBinExpansionCount, 0, sizeof(int)));
            CUDA_CHECK(cudaMemset(d_currentBinExpansionBuffer, -1, MAX_BIN_SIZE * sizeof(int)));

            std::cout<<nbKernels++<<"\n";

            aStarKernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_grid, width, height, goalNodeId,
                d_nodes,
                d_currentOpenListBins, d_currentBinCounts,
                d_currentBinBitMask,
                d_currentBinExpansionBuffer, d_currentBinExpansionCount,
                d_found,
                d_path, d_pathLength, currentBin, d_totalExpandedNodes);

            CUDA_KERNEL_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());

            // After kernel execution
            CUDA_CHECK(cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost));
            if (h_found) {
                break;
            }

            // Get currentBinExpansionCount from device
            int h_currentBinExpansionCount;
            CUDA_CHECK(cudaMemcpy(&h_currentBinExpansionCount, d_currentBinExpansionCount, sizeof(int), cudaMemcpyDeviceToHost));

            if (h_currentBinExpansionCount > 0) {
                // Update currentBinCounts[currentBin] to h_currentBinExpansionCount
                CUDA_CHECK(cudaMemcpy(&d_currentBinCounts[currentBin], &h_currentBinExpansionCount, sizeof(int), cudaMemcpyHostToDevice));

                // Copy elements from currentBinExpansionBuffer to currentOpenListBins[currentBin * MAX_BIN_SIZE]
                CUDA_CHECK(cudaMemcpy(&d_currentOpenListBins[currentBin * MAX_BIN_SIZE],
                                      d_currentBinExpansionBuffer,
                                      h_currentBinExpansionCount * sizeof(int),
                                      cudaMemcpyDeviceToDevice));
            } else {
                // Set currentBinCounts[currentBin] to 0
                int zero = 0;
                CUDA_CHECK(cudaMemcpy(&d_currentBinCounts[currentBin], &zero, sizeof(int), cudaMemcpyHostToDevice));

                // Clear the bit in currentBinBitMask
                int maskIndex = currentBin / 32;
                unsigned int mask = ~(1U << (currentBin % 32));
                h_binBitMask[maskIndex] &= mask;
                CUDA_CHECK(cudaMemcpy(&d_currentBinBitMask[maskIndex], &h_binBitMask[maskIndex], sizeof(unsigned int), cudaMemcpyHostToDevice));
            }

            // Reset currentBinExpansionCount and buffer for next iteration
            CUDA_CHECK(cudaMemset(d_currentBinExpansionCount, 0, sizeof(int)));
            CUDA_CHECK(cudaMemset(d_currentBinExpansionBuffer, -1, MAX_BIN_SIZE * sizeof(int)));
        } else {
            // Bin is empty, set currentBinCounts[currentBin] to 0
            int zero = 0;
            CUDA_CHECK(cudaMemcpy(&d_currentBinCounts[currentBin], &zero, sizeof(int), cudaMemcpyHostToDevice));

            // Clear the bit in currentBinBitMask
            int maskIndex = currentBin / 32;
            unsigned int mask = ~(1U << (currentBin % 32));
            h_binBitMask[maskIndex] &= mask;
            CUDA_CHECK(cudaMemcpy(&d_currentBinBitMask[maskIndex], &h_binBitMask[maskIndex], sizeof(unsigned int), cudaMemcpyHostToDevice));
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedSeconds = endTime - startTime;

    // Retrieve results
    int h_pathLength;
    int *h_path = (int *)malloc(gridSize * sizeof(int));
    if (h_path == NULL) {
        fprintf(stderr, "Failed to allocate host memory for h_path\n");
        exit(EXIT_FAILURE);
    }

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
    CUDA_CHECK(cudaFree(d_currentOpenListBins));
    CUDA_CHECK(cudaFree(d_currentBinCounts));
    CUDA_CHECK(cudaFree(d_currentBinBitMask));
    CUDA_CHECK(cudaFree(d_currentBinExpansionBuffer));
    CUDA_CHECK(cudaFree(d_currentBinExpansionCount));
    CUDA_CHECK(cudaFree(d_grid));

    // Free host memory
    free(h_grid);
    free(h_path);
    free(h_binCounts);
    free(h_openListBins);
    free(h_binBitMask);

    return 0;
}
