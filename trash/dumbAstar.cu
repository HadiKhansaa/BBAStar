// a_star_gpu_grid_8dir.cu

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <chrono>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <string>

#define INF_FLT 1e20f  // A large float value representing infinity
#define MAX_NEIGHBORS 8        // 8-directional movement
#define GRID_WIDTH 10001       // Adjusted for demonstration
#define GRID_HEIGHT 10001         // Adjusted for demonstration

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

// Mutex implementation
struct Mutex {
    int lock;

    __device__ void init() {
        lock = 0;
    }

    __device__ void acquire() {
        while (atomicCAS(&lock, 0, 1) != 0) {
            // Busy-wait
        }
    }

    __device__ void release() {
        atomicExch(&lock, 0);
    }
};

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

// Priority Queue implementation
struct PriorityQueue {
    int *heap;
    volatile int size;
    int capacity;
    Mutex *mutex;
    float *fValues;  // Corresponding f-values for nodes

    __device__ void init(int *buffer, int capacity, Mutex *mutex, float *fValues) {
        this->heap = buffer;
        this->size = 0;
        this->capacity = capacity;
        this->mutex = mutex;
        this->fValues = fValues;
    }

    __device__ void swap(int i, int j) {
        int tempNode = heap[i];
        heap[i] = heap[j];
        heap[j] = tempNode;

        float tempF = fValues[i];
        fValues[i] = fValues[j];
        fValues[j] = tempF;
    }

    __device__ void heapifyUp(int index) {
        while (index > 0) {
            int parent = (index - 1) / 2;
            if (fValues[parent] <= fValues[index])
                break;
            swap(parent, index);
            index = parent;
        }
    }

    __device__ void heapifyDown(int index) {
        while (true) {
            int left = 2 * index + 1;
            int right = 2 * index + 2;
            int smallest = index;

            if (left < size && fValues[left] < fValues[smallest])
                smallest = left;
            if (right < size && fValues[right] < fValues[smallest])
                smallest = right;
            if (smallest == index)
                break;
            swap(index, smallest);
            index = smallest;
        }
    }

    __device__ bool insert(int nodeId, float fValue) {
        mutex->acquire();

        if (size >= capacity) {
            printf("PriorityQueue is full\n");
            mutex->release();
            return false;
        }

        heap[size] = nodeId;
        fValues[size] = fValue;
        heapifyUp(size);
        size++;

        mutex->release();
        return true;
    }

    __device__ bool remove(int *nodeId) {
        mutex->acquire();

        if (size <= 0) {
            mutex->release();
            return false;
        }

        *nodeId = heap[0];
        size--;
        heap[0] = heap[size];
        fValues[0] = fValues[size];
        heapifyDown(0);

        mutex->release();
        return true;
    }
};

// Node structure
struct Node {
    int id;
    float g;
    float h;
    float f;
    int parent;
};

// Heuristic function (Octile distance)
__device__ float heuristic(int currentNodeId, int goalNodeId, int width) {
    int xCurrent = currentNodeId % width;
    int yCurrent = currentNodeId / width;
    int xGoal = goalNodeId % width;
    int yGoal = goalNodeId / width;

    int dx = abs(xCurrent - xGoal);
    int dy = abs(yCurrent - yGoal);
    float D = 1.0f;
    float D2 = sqrtf(2.0f);

    return dx + dy;
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy);
}

// Kernel to initialize nodes
__global__ void initializeNodes(Node* nodes, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        nodes[idx].g = INF_FLT;
        nodes[idx].h = 0.0f;
        nodes[idx].f = 0.0f;
        nodes[idx].parent = -1;
    }
}

// Kernel to initialize the priority queue
__global__ void initializePriorityQueue(PriorityQueue *pq, int *heap, int capacity, Mutex *mutex, float *fValues) {
    pq->init(heap, capacity, mutex, fValues);
}

// Kernel to initialize the mutex
__global__ void initializeMutex(Mutex *mutex) {
    mutex->init();
}

// A* algorithm kernel
__global__ void aStarKernel(int *grid, int width, int height, int startNodeId, int goalNodeId,
                            Node *nodes, PriorityQueue *openList, int *closedListFlags, bool *found,
                            int *path, int *pathLength, int* totalExpandedNodes) {
    int idx = threadIdx.x;

    // Shared variables for synchronization
    __shared__ int s_currentNodeId;
    __shared__ bool s_finished;

    // Initialize shared variables
    if (idx == 0) {
        s_finished = false;
    }
    __syncthreads();

    // Initialize open list with the start node
    if (idx == 0) {
        Node startNode;
        startNode.id = startNodeId;
        startNode.g = 0.0f;
        startNode.h = heuristic(startNodeId, goalNodeId, width);
        startNode.f = startNode.g + startNode.h;
        startNode.parent = -1;

        nodes[startNodeId] = startNode;

        bool inserted = openList->insert(startNodeId, startNode.f);
        if (!inserted) {
            printf("Failed to insert start node into open list\n");
            *found = false;
            return;
        }
    }

    __syncthreads();

    while (true) {
        __syncthreads();

        int currentNodeId = -1;
        if (idx == 0) {
            if (!openList->remove(&currentNodeId)) {
                // Open list is empty, no path found
                s_finished = true;
                currentNodeId = -1;
            } else {
                s_finished = false;
            }
        }

        __syncthreads();

        // Check if the open list is empty
        if (s_finished) {
            if (idx == 0) {
                *found = false;
            }
            break;
        }

        // Broadcast currentNodeId to all threads
        s_currentNodeId = currentNodeId;
        __syncthreads();
        currentNodeId = s_currentNodeId;

        // Check if the goal has been reached
        if (currentNodeId == goalNodeId) {
            if (idx == 0) {
                *found = true;
                // Reconstruct path (in reverse order)
                int tempId = goalNodeId;
                int count = 0;
                while (tempId != -1 && count < width * height) {
                    path[count++] = tempId;
                    tempId = nodes[tempId].parent;
                }
                *pathLength = count;
            }
            __syncthreads();
            break;
        }

        // Mark the current node as closed
        closedListFlags[currentNodeId] = 1;

        Node currentNode = nodes[currentNodeId];

        if(idx == 0)
        {
            atomicAdd(totalExpandedNodes, 1);
            // printf("Total Expanded Nodes: %d\n", *totalExpandedNodes);
        }

        // Expand neighbors
        if (idx < MAX_NEIGHBORS) {
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

            int dx = neighborOffsets[idx][0];
            int dy = neighborOffsets[idx][1];
            int xNeighbor = xCurrent + dx;
            int yNeighbor = yCurrent + dy;

            if (xNeighbor >= 0 && xNeighbor < width && yNeighbor >= 0 && yNeighbor < height) {
                int neighborId = yNeighbor * width + xNeighbor;

                // Check if neighbor is blocked
                if (grid[neighborId] == 0) {  // 0 indicates free cell
                    // Check if neighbor is in the closed list
                    if (closedListFlags[neighborId] == 0) {
                        // Determine movement cost
                        bool isDiagonal = (abs(dx) + abs(dy) == 2);
                        float movementCost = isDiagonal ? sqrtf(2.0f) : 1.0f;

                        float tentativeG = currentNode.g + movementCost;

                        // Use atomic operation to check and update g value
                        float oldG = atomicMinFloat(&nodes[neighborId].g, tentativeG);

                        if (tentativeG < oldG) {
                            // Update node information
                            nodes[neighborId].id = neighborId;
                            nodes[neighborId].parent = currentNodeId;
                            nodes[neighborId].h = heuristic(neighborId, goalNodeId, width);
                            nodes[neighborId].f = tentativeG + nodes[neighborId].h;
                            nodes[neighborId].g = tentativeG;

                            // Insert neighbor into the open list
                            bool inserted = openList->insert(neighborId, nodes[neighborId].f);
                            if (!inserted) {
                                printf("Failed to insert node %d into open list\n", neighborId);
                            }
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

// Function to place random rectangle-shaped obstacles on the grid
void applyRandomRectangleObstacles(int *grid, int width, int height, float rectangleRate) {
    // Seed the random number generator (if needed)
    srand(time(NULL));

    // Initialize all cells to free
    for (int i = 0; i < width * height; i++) {
        grid[i] = 0; // 0 indicates free cell
    }

    // Place random rectangle-shaped obstacles
    int maxRectWidth = width / 4;  // Define a maximum width for a rectangle (adjustable)
    int maxRectHeight = height / 4; // Define a maximum height for a rectangle (adjustable)

    for (int r = 0; r < (width/10) * rectangleRate ; r++) {
        int startX = rand() % width;
        int startY = rand() % height;
        int rectWidth = rand() % maxRectWidth + 1; // Minimum width of 1
        int rectHeight = rand() % maxRectHeight + 1; // Minimum height of 1

        // Place the rectangle on the grid, ensuring it doesn't go out of bounds
        for (int i = startY; i < startY + rectHeight && i < height; i++) {
            for (int j = startX; j < startX + rectWidth && j < width; j++) {
                grid[i * width + j] = 1; // 1 indicates an obstacle
            }
        }
    }

    for(int i=0; i<height; ++i) // clear top row
        grid[i] = 0;

    for(int i = 0; i<height; ++i) // clear right column
        grid[height * i + height - 1] = 0;

    // Ensure the starting and goal cells are free
    grid[0] = 0; // Starting cell
    grid[(height - 1) * width + (width - 1)] = 0; // Goal cell
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


int main(int argc, char** argv) {
    // Grid dimensions
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
    else if (gridType == "rectangle")
        applyRandomRectangleObstacles(h_grid, width, height, obstacleRate);
    else
        applyRandomObstacles(h_grid, width, height, obstacleRate);


    // Timing the data copy to GPU
    auto startCopyTime = std::chrono::high_resolution_clock::now();

    // Allocate and copy grid to device
    int *d_grid;
    CUDA_CHECK(cudaMalloc((void **)&d_grid, gridSize * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_grid, h_grid, gridSize * sizeof(int), cudaMemcpyHostToDevice));

    // Device variables
    Node *d_nodes;
    int *d_closedListFlags;
    int *d_path;
    int *d_pathLength;
    bool *d_found;

    CUDA_CHECK(cudaMalloc((void **)&d_nodes, gridSize * sizeof(Node)));
    CUDA_CHECK(cudaMalloc((void **)&d_closedListFlags, gridSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_path, gridSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_pathLength, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_found, sizeof(bool)));

    auto endCopyTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> copyTime = endCopyTime - startCopyTime;

    // Initialize nodes[].g to INF_FLT
    dim3 threadsPerBlockInit(16, 16);
    dim3 blocksPerGridInit((width + threadsPerBlockInit.x - 1) / threadsPerBlockInit.x,
                           (height + threadsPerBlockInit.y - 1) / threadsPerBlockInit.y);

    initializeNodes<<<blocksPerGridInit, threadsPerBlockInit>>>(d_nodes, width, height);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Initialize other device variables
    CUDA_CHECK(cudaMemset(d_closedListFlags, 0, gridSize * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_path, -1, gridSize * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_pathLength, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_found, 0, sizeof(bool)));

    // Initialize priority queue
    int *d_heap;
    float *d_fValues;
    Mutex *d_mutex;
    CUDA_CHECK(cudaMalloc((void **)&d_heap, gridSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_fValues, gridSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_mutex, sizeof(Mutex)));

    // Initialize mutex on device
    initializeMutex<<<1, 1>>>(d_mutex);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Initialize priority queue on device
    PriorityQueue *d_pq;
    CUDA_CHECK(cudaMalloc((void **)&d_pq, sizeof(PriorityQueue)));
    initializePriorityQueue<<<1, 1>>>(d_pq, d_heap, gridSize, d_mutex, d_fValues);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Display device properties (optional)
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Using device %s with compute capability %d.%d\n", deviceProp.name, deviceProp.major, deviceProp.minor);
    printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Max threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("Total global memory: %zu bytes\n", deviceProp.totalGlobalMem);

    int* d_totalExpandedNodes;
    CUDA_CHECK(cudaMalloc((void**)&d_totalExpandedNodes, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_totalExpandedNodes, 0, sizeof(int)));

    // Timing the execution of the A* algorithm
    auto startTime = std::chrono::high_resolution_clock::now();

    // Launch A* kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = 1;  // Single block for simplicity

    aStarKernel<<<blocksPerGrid, threadsPerBlock>>>(d_grid, width, height, startNodeId, goalNodeId,
                                                    d_nodes, d_pq, d_closedListFlags, d_found,
                                                    d_path, d_pathLength, d_totalExpandedNodes);

    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedSeconds = endTime - startTime;

    // Retrieve results
    bool h_found;
    int h_pathLength;
    int *h_path = (int *)malloc(gridSize * sizeof(int));
    if (h_path == NULL) {
        fprintf(stderr, "Failed to allocate host memory for h_path\n");
        exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost));

    if (h_found) {
        CUDA_CHECK(cudaMemcpy(&h_pathLength, d_pathLength, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_path, d_path, h_pathLength * sizeof(int), cudaMemcpyDeviceToHost));

        // Calculate total cost
        // Sum the movement costs along the path
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
    
        // Generate the image
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
    CUDA_CHECK(cudaFree(d_closedListFlags));
    CUDA_CHECK(cudaFree(d_path));
    CUDA_CHECK(cudaFree(d_pathLength));
    CUDA_CHECK(cudaFree(d_found));
    CUDA_CHECK(cudaFree(d_heap));
    CUDA_CHECK(cudaFree(d_fValues));
    CUDA_CHECK(cudaFree(d_mutex));
    CUDA_CHECK(cudaFree(d_pq));
    CUDA_CHECK(cudaFree(d_grid));

    // Free host memory
    free(h_grid);
    free(h_path);

    return 0;
}