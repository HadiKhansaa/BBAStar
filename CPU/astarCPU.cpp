// a_star_cpu_grid.cpp

#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>

#define INF_FLT 1e20f  // A large float value representing infinity
#define GRID_WIDTH 10001  // Adjusted for demonstration
#define GRID_HEIGHT 10001 // Adjusted for demonstration

// Node structure
struct Node {
    int id;
    float g;
    float h;
    float f;
    int parent;
    bool closed;
    bool opened;
};

// Heuristic function (Octile distance)
float heuristic(int currentNodeId, int goalNodeId, int width) {
    int xCurrent = currentNodeId % width;
    int yCurrent = currentNodeId / width;
    int xGoal = goalNodeId % width;
    int yGoal = goalNodeId / width;

    int dx = abs(xCurrent - xGoal);
    int dy = abs(yCurrent - yGoal);
    float D = 1.0f;
    float D2 = std::sqrt(2.0f);

    // return 0; // testing
    return dx + dy;
    return D * (dx + dy) + (D2 - 2 * D) * std::min(dx, dy);
}

// Function to create a zigzag pattern in the grid
void createZigzagPattern(std::vector<int>& grid, int width, int height) {
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

// Function to apply random obstacles and ensure a valid path exists
void applyRandomObstacles(std::vector<int>& grid, int width, int height, float obstacleRate) {
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

// A* algorithm implementation
bool aStarCPU(const std::vector<int>& grid, int width, int height, int startNodeId, int goalNodeId,
              std::vector<Node>& nodes, std::vector<int>& path, int &totalExpandedNodes) {
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

    // Initialize nodes
    for (int i = 0; i < width * height; i++) {
        nodes[i].id = i;
        nodes[i].g = INF_FLT;
        nodes[i].h = 0.0f;
        nodes[i].f = 0.0f;
        nodes[i].parent = -1;
        nodes[i].closed = false;
        nodes[i].opened = false;
    }

    // Lambda function for the priority queue comparator
    auto cmp = [](const Node* left, const Node* right) {
        return left->f > right->f;
    };

    // Priority queue (min-heap)
    std::priority_queue<Node*, std::vector<Node*>, decltype(cmp)> openList(cmp);

    // Initialize start node
    Node* startNode = &nodes[startNodeId];
    startNode->g = 0.0f;
    startNode->h = heuristic(startNodeId, goalNodeId, width);
    startNode->f = startNode->g + startNode->h;
    startNode->opened = true;
    openList.push(startNode);

    // int nodesExpanded = 0;
    while (!openList.empty()) {
        Node* currentNode = openList.top();
        openList.pop();

        ++totalExpandedNodes;

        currentNode->closed = true;
        // nodesExpanded++;

        // if (nodesExpanded % 100000 == 0) {
        //     std::cout << "Nodes expanded: " << nodesExpanded << std::endl;
        // }

        if (currentNode->id == goalNodeId) {
            // Path found
            int tempId = goalNodeId;
            while (tempId != -1) {
                path.push_back(tempId);
                tempId = nodes[tempId].parent;
            }
            return true;
        }

        int xCurrent = currentNode->id % width;
        int yCurrent = currentNode->id / width;

        // Expand neighbors
        for (int idx = 0; idx < 8; idx++) {
            int dx = neighborOffsets[idx][0];
            int dy = neighborOffsets[idx][1];
            int xNeighbor = xCurrent + dx;
            int yNeighbor = yCurrent + dy;

            if (xNeighbor >= 0 && xNeighbor < width && yNeighbor >= 0 && yNeighbor < height) {
                int neighborId = yNeighbor * width + xNeighbor;
                Node* neighborNode = &nodes[neighborId];

                if (grid[neighborId] == 0 && !neighborNode->closed) {
                    // Determine movement cost
                    bool isDiagonal = (abs(dx) + abs(dy) == 2);
                    float movementCost = isDiagonal ? std::sqrt(2.0f) : 1.0f;

                    float tentativeG = currentNode->g + movementCost;

                    if (!neighborNode->opened || tentativeG < neighborNode->g) {
                        neighborNode->parent = currentNode->id;
                        neighborNode->g = tentativeG;
                        neighborNode->h = heuristic(neighborId, goalNodeId, width);
                        neighborNode->f = neighborNode->g + neighborNode->h;

                        if (!neighborNode->opened) {
                            openList.push(neighborNode);
                            neighborNode->opened = true;
                        }
                    }
                }
            }
        }
    }

    // Path not found
    return false;
}


// Function to load the grid from a compressed file
bool loadCompressedGridFromFile(int *&grid, int &width, int &height, const std::string &filename) {
    std::ifstream ifs(filename, std::ios::in | std::ios::binary);
    if (!ifs) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return false;
    }

    // Read grid dimensions
    ifs.read(reinterpret_cast<char*>(&width), sizeof(int));
    ifs.read(reinterpret_cast<char*>(&height), sizeof(int));

    int gridSize = width * height;
    grid = (int *)malloc(gridSize * sizeof(int));
    if (grid == NULL) {
        std::cerr << "Failed to allocate memory for grid\n";
        ifs.close();
        return false;
    }

    // Read compressed grid data
    std::vector<unsigned char> compressedData((gridSize + 7) / 8, 0);
    ifs.read(reinterpret_cast<char*>(compressedData.data()), compressedData.size());

    // Decompress grid data
    for (int i = 0; i < gridSize; ++i) {
        grid[i] = (compressedData[i / 8] & (1 << (i % 8))) ? 1 : 0;
    }

    ifs.close();
    std::cout << "Compressed grid loaded from " << filename << std::endl;
    return true;
}


// Function to generate a PPM image from the grid and path
void generatePPMImage(const int *grid, int width, int height, const std::vector<int> &path, int pathLength, const std::string &filename) {
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
    float obstacleRate = 0.2;
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

    // Allocate and initialize grid
    std::vector<int> grid(gridSize, 0);

    int* tempGrid = (int*)malloc(height*width*sizeof(int));

    for(int i = 0; i< height*width; ++i)
        grid[i] = tempGrid[i];

    if(gridPath!="")
    {
        loadCompressedGridFromFile(tempGrid, width, height, gridPath);  // load grid from file
        for(int i = 0; i< height*width; ++i)
            grid[i] = tempGrid[i];
    }

    // Apply obstacles
    else if(gridType == "random")
        applyRandomObstacles(grid, width, height, obstacleRate);
    else if(gridType == "maze")
    {
        createMaze(tempGrid, height);
        for(int i = 0; i< height*width; ++i)
            grid[i] = tempGrid[i];
    }
    else if(gridType == "blockCenter")
    {
        createConcentratedObstacles(tempGrid, height);
        for(int i = 0; i< height*width; ++i)
            grid[i] = tempGrid[i];
    }
    else if(gridType == "zigzag")
        createZigzagPattern(grid, width, height);
    else if (gridType == "rectangle")
    {
        applyRandomRectangleObstacles(tempGrid, width, height, obstacleRate);
        for(int i = 0; i< height*width; ++i)
            grid[i] = tempGrid[i];
    }
    else
        applyRandomObstacles(grid, width, height, obstacleRate);

    // Allocate nodes
    std::vector<Node> nodes(gridSize);

    // Path vector
    std::vector<int> path;

    // variable to track the total number of expanded nodes
    int totalExpandedNodes = 0;

    // Timing the execution
    auto startTime = std::chrono::high_resolution_clock::now();

    // Run A* algorithm
    bool found = aStarCPU(grid, width, height, startNodeId, goalNodeId, nodes, path, totalExpandedNodes);

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedSeconds = endTime - startTime;

    if (found) {
        // Calculate total cost
        float totalCost = 0.0f;
        for (size_t i = path.size() - 1; i > 0; i--) {
            int currentNodeId = path[i];
            int nextNodeId = path[i - 1];

            int xCurrent = currentNodeId % width;
            int yCurrent = currentNodeId / width;
            int xNext = nextNodeId % width;
            int yNext = nextNodeId / width;

            int dx = abs(xNext - xCurrent);
            int dy = abs(yNext - yCurrent);
            bool isDiagonal = (dx + dy == 2);
            float movementCost = isDiagonal ? std::sqrt(2.0f) : 1.0f;

            totalCost += movementCost;
        }

        // Print the results
        std::cout << "Path found with length " << path.size() << " and total cost " << totalCost << std::endl;
        std::cout << "Execution time: " << elapsedSeconds.count() << " seconds" << std::endl;
        std::cout << "Total number of expanded nodes: " << totalExpandedNodes << std::endl;

        // Optionally, generate the image
        // std::string filename = "grid_path_visualization.ppm";
        // generatePPMImage(tempGrid, width, height, path, path.size(), filename);
        // std::cout << "Image generated " << std::endl;

    } else {
        std::cout << "Path not found." << std::endl;
        std::cout << "Execution time: " << elapsedSeconds.count() << " seconds" << std::endl;
        std::cout << "Total number of expanded nodes: " << totalExpandedNodes << std::endl;
    }

    // Optionally, print the grid and path
    /*
    // Mark the path on the grid
    for (size_t i = 0; i < path.size(); i++) {
        int idx = path[i];
        grid[idx] = 2;  // Mark the path
    }

    // Print the grid
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            if (grid[idx] == 1)
                std::cout << "#";  // Blocked cell
            else if (grid[idx] == 2)
                std::cout << "*";  // Path cell
            else
                std::cout << ".";  // Free cell
        }
        std::cout << std::endl;
    }
    */

    return 0;
}
