#include "grid_generation.cuh"

// Function to save the grid to a compressed file
void saveCompressedGridToFile(const int *grid, int width, int height, const std::string &filename) {
    std::ofstream ofs(filename, std::ios::out | std::ios::binary);
    if (!ofs) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    // Write grid dimensions as the header
    ofs.write(reinterpret_cast<const char*>(&width), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&height), sizeof(int));

    // Compress grid data (1 bit per cell)
    int gridSize = width * height;
    std::vector<unsigned char> compressedData((gridSize + 7) / 8, 0);

    for (int i = 0; i < gridSize; ++i) {
        if (grid[i] == 1) {
            compressedData[i / 8] |= (1 << (i % 8));
        }
    }

    // Write compressed grid data
    ofs.write(reinterpret_cast<const char*>(compressedData.data()), compressedData.size());

    ofs.close();
    std::cout << "Compressed grid saved to " << filename << std::endl;
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
            double probability = (1.0 - (distance / maxDistance));

            // double probability = (distance / maxDistance)/2;


            // Use the probability to decide if the cell should be an obstacle (0) or open (1)
            if (static_cast<double>(rand()) / RAND_MAX < probability) {
                grid[index(x, y, n)] = 1;  // Blocked cell (obstacle)
            } else {
                grid[index(x, y, n)] = 0;  // Open cell
            }
        }
    }


    for(int i=0; i<n; ++i) // clear top row
        grid[i] = 0;

    for(int i = 0; i<n; ++i) // clear right column
        grid[n * i + n - 1] = 0;
    

    grid[0] = 0;
    grid[(n - 1) * n + (n - 1)] = 0;

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

// int main(int argc, char** argv)
// {
//      // Grid dimensions and initialization code (same as before)
//     int width = 1001;  // Adjusted for demonstration
//     int height = 1001;
//     float obstacleRate = 0.2; // Default obstacle rate (percentage)
//     std::string gridType = "";
//     std::string output_path = "";

//     if (argc == 5)
//     {
//         height = atoi(argv[1]);
//         width = atoi(argv[1]);
//         obstacleRate = atoi(argv[2])/100.0;
//         gridType = argv[3];
//         output_path = argv[4];
//     }
//     else
//     {
//         std::cout<<"USAGE: ./urProgram [N] [ObstacleRate] [gridType] [outputPath]\n";
//         return EXIT_FAILURE;
//     }

//     int gridSize = width * height;

//     // Start and goal nodes
//     int startNodeId = 0;                 // Top-left corner
//     int goalNodeId = width * height - 1; // Bottom-right corner

//     // Allocate and initialize grid on host
//     int *h_grid = (int *)malloc(gridSize * sizeof(int));
//     if (h_grid == NULL) {
//         fprintf(stderr, "Failed to allocate host memory for h_grid\n");
//         exit(EXIT_FAILURE);
//     }

//     // Apply obstacles
//     if(gridType == "random")
//         applyRandomObstacles(h_grid, width, height, obstacleRate);
//     else if(gridType == "maze")
//         createMaze(h_grid, height);
//     else if(gridType == "blockCenter")
//         createConcentratedObstacles(h_grid, height);
//     else if(gridType == "zigzag")
//         createZigzagPattern(h_grid, width, height);
//     else if (gridType == "rectangle")
//         applyRandomRectangleObstacles(h_grid, width, height, obstacleRate);
//     else
//         applyRandomObstacles(h_grid, width, height, obstacleRate);
    
//     // save the grid to PATH
//     saveCompressedGridToFile(h_grid, width, height, output_path);
// }