#include "grid_generation.cuh"
#include "constants.cuh"
#include <algorithm>

// Include stb_image_write for PNG output
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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
        grid[i] = PASSABLE; // 0 indicates free cell
    }

    // Place random obstacles
    for (int i = 0; i < width * height; i++) {
        float randValue = static_cast<float>(rand()) / RAND_MAX;
        if (randValue < obstacleRate) {
            grid[i] = OBSTACLE; // 1 indicates an obstacle
        }
    }

    for(int i=0; i<width; ++i) // clear top row
        grid[i] = PASSABLE;

    for(int i = 0; i<height; ++i) // clear right column
        grid[width * i + width - 1] = PASSABLE;

    // Ensure the goal cell is free
    grid[0] = PASSABLE; // Starting cell
    grid[(height - 1) * width + (width - 1)] = PASSABLE; // Goal cell
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
            grid[index(x, wallY, n)] = OBSTACLE;
        }

        // Create a passage in the wall
        int passageX = startX + rand() % width;
        grid[index(passageX, wallY, n)] = PASSABLE;

        // Recursively divide the top and bottom areas
        divide(grid, startX, startY, width, wallY - startY, !horizontal, n);  // Top half
        divide(grid, startX, wallY + 1, width, startY + height - wallY - 1, !horizontal, n);  // Bottom half
    } else {
        // Create a vertical wall
        int wallX = startX + rand() % (width - 2) + 1;
        for (int y = startY; y < startY + height; ++y) {
            grid[index(wallX, y, n)] = OBSTACLE;
        }

        // Create a passage in the wall
        int passageY = startY + rand() % height;
        grid[index(wallX, passageY, n)] = PASSABLE;

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
        grid[i] = PASSABLE; // 0 indicates free cell
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
                grid[index(x, y, n)] = OBSTACLE;  // Blocked cell (obstacle)
            } else {
                grid[index(x, y, n)] = PASSABLE;  // Open cell
            }
        }
    }


    for(int i=0; i<n; ++i) // clear top row
        grid[i] = PASSABLE;

    for(int i = 0; i<n; ++i) // clear right column
        grid[n * i + n - 1] = PASSABLE;
    

    grid[0] = PASSABLE;
    grid[(n - 1) * n + (n - 1)] = PASSABLE; // Goal cell

}

// Function to place random rectangle-shaped obstacles on the grid
void applyRandomRectangleObstacles(int *grid, int width, int height, float rectangleRate) {
    // Seed the random number generator (if needed)
    srand(time(NULL));

    // Initialize all cells to free
    for (int i = 0; i < width * height; i++) {
        grid[i] = PASSABLE; // 0 indicates free cell
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
                grid[i * width + j] = OBSTACLE; // 1 indicates an obstacle
            }
        }
    }

    for(int i=0; i<width; ++i) // clear top row
        grid[i] = PASSABLE;

    for(int i = 0; i<height; ++i) // clear right column
        grid[width * i + width - 1] = PASSABLE;

    // Ensure the starting and goal cells are free
    grid[0] = 0; // Starting cell
    grid[(height - 1) * width + (width - 1)] = PASSABLE; // Goal cell
}

// Function to create a zigzag pattern in the grid
void createZigzagPattern(int *grid, int width, int height) {
    // Clear the grid
    for (int i = 0; i < width * height; i++) {
        grid[i] = PASSABLE;  // 0 indicates free cell
    }

    // Create the zigzag pattern
    int row = 0;
    while (row < height) {
        if (row % 4 == 1) {
            // Block cells except the last column
            for (int col = 0; col < width - 1; col++) {
                grid[row * width + col] = OBSTACLE;  // 1 indicates blocked cell
            }
        } else if (row % 4 == 3) {
            // Block cells except the first column
            for (int col = 1; col < width; col++) {
                grid[row * width + col] = OBSTACLE;  // 1 indicates blocked cell
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

// visualizeAStarPathOnGrid with:
// - thick red path (output-space dilation)
// - start/end as big blue dots (8x8 by default)
void visualizeAStarPathOnGrid(const int *grid, int width, int height,
                              const int *path, int pathLength,
                              const int *expandedNodes, int expandedLength,
                              const std::string &filename) {
    // Flags for original-grid cells
    std::vector<bool> isInPath(width * height, false);
    std::vector<bool> isExpanded(width * height, false);

    for (int i = 0; i < pathLength; ++i) {
        int nodeId = path[i];
        if (0 <= nodeId && nodeId < width * height) isInPath[nodeId] = true;
    }
    for (int i = 0; i < expandedLength; ++i) {
        int nodeId = expandedNodes[i];
        if (0 <= nodeId && nodeId < width * height) isExpanded[nodeId] = true;
    }

    // Infer start/end from the path (if present)
    int startId = (pathLength > 0 ? path[0] : -1);
    int goalId  = (pathLength > 0 ? path[pathLength - 1] : -1);

    // Determine output resolution (max 1000×1000)
    int outWidth  = (width  > 1000 ? 1000 : width);
    int outHeight = (height > 1000 ? 1000 : height);

    // Map original grid -> output pixels
    int blockW = (width  > 1000 ? width  / outWidth  : 1);
    int blockH = (height > 1000 ? height / outHeight : 1);

    // RGB buffer, default white
    std::vector<unsigned char> image(outWidth * outHeight * 3, 255);

    auto setPixel = [&](int x, int y, unsigned char r, unsigned char g, unsigned char b) {
        if (x < 0 || x >= outWidth || y < 0 || y >= outHeight) return;
        int idx = (y * outWidth + x) * 3;
        image[idx]     = r;
        image[idx + 1] = g;
        image[idx + 2] = b;
    };

    // auto getPixel = [&](int x, int y, unsigned char &r, unsigned char &g, unsigned char &b) {
    //     int idx = (y * outWidth + x) * 3;
    //     r = image[idx];
    //     g = image[idx + 1];
    //     b = image[idx + 2];
    // };

    auto toOutXY = [&](int nodeId, int &ox, int &oy) -> bool {
        if (nodeId < 0 || nodeId >= width * height) return false;
        int x = nodeId % width;
        int y = nodeId / width;
        ox = x / blockW;
        oy = y / blockH;
        // clamp just in case integer division hits boundary
        if (ox < 0) ox = 0; if (ox >= outWidth)  ox = outWidth - 1;
        if (oy < 0) oy = 0; if (oy >= outHeight) oy = outHeight - 1;
        return true;
    };

    // Base render (same logic as yours)
    for (int by = 0; by < outHeight; ++by) {
        for (int bx = 0; bx < outWidth; ++bx) {
            bool redFound    = false;
            bool orangeFound = false;
            long sumR = 0, sumG = 0, sumB = 0;
            int count = 0;

            int y0 = by * blockH;
            int x0 = bx * blockW;
            int y1 = std::min(y0 + blockH, height);
            int x1 = std::min(x0 + blockW, width);

            for (int y = y0; y < y1; ++y) {
                for (int x = x0; x < x1; ++x) {
                    int idx = y * width + x;
                    unsigned char r, g, b;

                    if (isInPath[idx]) { r = 255; g =   0; b =   0; redFound = true; }
                    else if (isExpanded[idx]) { r = 255; g = 165; b =   0; orangeFound = false; }
                    else if (grid[idx] == 1) { r = g = b = 0; }
                    else { r = g = b = 255; }

                    sumR += r; sumG += g; sumB += b; ++count;
                }
            }

            unsigned char outR, outG, outB;
            if (redFound) { outR = 255; outG = 0;   outB = 0; }
            else if (orangeFound) { outR = 255; outG = 165; outB = 0; }
            else {
                outR = static_cast<unsigned char>(sumR / count);
                outG = static_cast<unsigned char>(sumG / count);
                outB = static_cast<unsigned char>(sumB / count);
            }

            setPixel(bx, by, outR, outG, outB);
        }
    }

    // --- Make the red path thicker (output-space dilation) ---
    // radius=1 => ~3px thick, radius=2 => ~5px thick
    const int pathRadius = 1; // change to 2 if you want thicker than ~3px
    std::vector<unsigned char> redMask(outWidth * outHeight, 0);

    // Build an output mask from original path nodes
    for (int i = 0; i < pathLength; ++i) {
        int nodeId = path[i];
        int ox, oy;
        if (toOutXY(nodeId, ox, oy)) redMask[oy * outWidth + ox] = 1;
    }

    // Dilate
    std::vector<unsigned char> dilated(outWidth * outHeight, 0);
    for (int y = 0; y < outHeight; ++y) {
        for (int x = 0; x < outWidth; ++x) {
            if (!redMask[y * outWidth + x]) continue;
            for (int dy = -pathRadius; dy <= pathRadius; ++dy) {
                for (int dx = -pathRadius; dx <= pathRadius; ++dx) {
                    int nx = x + dx, ny = y + dy;
                    if (nx < 0 || nx >= outWidth || ny < 0 || ny >= outHeight) continue;
                    dilated[ny * outWidth + nx] = 1;
                }
            }
        }
    }

    // Paint thick path in red (overrides whatever is underneath)
    for (int y = 0; y < outHeight; ++y) {
        for (int x = 0; x < outWidth; ++x) {
            if (dilated[y * outWidth + x]) setPixel(x, y, 255, 0, 0);
        }
    }

    // --- Draw start/end as big blue "dots" (8x8 squares) ---
    const int dotSize = 32;      // 8x8
    const int halfA   = dotSize / 2;     // 4
    const int halfB   = dotSize - halfA; // 4 (so even sizes stay 8 total)

    auto drawBlueDot = [&](int nodeId) {
        int cx, cy;
        if (!toOutXY(nodeId, cx, cy)) return;
        for (int y = cy - halfA; y < cy + halfB; ++y) {
            for (int x = cx - halfA; x < cx + halfB; ++x) {
                setPixel(x, y, 0, 0, 255);
            }
        }
    };

    drawBlueDot(startId);
    drawBlueDot(goalId);

    // Write PNG
    if (stbi_write_png(filename.c_str(), outWidth, outHeight, 3, image.data(), outWidth * 3)) {
        std::cout << "Image saved to " << filename << "\n";
    } else {
        std::cerr << "Failed to save image to " << filename << "\n";
    }
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