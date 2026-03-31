// a_star_cpu_grid_custom_pq.cpp

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <algorithm>

#define INF_FLT 1e20f  // A large float value representing infinity
#define GRID_WIDTH 5001  // Adjusted for demonstration
#define GRID_HEIGHT 5001 // Adjusted for demonstration

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

// Custom MinHeap implementation
class MinHeap {
public:
    MinHeap(int capacity) {
        heap.reserve(capacity);
        size = 0;
    }

    void push(Node* node) {
        heap.push_back(node);
        size++;
        heapifyUp(size - 1);
    }

    Node* pop() {
        if (size == 0) return nullptr;
        Node* minNode = heap[0];
        heap[0] = heap[size - 1];
        size--;
        heap.pop_back();
        heapifyDown(0);
        return minNode;
    }

    bool empty() const {
        return size == 0;
    }

private:
    std::vector<Node*> heap;
    int size;

    void heapifyUp(int index) {
        while (index > 0) {
            int parent = (index - 1) / 2;
            if (heap[parent]->f <= heap[index]->f)
                break;
            std::swap(heap[parent], heap[index]);
            index = parent;
        }
    }

    void heapifyDown(int index) {
        while (index < size) {
            int left = 2 * index + 1;
            int right = 2 * index + 2;
            int smallest = index;
            if (left < size && heap[left]->f < heap[smallest]->f)
                smallest = left;
            if (right < size && heap[right]->f < heap[smallest]->f)
                smallest = right;
            if (smallest == index)
                break;
            std::swap(heap[index], heap[smallest]);
            index = smallest;
        }
    }
};

// A* algorithm implementation
bool aStarCPU(const std::vector<int>& grid, int width, int height, int startNodeId, int goalNodeId,
              std::vector<Node>& nodes, std::vector<int>& path) {
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

    // Custom MinHeap
    MinHeap openList(width * height);

    // Initialize start node
    Node* startNode = &nodes[startNodeId];
    startNode->g = 0.0f;
    startNode->h = heuristic(startNodeId, goalNodeId, width);
    startNode->f = startNode->g + startNode->h;
    startNode->opened = true;
    openList.push(startNode);

    int nodesExpanded = 0;
    while (!openList.empty()) {
        Node* currentNode = openList.pop();
        currentNode->closed = true;
        nodesExpanded++;

        if (nodesExpanded % 100000 == 0) {
            std::cout << "Nodes expanded: " << nodesExpanded << std::endl;
        }

        if (currentNode->id == goalNodeId) {
            // Path found
            int tempId = goalNodeId;
            while (tempId != -1) {
                path.push_back(tempId);
                tempId = nodes[tempId].parent;
            }
            std::cout << "Total nodes expanded: " << nodesExpanded << std::endl;
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
                        } else {
                            // Since we don't have a decrease-key operation, we can push it again
                            openList.push(neighborNode);
                        }
                    }
                }
            }
        }
    }

    // Path not found
    std::cout << "Total nodes expanded: " << nodesExpanded << std::endl;
    return false;
}

int main() {
    // Grid dimensions
    const int width = GRID_WIDTH;
    const int height = GRID_HEIGHT;
    const int gridSize = width * height;

    // Start and goal nodes
    int startNodeId = 0;                 // Top-left corner
    int goalNodeId = width * height - 1; // Bottom-right corner

    // Allocate and initialize grid
    std::vector<int> grid(gridSize, 0);

    // Create the zigzag pattern in the grid
    createZigzagPattern(grid, width, height);

    // Allocate nodes
    std::vector<Node> nodes(gridSize);

    // Path vector
    std::vector<int> path;

    // Timing the execution
    auto startTime = std::chrono::high_resolution_clock::now();

    // Run A* algorithm
    bool found = aStarCPU(grid, width, height, startNodeId, goalNodeId, nodes, path);

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
    } else {
        std::cout << "Path not found." << std::endl;
        std::cout << "Execution time: " << elapsedSeconds.count() << " seconds" << std::endl;
    }

    return 0;
}
