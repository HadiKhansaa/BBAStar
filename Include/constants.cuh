#pragma once

#define MAX_NEIGHBORS 8 // 8-directional movement

#define BUCKET_F_RANGE 3000
#define MAX_BINS 3000         // Maximum number of bins (adjust as needed)
#define MAX_BIN_SIZE 3000     // Maximum number of nodes per bin (adjust as needed)
#define SCALE_FACTOR 1000     // Scale factor for cost
#define DIAGONAL_COST 1414      // Cost for diagonal movement (sqrt(2) * SCALE_FACTOR)

// #define FRONTIER_SIZE 512
#define MAX_PATH_LENGTH 10000000

#define PASSABLE 0
#define OBSTACLE 1

// #define SH_MAX_RANGE 15
// #define MAX_SHARED_BIN_SIZE 100
// #define TILE_WIDTH  16
// #define TILE_HEIGHT 16

// Global variables for inter-block communication
// extern __device__ bool d_done_forward = false;
// extern __device__ bool d_done_backward = false;

// extern __device__ unsigned int globalBestCost = INT_MAX;
// extern __device__ BiNode globalBestNode = {-1, INT_MAX, 0, INT_MAX, INT_MAX, 0, INT_MAX, -1, -1 };

// // global variables for the active bucket range
// extern __device__ int global_forward_bucketRangeStart = -1;
// extern __device__ int global_forward_bucketRangeEnd = -1;
// extern __device__ int global_forward_totalElementsInRange = 0;

// extern __device__ int global_backward_bucketRangeStart = -1;
// extern __device__ int global_backward_bucketRangeEnd = -1;
// extern __device__ int global_backward_totalElementsInRange = 0;


// ANSI escape codes for color
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define RED "\033[31m"
#define BLUE "\033[34m"
#define RESET "\033[0m"
#define PURPLE "\033[35m"
