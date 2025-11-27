#pragma once

#define MAX_NEIGHBORS 8 // 8-directional movement

#define BUCKET_F_RANGE 100
#define MAX_BINS 20000         // Maximum number of bins (adjust as needed) // should be adjusted dynamically
#define MAX_BIN_SIZE 20000     // Maximum number of nodes per bin (adjust as needed) // should be adjusted dynamically
#define SCALE_FACTOR 1000     // Scale factor for cost
#define DIAGONAL_COST 1414      // Cost for diagonal movement (sqrt(2) * SCALE_FACTOR)

// #define FRONTIER_SIZE 512
#define MAX_PATH_LENGTH 10000000

#define PASSABLE 0
#define OBSTACLE 1

#define TOTAL_THREADS 40000

// #define SH_MAX_RANGE 15
// #define MAX_SHARED_BIN_SIZE 100
// #define TILE_WIDTH  16
// #define TILE_HEIGHT 16

// ANSI escape codes for color
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define RED "\033[31m"
#define BLUE "\033[34m"
#define RESET "\033[0m"
#define PURPLE "\033[35m"
