#pragma once
#include <iostream>
#include <math.h>
#include <vector>
#include <fstream>
#include <cstdlib>

void saveCompressedGridToFile(const int *grid, int width, int height, const std::string &filename);

bool loadCompressedGridFromFile(int *&grid, int &width, int &height, const std::string &filename);

void divide(int* grid, int startX, int startY, int width, int height, bool horizontal, int n);

void createMaze(int* grid, int n);

void printMaze(int* grid, int n);

void createZigzagPattern(int *grid, int width, int height);

void applyRandomRectangleObstacles(int *grid, int width, int height, float obstacleRate);

void applyRandomObstacles(int *grid, int width, int height, float obstacleRate);

void createConcentratedObstacles(int *grid, int height);

void generatePPMImage(const int *grid, int width, int height, const int *path, int pathLength, const std::string &filename);