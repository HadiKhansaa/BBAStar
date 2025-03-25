#include <iostream>
#include <fstream>
#include <string>
using namespace std;

// ANSI escape codes for color
std::string green = "\033[32m";
std::string yellow = "\033[33m";
std::string red = "\033[31m";
std::string blue = "\033[34m";
std::string reset = "\033[0m";
std::string purple = "\033[35m";

void writeArrayToFile(int* array, int arraySize, string filename) {
    // Open the file for writing
    ofstream outFile(filename);
    if (!outFile.is_open()) {
        cerr << "Error: Could not open file " << filename << " for writing." << endl;
        return;
    }

    // Write each element of the array to the file, one per line
    for (int i = 0; i < arraySize; ++i) {
        outFile << array[i] << endl;
    }

    // Close the file
    outFile.close();
}
