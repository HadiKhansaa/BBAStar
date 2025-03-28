#include "utils.cuh"

void writeArrayToFile(int* array, int arraySize, std::string filename) {
    // Open the file for writing
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    // Write each element of the array to the file, one per line
    for (int i = 0; i < arraySize; ++i) {
        outFile << array[i] << std::endl;
    }

    // Close the file
    outFile.close();
}
