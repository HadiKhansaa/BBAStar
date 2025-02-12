#include <iostream>
#include <nvgraph.h>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <cstdlib>

#define CHECK_NVGRAPH(call) { \
    nvgraphStatus_t status = call; \
    if (status != NVGRAPH_STATUS_SUCCESS) { \
        std::cerr << "nvGRAPH error: " << status << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

void generate_random_graph(int num_vertices, int num_edges, std::vector<int>& h_edge_indices, std::vector<float>& h_edge_weights) {
    h_edge_indices.resize(num_edges * 2);  // 2 arrays for source and destination
    h_edge_weights.resize(num_edges);

    for (int i = 0; i < num_edges; i++) {
        int src = rand() % num_vertices;
        int dest = rand() % num_vertices;
        h_edge_indices[i] = src;
        h_edge_indices[i + num_edges] = dest;
        h_edge_weights[i] = static_cast<float>(rand()) / RAND_MAX;  // Random weight
    }
}

int main() {
    // Initialize nvGRAPH
    nvgraphHandle_t handle;
    CHECK_NVGRAPH(nvgraphCreate(&handle));

    // Define the graph size
    int num_vertices = 10000; // Adjust as needed
    int num_edges = 20000;    // Adjust as needed
    int source = 0;

    // Prepare edge data
    std::vector<int> h_edge_indices;
    std::vector<float> h_edge_weights;
    generate_random_graph(num_vertices, num_edges, h_edge_indices, h_edge_weights);

    // Create nvGRAPH graph structure
    nvgraphGraphDescr_t graph;
    CHECK_NVGRAPH(nvgraphCreateGraphDescr(handle, &graph));

    // Define the graph structure
    nvgraphCSR_t csr;
    csr.nvertices = num_vertices;
    csr.nedges = num_edges;
    csr.deg = new int[num_vertices + 1]();

    // Fill the CSR format
    for (int i = 0; i < num_edges; i++) {
        csr.src_indices[i] = h_edge_indices[i];
        csr.dest_indices[i] = h_edge_indices[i + num_edges];
        csr.weights[i] = h_edge_weights[i];
        csr.deg[csr.src_indices[i] + 1]++;
    }

    // Accumulate degree counts to create the CSR structure
    for (int i = 1; i <= num_vertices; i++) {
        csr.deg[i] += csr.deg[i - 1];
    }

    // Allocate memory for CSR arrays
    csr.src_indices = new int[num_edges];
    csr.dest_indices = new int[num_edges];
    csr.weights = new float[num_edges];

    // Fill src_indices and dest_indices in CSR format
    for (int i = 0; i < num_edges; i++) {
        csr.src_indices[csr.deg[h_edge_indices[i]]++] = h_edge_indices[i]; 
        csr.dest_indices[csr.deg[h_edge_indices[i] + num_vertices]] = h_edge_indices[i + num_edges];
    }

    // Reset the degree array for the next use
    std::fill(csr.deg, csr.deg + num_vertices + 1, 0);
    for (int i = 0; i < num_edges; i++) {
        csr.deg[h_edge_indices[i] + 1]++;
    }
    for (int i = 1; i <= num_vertices; i++) {
        csr.deg[i] += csr.deg[i - 1];
    }

    // Allocate device memory for the graph
    CHECK_NVGRAPH(nvgraphAllocateGraph(handle, graph, &csr));

    // Allocate memory for SSSP result
    float *d_distances;
    cudaMalloc((void**)&d_distances, num_vertices * sizeof(float));
    cudaMemset(d_distances, 0, num_vertices * sizeof(float));

    // Timing the SSSP algorithm
    int num_runs = 10; // Number of runs for averaging
    double total_time = 0.0;

    for (int run = 0; run < num_runs; run++) {
        auto start = std::chrono::high_resolution_clock::now();

        // Run the SSSP algorithm
        CHECK_NVGRAPH(nvgraphSssp(handle, graph, source, d_distances, NULL, NULL));

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        total_time += elapsed.count();
        std::cout << "Run " << run + 1 << " elapsed time: " << elapsed.count() << " seconds.\n";
    }

    // Output average time
    std::cout << "Average elapsed time: " << (total_time / num_runs) << " seconds.\n";

    // Clean up
    cudaFree(d_distances);
    delete[] csr.deg;
    delete[] csr.src_indices;
    delete[] csr.dest_indices;
    delete[] csr.weights;
    CHECK_NVGRAPH(nvgraphDestroyGraphDescr(handle, graph));
    CHECK_NVGRAPH(nvgraphDestroy(handle));

    return 0;
}

