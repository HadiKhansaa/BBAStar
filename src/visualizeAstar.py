import numpy as np
import matplotlib.pyplot as plt
import math


def read_array_from_file(filename):
    """
    Read an array from a text file.
    The file should contain a list of numbers separated by spaces.
    """
    with open(filename, 'r') as f:
        array = f.read().split("\n")
    return np.array(array, dtype=int)

def majority_color(block):
    """
    Given a block (an array of shape (m, n, 3)), return the majority color.
    Colors are compared exactly (our drawing colors are predefined).
    """
    # Reshape block into a list of colors (each of length 3)
    block_reshaped = block.reshape(-1, 3)
    # Find unique colors and their counts
    colors, counts = np.unique(block_reshaped, axis=0, return_counts=True)
    majority = colors[np.argmax(counts)]
    return majority

def visualizeAStar(grid, width, height, path, startId, endId, nodesExplored, max_size=10000):
    """
    Visualizes the A* algorithm over a grid.
    
    Parameters:
      grid          : 1D list or array of 0's and 1's (1 indicates an obstacle).
      width, height : Dimensions of the grid.
      path          : List of node ids (based on nodeId = width*row + col) that form the final path.
      startId       : Node id for the start position.
      endId         : Node id for the end position.
      nodesExplored : List of node ids that were explored during the search.
      max_size      : Maximum size for the visualization. If the grid is larger than this,
                      the image is compressed by grouping blocks of cells.
    """
    # Create a color matrix with shape (height, width, 3); start with all white.
    color_matrix = np.ones((height, width, 3))
    
    # Define RGB colors (using values between 0 and 1)
    obstacle_color = np.array([0, 0, 0])    # Black
    path_color     = np.array([0, 1, 0])      # Green
    explored_color = np.array([1, 0.65, 0])   # Orange
    start_color    = np.array([0, 0, 1])      # Blue
    end_color      = np.array([1, 0, 0])      # Red
    
    # Mark obstacles in the grid
    for i, cell in enumerate(grid):
        row = i // width
        col = i % width
        if cell:
            color_matrix[row, col] = obstacle_color
    
    # Mark explored nodes (only if the cell is still white, i.e. not an obstacle)
    for node in nodesExplored:
        row = node // width
        col = node % width
        if np.array_equal(color_matrix[row, col], np.array([1, 1, 1])):
            color_matrix[row, col] = explored_color

    # Mark the final path (this will override explored colors)
    for node in path:
        row = node // width
        col = node % width
        color_matrix[row, col] = path_color

    # Mark start and end nodes (override any other color)
    row, col = divmod(startId, width)
    color_matrix[row, col] = start_color
    row, col = divmod(endId, width)
    color_matrix[row, col] = end_color

    # Determine a compression factor so that the largest dimension is at most max_size pixels.
    compress_factor = max(1, int(max(width, height) / max_size))
    if compress_factor > 1:
        new_height = height // compress_factor
        new_width  = width // compress_factor
        compressed_matrix = np.zeros((new_height, new_width, 3))
        for i in range(new_height):
            for j in range(new_width):
                block = color_matrix[i*compress_factor:(i+1)*compress_factor,
                                     j*compress_factor:(j+1)*compress_factor]
                compressed_matrix[i, j] = majority_color(block)
        color_matrix = compressed_matrix

    # Display the visualization
    plt.figure(figsize=(8,8))
    plt.imshow(color_matrix, interpolation='nearest')
    plt.xticks([]), plt.yticks([])
    plt.title("A* Algorithm Visualization")
    plt.show()

# ---------------------------
# Example usage:
if __name__ == "__main__":
    # Grid dimensions
    width, height = 10000, 1000
    grid = [0] * (width * height)
    
    # Randomly add obstacles
    import random
    random.seed(42)
    obstacles = random.sample(range(width * height), 60)
    for idx in obstacles:
        grid[idx] = 1
    
    # Create a sample path (a diagonal line from top-left to bottom-right)
    path = [i * width + i for i in range(min(width, height))]
    
    # Simulate explored nodes: for instance, a small band around the diagonal
    nodesExplored = []
    for i in range(height):
        for j in range(max(0, i-1), min(width, i+2)):
            node = i * width + j
            if node not in path:
                nodesExplored.append(node)
    
    startId = 0                   # Top-left cell
    endId = width * height - 1      # Bottom-right cell
    
    visualizeAStar(grid, width, height, path, startId, endId, nodesExplored)
