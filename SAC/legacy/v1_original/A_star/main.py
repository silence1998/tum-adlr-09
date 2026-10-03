import algorithm as algo

import numpy as np

def main():
    #Customize the grid here, placing 1's where you want to add obstacles.
    grid = np.zeros((10,10))
    grid[0,4]=1
    grid[1,4]=1
    #Start position
    StartNode = (0, 0)
    #Goal position
    EndNode = (0, 9)
    print(algo.algorithm(grid, StartNode, EndNode))


if __name__ == '__main__':
    main()