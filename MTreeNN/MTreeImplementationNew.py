import pandas as pd
import numpy as np
import mtree
import random
import utils
import time
import matplotlib.pyplot as plt


# 10 dimensions hardcoded
def calcEuclideanDistance(x, y):
    return ((x[0] - y[0])**2 + 
            (x[1] - y[1])**2 + 
            (x[2] - y[2])**2 + 
            (x[3] - y[3])**2 + 
            (x[4] - y[4])**2 + 
            (x[5] - y[5])**2 + 
            (x[6] - y[6])**2 + 
            (x[7] - y[7])**2 + 
            (x[8] - y[8])**2 + 
            (x[9] - y[9])**2
            )**(0.5)

"""
Main class for MTree multidimensional data structure indexing on our travel reviews dataset for nearest neighbor cost/complexity exploration
With the open source package (see MTree module) from https://github.com/tburette/mtree, we use the BallTree data structure to explore nearest neighbor runtime
"""
class MTreeExplore:
    #numDimensions between 1 and 10 for our travelreview dataset
    def readDataset(self, fileName, numDimensions):
        df = pd.read_csv(fileName)
        data = df.iloc[:,1:numDimensions+1].to_numpy()
        self.data = data
        self.numDimensions = numDimensions

    def createMTree(self, max_node_size):
        m_tree = mtree.MTree(calcEuclideanDistance, max_node_size=max_node_size)
        m_tree.add_all(self.data.tolist()) 
        return m_tree

    def nnSearch(self, queriesArray, m_tree, k):
        indList = []

        start = time.time()

        for i in range(50000):
            if k == 1:
                indList.append(list(m_tree.search(list(queriesArray[i])))[0]) 
            else:
                indList.append(list(m_tree.search(list(queriesArray[i])))) 

        end = time.time()

        timeTaken = end - start

        return indList, timeTaken

# Test procedure to output k=1 nearest neighbor for queries1.txt (50k queries) on bt tree of leaf size 10 built from the dataset
# save to "MTreeNN_Indexes"
# Procedure cannot work as a check since the open source code only returns the value and not the index. Difficult to verify.
def checkNearestNeighbors(fileName):
    mClass = MTreeExplore()
    mClass.readDataset("tripadvisor_review.csv", 10) #10 dimensions

    m_tree = mClass.createMTree(10) #Initialize kd tree with the dataset, max_node_size of 10

    queriesArray = utils.importQueries("queries1.txt")

    indList, timeTaken = mClass.nnSearch(queriesArray, m_tree, 1)

    print("For M tree of max_node_size 10 built from the dataset (10 dimensions), K=1 nearest neighbors for queries1.txt 50k queries are outputted to " + fileName)
    
    np.savetxt(fileName, indList, fmt='%d')

    print("\n")

# Procedure to iterate through number of max_node_size for M Tree and plot how number affect nn search time for 50k queries (queries1.txt)
def numLeaves(plotName):
    mClass = MTreeExplore()
    mClass.readDataset("tripadvisor_review.csv", 10) #10 dimensions

    maxNodeSizeList = [32, 64]

    timeTakenList = []
    for x in maxNodeSizeList:
        m_tree = mClass.createMTree(x) #Initialize M tree with the dataset, max node size varying

        queriesArray = utils.importQueries("queries1.txt")

        indList, timeTaken = mClass.nnSearch(queriesArray, m_tree, 1)

        print("For 50k queries and 10 dimensions and M tree with max node size of " + str(x) + ", the time taken for nearest neighbor search (k=1) was ", timeTaken)
        timeTakenList.append(timeTaken)
    
    plt.figure(figsize=(10,5))
    plt.title('MTree: Max Node Size vs. NN (k=1) search time taken (seconds) for 50k queries (10 dimensions)')
    plt.xlabel('Max Node Size')
    plt.ylabel('NN Search Time (seconds)')
    plt.plot(maxNodeSizeList, timeTakenList)

    plt.savefig(plotName)

    print("\n")


def main():
    # Generate and save queries. 1 time.
    # queriesArray = utils.generateQueries(50000)
    # utils.saveQueries("queries1.txt", queriesArray)

    #checkNearestNeighbors("MTree_NN_indexes.txt")
    numLeaves("images/numLeafsVsSearchTime.png")

if __name__=="__main__":
    main()


