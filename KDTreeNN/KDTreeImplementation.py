import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import random
import utils
import time

"""
Main class for KDTree multidimensional data structure indexing on our travel reviews dataset for nearest neighbor cost/complexity exploration
With the scikiit-learn package, KDTree data structure only supports nearest neighbor queries. For KDTree data structure, inserts and deletes
often causes large reformatting changes to the tree structure that is not ideal; thus for KDTree, inserts and deletes are not recommended
and if there are big changes to the dataset, then just rebuilding of the tree is recommended.

For our purpose, with the KDTree, we will explore how the number of dimensions of the dataset, and parameter values (such as leaf size and distance metric) of
the KDTree affect the time of nearest neighbor queries, and compare with the brute force approach and other multidimensional index structures explored.
"""
class KDTreeExplore:
    #numDimensions between 1 and 10 for our travelreview dataset
    def readDataset(self, fileName, numDimensions):
        df = pd.read_csv(fileName)
        data = df.iloc[:,1:numDimensions+1].to_numpy()
        self.data = data
        self.numDimensions = numDimensions

    def createKDTree(self, leaf_size, metric):
        kdt = KDTree(self.data, leaf_size=leaf_size, metric=metric)
        return kdt

    def nnSearch(self, queriesArray, kdt):
        distList = []
        indList = []

        start = time.time()
        distList, indList  = kdt.query(queriesArray, k=1, return_distance=True) 
        end = time.time()

        timeTaken = end - start

        return distList, indList, timeTaken

def main():
    kdClass = KDTreeExplore()
    kdClass.readDataset("tripadvisor_review.csv", 10)

    kd1 = kdClass.createKDTree(10, "euclidean")

    queriesArray = utils.generateQueries(50000)
    print(queriesArray[0])
    utils.saveQueries("queries1.txt", queriesArray)

    distList, indList, timeTaken = kdClass.nnSearch(queriesArray, kd1)

    print("Time taken for nearest neighbor search was ", timeTaken)

    # Next, I will implement a function to loop thorugh the number of dimensions, and different parameters of kd tree to see the effect on the time.
    # I will also plot the results in line graphs.

if __name__=="__main__":
    main()


