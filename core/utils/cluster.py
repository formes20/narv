import numpy as np
import matplotlib.pyplot as plt


# 两点距离
def distEclud(vecA, vecB):
    # print("vecA",np.array(vecA)[0])
    # print("vecB",np.array(vecB)[0])
    return np.sqrt(sum(np.power(np.array(vecA)[0] - np.array(vecB)[0], 2)))


# 集合中心
def means(arr):
    return np.array([np.mean([e[0] for e in arr]), np.mean([e[1] for e in arr])])


# 构建聚簇中心，取k个随机质心
def randCent(dataSet, k):
    n = dataSet.shape[1]
    centroids = np.mat(np.zeros((k, n)))  # 每个质心有n个坐标值，总共要k个质心
    for j in range(n):
        minJ = min(dataSet[:, j])
        maxJ = max(dataSet[:, j])
        rangeJ = float(maxJ - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids


# k-means 聚类算法
def kMeans(dataSet, k, distMeans=distEclud, createCent=randCent):
    m = dataSet.shape[0]
    clusterAssment = np.mat(np.zeros((m, 2)))  # 用于存放该样本属于哪类及质心距离
    # clusterAssment第一列存放该数据所属的中心点，第二列是该数据到中心点的距离
    centroids = createCent(dataSet, k)
    clusterChanged = True  # 用来判断聚类是否已经收敛
    count = 0
    while clusterChanged:
        count += 1
        clusterChanged = False
        for i in range(m):  # 把每一个数据点划分到离它最近的中心点
            minDist = np.inf;
            minIndex = -1
            for j in range(k):
                distJI = distMeans(centroids[j, :], dataSet[i])
                if distJI < minDist:
                    minDist = distJI;
                    minIndex = j  # 如果第i个数据点到第j个中心点更近，则将i归属为j
            if clusterAssment[i, 0] != minIndex: clusterChanged = True;  # 如果分配发生变化，则需要继续迭代
            clusterAssment[i, :] = minIndex, minDist ** 2  # 并将第i个数据点的分配情况存入字典
        # print(centroids)
        for cent in range(k):  # 重新计算中心点
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            if (len(ptsInClust) == 0):
                centroids[cent, :] = np.mean(centroids, axis=0)
            else:
                centroids[cent, :] = np.mean(ptsInClust, axis=0)  # 算出这些数据的中心点
    # print("count",count)
    return centroids, clusterAssment

# datMat = np.random.rand(20,100)
# myCentroids,clustAssing = kMeans(datMat,10)
# print("myCentroids",myCentroids)
# print("clustAssing",clustAssing)
