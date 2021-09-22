import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt
 
def hierarchy_cluster(data, method='average', threshold=5.0):
    '''层次聚类
    
    Arguments:
        data [[0, float, ...], [float, 0, ...]] -- 文档 i 和文档 j 的距离
    
    Keyword Arguments:
        method {str} -- [linkage的方式： single、complete、average、centroid、median、ward] (default: {'average'})
        threshold {float} -- 聚类簇之间的距离
    Return:
        cluster_number int -- 聚类个数
        cluster [[idx1, idx2,..], [idx3]] -- 每一类下的索引
    '''
    data = np.array(data)
 
    Z = linkage(data, method=method)
    cluster_assignments = fcluster(Z, threshold, criterion='distance')
    print (type(cluster_assignments))
    num_clusters = cluster_assignments.max()
    indices = get_cluster_indices(cluster_assignments)
 
    return num_clusters, indices
 
 
 
def get_cluster_indices(cluster_assignments):
    '''映射每一类至原数据索引
    
    Arguments:
        cluster_assignments 层次聚类后的结果
    
    Returns:
        [[idx1, idx2,..], [idx3]] -- 每一类下的索引
    '''
    n = cluster_assignments.max()
    indices = []
    for cluster_number in range(1, n + 1):
        indices.append(np.where(cluster_assignments == cluster_number)[0])
    
    return indices
 
 
if __name__ == '__main__':
    
 
    arr = [[0., 21.6, 22.6, 63.9, 65.1, 17.7, 99.2],
    [21.6, 0., 1., 42.3, 43.5, 3.9, 77.6],
    [22.6, 1., 0, 41.3, 42.5, 4.9, 76.6],
    [63.9, 42.3, 41.3, 0., 1.2, 46.2, 35.3],
    [65.1, 43.5, 42.5, 1.2, 0., 47.4, 34.1],
    [17.7, 3.9, 4.9, 46.2, 47.4, 0, 81.5],
    [99.2, 77.6, 76.6, 35.3, 34.1, 81.5, 0.]]
 
    arr = np.array(arr)
    r, c = arr.shape
    for i in range(r):
        for j in range(i, c):
            if arr[i][j] != arr[j][i]:
                arr[i][j] = arr[j][i]
    for i in range(r):
        for j in range(i, c):
            if arr[i][j] != arr[j][i]:
                print(arr[i][j], arr[j][i])
 
    num_clusters, indices = hierarchy_cluster(arr)
 
 
    print ("%d clusters" % num_clusters)
    for k, ind in enumerate(indices):
        print ("cluster", k + 1, "is", ind)
