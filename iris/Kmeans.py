import numpy as np
import time

from sklearn import metrics
from sklearn import datasets
from sklearn import cluster

# Purity:計算
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

# 讀檔
iris = datasets.load_iris()
X = iris.data

# KMeans 演算法
start = time.time();
kmeans_fit = cluster.KMeans(n_clusters = 3).fit(X)

# 分群結果cluster_labels
cluster_labels = kmeans_fit.labels_
end = time.time();

# KMeans花費時間
print("執行時間：%f 豪秒" % ((end - start)/0.001))

# iris確切品種
iris_y = iris.target

print("Purity:",purity_score(iris_y,cluster_labels))