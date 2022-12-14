import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
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

# Hierarchical Clustering 演算法
start = time.time();
hclust = cluster.AgglomerativeClustering(n_clusters = 3,affinity = 'euclidean',linkage = 'ward')

# 分群結果cluster_labels
hclust.fit(X)
cluster_labels = hclust.labels_
end = time.time();

# Hierarchical Clustering花費時間
print("執行時間：%f 豪秒" % ((end - start)/0.001))
# iris確切品種
iris_y = iris.target

print("Purity:",purity_score(iris_y,cluster_labels))

# Hierarchical Clustering的階層樹(Dendrogram)
dis = sch.linkage(X,metric = 'euclidean',method = 'ward')
plt.figure(figsize=(18,18))
sch.dendrogram(dis)
plt.xlabel('Index')
plt.ylabel('Distance')
plt.suptitle("Dendrogram",fontsize=18) 
#plt.savefig("Dendrogram.pdf")
plt.show()

