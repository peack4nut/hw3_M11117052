import numpy as np
import time
import pandas as pd

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
x = pd.read_csv('homework.csv')
x_split = x.drop(columns=['Gender','NObeyesdad'])
x_target = x.drop(columns=['Gender','Age','Height','Weight','family_history_with_overweight','FAVC','FCVC','NCP','CAEC','SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS'])
array_split = np.array(x_split)
array_target = np.array(x_target)

# KMeans 演算法
start = time.time();
kmeans_fit = cluster.KMeans(n_clusters = 6).fit(x_split)

# 分群結果cluster_labels
cluster_labels = kmeans_fit.labels_
end = time.time();

# KMeans花費時間
print("執行時間：%f 豪秒" % ((end - start)/0.001))
print("Purity:",purity_score(x_target,cluster_labels))