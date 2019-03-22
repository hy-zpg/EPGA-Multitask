# -*- coding: utf-8 -*-
# ----------------------------------------------------
# Copyright (c) 2017, Wray Zheng. All Rights Reserved.
# Distributed under the BSD License.
# ----------------------------------------------------

import matplotlib.pyplot as plt
from gmm import *


# 载入数据
Y = np.loadtxt("gmm.data")
matY = np.matrix(Y, copy=True)

# 模型个数，即聚类的类别个数
K = 4

# 计算 GMM 模型参数
# mu, cov, alpha = GMM_EM(matY, K, 100)
# print("mu:", mu)
# print("cov:", cov)
# print("alpha:", alpha)



category,possibility_label = gmm_cluster_possibility(matY,K)
print('category,possibility',np.shape(category),np.shape(possibility_label))
# print(category[0],possibility_label[0])

classified_data,classified_possibility = gmm_cluster_distribution(matY,K)
print('classified result:',np.shape(classified_data),np.shape(classified_possibility))
print(classified_data[0][0],classified_possibility[0][0])
# 根据 GMM 模型，对样本数据进行聚类，一个模型对应一个类别
N = Y.shape[0]
# 求当前模型参数下，各模型对样本的响应度矩阵
# gamma = getExpectation(matY, mu, cov, alpha)
# # 对每个样本，求响应度最大的模型下标，作为其类别标识
# category = gamma.argmax(axis=1).flatten().tolist()[0]
# 将每个样本放入对应类别的列表中
class1 = np.array([Y[i] for i in range(N) if category[i] == 0])
class2 = np.array([Y[i] for i in range(N) if category[i] == 1])
class3 = np.array([Y[i] for i in range(N) if category[i] == 2])
class4 = np.array([Y[i] for i in range(N) if category[i] == 3])


# 绘制聚类结果
plt.plot(class1[:, 0], class1[:, 1], 'rs', label="class1")
plt.plot(class2[:, 0], class2[:, 1], 'bo', label="class2")
plt.plot(class3[:, 0], class3[:, 1], 'go', label="class3")
plt.plot(class4[:, 0], class4[:, 1], 'gs', label="class4")
plt.legend(loc="best")
plt.title("GMM Clustering By EM Algorithm")
plt.show()
