# -*- coding: utf-8 -*-
# ----------------------------------------------------
# Copyright (c) 2017, Wray Zheng. All Rights Reserved.
# Distributed under the BSD License.
# ----------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# DEBUG = True

######################################################
# 调试输出函数
# 由全局变量 DEBUG 控制输出
######################################################
# def debug(*args, **kwargs):
#     global DEBUG
#     if DEBUG:
#         print(*args, **kwargs)


######################################################
# 第 k 个模型的高斯分布密度函数
# 每 i 行表示第 i 个样本在各模型中的出现概率
# 返回一维列表
######################################################
def phi(Y, mu_k, cov_k):
    norm = multivariate_normal(mean=mu_k, cov=cov_k)
    return norm.pdf(Y)


######################################################
# E 步：计算每个模型对样本的响应度
# Y 为样本矩阵，每个样本一行，只有一个特征时为列向量
# mu 为均值多维数组，每行表示一个样本各个特征的均值
# cov 为协方差矩阵的数组，alpha 为模型响应度数组
######################################################
def getExpectation(Y, mu, cov, alpha):
    # 样本数
    N = Y.shape[0]
    # 模型数
    K = alpha.shape[0]

    # 为避免使用单个高斯模型或样本，导致返回结果的类型不一致
    # 因此要求样本数和模型个数必须大于1
    assert N > 1, "There must be more than one sample!"
    assert K > 1, "There must be more than one gaussian model!"

    # 响应度矩阵，行对应样本，列对应响应度
    gamma = np.mat(np.zeros((N, K)))

    # 计算各模型中所有样本出现的概率，行对应样本，列对应模型
    prob = np.zeros((N, K))
    for k in range(K):
        prob[:, k] = phi(Y, mu[k], cov[k])
    prob = np.mat(prob)

    # 计算每个模型对每个样本的响应度
    for k in range(K):
        gamma[:, k] = alpha[k] * prob[:, k]
    for i in range(N):
        gamma[i, :] /= np.sum(gamma[i, :])
    return gamma


######################################################
# M 步：迭代模型参数
# Y 为样本矩阵，gamma 为响应度矩阵
######################################################
def maximize(Y, gamma):
    # 样本数和特征数
    N, D = Y.shape
    # 模型数
    K = gamma.shape[1]

    #初始化参数值
    mu = np.zeros((K, D))
    cov = []
    alpha = np.zeros(K)

    # 更新每个模型的参数
    for k in range(K):
        # 第 k 个模型对所有样本的响应度之和
        Nk = np.sum(gamma[:, k])
        # 更新 mu
        # 对每个特征求均值
        mu[k, :] = np.sum(np.multiply(Y, gamma[:, k]), axis=0) / Nk
        # 更新 cov
        cov_k = (Y - mu[k]).T * np.multiply((Y - mu[k]), gamma[:, k]) / Nk
        cov.append(cov_k)
        # 更新 alpha
        alpha[k] = Nk / N
    cov = np.array(cov)
    return mu, cov, alpha


######################################################
# 数据预处理
# 将所有数据都缩放到 0 和 1 之间
######################################################
def scale_data(Y):
    # 对每一维特征分别进行缩放
    for i in range(Y.shape[1]):
        max_ = Y[:, i].max()
        min_ = Y[:, i].min()
        Y[:, i] = (Y[:, i] - min_) / (max_ - min_)
    print("Data scaled.")
    return Y


######################################################
# 初始化模型参数
# shape 是表示样本规模的二元组，(样本数, 特征数)
# K 表示模型个数
######################################################
def init_params(shape, K):
    N, D = shape
    mu = np.random.rand(K, D)
    cov = np.array([np.eye(D)] * K)
    alpha = np.array([1.0 / K] * K)
    # print("Parameters initialized.")
    # print("mu:", mu)
    # print("cov:", cov)
    # print("alpha:", alpha)
    return mu, cov, alpha


######################################################
# 高斯混合模型 EM 算法
# 给定样本矩阵 Y，计算模型参数
# K 为模型个数
# times 为迭代次数
######################################################
def GMM_EM(Y, K, times):
    Y = scale_data(Y)
    # print('1',Y.shape)
    mu, cov, alpha = init_params(Y.shape, K)
    # print('data shape:',np.shape(Y))
    # print('corresponding mu,cov,alpha shape',np.shape(mu),np.shape(cov),np.shape(alpha))
    for i in range(times):
        gamma = getExpectation(Y, mu, cov, alpha)
        mu, cov, alpha = maximize(Y, gamma)
    # print("{sep} Result {sep}".format(sep="-" * 20))
    # print("mu:", mu)
    # print("cov:", cov)
    # print("alpha:", alpha)
    return mu, cov, alpha

def gmm_cluster_possibility(input_data,K):
    mu, cov, alpha = GMM_EM(input_data, K, 100)
    N = input_data.shape[0]
    possibility_label = np.ones(N)
    gamma = getExpectation(input_data, mu, cov, alpha)
    category = gamma.argmax(axis=1).flatten().tolist()[0]
    for i in range(N):
        possibility_label[i] = alpha[category[i]]
    return category,possibility_label



def gmm_cluster_distribution(input_data,K):
# alpha is the possibility of data in category 
# data density can be calculated from the variance matrix  how?????
# i think the data density is from the alpha
    category,possibility_label = gmm_cluster_possibility(input_data,K)
    N = input_data.shape[0]
    classified_data = np.zeros([K,np.shape(input_data)[0],np.shape(input_data)[1]])
    classified_possibility = np.zeros([K,np.shape(possibility_label)[0]])
    classified_data=[]
    classified_possibility = []
    for j in range(K):
        classified_data.append(np.array([input_data[i] for i in range(N) if category[i] == j]))
        classified_possibility.append(np.array([possibility_label[i] for i in range(N) if category[i] == j]))
    return classified_data,classified_possibility