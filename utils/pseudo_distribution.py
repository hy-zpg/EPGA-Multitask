import time
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.decomposition import PCA
from sklearn.utils import check_array, check_consistent_length, gen_batches
from scipy.stats import multivariate_normal
import scipy.optimize
from sklearn.utils import check_array, check_consistent_length, gen_batches


###GMM clustering for data distribution
def phi(Y, mu_k, cov_k):
    norm = multivariate_normal(mean=mu_k, cov=cov_k)
    return norm.pdf(Y)
def getExpectation(Y, mu, cov, alpha):
    N = Y.shape[0]
    K = alpha.shape[0]
    assert N > 1, "There must be more than one sample!"
    assert K > 1, "There must be more than one gaussian model!"
    gamma = np.mat(np.zeros((N, K)))
    prob = np.zeros((N, K))
    for k in range(K):
        prob[:, k] = phi(Y, mu[k], cov[k])
    prob = np.mat(prob)
    for k in range(K):
        gamma[:, k] = alpha[k] * prob[:, k]
    for i in range(N):
        gamma[i, :] /= np.sum(gamma[i, :])
    return gamma
def maximize(Y, gamma):
    N, D = Y.shape
    K = gamma.shape[1]
    mu = np.zeros((K, D))
    cov = []
    alpha = np.zeros(K)
    for k in range(K):
        Nk = np.sum(gamma[:, k])
        mu[k, :] = np.sum(np.multiply(Y, gamma[:, k]), axis=0) / Nk
        cov_k = (Y - mu[k]).T * np.multiply((Y - mu[k]), gamma[:, k]) / Nk
        cov.append(cov_k)
        alpha[k] = Nk / N
    cov = np.array(cov)
    return mu, cov, alpha
def scale_data(Y):
    for i in range(Y.shape[1]):
        max_ = Y[:, i].max()
        min_ = Y[:, i].min()
        Y[:, i] = (Y[:, i] - min_) / (max_ - min_)
    return Y
def init_params(shape, K):
    N, D = shape
    mu = np.random.rand(K, D)
    cov = np.array([np.eye(D)] * K)
    alpha = np.array([1.0 / K] * K)
    return mu, cov, alpha
def GMM_EM(Y, K, times):
    Y = scale_data(Y)
    print('1',Y.shape)
    mu, cov, alpha = init_params(Y.shape, K)
    # print('data shape:',np.shape(Y))
    # print('corresponding mu,cov,alpha shape',np.shape(mu),np.shape(cov),np.shape(alpha))
    for i in range(times):
        gamma = getExpectation(Y, mu, cov, alpha)
        mu, cov, alpha = maximize(Y, gamma)
    return mu, cov, alpha

def gmm_cluster_possibility(input_data,K):
    input_data_mat =  np.matrix(input_data, copy=True)
    print('feature shape:',np.shape(input_data_mat))
    mu, cov, alpha = GMM_EM(input_data_mat, K, 100)
    N = input_data.shape[0]
    possibility_label = np.ones(N)
    gamma = getExpectation(input_data_mat, mu, cov, alpha)
    category = gamma.argmax(axis=1).flatten().tolist()[0]
    for i in range(N):
        possibility_label[i] = alpha[category[i]]
    return category,possibility_label



def gmm_cluster_distribution(input_data,label,K):
# alpha is the possibility of data in category 
# data density can be calculated from the variance matrix  how?????
# i think the data density is from the alpha
    category,possibility_label = gmm_cluster_possibility(input_data,K)
    N = input_data.shape[0]
    classified_data = np.zeros([K,np.shape(input_data)[0],np.shape(input_data)[1]])
    classified_possibility = np.zeros([K,np.shape(possibility_label)[0]])

    classified_data=[]
    classified_possibility = []
    classified_label = []
    classified_list = []

    for j in range(K):
        # recording data index
        classified_list.append([i for i, label in enumerate(category) if label == j])
        classified_data.append(np.array([input_data[i] for i in range(N) if category[i] == j]))
        classified_possibility.append(np.array([possibility_label[i] for i in range(N) if category[i] == j]))
        classified_label.append(np.array([label[i] for i in range(N) if category[i] == j]))
    return classified_data,classified_possibility,classified_label,classified_list


###input [cluster,weights]
def positivity(f):
    return f 
def fromSrc(f, wp, i, shape):
    fr = np.reshape(f, shape)
    f_sumColi = np.sum(fr[i,:])
    return wp[i] - f_sumColi
def toTgt(f, wq, j, shape):
    fr = np.reshape(f, shape)
    f_sumRowj = np.sum(fr[:,j])
    return wq[j] - f_sumRowj
def maximiseTotalFlow(f, wp, wq): 
    return f.sum() - np.minimum(wp.sum(), wq.sum())
def flow(f, D):
    f = np.reshape(f, D.shape)
    return (f * D).sum()
def groundDistance(x1, x2, norm = 2):
    return np.linalg.norm(x1-x2, norm)
def getDistMatrix(s1, s2, norm = 2):
    numFeats1 = s1.shape[0]
    numFeats2 = s2.shape[0]
    distMatrix = np.zeros((numFeats1, numFeats2))
    for i in range(0, numFeats1):
        for j in range(0, numFeats2):
            distMatrix[i,j] = groundDistance(s1[i], s2[j], norm)
    return distMatrix
def getFlowMatrix(P, Q, D):
    numFeats1 = P[0].shape[0]
    numFeats2 = Q[0].shape[0]
    shape = (numFeats1, numFeats2)
    cons1 = [{'type':'ineq', 'fun' : positivity},
             {'type':'eq', 'fun' : maximiseTotalFlow, 'args': (P[1], Q[1],)}]
    
    cons2 = [{'type':'ineq', 'fun' : fromSrc, 'args': (P[1], i, shape,)} for i in range(numFeats1)]
    cons3 = [{'type':'ineq', 'fun' : toTgt, 'args': (Q[1], j, shape,)} for j in range(numFeats2)]
    cons = cons1 + cons2 + cons3
    F_guess = np.zeros(D.shape)
    F = scipy.optimize.minimize(flow, F_guess, args=(D,), constraints=cons)
    F = np.reshape(F.x, (numFeats1,numFeats2))
    return F
def EMD(F, D):  
    return (F * D).sum() / F.sum()
def getEMD(P,Q, norm = 2):
    D = getDistMatrix(P[0], Q[0], norm)
    F = getFlowMatrix(P, Q, D)
    return EMD(F, D)

def weights_dataset_distribution(norm,feature,pseudo_feature,pseudo_labels,K):
    ## gmm category for feature and pseudo feature
    category,possibility_label = gmm_cluster_possibility(feature,K)
    pseudo_category,pseudo_possibility_label = gmm_cluster_possibility(pseudo_feature,K)

    ## pseudo feature K clusters
    pseudo_classified_data,pseudo_classified_possibility,pseudo_classified,pseudo_classified_list = gmm_cluster_distribution(pseudo_feature,pseudo_labels,K)
    
    ## calculated data distribution distance between pseudo features and feature
    Q = (feature, possibility_label)
    emd = []
    for i in range(K):
        emd.append(getEMD((pseudo_classified_data[i],pseudo_classified_possibility[i]),Q))
    return emd

    new_pseudo = np.ones(len(pseudo_labels))
    for i in range(K):
        print('the length of list and the length of classified pseudo label',len(pseudo_classified_list[i]),len(pseudo_classified))
        emd = np.array(emd/sum(emd))
        pseudo_classified[i] = pseudo_classified[i]*emd[i]
        new_pseudo[pseudo_classified_list[i]] = pseudo_classified[i]
    return new_pseudo




