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
    mu, cov, alpha = init_params(Y.shape, K)
    # print('data shape:',np.shape(Y))
    # print('corresponding mu,cov,alpha shape',np.shape(mu),np.shape(cov),np.shape(alpha))
    for i in range(times):
        # print('times:',i)
        gamma = getExpectation(Y, mu, cov, alpha)
        mu, cov, alpha = maximize(Y, gamma)
        # print('{}_times_finished'.format(i))
    return mu, cov, alpha,gamma

def gmm_cluster_possibility(input_data,K):
    # input_data_mat =  np.matrix(input_data, copy=True)
    # print('feature shape:',np.shape(input_data_mat))
    mu, cov, alpha, gamma = GMM_EM(input_data, K, 3)
    N = input_data.shape[0]
    # possibility_label = np.ones(N)
    # print('starting to generating alpha after trained')
    # gamma = getExpectation(input_data, mu, cov, alpha)
    # print('generated alpha after trained')
    category = gamma.argmax(axis=1).flatten().tolist()[0]
    # print(len(category),N)
    # print('generated alpha after trained')
    possibility_label = np.array([alpha[category[i]] for i in range(N)])
    # for i in range(N):
    #     possibility_label[i] = alpha[category[i]]
    # print('generated possibility')
    return category,possibility_label





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
    # print('2')
    F = np.reshape(F.x, (numFeats1,numFeats2))
    return F
def EMD(F, D):  
    return (F * D).sum() / F.sum()
def getEMD(P,Q, norm = 2):
    D = getDistMatrix(P[0], Q[0], norm)
    F = getFlowMatrix(P, Q, D)
    return EMD(F, D)


# def gmm_cluster_distribution(input_data,label,K):
# # alpha is the possibility of data in category 
# # data density can be calculated from the variance matrix  how?????
# # i think the data density is from the alpha
#     category,possibility_label = gmm_cluster_possibility(input_data,K)
#     # print('category is:',category)
#     N = input_data.shape[0]
#     classified_data = np.zeros([K,np.shape(input_data)[0],np.shape(input_data)[1]])
#     classified_possibility = np.zeros([K,np.shape(possibility_label)[0]])

#     classified_data=[]
#     classified_possibility = []
#     classified_label = []
#     classified_list = []

#     for j in range(K):
#         # recording data index
#         classified_list.append([i for i, label in enumerate(category) if label == j])
#         classified_data.append(np.array([input_data[i] for i in range(N) if category[i] == j]))
#         classified_possibility.append(np.array([possibility_label[i] for i in range(N) if category[i] == j]))
#         classified_label.append(np.array([label[i] for i in range(N) if category[i] == j]))
#     return classified_data,classified_possibility,classified_label,classified_list


# def each_category_distribution(norm,feature,pseudo_feature,pseudo_labels,K):
#     ## gmm category for feature and pseudo feature
#     # print('feature clustering')
#     category,possibility_label = gmm_cluster_possibility(feature,K)
#     # print('pseuo feature clustering')
#     # pseudo_category,pseudo_possibility_label = gmm_cluster_possibility(pseudo_feature,K)
#     del category

#     ## pseudo feature K clusters
#     pseudo_classified_data,pseudo_classified_possibility,pseudo_classified,pseudo_classified_list = gmm_cluster_distribution(pseudo_feature,pseudo_labels,K)
    
#     ## calculated data distribution distance between pseudo features and feature
#     Q = (feature, possibility_label)
#     print('Q shape:',np.shape(feature),np.shape(possibility_label),type(feature),type(possibility_label))
#     emd = []
#     for i in range(K):
#         print('P shape:',np.shape(pseudo_classified_data[i]),np.shape(pseudo_classified_possibility[i]),type(pseudo_classified_data[i]),type(pseudo_classified_possibility[i]))
#         emd.append(getEMD((np.array(pseudo_classified_data[i]),np.array(pseudo_classified_possibility[i])),Q))
#     # return emd

#     new_pseudo = np.ones(len(pseudo_feature))
#     for i in range(K):
#         print('the length of classified pseudo label',len(pseudo_classified_list[i]),len(pseudo_classified))
#         emd = np.array(emd)
#         emd = emd/np.sum(emd)
#         pseudo_classified[i] = pseudo_classified[i]*emd[i]
#         new_pseudo[pseudo_classified_list[i]] = pseudo_classified[i]
#     return new_pseudo

def categoried_gmm_distribution(norm,feature,pseudo_feature,pseudo_labels,K):
    # preparing clusted P
    category_pseudo,possibility_label_pseudo = gmm_cluster_possibility(pseudo_feature,K)
    N = np.shape(pseudo_labels)[0]
    pseudo_classified_list=[]
    pseudo_classified_label=[]
    emd=[]
    for j in range(K):
         # preparing Q
        category,possibility_label = gmm_cluster_possibility(feature,K)
        del category
        Q = (feature, possibility_label)
        print('Q shape:',np.shape(feature),np.shape(possibility_label),type(feature),type(possibility_label))
        
        pseudo_classified_data = np.array([pseudo_feature[i] for i in range(N) if category_pseudo[i] == j])
        pseudo_classified_possibility = np.array([possibility_label_pseudo[i] for i in range(N) if category_pseudo[i] == j])
        pseudo_classified_list.append([i for i, label in enumerate(category_pseudo) if label == j])
        pseudo_classified_label.append(np.array([pseudo_labels[i] for i in range(N) if category_pseudo[i] == j])) 
        print('P shape:',np.shape(pseudo_classified_data),np.shape(pseudo_classified_possibility),type(pseudo_classified_data),type(pseudo_classified_possibility))
        #calculating emd
        emd.append(getEMD((np.array(pseudo_classified_data),np.array(pseudo_classified_possibility)),Q))
        del pseudo_classified_data, pseudo_classified_possibility, Q

    new_pseudo = np.ones(len(pseudo_feature))
    for j in range(K):
        emd = np.array(emd)
        emd = emd/np.sum(emd)
        pseudo_classified_label[i] = pseudo_classified_label[i]*emd[i]
        new_pseudo[pseudo_classified_list[i]] = pseudo_classified_label[i]
        del pseudo_classified_label,pseudo_classified_list
    return new_pseudo

def weights_gmm_dataset_distribution(norm,feature,pseudo_feature,pseudo_labels,labels,K):
    pseudo_labels_distribution = np.ones(len(pseudo_labels))
    categoty_n = len(set(labels))
    # category_feature=[]
    # category_label=[]
    # category_list=[]

    # category_pseudo_feature=[]
    # category_pseudo_label=[]
    # category_pseudo_list=[]
    ### each category,list,feature,label
    for j in range(categoty_n):
        # category_list = [i for i, label in enumerate(labels) if label == j]
        category_feature = np.array([feature[i] for i in range(len(labels)) if labels[i] == j])
        # category_label = np.array([labels[i] for i in range(len(labels)) if labels[i] == j])

        category_pseudo_list = [i for i, label in enumerate(pseudo_labels) if label == j]
        category_pseudo_feature = np.array([pseudo_feature[i] for i in range(len(pseudo_labels)) if pseudo_labels[i] == j])
        category_pseudo_label = np.array([pseudo_labels[i] for i in range(len(pseudo_labels)) if pseudo_labels[i] == j])
        # distribution possibility
        pseudo_labels_distribution[category_pseudo_list] = categoried_gmm_distribution(norm,category_feature,category_pseudo_feature,category_pseudo_label,K)
    return pseudo_labels_distribution



###### distribution distance based density
def categoried_density_distribution(feature,density_labels,labels,pseudo_feature,pseudo_density_labels,pseudo_labels,K):
    # cluster_n = len(list(set(density_labels)))
    categoried_distribution_distance = np.zeros(len(pseudo_feature))
    N = np.shape(pseudo_labels)[0]
    pseudo_classified_list=[]
    pseudo_classified_label=[]
    emd=[]
    print(len(list(set(pseudo_labels))))
    for j in range(K):
         # preparing Q
        density_labels_weights=density_labels/sum(list(set(density_labels)))
        density_labels_weights = np.ones((len(density_labels)), dtype=np.float32) - density_labels_weights
        Q = (feature, density_labels_weights)
        print('Q shape:',np.shape(feature),np.shape(density_labels_weights),type(feature),type(density_labels_weights))
        
        pseudo_classified_data = np.array([pseudo_feature[i] for i in range(N) if pseudo_density_labels[i] == j])
        pseudo_classified_list = np.array([i for i, label in enumerate(pseudo_density_labels) if label == j])
        pseudo_classified_label_current = np.array([pseudo_density_labels[i] for i in range(N) if pseudo_density_labels[i] == j])
        # pseudo_classified_label.append(pseudo_classified_label_current) 
        pseudo_classified_density_weights = pseudo_classified_label_current/sum(list(set(list(pseudo_classified_label_current))))
        pseudo_classified_density_weights = np.ones((len(pseudo_classified_density_weights)), dtype=np.float32) - pseudo_classified_density_weights

        print('P shape:',np.shape(pseudo_classified_data),np.shape(pseudo_classified_density_weights),type(pseudo_classified_data),type(pseudo_classified_density_weights))
        #calculating emd
        emd = getEMD((np.array(pseudo_classified_data),np.array(pseudo_classified_density_weights)),Q)
        categoried_distribution_distance[pseudo_classified_list] = emd
        del pseudo_classified_data, pseudo_classified_density_weights, Q
        return categoried_distribution_distance

    # new_pseudo = np.ones(len(pseudo_feature))
    # for j in range(K):
    #     emd = np.array(emd)
    #     emd = emd/np.sum(emd)
    #     pseudo_classified_label[i] = pseudo_classified_label[i]*emd[i]
    #     new_pseudo[pseudo_classified_list[i]] = pseudo_classified_label[i]
    #     del pseudo_classified_label,pseudo_classified_list
    # return new_pseudo

def weights_density_dataset_distribution(feature,pseudo_feature,density_labels,pseudo_density_labels,pseudo_labels,labels,K):
    distribution_distance = np.zeros(len(pseudo_labels))
    categoty_n = len(set(labels))
    # print('1',len(set(pseudo_labels)))
    
    ### each category,list,feature,label
    for j in range(categoty_n):
        category_list = [i for i, label in enumerate(labels) if label == j]
        category_feature = np.array([feature[i] for i in range(len(labels)) if labels[i] == j])
        category_label = np.array([labels[i] for i in range(len(labels)) if labels[i] == j])

        category_pseudo_list = [i for i, label in enumerate(pseudo_labels) if label == j]
        category_pseudo_feature = np.array([pseudo_feature[i] for i in range(len(pseudo_labels)) if pseudo_labels[i] == j])
        category_pseudo_density_label = np.array([pseudo_density_labels[i] for i in range(len(pseudo_labels)) if pseudo_labels[i] == j])
        category_pseudo_label = np.array([pseudo_labels[i] for i in range(len(pseudo_labels)) if pseudo_labels[i] == j])
        # distribution possibility
        distribution_distance[category_pseudo_list] = categoried_density_distribution(feature,density_labels,labels,category_pseudo_feature,category_pseudo_density_label,category_pseudo_label,K)
    return distribution_distance










'''
1. pesudo feature are categoried into specific category
2. in each category of feature and pseudo feature, clustering using gmm, generating clustered feature and clustered labels
3. in same categoty, calculting the distance of each cluster of pseudo feature and all feature cluster, generating emd distance to obtain corresponding possibility
'''
    



