import time
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.decomposition import PCA
from sklearn.utils import check_array, check_consistent_length, gen_batches
from scipy.stats import multivariate_normal
import scipy.optimize


###gmm generation###
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
        gamma = getExpectation(Y, mu, cov, alpha)
        mu, cov, alpha = maximize(Y, gamma)
    return mu, cov, alpha

#generating  k cluster based on gmm algorithm 
def gmm_cluster(input_data,K):
# alpha is the possibility of data in category 
# data density can be calculated from the variance matrix  how?????
# i think the data density is from the alpha
    mu, cov, alpha = GMM_EM(input_data, K, 100)
    N = Y.shape[0]
    gamma = getExpectation(input_data, mu, cov, alpha)
    category = gamma.argmax(axis=1).flatten().tolist()[0]
    #generating index_category
    for i in range(K):
        cat[i]=[]
        cat_index[i]=[]
    for i in range(N):
        cat[category[i]].append(input_data[i])
        cat_index[category[i]].append(i)
        mu[category[i]].append
    for i in range(K):
        cat[i]=np.array(cat[i])
        cat_index[i]=np.array(cat_index[i])
    return cat,cat_index,mu,cov,alpha



### data distribution distance ###
#input [cluster,weights]
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

# def dataset_distribution_distance(norm,feature_1,weights_1,feature_2,weights_2):
#     k=np.shape(feature_1)[0]
#     Q = (feature_2, weights_2)
#     emd=np.zeros(k)
#     for i in range(k):
#         emd[i] = getEMD((feature_1[i],weights_1[i]),Q)
#     return emd


def data_distribution_subsets(X_t, y_t,X_p,y_p, n_clusters=4,batch_max=500000):
    X_t = check_array(X_t, accept_sparse='csr')
    X_p = check_array(X_p, accept_sparse='csr')
    check_consistent_length(X_t, y_t)
    check_consistent_length(X_p, y_p)
    
    # total category
    unique_categories_t = set(list(y_t))
    unique_categories_p = set(list(y_p))
    t0 = None

    # Initialize all labels as negative one which represents un-clustered 'noise'.
    # Post-condition: after clustering, there should be no negatives in the label output.
    # all_clustered_labels_t = np.full(len(y_t), 1, dtype=np.intp)
    all_clustered_labels_p = np.full(len(y_p), 1, dtype=np.intp)
    
    for cluster_idx_t, current_category_t in enumerate(unique_categories_t):
        for cluster_idx_p, current_category_p in enumerate(unique_categories_p):
        # Collect the "learning material" for this particular category
            if current_category_t==current_category_p:
                dist_list_t = [i for i, label in enumerate(y_t) if label == current_category_t]
                dist_list_p = [i for i, label in enumerate(y_p) if label == current_category_p]
                # recording the index
                for batch_range in gen_batches(len(dist_list_t), batch_size=batch_max):
                    batch_dist_list_t = dist_list_t[batch_range]
                for batch_range in gen_batches(len(dist_list_p), batch_size=batch_max):
                    batch_dist_list_p = dist_list_p[batch_range]

                # Load data subset
                subset_vectors_t = np.zeros((len(batch_dist_list_t), X_t.shape[1]), dtype=np.float32)
                subset_vectors_p = np.zeros((len(batch_dist_list_p), X_p.shape[1]), dtype=np.float32)
                for subset_idx, global_idx in enumerate(batch_dist_list_t):
                    subset_vectors_t[subset_idx, :] = X_t[global_idx, :]
                for subset_idx, global_idx in enumerate(batch_dist_list_p):
                    subset_vectors_p[subset_idx, :] = X_p[global_idx, :]

                cat,cat_index,mu,cov,alpha=gmm_cluster(subset_vectors_p,K=n_clusters)
                cat_p,cat_index_p,mu_p,cov_p,alpha_p=gmm_cluster(subset_vectors_p,K=n_clusters)
                

                ### data purity 
                all_clustered_labels_p[batch_dist_list_p] = all_clustered_labels_p[batch_dist_list_p]*cat_p            
                #calculating the distance between each pesudo clusters and all ground-truth clusters and calculating weights
                # for i in range(n_clusters):
                #     print('feature shape:',np.shape(subset_vectors_p),np.shape(subset_vectors_t))
                #     print('clustered data:',np.shape(cat),np.shape(cat_index),np.shape(mu),np.shape(cov),np.shape(alpha))
                #     emd=dataset_distribution_distance(2,subset_vectors_p[i],cat_p[i],subset_vectors_t,cat)
                #     all_clustered_labels_p[cat_index_p]=all_clustered_labels_p[cat_index_p]*emd
                
                ###  distribution distance
                emd = getEMD((subset_vectors_p,cat_p),(subset_vectors_t,cat))
                all_clustered_labels_p[batch_dist_list_p] = all_clustered_labels_p[batch_dist_list_p]*emd
    return all_clustered_labels_p



class Data_distribution(BaseEstimator, ClusterMixin):

    def __init__(self, n_clusters=3,batch_max=500000):
        self.n_clusters = n_clusters
        self.batch_max = batch_max
        self.output_labels=None
    def fit(self,X_t, y_t,X_p,y_p):
        X_t = check_array(X_t, accept_sparse='csr')
        X_p = check_array(X_p, accept_sparse='csr')
        check_consistent_length(X_t, y_t)
        check_consistent_length(X_p, y_p)
        self.output_labels = data_distribution_subsets(X_t, y_t,X_p,y_p, **self.get_params())
        return self

    # def fit_predict(self, X, y=None):
    #     self.fit(X, y)
    #     return self.output_labels