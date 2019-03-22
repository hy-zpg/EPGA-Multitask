# import time
# import numpy as np
# from sklearn.base import BaseEstimator, ClusterMixin
# from sklearn.decomposition import PCA
# from sklearn.utils import check_array, check_consistent_length, gen_batches
# from scipy.stats import multivariate_normal
# import scipy.optimize


# ###gmm generation###
# def phi(Y, mu_k, cov_k):
#     norm = multivariate_normal(mean=mu_k, cov=cov_k)
#     return norm.pdf(Y)

# def getExpectation(Y, mu, cov, alpha):
#     N = Y.shape[0]
#     K = alpha.shape[0]
#     assert N > 1, "There must be more than one sample!"
#     assert K > 1, "There must be more than one gaussian model!"
#     gamma = np.mat(np.zeros((N, K)))
#     prob = np.zeros((N, K))
#     for k in range(K):
#         prob[:, k] = phi(Y, mu[k], cov[k])
#     prob = np.mat(prob)
#     for k in range(K):
#         gamma[:, k] = alpha[k] * prob[:, k]
#     for i in range(N):
#         gamma[i, :] /= np.sum(gamma[i, :])
#     return gamma

# def maximize(Y, gamma):
#     N, D = Y.shape
#     K = gamma.shape[1]
#     mu = np.zeros((K, D))
#     cov = []
#     alpha = np.zeros(K)
#     for k in range(K):
#         Nk = np.sum(gamma[:, k])
#         mu[k, :] = np.sum(np.multiply(Y, gamma[:, k]), axis=0) / Nk
#         cov_k = (Y - mu[k]).T * np.multiply((Y - mu[k]), gamma[:, k]) / Nk
#         cov.append(cov_k)
#         alpha[k] = Nk / N
#     cov = np.array(cov)
#     return mu, cov, alpha

# def scale_data(Y):
#     for i in range(Y.shape[1]):
#         max_ = Y[:, i].max()
#         min_ = Y[:, i].min()
#         Y[:, i] = (Y[:, i] - min_) / (max_ - min_)
#     return Y

# def init_params(shape, K):
#     N, D = shape
#     mu = np.random.rand(K, D)
#     cov = np.array([np.eye(D)] * K)
#     alpha = np.array([1.0 / K] * K)
#     return mu, cov, alpha

# def GMM_EM(Y, K, times):
#     Y = scale_data(Y)
#     mu, cov, alpha = init_params(Y.shape, K)
#     # print('data shape:',np.shape(Y))
#     # print('corresponding mu,cov,alpha shape',np.shape(mu),np.shape(cov),np.shape(alpha))
#     for i in range(times):
#         gamma = getExpectation(Y, mu, cov, alpha)
#         mu, cov, alpha = maximize(Y, gamma)
#     return mu, cov, alpha

# #generating  k cluster based on gmm algorithm 
# def gmm_cluster(input_data,K):
# # alpha is the possibility of data in category 
# # data density can be calculated from the variance matrix  how?????
# # i think the data density is from the alpha
#     mu, cov, alpha = GMM_EM(input_data, K, 100)
#     N = Y.shape[0]
#     gamma = getExpectation(input_data, mu, cov, alpha)
#     category = gamma.argmax(axis=1).flatten().tolist()[0]
#     #generating index_category
#     for i in range(K):
#         cat[i]=[]
#         cat_index[i]=[]
#     for i in range(N):
#         cat[category[i]].append(input_data[i])
#         cat_index[category[i]].append(i)
#         mu[category[i]].append
#     for i in range(K):
#         cat[i]=np.array(cat[i])
#         cat_index[i]=np.array(cat_index[i])
#     return cat,cat_index,mu,cov,alpha



# ### data distribution distance ###
# #input [cluster,weights]
# def positivity(f):
#     return f 

# def fromSrc(f, wp, i, shape):
#     fr = np.reshape(f, shape)
#     f_sumColi = np.sum(fr[i,:])
#     return wp[i] - f_sumColi
# def toTgt(f, wq, j, shape):
#     fr = np.reshape(f, shape)
#     f_sumRowj = np.sum(fr[:,j])
#     return wq[j] - f_sumRowj
# def maximiseTotalFlow(f, wp, wq): 
#     return f.sum() - np.minimum(wp.sum(), wq.sum())
# def flow(f, D):
#     f = np.reshape(f, D.shape)
#     return (f * D).sum()
# def groundDistance(x1, x2, norm = 2):
#     return np.linalg.norm(x1-x2, norm)
# def getDistMatrix(s1, s2, norm = 2):
#     numFeats1 = s1.shape[0]
#     numFeats2 = s2.shape[0]
#     distMatrix = np.zeros((numFeats1, numFeats2))
#     for i in range(0, numFeats1):
#         for j in range(0, numFeats2):
#             distMatrix[i,j] = groundDistance(s1[i], s2[j], norm)
#     return distMatrix
# def getFlowMatrix(P, Q, D):
#     numFeats1 = P[0].shape[0]
#     numFeats2 = Q[0].shape[0]
#     shape = (numFeats1, numFeats2)
#     cons1 = [{'type':'ineq', 'fun' : positivity},
#              {'type':'eq', 'fun' : maximiseTotalFlow, 'args': (P[1], Q[1],)}]
    
#     cons2 = [{'type':'ineq', 'fun' : fromSrc, 'args': (P[1], i, shape,)} for i in range(numFeats1)]
#     cons3 = [{'type':'ineq', 'fun' : toTgt, 'args': (Q[1], j, shape,)} for j in range(numFeats2)]
#     cons = cons1 + cons2 + cons3
#     F_guess = np.zeros(D.shape)
#     F = scipy.optimize.minimize(flow, F_guess, args=(D,), constraints=cons)
#     F = np.reshape(F.x, (numFeats1,numFeats2))
#     return F
# def EMD(F, D):  
#     return (F * D).sum() / F.sum()
# def getEMD(P,Q, norm = 2):
#     D = getDistMatrix(P[0], Q[0], norm)
#     F = getFlowMatrix(P, Q, D)
#     return EMD(F, D)

# # def dataset_distribution_distance(norm,feature_1,weights_1,feature_2,weights_2):
# #     k=np.shape(feature_1)[0]
# #     Q = (feature_2, weights_2)
# #     emd=np.zeros(k)
# #     for i in range(k):
# #         emd[i] = getEMD((feature_1[i],weights_1[i]),Q)
# #     return emd


# def data_distribution_subsets(X_t, y_t,X_p,y_p, n_subsets=3,batch_max=500000,method='default', density_t=0.6,is_distribution):
#     X_t = check_array(X_t, accept_sparse='csr')
#     X_p = check_array(X_p, accept_sparse='csr')
#     check_consistent_length(X_t, y_t)
#     check_consistent_length(X_p, y_p)
    
#     # total category
#     unique_categories_t = set(list(y_t))
#     unique_categories_p = set(list(y_p))
#     t0 = None

#     # Initialize all labels as negative one which represents un-clustered 'noise'.
#     # Post-condition: after clustering, there should be no negatives in the label output.
#     # all_clustered_labels_t = np.full(len(y_t), 1, dtype=np.intp)
#     all_clustered_labels_p = np.full(len(y_p), 1, dtype=np.intp)
#     all_distribution_labels_p = np.full(len(y_p), 1, dtype=np.intp)
    
#     for cluster_idx_t, current_category_t in enumerate(unique_categories_t):
#         for cluster_idx_p, current_category_p in enumerate(unique_categories_p):
#         # Collect the "learning material" for this particular category
#             if current_category_t==current_category_p:
#                 dist_list_t = [i for i, label in enumerate(y_t) if label == current_category_t]
#                 dist_list_p = [i for i, label in enumerate(y_p) if label == current_category_p]
#                 # recording the index
#                 for batch_range in gen_batches(len(dist_list_t), batch_size=batch_max):
#                     batch_dist_list_t = dist_list_t[batch_range]
#                 for batch_range in gen_batches(len(dist_list_p), batch_size=batch_max):
#                     batch_dist_list_p = dist_list_p[batch_range]

#                 # Load data subset
#                 subset_vectors_t = np.zeros((len(batch_dist_list_t), X_t.shape[1]), dtype=np.float32)
#                 subset_vectors_p = np.zeros((len(batch_dist_list_p), X_p.shape[1]), dtype=np.float32)
#                 for subset_idx, global_idx in enumerate(batch_dist_list_t):
#                     subset_vectors_t[subset_idx, :] = X_t[global_idx, :]
#                 for subset_idx, global_idx in enumerate(batch_dist_list_p):
#                     subset_vectors_p[subset_idx, :] = X_p[global_idx, :]

#                 m_t = np.dot(subset_vectors_t, np.transpose(subset_vectors_t))
#                 t_t = np.square(subset_vectors_t).sum(axis=1)
#                 distance_t = np.sqrt(np.abs(-2 * m_t + t_t + np.transpose(np.array([t_t]))))

#                 m_p = np.dot(subset_vectors_p, np.transpose(subset_vectors_p))
#                 t_p = np.square(subset_vectors_p).sum(axis=1)
#                 distance_p = np.sqrt(np.abs(-2 * m_p + t_p + np.transpose(np.array([t_p]))))

#                 if method == 'gaussian':
#                     densities_t = np.zeros((len(subset_vectors_t)), dtype=np.float32)
#                     distance_t = distance_t / np.max(distance_t)
#                     for i in range(len(subset_vectors_t)):
#                         densities_t[i] = np.sum(1 / np.sqrt(2 * np.pi) * np.exp((-1) * np.power(distance_t[i], 2) / 2.0))

#                     densities_p = np.zeros((len(subset_vectors_p)), dtype=np.float32)
#                     distance_p = distance_p / np.max(distance_p)
#                     for i in range(len(subset_vectors_p)):
#                         densities_p[i] = np.sum(1 / np.sqrt(2 * np.pi) * np.exp((-1) * np.power(distance_p[i], 2) / 2.0))
#                 else:
#                     densities_t = np.zeros((len(subset_vectors_t)), dtype=np.float32)
#                     flat_distance_t = distance_t.reshape(distance_t.shape[0] * distance_t.shape[1])
#                     dist_cutoff_t = np.sort(flat_distance_t)[int(distance_t.shape[0] * distance_t.shape[1] * density_t)]
#                     for i in range(len(batch_dist_list)):
#                         densities_t[i] = len(np.where(distance_t[i] < dist_cutoff_t)[0]) - 1  # remove itself

#                     densities_p = np.zeros((len(subset_vectors_p)), dtype=np.float32)
#                     flat_distance_p = distance_t.reshape(distance_p.shape[0] * distance_p.shape[1])
#                     dist_cutoff_p = np.sort(flat_distance_p)[int(distance_p.shape[0] * distance_p.shape[1] * density_t)]
#                     for i in range(len(batch_dist_list)):
#                         densities_t[i] = len(np.where(distance_t[i] < dist_cutoff_t)[0]) - 1  # remove itself


#                 if len(densities_t) < n_subsets or len(densities_p) < n_subsets:
#                     raise ValueError("Cannot cluster into {} subsets due to lack of density diversification,"
#                                      " please try a smaller n_subset number.".format(n_subsets))
                

#                 model_t = KMeans(n_clusters=n_subsets, random_state=random_state)
#                 model_p = KMeans(n_clusters=n_subsets, random_state=random_state)
#                 model_t.fit(densities_t.reshape(len(densities_t), 1))
#                 model_p.fit(densities_p.reshape(len(densities_p), 1))
#                 clusters_t = [densities_t[np.where(model_t.labels_ == i)] for i in range(n_subsets)]
#                 clusters_p = [densities_p[np.where(model_p.labels_ == i)] for i in range(n_subsets)]
#                 n_clusters_made_t = len(set([k for j in clusters_t for k in j]))
#                 n_clusters_made_p = len(set([k for j in clusters_p for k in j]))
                
#                 if n_clusters_made_t < n_subsets or n_clusters_made_p < n_subsets:
#                     raise ValueError("Cannot cluster into {} subsets, please try a smaller n_subset number, such as {}.".
#                                      format(n_subsets, n_clusters_made_t))

#                 cluster_mins_t = [np.min(c) for c in clusters_t]
#                 bound_t = np.sort(np.array(cluster_mins_t))
#                 cluster_mins_p = [np.min(c) for c in clusters_p]
#                 bound_p = np.sort(np.array(cluster_mins_p))


#                 # Distribute into curriculum subsets, and package into global adjusted returnable array, optionally aux too
#                 other_bounds = range(n_subsets - 1)
#                 for i in range(len(densities_t)):
#                     if densities_t[i] >= bound_t[n_subsets - 1]:
#                         all_clustered_labels_t[batch_dist_list_t[i]] = 0
#                     else:
#                         for j in other_bounds:
#                             if bound_t[j] <= densities_t[i] < bound_t[j + 1]:
#                                 all_clustered_labels_t[batch_dist_list_t[i]] = len(bound_t) - j - 1
                
#                 for i in range(len(densities_p)):
#                     if densities_p[i] >= bound_p[n_subsets - 1]:
#                         all_clustered_labels_p[batch_dist_list_p[i]] = 0
#                     else:
#                         for j in other_bounds:
#                             if bound_p[j] <= densities_p[i] < bound_p[j + 1]:
#                                 all_clustered_labels_p[batch_dist_list_p[i]] = len(bound_p) - j - 1
                
#                 if is_distribution_distance:
#                     ### data purity 
#                     # all_distribution_labels_p[batch_dist_list_p] = all_clustered_labels_p[batch_dist_list_p]*cat_p            
#                     #calculating the distance between each pesudo clusters and all ground-truth clusters and calculating weights
#                     # for i in range(n_clusters):
#                     #     print('feature shape:',np.shape(subset_vectors_p),np.shape(subset_vectors_t))
#                     #     print('clustered data:',np.shape(cat),np.shape(cat_index),np.shape(mu),np.shape(cov),np.shape(alpha))
#                     #     emd=dataset_distribution_distance(2,subset_vectors_p[i],cat_p[i],subset_vectors_t,cat)
#                     #     all_clustered_labels_p[cat_index_p]=all_clustered_labels_p[cat_index_p]*emd
                    
#                     ###  distribution distance
#                     for i in range(n_subsets):

#                     # emd = getEMD((subset_vectors_p,cat_p),(subset_vectors_t,cat))
#                     # all_clustered_labels_p[batch_dist_list_p] = all_clustered_labels_p[batch_dist_list_p]*emd
#     # return all_clustered_labels_p



# # class Data_distribution(BaseEstimator, ClusterMixin):

# #     def __init__(self, n_clusters=3,batch_max=500000):
# #         self.n_clusters = n_clusters
# #         self.batch_max = batch_max
# #         self.output_labels=None
# #     def fit(self,X_t, y_t,X_p,y_p):
# #         X_t = check_array(X_t, accept_sparse='csr')
# #         X_p = check_array(X_p, accept_sparse='csr')
# #         check_consistent_length(X_t, y_t)
# #         check_consistent_length(X_p, y_p)
# #         self.output_labels = data_distribution_subsets(X_t, y_t,X_p,y_p, **self.get_params())
# #         return self

#     # def fit_predict(self, X, y=None):
#     #     self.fit(X, y)
#     #     return self.output_labels