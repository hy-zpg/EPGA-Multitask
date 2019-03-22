import time
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.decomposition import PCA
from sklearn.utils import check_array, check_consistent_length, gen_batches
from scipy.stats import multivariate_normal
import scipy.optimize
from sklearn.mixture import GaussianMixture as GMM
import cv2
from pyemd import emd_samples
from scipy.stats import wasserstein_distance


from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

store_path = '/home/user/Desktop/'
#tsne
def plot_embedding(data,origin, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    fig.patch.set_facecolor('none')
    ax = plt.subplot(111)
    sorted_result = np.array(sorted(origin))
    for i in range(data.shape[0]):
        # if origin[i]>sorted_result[-6]:
        #     plt.text(data[i, 0], data[i, 1], '$\bm{\checkmark}$',
        #              color=plt.cm.Set1(label[i] / 10.),
        #              fontdict={'weight': 'bold', 'size': 9})
        # elif origin[i]<sorted_result[4]:
        #     plt.text(data[i, 0], data[i, 1], 'x',
        #              color=plt.cm.Set1(label[i]+2 / 10.),
        #              fontdict={'weight': 'bold', 'size': 9})
        # else:
        plt.text(data[i, 0], data[i, 1], '.',
                 color=plt.cm.Set1(label[i]+4/ 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    # plt.title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return fig

def obtain_pseudo_density(data,label,store_path,fig_name,title,density):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    # fig = plot_embedding(result, label,
    #                      't-SNE embedding of the digits (time %.2fs)'
    #                      % (time() - t0))
    fig = plot_embedding(result, density,label,title)
    plt.savefig(store_path+fig_name,transparent=True,dpi=512)




def plot_distribution_embedding(data_1, label_1,data_2,label_2, title):
    x_min_1, x_max_1 = np.min(data_1, 0), np.max(data_1, 0)
    data_1 = (data_1 - x_min_1) / (x_max_1 - x_min_1)

    x_min_2, x_max_2 = np.min(data_2, 0), np.max(data_2, 0)
    data_2 = (data_2 - x_min_2) / (x_max_2 - x_min_2)
    
    fig = plt.figure()
    fig.patch.set_facecolor('none')
    ax = plt.subplot(111)
    for i in range(data_1.shape[0]):
        plt.text(data_1[i, 0], data_1[i, 1], '.',
                 color=plt.cm.Set1(1 / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    for i in range(data_2.shape[0]):
        plt.text(data_2[i, 0], data_2[i, 1], '.',
                 color=plt.cm.Set1(2 / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    # plt.title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return fig

def obtain_distribution(data_1,label_1,data_2,label_2,store_path,fig_name,title):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result_1 = tsne.fit_transform(data_1)
    result_2 = tsne.fit_transform(data_2)
    # fig = plot_embedding(result, label,
    #                      't-SNE embedding of the digits (time %.2fs)'
    #                      % (time() - t0))
    fig = plot_distribution_embedding(result_1, label_1,result_2,label_2, title)
    plt.savefig(store_path+fig_name,transparent=True,dpi=512)
'''
EMD calculation
need P([pi],[w_pi]), Q([qi],[w_qi])
1. [pi],[qi]
2. [w_pi],[w_qi]
'''
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


def data_density_distribution_weight(X_t, y_t,X_p,y_p,n_subsets,batch_max,density_t,is_density,is_distribution,method='default'):
    # c=Checking data [data,label]
    print('batchsize:',batch_max)
    X_t = check_array(X_t, accept_sparse='csr')
    X_p = check_array(X_p, accept_sparse='csr')
    check_consistent_length(X_t, y_t)
    check_consistent_length(X_p, y_p)
    
    # Total category
    unique_categories_t = set(list(y_t))
    unique_categories_p = set(list(y_p))

    # Initialize all d_weighs and g_weighs labels as -1
    # Post-condition: after clustering, there should be no negatives in the label output.

    densities_weights = np.full(np.shape(X_p)[0],-1, dtype=np.float32)
    distribution_distance = np.full(np.shape(X_p)[0],-1, dtype=np.float32)



    ####distribution distance#####
    # Preparing Q for emd
    # obtain_pseudo_density(X_t,y_t,store_path,'data_1_distribution.png','Distribution of training samples with ground truth')
    # obtain_pseudo_density(X_p,y_p,store_path,'data_2_distribution.png','Distribution of training samples with pseudo labels')
    
    obtain_distribution(X_t[:1000],y_t,X_p[:1000],y_p,store_path,'distribution-1.png','Distribution distance between training samples with pseudo labels and ground truth')
    obtain_distribution(X_t[:100],y_t,X_p[:100],y_p,store_path,'distribution-2.png','Distribution distance between training samples with pseudo labels and ground truth')
    obtain_distribution(X_t[:5000],y_t,X_p[:5000],y_p,store_path,'distribution-3.png','Distribution distance between training samples with pseudo labels and ground truth')
    obtain_distribution(X_t[:50],y_t,X_p[:50],y_p,store_path,'distribution-4.png','Distribution distance between training samples with pseudo labels and ground truth')

    if is_distribution:
		
        gmm_t = GMM(n_components=n_subsets).fit(X_t)
        category_labels_t = gmm_t.predict(X_t)
        category_probs_t = gmm_t.predict_proba(X_t)
        weights_t = gmm_t.weights_
        mean_tt=gmm_t.means_
        Q = (mean_tt,weights_t)

        gmm_p = GMM(n_components=n_subsets).fit(X_p)
        category_labels_p = gmm_p.predict(X_p)
        category_probs_p = gmm_p.predict_proba(X_p)
        weights_p = gmm_p.weights_
        # print('weights',weights_p[:10])
        # print('cate_pro',category_probs_p[:10])
        # print('cate:',category_labels_p[:10])
        mean_pp = gmm_p.means_
        N = np.shape(X_p)[0]
        emds = []
        for j in range(n_subsets):
            # classified_X_p = np.array([X_p[i] for i in range(N) if category_labels_p[i] == j])
            classified_list_p = np.array([i for i, label in enumerate(category_labels_p) if label == j])
            classified_mean_pp = np.array([mean_pp[j]])
            classified_weights_pp = np.array([weights_p[j]])
            P = (classified_mean_pp,classified_weights_pp)
            emd = getEMD(P,Q)
            # emds.append(emd)
            distribution_distance[classified_list_p] = emd
            print('cluster-{}-emd-{}'.format(j,emd))
        # print(np.shape(category_probs_p),np.shape(np.array(emds))) 
        # distribution_distance = np.dot(np.array(category_probs_p),np.array(emds))

        distribution_distance = np.ones(len(distribution_distance))/distribution_distance
        distribution_distance = distribution_distance/np.max(distribution_distance)




    if is_density:
        # Collect the "learning material" for this particular category
        j=1
        for cluster_idx_p, current_category_p in enumerate(unique_categories_p):
            j=j+1
            # Collecting pseudo data and ground truth data from the same category
            dist_list_p = [i for i, label in enumerate(y_p) if label == current_category_p]
            # Recording the index
            for batch_range in gen_batches(len(dist_list_p), batch_size=batch_max):
                batch_dist_list_p = dist_list_p[batch_range]

            # Load data subset
            subset_vectors_p = np.zeros((len(batch_dist_list_p), X_p.shape[1]), dtype=np.float32)
            subset_label_p = np.full(np.shape(X_p)[0],current_category_p, dtype=np.float32)
            for subset_idx, global_idx in enumerate(batch_dist_list_p):
                subset_vectors_p[subset_idx, :] = X_p[global_idx, :]
            
            # Calculating Eaulidean distance 
            m_p = np.dot(subset_vectors_p, np.transpose(subset_vectors_p))
            t_p = np.square(subset_vectors_p).sum(axis=1)
            distance_p = np.sqrt(np.abs(-2 * m_p + t_p + np.transpose(np.array([t_p]))))
            if method == 'gaussian':
                # densities_t = np.zeros((len(subset_vectors_t)), dtype=np.float32)
                # distance_t = distance_t / np.max(distance_t)
                # for i in range(len(subset_vectors_t)):
                #     densities_t[i] = np.sum(1 / np.sqrt(2 * np.pi) * np.exp((-1) * np.power(distance_t[i], 2) / 2.0))
                densities_p = np.zeros((len(subset_vectors_p)), dtype=np.float32)
                distance_p = distance_p / np.max(distance_p)
                for i in range(len(subset_vectors_p)):
                    densities_p[i] = np.sum(1 / np.sqrt(2 * np.pi) * np.exp((-1) * np.power(distance_p[i], 2) / 2.0))
            else:
                densities_p = np.zeros((len(subset_vectors_p)), dtype=np.float32)
                flat_distance_p = distance_p.reshape(distance_p.shape[0] * distance_p.shape[1])
                dist_cutoff_p = np.sort(flat_distance_p)[int(distance_p.shape[0] * distance_p.shape[1] * density_t)]
                for i in range(len(subset_vectors_p)):
                    densities_p[i] = len(np.where(distance_p[i] < dist_cutoff_p)[0]) - 1  # remove itself
            
            obtain_pseudo_density(subset_vectors_p,subset_label_p,store_path,('pseudo_{}_density.png').format(current_category_p),'Density of training samples with pseudo labels',densities_p)
            # obtain_pseudo_density(subset_vectors_p[:50],subset_label_p[:50],store_path,('pseudo_{}_density-1.png').format(current_category_p),'Density of training samples with pseudo labels',densities_p[:50])
            # obtain_pseudo_density(subset_vectors_p[:100],subset_label_p[:100],store_path,('pseudo_{}_density-2.png').format(current_category_p),'Density of training samples with pseudo labels',densities_p[:100])
            obtain_pseudo_density(subset_vectors_p[:1000],subset_label_p[:1000],store_path,('pseudo_{}_density-3.png').format(current_category_p),'Density of training samples with pseudo labels',densities_p[:1000])
            obtain_pseudo_density(subset_vectors_p[:5000],subset_label_p[:5000],store_path,('pseudo_{}_density-4.png').format(current_category_p),'Density of training samples with pseudo labels',densities_p[:5000])


            print('category-{}-density-{}'.format(current_category_p,densities_p[:5]))
                # if len(densities_p) < n_subsets:
                #     raise ValueError("Cannot cluster into {} subsets due to lack of density diversification,"
                #                      " please try a smaller n_subset number.".format(n_subsets))
                # Generating density weights, not normalization
                # print('batch density:',densities_p)
            densities_p =  densities_p - np.min( densities_p)
            densities_p = densities_p/np.max(densities_p)
            densities_p = np.where(densities_p > 0.00001,densities_p, 0)
            densities_weights[batch_dist_list_p] = densities_p
                
    if is_density and not is_distribution:
        print('checking_d:',(densities_weights > -1).all())
        print('single density weights:',densities_weights[:10])
        return densities_weights


    elif is_distribution and not is_density:
        print('single distance weights:',distribution_distance[:10])
        print('checking_g:',(distribution_distance > -1).all())
        return distribution_distance
    elif is_density and is_distribution:
        print('checking_d:',(densities_weights > -1).all())
        print('density minimun',np.min(densities_weights))
        print('checking_g:',(distribution_distance > -1).all())
        print('single density weights:',densities_weights[:10])
        # distribution_distance = np.ones(len(distribution_distance))/distribution_distance
        print('single distance weights:',distribution_distance[:10])
        combined_weights = densities_weights * distribution_distance
        print('density distribution combined:',combined_weights[:10])
        # combined_weights = combined_weights/np.max(combined_weights)
        # print('density distribution normalization combined:',combined_weights[:30])
        return combined_weights
    else:
        return np.full(np.shape(X_p)[0],1, dtype=np.float32)





class Density_gmm_distribution_weights(BaseEstimator, ClusterMixin):

    def __init__(self, n_subsets,density_t,is_density,is_distribution,batch_max=500000):
        self.n_subsets = n_subsets
        self.density_t = density_t
        self.is_density = is_density
        self.is_distribution = is_distribution
        self.batch_max = batch_max
        self.output_labels=None
    def fit(self,X_t,y_t,X_p,y_p):
        X_t = check_array(X_t, accept_sparse='csr')
        X_p = check_array(X_p, accept_sparse='csr')
        check_consistent_length(X_t, y_t)
        check_consistent_length(X_p, y_p)
        self.output_labels = data_density_distribution_weight(X_t, y_t,X_p,y_p, **self.get_params())
        return self

    # def fit_predict(self, X, y=None):
    #     self.fit(X, y)
    #     return self.output_labels