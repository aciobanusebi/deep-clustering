import numpy as np
from sklearn import metrics

def purity_score(y_true, y_pred): # from https://stackoverflow.com/a/51672699/7947996; in [0,1]; 0-bad,1-good
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

from sklearn.metrics.cluster import adjusted_rand_score # in [0,1]; 0-bad,1-good
from sklearn.metrics.cluster import normalized_mutual_info_score # in [0,1]; 0-bad,1-good

from coclust.evaluation.external import accuracy # in [0,1]; 0-bad,1-good; install it via "!pip install coclust" in Google Colab