from sklearn.cluster import KMeans
import numpy as np

MAX_K = 10

def convert_lsi_to_vector(corpus_lsi):
    vector = []
    for token_vector in corpus_lsi:
        vector.append([v for k,v in token_vector])
    return vector

def find_elbow_point(data_to_fit, max_k=MAX_K):
    inertias = np.zeros(MAX_K)
    diff = np.zeros(MAX_K)
    diff2 = np.zeros(MAX_K)
    diff3 = np.zeros(MAX_K)
    for k in range(1,max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data_to_fit)
        inertias[k - 1] = kmeans.inertia_
        if k > 1:
            diff[k - 1] = inertias[k - 1] - inertias[k - 2]
        if k > 2:
            diff2[k - 1] = diff[k - 1] - diff[k - 2]
        if k > 3:
            diff3[k - 1] = diff2[k - 1] - diff2[k - 2]

    return np.argmin(diff3[3:]) + 3

def run_k_mean_and_get_optimal_k(corpus_lsi, initial_dim):
    # data = convert_lsi_to_vector(corpus_lsi)
    data = convert_lda_to_vector(corpus_lsi, initial_dim)
    return find_elbow_point(data)

def run_k_mean_with_k(corpus_lda, k):
    data_to_fit = convert_lda_to_vector(corpus_lda, k)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data_to_fit)
    return kmeans.labels_

def convert_lda_to_vector(corpus_lda, num_topics):
    vectors = []
    for token_vector in corpus_lda:
        vector = [0]*num_topics
        for k,v in token_vector:
            vector[k] = v
        vectors.append(vector)
    return vectors
