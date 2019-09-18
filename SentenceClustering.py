import collections
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

class SentenceClustering:
    def __init__(self, sents, nclusters,visualization=False):
        self.sents = sents
        self.nclusters = nclusters
        self.visualization = visualization

    def cluster_visualization_tsne(self,algo,tfidf_matrix):
        tsne_perplexity = 20.0
        tsne_early_exaggeration = 4.0
        tsne_learning_rate = 1000
        random_state = 1
        model = TSNE(n_components=2, random_state=random_state,
                     perplexity=tsne_perplexity,
                     early_exaggeration=tsne_early_exaggeration,
                     learning_rate=tsne_learning_rate,
                     )

        transformed_centroids = model.fit_transform(tfidf_matrix.toarray())
        plt.scatter(transformed_centroids[:, 0], transformed_centroids[:, 1],
                    c="g")
        plt.show()

    def kmeans_clustering(self):
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.sents)
        kmeans = KMeans(n_clusters=self.nclusters)
        kmeans.fit(tfidf_matrix)
        clusters = collections.defaultdict(list)
        for i, label in enumerate(kmeans.labels_):
            clusters[label].append(i)
        self.cluster_visualization_tsne(kmeans, tfidf_matrix)
        return dict(clusters)

    def affinity_clustering(self):
        tfidf_vectorizer = TfidfVectorizer()
        tf_idf_matrix = tfidf_vectorizer.fit_transform(self.sents)
        similarity_matrix = (tf_idf_matrix * tf_idf_matrix.T).A
        affinity_propagation = AffinityPropagation(affinity="precomputed",
                                                   damping=0.5)
        affinity_propagation.fit(similarity_matrix)
        clusters = collections.defaultdict(list)
        for i, label in enumerate(affinity_propagation.labels_):
            clusters[label].append(i)
        return dict(clusters)

    def Agglomerative_clustering(self):
        tfidf_vectorizer = TfidfVectorizer()
        tf_idf_matrix = tfidf_vectorizer.fit_transform(self.sents)
        similarity_matrix = (tf_idf_matrix * tf_idf_matrix.T).A
        agglomerativeclustering = AgglomerativeClustering()
        agglomerativeclustering.fit(similarity_matrix)
        clusters = collections.defaultdict(list)
        for i, label in enumerate(agglomerativeclustering.labels_):
            clusters[label].append(i)
        return dict(clusters)




