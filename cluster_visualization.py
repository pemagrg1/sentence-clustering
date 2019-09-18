import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class ClusterVisualization:
    def __init__(self,tfidf_matrix):
        self.tfidf_matrix = tfidf_matrix

    def cluster_visualization_tsne(self):
        tsne_perplexity = 20.0
        tsne_early_exaggeration = 4.0
        tsne_learning_rate = 1000
        random_state = 1
        model = TSNE(n_components=2, random_state=random_state,
                     perplexity=tsne_perplexity,
                     early_exaggeration=tsne_early_exaggeration,
                     learning_rate=tsne_learning_rate,
                     )

        transformed_centroids = model.fit_transform(self.tfidf_matrix.toarray())
        plt.scatter(transformed_centroids[:, 0], transformed_centroids[:, 1],
                    c="g")
        plt.show()