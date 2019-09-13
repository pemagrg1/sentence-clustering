import collections
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AffinityPropagation



class SentenceClustering:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def word_tokenizer(self,text):
            # tokenizes and stems the text
            tokens = word_tokenize(text)
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(t) for t in tokens if
                      t not in stopwords.words('english')]
            return tokens

    def kmeas_clustering(self,sentences, nb_of_clusters=5):
        tfidf_vectorizer = TfidfVectorizer(tokenizer=self.word_tokenizer,
                                           stop_words=stopwords.words(
                                               'english'),
                                           max_df=0.9,
                                           min_df=0.1,
                                           lowercase=True)
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
        kmeans = KMeans(n_clusters=nb_of_clusters)
        kmeans.fit(tfidf_matrix)
        clusters = collections.defaultdict(list)
        for i, label in enumerate(kmeans.labels_):
            clusters[label].append(i)
        return dict(clusters)

    def affinity_clustering(self,sentences, nb_of_clusters=5):
        tfidf_vectorizer = TfidfVectorizer(tokenizer=self.word_tokenizer,
                                           stop_words=stopwords.words(
                                               'english'),
                                           max_df=0.9,
                                           min_df=0.1,
                                           lowercase=True)
        tf_idf_matrix = tfidf_vectorizer.fit_transform(sentences)
        similarity_matrix = (tf_idf_matrix * tf_idf_matrix.T).A
        affinity_propagation = AffinityPropagation(affinity="precomputed",
                                                   damping=0.5)
        affinity_propagation.fit(similarity_matrix)
        clusters = collections.defaultdict(list)
        for i, label in enumerate(affinity_propagation.labels_):
            clusters[label].append(i)
        return dict(clusters)




