import collections
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
import nltk


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

    def cluster_sentences(self,sentences, nb_of_clusters=5):
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




