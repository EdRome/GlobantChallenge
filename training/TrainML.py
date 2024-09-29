from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.metrics import silhouette_score

class ML:

    def __init__(self, tfidf):
        self.__X = tfidf
        self.__num_clusters = 0
        self.__model = None

    def define_clusters(self):
        visualizer = KElbowVisualizer(KMeans(), k=(5,20), metric='silhouette')
        visualizer.fit(self.__X)
        self.__num_clusters = visualizer.elbow_value_

    def train_kmeans(self):
        self.define_clusters()

        kmeans = KMeans(n_clusters=self.__num_clusters)
        kmeans.fit(self.__X)

        score = silhouette_score(self.__X, kmeans.labels_)
        print("Training score: ", score)

        self.__model = kmeans

    def predict(self, X=None):
        if X is not None:
            y = self.__model.predict(X)
        else:
            y = self.__model.predict(self.__X)

        return y