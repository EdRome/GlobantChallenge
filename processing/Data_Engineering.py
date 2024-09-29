from collections import Counter

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

class Prepare:
    
    def __init__(self, df):
       self.__df = df

    def create_tfidf(self):
        # Vectorize the text data using TF-IDF
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.__df['AbstractNarration_clean'])
        tfidf_title_info = np.hstack([tfidf_matrix.toarray(), self.__df.filter(like='_specific').values.astype(int)])

        # Apply PCA to reduce dimensionality
        pca = PCA(n_components=2)
        return pca.fit_transform(tfidf_title_info)