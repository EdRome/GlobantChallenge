from collections import Counter
import numpy as np

class Topic:

    def __init__(self, labels, df):
        self.__labels = labels
        self.__df = df

        self.__df['topic'] = labels

        self.__topics = {}

    def model(self):
        topic_cluster = {}

        word_counting = Counter(" ".join(self.__df.AbstractNarration_clean.values).split())
        new_word_counting = {key: val for key, val in word_counting.items() if val >= 880}

        for cluster in np.unique(self.__labels):
            topic_word_counting = Counter(" ".join(self.__df.query("topic == @cluster").AbstractNarration_clean.values).split())
            filtering_word_counting = {key: val for key, val in topic_word_counting.items() if key not in new_word_counting}
            relevant_words = sorted(filtering_word_counting.items(), key=lambda x: x[1], reverse=True)[:5]
            self.__topics[cluster] = [key for key, _ in relevant_words]
        
    def get_topics(self):
        return self.__topics
    
    def get_df(self):
        return self.__df

        