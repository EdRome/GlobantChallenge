from collections import Counter

class Topic:

    def __init__(self, labels, df):
        self.__labels = labels
        self.__df = df

    def modeling(self):
        word_counting = Counter(" ".join(self.__df.AbstractNarration_clean.values).split())
        new_word_counting = {key: val for key, val in word_counting.items() if val >= 880}

        topic_cluster = {}

        for cluster in self.__labels:
            topic_word_counting = Counter(" ".join(self.__df.query("kmeans_cluster == @cluster").AbstractNarration_clean.values).split())
            filtering_word_counting = {key: val for key, val in word_counting.items() if key not in new_word_counting}
            topic_cluster[cluster] = sorted(new_word_counting.items(), key=lambda x: x[1], reverse=True)[:5]

        return topic_cluster