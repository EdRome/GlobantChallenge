import re
import string
import subprocess
from collections import Counter

import nltk
from nltk.stem import WordNetLemmatizer

import numpy as np
import pandas as pd

class DataFrameProcessing:

    def __init__(self, df, download_dir):
        self.__df = df
        self.__fixed_divisions = None
        self.__download_dir = download_dir
        # Download and unzip wordnet
        try:
            nltk.data.find('wordnet.zip')
        except:
            nltk.download('wordnet', download_dir=self.__download_dir)
            command = "unzip " + self.__download_dir + "/corpora/wordnet.zip -d " + self.__download_dir + "/corpora"
            subprocess.run(command.split())
            nltk.data.path.append(self.__download_dir)

        try:
            nltk.data.find('stopwords.zip')
        except:
            nltk.download('stopwords', download_dir=self.__download_dir)
            nltk.data.path.append(self.__download_dir)
            from nltk.corpus import stopwords

        self.__lemmatizer = WordNetLemmatizer()
        self.__stop_words = set(stopwords.words("english"))

    def remove_empty(self):
        self.__df.dropna(subset=['AbstractNarration'], inplace=True)

    def clean_text(self, text):
        text = text.lower().translate(str.maketrans('','',string.punctuation))
        text = ' '.join([self.__lemmatizer.lemmatize(word) for word in text.split() if word not in self.__stop_words])
        return text
    
    def clean_df_division(self):
        dict_div_title = {}

        # get the top divisions
        self.get_top_divisions()


        self.__df = self.__df.assign(
            fixed_division = lambda df: self.fix_division_name(df)
        )

        most_common_award_title = Counter(' '.join(self.__df['AwardTitle_clean']).split()).most_common(20)

        dict_div_title = {}

        for division in self.__df.fixed_division.unique():
            subdivision = self.__df.query(
                "fixed_division == @division"
            )
            
            words_award_title = Counter(' '.join(subdivision['AwardTitle_clean']).split())
            
            most_common_words = words_award_title.most_common(20)
            dict_div_title[division] = [word for word, _ in words_award_title.items() if word not in most_common_words]

        # Loop over the division's title words and get most common words based on divisions.
        for division, words in dict_div_title.items():
            self.__df[f'{division}_specific'] = self.__df['AwardTitle_clean'].apply(lambda x: any(word in x for word in words))

    def clean_df(self):

        self.__df.dropna(subset=['AbstractNarration','AwardTitle','Division'], inplace=True)


        self.__df['AwardTitle_clean'] = self.__df.AwardTitle.apply(self.clean_text)
        self.__df['Division_clean'] = self.__df.Division.apply(self.clean_text)
        self.__df['AbstractNarration_clean'] = self.__df.AbstractNarration.apply(self.clean_text)

        self.clean_df_division()

        return self.__df

    def get_top_divisions(self):
        division_plot = self.__df.Division_clean.value_counts(

        ).reset_index(

        ).assign(
            cumulative_sum = lambda df: df['count'].cumsum(),
            ratio = lambda df: df.cumulative_sum/df['count'].sum()*100
        )

        self.__fixed_divisions = division_plot.query(
            "ratio <= 80"
        )

    def fix_division_name(self, df):
        return np.where(
            df.Division_clean.isin(self.__fixed_divisions.Division_clean),
            df.Division_clean,
            "OTHER"
        )
    
    @staticmethod
    def remove_words(text, words_to_remove):
        # Join words with | to create a regex pattern
        pattern = r'\b(?:' + '|'.join([word for word, _ in words_to_remove]) + r')\b'
        
        # Use regex to replace the pattern with an empty string
        return re.sub(pattern, '', text)