import os
from sklearn.model_selection import train_test_split
from loading import Data_Loading
from processing import XML_Processing, Data_Processing, Data_Engineering
from training import TrainML

xml_list_files = os.listdir('/kaggle/input/nsf-research-awards-abstracts')

train_list, test_list = train_test_split(xml_list_files, test_size=0.33, shuffle=True)

batch_gen = Data_Loading.CreateTrainingBatches(train_list)
it = batch_gen.create_batches()

xml_pr = XML_Processing.XMLProcessor(it)
df1 = xml_pr.process()

processor = Data_Processing(df1)
df1 = processor.clean_df()

engineer = Data_Engineering(df1)
tfidf = engineer.create_tfidf()

trainer = TrainML(tfidf)
trainer.train_kmeans()
topics = trainer.predict()

