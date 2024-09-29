import os
from sklearn.model_selection import train_test_split
from loading import Data_Loading
from processing import XML_Processing, Data_Processing, Data_Engineering
from model import TrainML, TopicModeling

xml_list_files = os.listdir('/content/input_data')

train_list, test_list = train_test_split(xml_list_files, test_size=0.33, shuffle=True)

batch_gen = Data_Loading.CreateTrainingBatches(train_list)
it = batch_gen.create_batches()

xml_pr = XML_Processing.XMLProcessor(it, '/content/input_data')
df1 = xml_pr.process()

processor = Data_Processing.DataFrameProcessing(df1)
df1 = processor.clean_df()

engineer = Data_Engineering.Prepare(df1)
tfidf = engineer.create_tfidf()

trainer = TrainML(tfidf)
trainer.train_kmeans()
topics = trainer.predict()

topic_modeling = TopicModeling.Topic(topics, df1)
topic_modeling.model()
print("Topics in text are:\n",topic_modeling.get_topics())