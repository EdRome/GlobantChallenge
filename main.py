import os
import zipfile
from sklearn.model_selection import train_test_split
from loading import Data_Loading
from processing import XML_Processing, Data_Processing, Data_Engineering
from model import TrainML, TopicModeling

# Unzip 2020.zip file in input_data folder
with zipfile.ZipFile('./input_data/2020.zip', 'r') as zip_ref:
  zip_ref.extractall('./input_data')

# list all XML inside the folder
xml_list_files = os.listdir('./input_data')

# Split the data into training and testing
train_list, test_list = train_test_split(xml_list_files, test_size=0.33, shuffle=True)

# Create batches on training list to make it more easy to manipulate
batch_gen = Data_Loading.CreateTrainingBatches(train_list)
it = batch_gen.create_batches()

# Process each XML and store the content in pandas DataFrame
xml_pr = XML_Processing.XMLProcessor(it, './input_data')
df1 = xml_pr.process()

# Clean the Dataframe, removing stopword and lemmatizing.
processor = Data_Processing.DataFrameProcessing(df1)
df1 = processor.clean_df()

# Create the TF-IDF matrix
engineer = Data_Engineering.Prepare(df1)
tfidf = engineer.create_tfidf()

# Train the ML and predict the labels
trainer = TrainML(tfidf)
trainer.train_kmeans()
topics = trainer.predict()

# For each label, get the most relevant words out of each topic
topic_modeling = TopicModeling.Topic(topics, df1)
topic_modeling.model()
print("Topics in text are:\n",topic_modeling.get_topics())