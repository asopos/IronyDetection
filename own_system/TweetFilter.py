import pandas as pd
from nltk.tokenize import TweetTokenizer
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords



data_path_training='../benchmark_system/SemEval2018-T3-train-taskA.txt'
trainingData = pd.read_csv(data_path_training, sep='\t')

data_path_test='../datasets/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt'
testData = pd.read_csv(data_path_test, sep='\t')
trainingData.drop('Tweet index', axis=1, inplace=True)
testData.drop('Tweet index', axis=1, inplace=True)


arrayTraing = trainingData.values
X_training = arrayTraing[:, 1]
Y_training = arrayTraing[:, 0].astype(int)

arrayTest = testData.values
X_test = arrayTest[:, 1]
Y_test = arrayTest[:, 0].astype(int)