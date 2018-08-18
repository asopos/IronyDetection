import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.pipeline import Pipeline
import sklearn.metrics
from sklearn.model_selection import GridSearchCV
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer().tokenize

data_path_training = '../benchmark_system/SemEval2018-T3-train-taskA.txt'
trainingData = pd.read_csv(data_path_training, sep='\t')
trainingData.drop('Tweet index', axis=1, inplace=True)

data_path_test = '../datasets/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt'
testData = pd.read_csv(data_path_test, sep='\t')
testData.drop('Tweet index', axis=1, inplace=True)

arrayTraining = trainingData.values
X_training = arrayTraining[:, 1]
Y_training = arrayTraining[:, 0].astype(int)

parameters = {'vect__ngram_range': [(1, 1), (1, 3)],
               'vect__use_idf': (True, False)}
              # 'clf__alpha': (1e-2, 1e-3)


arrayTest = testData.values
X_test = arrayTest[:, 1]
Y_test = arrayTest[:, 0].astype(int)

vec = TfidfVectorizer()

svm_clf = svm.LinearSVC()
vec_clf = Pipeline([('vect', vec), ('clf', svm_clf)])

gs_clf = GridSearchCV(vec_clf, parameters, n_jobs=-1)

gs_clf.fit(X_training, Y_training)
#joblib.dump(vec_clf, 'svmClassifier.pkl', compress=3)

predicted = gs_clf.predict(X_test)


print(sklearn.metrics.classification_report(Y_test, predicted,target_names=['no Irony','Irony']))