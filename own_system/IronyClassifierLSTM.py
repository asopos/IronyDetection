import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout


data_path_training='../benchmark_system/SemEval2018-T3-train-taskA.txt'
trainingData = pd.read_csv(data_path_training, sep='\t')

data_path_test='../datasets/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt'
testData = pd.read_csv(data_path_test, sep='\t')
trainingData.drop('Tweet index', axis=1, inplace=True)
testData.drop('Tweet index', axis=1, inplace=True)


arrayTraing = trainingData.values
X_training = arrayTraing[:, 1]
Y_training = arrayTraing[:, 0].astype(int)
max_features = 20


arrayTest = testData.values
X_test = arrayTest[:, 1]
Y_test = arrayTest[:, 0].astype(int)

tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(X_training)
X_training = tokenizer.texts_to_sequences(X_training)
X_training = pad_sequences(X_training)

tokenizer.fit_on_texts(X_test)
X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test)


embed_dim = 128
lstm_out = 196

model = Sequential()


model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer='rmsprop',metrics = ['accuracy'])

batch_size = 32

model.fit(X_training, Y_training, batch_size=batch_size, epochs=0)

score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size)
predicted = model.predict_classes(X_test, batch_size, 0)
print(score)
print(acc)
#predicted = predicted.tolist()
with open("../evaluation/res/predictions-TaskA.txt","w") as f:
    for score in predicted:
        for s in score:
            f.write(str(s) + '\n')



nlp = spacy.load('en')
#x = np.array([])
#tweet ='UHHHH i Looove all the tittis'

features = []
X_training = X_training.tolist()
for tweet in trainingData['Tweet text']:
    nounCounter = 0
    doc = nlp(tweet)
    for token in doc:
        if(token.pos_== 'NOUN'):
            nounCounter = nounCounter + 1
            print(token)
    features.append([nounCounter])


print(features[0])



#training = trainingData['Tweet text'].apply(lambda x: nlp.pipe(x[1]), print())