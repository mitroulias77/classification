from itertools import chain
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import numpy as np
import  re
import greek_stemmer as gr_stemm
from os import path

from keras import Sequential, optimizers
from keras.layers import Embedding, LSTM, Dense
from keras_preprocessing.text import Tokenizer
from scipy.sparse import csr_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from classification.utils import remove_emphasis

file = path.join('data', 'nsk_multiclass.xlsx')
xl = pd.ExcelFile(file)
df = xl.parse('Sheet1')
df.head()

nsk_list = df['Category'].tolist()
nsk_list = [(str(x)).split(',') for x in nsk_list]

for idx, lst in enumerate(nsk_list):
    cats = [x.strip() for x in lst]
    df.at[idx, 'Category'] = cats
nsk_list = list(chain.from_iterable(nsk_list))
nsk_list = [x.strip() for x in nsk_list]
nsk_set = set(nsk_list)

categories = (list(nsk_set))

categories.sort()
categories_accumulator = [0]*len(categories)
for index, row in df.iterrows():
    cats = row['Category']
    for cat in cats:
        accumulator_idx = categories.index(cat)
        categories_accumulator[accumulator_idx] +=1

sorted_indexes = sorted(range(len(categories_accumulator)), key=lambda k: categories_accumulator[k], reverse=True)
categories_accumulator.sort(reverse=True)

new_labels = [categories[x] for x in sorted_indexes[:50]]
for idx, row in df.iterrows():
    cats = row['Category']
    new_cats=[]
    for cat in cats:
        if cat in new_labels:
            new_cats.append(cat)
    df.at[idx, 'Category'] = new_cats

y = np.array([np.array(x) for x in df['Category'].values.tolist()])
mlb = MultiLabelBinarizer()
y_1 = mlb.fit_transform(y)
mlb_cats = mlb.classes_

max_features = 30000
tokenizer = Tokenizer(num_words=max_features, filters='!"#$%&amp;()*+,-./:;&lt;=>?@[\]^_`{|}~',lower=True,split=' ')
tokenizer.fit_on_texts(df['Concultatory'].values)
X1 = tokenizer.texts_to_sequences(df['Concultatory'].values)
X1 = pad_sequences(X1)


X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1,y_1, test_size=0.2, random_state = 66)
print(X1_train.shape,Y1_train.shape)
print(X1_test.shape,Y1_test.shape)

voc_size = X1.max()+1
embed_dim = 128
lstm_out = 200
batch_size = 32

model_lstm = Sequential()
model_lstm.add(Embedding(voc_size, embed_dim,input_length = X1.shape[1], dropout = 0.2))
model_lstm.add(LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2))
model_lstm.add(Dense(50,activation='softmax'))
#sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
rms=optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model_lstm.compile(loss = 'binary_crossentropy', optimizer=rms,metrics = ['accuracy'])
print(model_lstm.summary())

model_lstm.fit(X1_train, Y1_train, epochs = 10, batch_size=batch_size, verbose = 2,  validation_data=(X1_test, Y1_test))

y_pred = model_lstm.predict(X1_test)
loss,acc = model_lstm.evaluate(X1_test , Y1_test, verbose = 1, batch_size = batch_size)

print("Test val_loss: %.2f" % (loss))

#Το ποσοστό των προβλέψεων για τις κατηγορίες από το σύνολο ελέγχου
print("Test val_accuracy: %.2f" % (acc))
'''
dataset = []
y = []
ids = []
label_dict = {"word2idx": {}, "idx2word": []}
idx = 0
label_per_cat = df ["Category"].apply (lambda x: str (x).split (","))
for l in [g for d in label_per_cat for g in d]:
    if l in label_dict ["idx2word"]:
        pass
    else:
        label_dict ["idx2word"].append (l)
        label_dict ["word2idx"] [l] = idx
        idx += 1
n_classes = len (label_dict ["idx2word"])
print ("identified {} classes".format (n_classes))


def show_example(idx):
    N_true = int(np.sum(Y1_test[idx]))
    print("Prediction: {}".format("|".join(["{} ({:.3})".format(y_pred[idx][s])
                                            for s in y_pred[idx].argsort()[-N_true:][::-1]])))
'''