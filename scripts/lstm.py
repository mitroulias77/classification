import re
from os import path
from keras import Sequential, optimizers
from keras.layers import LSTM, Embedding, Dense
from keras.preprocessing.text import Tokenizer
import greek_stemmer as gr_stemm
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from classification.utils import remove_emphasis

file = path.join('data', 'nsk_multiclass.xlsx')
xl = pd.ExcelFile(file)
df = xl.parse('Sheet1')
df.head()

nsk_list = df['Category'].values.tolist()
nsk_list = df['Category'].astype(str)
nsk_list = [x.split(',')[0] for x in nsk_list]

import matplotlib.pyplot as plt
df['Label'] = pd.Series(nsk_list)
'''
STOPWORDS = set(stopwords.words('greek'))
corpus = []

for i in range(0, df.shape[0]):
    subject = re.sub(r"\d+", '', df['Concultatory'][i],flags=re.I)
    subject = re.sub(r"[-,()/@\'?\.$%_+\d]", '', df['Concultatory'][i],flags=re.I)
    stemmer = gr_stemm.GreekStemmer()
    subject = subject.split()
    subject = [remove_emphasis(x) for x in subject]
    subject = [x.upper() for x in subject]
    subject = [stemmer.stem(word) for word in subject if not word in STOPWORDS and len(word)>=3]
    subject = [x.lower() for x in subject]
    subject = " ".join(subject)
    corpus.append(subject)
'''
nsk=pd.DataFrame(df, columns=['Concultatory'])
nsk['Category'] = df['Label']
nsk.head()

value_counts = nsk['Category'].value_counts()
to_remove = value_counts[value_counts < 150].index
nsk = nsk[~nsk.Category.isin(to_remove)]
nsk = nsk.reset_index(drop=True)

fig = plt.figure(figsize=(8,6))
nsk.groupby('Category').Concultatory.count().plot.bar(ylim=0)
plt.show()
#https://androidkt.com/saved-keras-model-to-predict-text-from-scratch/
#lstm

max_features = 30000
tokenizer = Tokenizer(num_words=max_features, filters='!"#$%&amp;()*+,-./:;&lt;=>?@[\]^_`{|}~',lower=True,split=' ')
tokenizer.fit_on_texts(nsk['Concultatory'].values)
X1 = tokenizer.texts_to_sequences(nsk['Concultatory'].values)
X1 = pad_sequences(X1)


Y1 = pd.get_dummies(nsk['Category']).values
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1,Y1, test_size=0.2, random_state = 66)
print(X1_train.shape,Y1_train.shape)
print(X1_test.shape,Y1_test.shape)

voc_size = X1.max()+1

embed_dim = 128
lstm_out = 200
batch_size = 32

model_lstm = Sequential()
model_lstm.add(Embedding(voc_size, embed_dim,input_length = X1.shape[1], dropout = 0.2))
model_lstm.add(LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2))
model_lstm.add(Dense(9,activation='softmax'))
#sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
rms=optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model_lstm.compile(loss = 'binary_crossentropy', optimizer=rms,metrics = ['accuracy'])
print(model_lstm.summary())


model_lstm.fit(X1_train, Y1_train, epochs = 10, batch_size=batch_size, verbose = 2,  validation_data=(X1_test, Y1_test))

y_pred = model_lstm.predict(X1_test)
loss,acc = model_lstm.evaluate(X1_test , Y1_test, verbose = 1, batch_size = batch_size)

#Η Εκτίμηση της loss function ή αλλιώς η συνάρτηση κόστους(όσο πιο μικρό είναι το score , τόσο χαμηλότερο είναι τολάθος πρόβλεψης)
#
print("Test val_loss: %.2f" % (loss))

#Το ποσοστό των προβλέψεων για τις κατηγορίες από το σύνολο ελέγχου
print("Test val_accuracy: %.2f" % (acc))
model_lstm.save('data/concultatories.h5')

