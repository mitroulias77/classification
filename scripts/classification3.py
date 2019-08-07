import itertools
import re
from os import path
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Activation, Dropout, np
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import LabelBinarizer

EMBEDDING_DIM = 100
num_labels = 8
vocab_size = 15000
batch_size = 100
num_epochs = 30


file = path.join('E:\\Python\\classification\\data', 'nsk_all.xlsx')
xl = pd.ExcelFile(file)
df = xl.parse('nsk_prakseis')
df.head()

corpus = []
STOPWORDS = set(stopwords.words('greek'))

for i in range(0, 6756):
    subject = re.sub(r"[,()/@\'?\.$%_+\d]", '', df['Θέμα'][i],flags=re.I)
    subject = subject.lower()
    subject = subject.split()
    ps = nltk.PorterStemmer( )
    subject = [ps.stem(word) for word in subject if not word in STOPWORDS]
    subject = ' '.join(subject)
    corpus.append(subject)

nsk=pd.DataFrame(corpus, columns=['Θέμα'])
nsk.head()

nsk = nsk.join(df[['Τύπος Πράξης','Κατηγορία']])
nsk.groupby(['Κατηγορία']).size()

nsk.columns = ['Subject','Type','Category']

fig = plt.figure(figsize=(8,6))
nsk.groupby('Category').Subject.count().plot.bar(ylim=0)
plt.show()

value_counts = nsk['Category'].value_counts()

to_remove = value_counts[value_counts <= 250].index
# nsk = nsk[~nsk.Category.isin(to_remove)]
for idx, row in nsk.iterrows():
    if row['Category'] in to_remove.tolist():
        nsk.ix[idx, 'Category'] = 'ΔΙΑΦΟΡΑ'

nsk['cat_id'] = nsk['Category'].factorize()[0]
cat_id_df = nsk[['Category', 'cat_id']].drop_duplicates().sort_values('cat_id')
cat_to_id = dict(cat_id_df.values)
id_to_cat = dict(cat_id_df[['cat_id', 'Category']].values)
nsk.head()

fig = plt.figure(figsize=(8,6))
nsk.groupby('Category').Subject.count().plot.bar(ylim=0)
plt.show()

nsk = nsk.reset_index(drop=True)
print(nsk)

# lets take 80% data as training and remaining 20% for test.
train_size = int(len(nsk) * .8)

train_concultatories = nsk['Subject'][:train_size]
train_types = nsk['Type'][:train_size]
train_category = nsk['Category'][:train_size]

test_concultatories = nsk['Subject'][train_size:]
test_types = nsk['Type'][train_size:]
test_category = nsk['Category'][train_size:]

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_concultatories)

X1 = tokenizer.texts_to_sequences(nsk['Subject'].values)
X1 = pad_sequences(X1)

x_train = tokenizer.texts_to_matrix(train_concultatories)
x_test = tokenizer.texts_to_matrix(test_concultatories)

encoder = LabelBinarizer()
encoder.fit(train_category)

y_train = encoder.transform(train_category)
y_test = encoder.transform(test_category)

#Δημιουργία μοντέλου
model = Sequential()
model.add(Dense(512, input_shape=(vocab_size,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

num_epochs =10
batch_size = 128
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    verbose=2,
                    validation_split=0.2)

score, acc = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=2)

print('Test accuracy:', acc)

#another approach using GRU model, takes longer time
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.layers.embeddings import Embedding

EMBEDDING_DIM = 100

print('Build model...')

model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=x_test[i]))
model.add(GRU(units=32,  dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(num_labels, activation='softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Summary of the built model...')
print(model.summary())

text_labels = encoder.classes_

for i in range(10):
    prediction = model.predict(np.array([x_test[i]]))
    predicted_label = text_labels[np.argmax(prediction[0])]
    #print(test_files_names.iloc[i])
    print('Actual label:' + test_category.iloc[i])
    print("Predicted label: " + predicted_label)


