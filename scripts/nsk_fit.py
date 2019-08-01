import re
import string

import matplotlib.pyplot as plt
from nltk import PorterStemmer
from nltk.corpus import stopwords
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Dropout, GRU, Conv1D, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from pandas import np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

file = 'nsk_from_2000.xlsx'
xl = pd.ExcelFile(file)
df = xl.parse('decisions')
df.head()

corpus = []
STOPWORDS = set(stopwords.words('greek'))


for i in range(0, 10157):
    subject = re.sub(r"\d+", '', df['Concultatory'][i],flags=re.I)
    subject = re.sub(r"[-,()/@\'?\.$%_+\d]", '', df['Concultatory'][i],flags=re.I)
    stemmer = PorterStemmer()
    subject = subject.lower().split()
    subject = [stemmer.stem(word) for word in subject if not word in STOPWORDS and len(word)>=3]
    subject = [word for word in subject if word not in STOPWORDS and len(word)>=3]
    subject = " ".join(subject)
    corpus.append(subject)
    #words_ = word_tokenize(subject)

corpus=pd.DataFrame(corpus, columns=['Concultatory'])

corpus.head()

result = corpus.join(df[['Status']])
result.groupby(['Status']).size()

result.head()
result.columns = ['Concultatory','Status']
result.info()

tfidf = TfidfVectorizer()
tfidf.fit(result['Concultatory'])

X = tfidf.transform(result['Concultatory'])
result['Concultatory'][1]

#print([X[1, tfidf.vocabulary_['διοίκησης']]])
#print([X[1, tfidf.vocabulary_['βαθμό']]])
#print([X[1, tfidf.vocabulary_['αποσπάσεως']]])

#Sentiment Classification
#Θετικές 1,2 , Αρνητικές 3,4
'''result.dropna(inplace=True)
#result[result['Score'] != 1]
result['Positivity'] = np.where(result['Status'] < 2, 1, 0)
cols = ['Status']
result.drop(cols, axis=1, inplace=True)
result.head()

result.groupby('Positivity').size()
'''
fig = plt.figure(figsize=(8,6))
result.groupby('Status').Concultatory.count().plot.bar(ylim=0)
plt.show()

X = result.Concultatory
y = result.Status

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
print("Το σύνολο Εκπαίδευσης έχει συνολικά {0} Γνωμοδοτήσεις με {1:.2f}% μη-αποδεκτές, {2:.2f}% αποδεκτές".format(len(X_train),
                                                                             (len(X_train[y_train == 0]) / (len(X_train)*1.))*100,
                                                                        (len(X_train[y_train == 1]) / (len(X_train)*1.))*100))
print("Το σύνολο Ελέγχου έχει συνολικά {0} Γνωμοδοτήσεις με {1:.2f}% μη-αποδεκτές , {2:.2f}% αποδεκτές".format(len(X_test),
                                                                             (len(X_test[y_test == 0]) / (len(X_test)*1.))*100,
                                                                            (len(X_test[y_test == 1]) / (len(X_test)*1.))*100))
'''
result['status_id'] = result['Positivity'].factorize()[0]
status_id_df = result[['Positivity', 'status_id']].drop_duplicates().sort_values('status_id')
status_to_id = dict(status_id_df.values)
id_to_status = dict(status_id_df[['status_id', 'Positivity']].values)
result.head()
'''
def accuracy_summary(pipeline, X_train, y_train, X_test, y_test):
    sentiment_fit = pipeline.fit(X_train, y_train)
    y_pred = sentiment_fit.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\n\nΒαθμολογία Ακρίβειας: {0:.2f}%".format(accuracy*100))
    return accuracy

cv = CountVectorizer()
rf = RandomForestClassifier(class_weight="balanced")
n_features = np.arange(10000,25001,5000)

def nfeature_accuracy_checker(vectorizer=cv, n_features=n_features, stop_words=None, ngram_range=(1, 1), classifier=rf):
    result = []
    print(classifier)
    print("\n")
    for n in n_features:
        vectorizer.set_params(stop_words=STOPWORDS, max_features=n, ngram_range=ngram_range)
        checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        print("Αποτελέσματα Ελέγχου για {} χαρακτηριστικά".format(n))
        nfeature_accuracy = accuracy_summary(checker_pipeline, X_train, y_train, X_test, y_test)
        result.append((n,nfeature_accuracy))
    return result

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()

#print("Αποτελέσματα για 3-gram με stop words (Tfidf)\n")

#feature_result_tgt = nfeature_accuracy_checker(vectorizer=tfidf,ngram_range=(1, 3))

from sklearn.metrics import classification_report

cv = CountVectorizer(max_features=30000,ngram_range=(1, 3))
pipeline = Pipeline([
        ('vectorizer', cv),
        ('classifier', rf)
    ])
#sentiment_fit = pipeline.fit(X_train, y_train)
#y_pred = sentiment_fit.predict(X_test)
#print(classification_report(y_test, y_pred, target_names=['Μη-Αποδεκτές','Αποδεκτές']))


from sklearn.feature_selection import chi2

tfidf = TfidfVectorizer(max_features=30000,ngram_range=(1, 3))

X_tfidf = tfidf.fit_transform(result.Concultatory)

y = result.Status
'''
chi2score = chi2(X_tfidf, y)[0]
plt.figure(figsize=(16,8))
scores = list(zip(tfidf.get_feature_names(), chi2score))
chi2 = sorted(scores, key=lambda x:x[1])
topchi2 = list(zip(*chi2[-20:]))
x = range(len(topchi2[1]))
labels = topchi2[0]
plt.barh(x,topchi2[1], align='center', alpha=0.5)
plt.plot(topchi2[1], x, '-o', markersize=5, alpha=0.8)
plt.yticks(x, labels)
plt.xlabel('$\chi^2$')
plt.show()
'''

#LSTM FRAMEWORK
max_features = 20000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(result['Concultatory'].values)
X1 = tokenizer.texts_to_sequences(result['Concultatory'].values)
X1 = pad_sequences(X1)

Y1 = pd.get_dummies(result['Status']).values
#X1.shape[1]
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1,Y1, test_size= 0.4 ,random_state = 42)
print(X1_train.shape,Y1_train.shape)
print(X1_test.shape,Y1_test.shape)

embed_dim = 100
lstm_out = 200
max_features=20000


model_lstm = Sequential()
model_lstm.add(Embedding(max_features, 100,input_length =X1.shape[1]))
#model_lstm.add(Dropout(0.2))
#model_lstm.add(Conv1D(64, 5, activation='relu'))
#model_lstm.add(MaxPooling1D(pool_size=4))
model_lstm.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model_lstm.add(Dense(4,activation='softmax'))
model_lstm.compile(loss ='mean_squared_error', optimizer='adam',metrics = ['accuracy'])
print(model_lstm.summary())

#Εκπαίδευση  και αξιολόγηση του μοντέλου
batch_size = 32
model_lstm.fit(X1_train,Y1_train,epochs = 30,batch_size=batch_size, verbose = 2, validation_data=(X1_test,Y1_test))

y_pred = model_lstm.predict(X1_test)

loss,acc = model_lstm.evaluate(X1_test, Y1_test, verbose = 1, batch_size = batch_size)

print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))



'''
pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0

for x in range(len(X1_test)):

    result = model_lstm.predict(X1_test[x].reshape(1, X1_test.shape[1]), batch_size=1, verbose=2)[0]

    if np.argmax(result) == np.argmax(Y1_test[x]):
        if np.argmax(Y1_test[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1

    if np.argmax(Y1_test[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1

print("Ακρίβεια Αποδεκτών Γνωμοδοτήσεων", pos_correct / pos_cnt * 100, "%")
print("Ακρίβεια Μη - Αποδεκτών Γνωμοδοτήσεων", neg_correct / neg_cnt * 100, "%")
'''