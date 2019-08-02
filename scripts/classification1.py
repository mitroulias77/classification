import re
from os import path

import matplotlib.pyplot as plt
import nltk
from warnings import simplefilter
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from pandas import np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk import PorterStemmer

file = path.join('E:\\Python\\classification\\data', 'nsk_all.xlsx')
xl = pd.ExcelFile(file)
df = xl.parse('nsk_prakseis')
df.head()

#nltk.download('stopwords')
STOPWORDS = set(stopwords.words('greek'))
corpus = []

for i in range(0, 6756):
    subject = re.sub(r"[,()/@\'?\.$%_+\d]", '', df['Θέμα'][i],flags=re.I)
    subject = subject.lower()
    subject = subject.split()
    ps = PorterStemmer()
    subject = [ps.stem(word) for word in subject if not word in STOPWORDS]
    subject = ' '.join(subject)
    corpus.append(subject)

df1=pd.DataFrame(corpus, columns=['Θέμα'])
df1.head()

df1 = df1.join(df[['Τύπος Πράξης','Κατηγορία']])
df1.groupby(['Κατηγορία']).size()

df1.columns = ['Subject','Type','Category']
'''
value_counts = df1['Category'].value_counts()

to_remove = value_counts[value_counts <= 250].index
df1 = df1[~df1.Category.isin(to_remove)]
df1 = df1.reset_index(drop=True)
print(df1)
'''
value_counts = df1['Category'].value_counts()

to_remove = value_counts[value_counts <= 500].index
# df1 = df1[~df1.Category.isin(to_remove)]
for idx, row in df1.iterrows():
    if row['Category'] in to_remove.tolist():
        df1.ix[idx, 'Category'] = 'ΔΙΑΦΟΡΑ'
df1 = df1.reset_index(drop=True)
print(df1)

fig = plt.figure(figsize=(8,6))
df1.groupby('Category').Subject.count().plot.bar(ylim=0)
plt.show()


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8', ngram_range=(1, 2), stop_words=stopwords.words('greek'))
features = tfidf.fit_transform(df1.Category).toarray()
tfidf.fit(df1['Category'])

X = df1.Subject
y = df1.Category
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

print("\n\nΤο σύνολο Εκπαίδευσης έχει συνολικά {0} θέματα που είναι το {1:.2f}% "
      "του συνόλου των θεμάτων".format(len(X_train),(len(X_train) / (len(X)))*100))
print("\n\nΤο σύνολο Ελέγχου έχει συνολικά {0} θέματα που είναι το {1:.2f}%"
      " του συνόλου των θεμάτων ".format(len(X_test),(len(X_test) / (len(X)))*100))

#Αγνόησε τα future warnings
simplefilter(action='ignore', category=FutureWarning)


def accuracy_summary(pipeline, X_train, y_train, X_test, y_test):
    sentiment_fit = pipeline.fit(X_train, y_train)
    y_pred = sentiment_fit.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Η Ακρίβεια του μοντέλου είναι: {0:.2f}%".format(accuracy*100))
    return accuracy

cv = CountVectorizer()
rf = RandomForestClassifier(class_weight='balanced')
n_features = np.arange(10000,30001,5000)

def nfeature_accuracy_checker(vectorizer=cv, n_features=n_features, stop_words=None, ngram_range=(1, 1), classifier=rf):
    result = []
    print(classifier)
    print("\n")
    for n in n_features:
        vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
        checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        print("Αποτελέσματα ελέγχου για {} χαρακτηριστικά".format(n))
        nfeature_accuracy = accuracy_summary(checker_pipeline, X_train, y_train, X_test, y_test)
        result.append((n,nfeature_accuracy))
    return result

###################################
#tfidf = TfidfVectorizer()
print('\n\n')
print("Αποτελέσματα για trigram με τα stopwords\n")
feature_result_tgt = nfeature_accuracy_checker(vectorizer=tfidf,ngram_range=(1, 3))

cv = CountVectorizer(max_features=30000,ngram_range=(1, 3))
pipeline = Pipeline([
        ('vectorizer', cv),
        ('classifier', rf)
    ])
sentiment_fit = pipeline.fit(X_train, y_train)
y_pred = sentiment_fit.predict(X_test)
print(classification_report(y_test, y_pred))

## K-fold Cross Validation
accuracies = cross_val_score(estimator = pipeline, X= X_train, y = y_train, cv = 5)
print("Ακρίβεια Random Forest Κατηγοριοποιητή: %0.2f (+/- %0.2f)"  % (accuracies.mean(), accuracies.std() * 2))

#X^2 Επιλογή χαρακτηριστικών

tfidf2 = TfidfVectorizer(max_features=30000,ngram_range=(1, 3))
X_tfidf = tfidf.fit_transform(df1.Subject)
y = df1.Category
Xi2score = chi2(X_tfidf, y)[0]

plt.figure(figsize=(8,6))
scores = list(zip(tfidf.get_feature_names(), Xi2score))
chi2 = sorted(scores, key=lambda x:x[1])
topchi2 = list(zip(*chi2[-20:]))
x = range(len(topchi2[1]))
labels = topchi2[0]
plt.barh(x,topchi2[1], align='center', alpha=0.5)
plt.plot(topchi2[1], x, '-o', markersize=5, alpha=0.8)
plt.yticks(x, labels)
plt.xlabel('$\chi^2$')
plt.show()

#lstm
max_fatures = 30000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(df1['Subject'].values)
X1 = tokenizer.texts_to_sequences(df1['Subject'].values)
X1 = pad_sequences(X1)

Y1 = pd.get_dummies(df1['Category']).values
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1,Y1, random_state = 42)
print(X1_train.shape,Y1_train.shape)
print(X1_test.shape,Y1_test.shape)
embed_dim = 150
lstm_out = 200
model_lstm = Sequential()
model_lstm.add(Embedding(max_fatures, embed_dim,input_length = X1.shape[1]))
model_lstm.add(SpatialDropout1D(0.2))
model_lstm.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model_lstm.add(Dense(5,activation='softmax'))
model_lstm.compile(loss = 'mean_squared_error', optimizer='adam',metrics = ['accuracy'])
print(model_lstm.summary())

batch_size = 32
model_lstm.fit(X1_train, Y1_train, epochs = 5, batch_size=batch_size, verbose = 2,  validation_data=(X1_test,Y1_test))

y_pred = model_lstm.predict(X1_test)
loss,acc = model_lstm.evaluate(X1_test , Y1_test, verbose = 1, batch_size = batch_size)


#Η Εκτίμηση της loss function ή αλλιώς η συνάρτηση κόστους(όσο πιο μικρό είναι το score , τόσο χαμηλότερο είναι τολάθος πρόβλεψης)
#
print("Test val_loss: %.2f" % (loss))

#Το ποσοστό των προβλέψεων για τις κατηγορίες από το σύνολο ελέγχου
print("Test val_accuracy: %.2f" % (acc))