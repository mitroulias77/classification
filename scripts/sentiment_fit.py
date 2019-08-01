import re
import matplotlib.pyplot as plt
import nltk
import view as view
from nltk.corpus import stopwords
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from pandas import np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from nltk import PorterStemmer
import seaborn as sns
import spacy
nlp = spacy.load("el_core_web_sm")

doc = nlp(u'Ονομάζομαι Δημήτρης Μητρούλιας και κατοικώ στην Ελλάδα')

#from nsk import model
file = nlp('nsk_all_2.xlsx')
xl = pd.ExcelFile(file)
df = xl.parse('decisions')
df.head()

nltk.download('stopwords')
corpus = []
#view.encoding()
from greek_stemmer import GreekStemmer
for i in range(0, 1000):
    subject = re.sub(r"[-,()/@\'?\.$%_+\d]", '', df['Subject'][i],flags=re.I)
    subject = subject.lower()
    subject = subject.split()
    #stemmer = GreekStemmer()
    #subject = [stemmer.stem(word) for word in subject if not word in set(stopwords.words('greek'))]
    subject = ' '.join(subject)
    corpus.append(subject)

corpus=pd.DataFrame(corpus, columns=['Subject'])
corpus.head()

result = corpus.join(df[['State']])
result.groupby(['State']).size()

result.head()
result.columns = ['Subject','Score']
result.info()

tfidf = TfidfVectorizer()
tfidf.fit(result['Subject'])

X = tfidf.transform(result['Subject'])
result['Subject'][1]

#print([X[1, tfidf.vocabulary_['δημόσιας']]])
#print([X[1, tfidf.vocabulary_['κατάταξη']]])
#print([X[1, tfidf.vocabulary_['βαθμό']]])

#Sentiment Classification

result.dropna(inplace=True)
result[result['Score'] != 3]
result['Positivity'] = np.where(result['Score'] >= 2, 1, 0)
cols = [ 'Score']
result.drop(cols, axis=1, inplace=True)
result.head()

result.groupby('Positivity').size()

X = result.Subject
y = result.Positivity
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
print("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(X_train),
                                                                             (len(X_train[y_train == 0]) / (len(X_train)*1.))*100,
                                                                        (len(X_train[y_train == 1]) / (len(X_train)*1.))*100))
print("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(X_test),
                                                                             (len(X_test[y_test == 0]) / (len(X_test)*1.))*100,
                                                                            (len(X_test[y_test == 1]) / (len(X_test)*1.))*100))
def accuracy_summary(pipeline, X_train, y_train, X_test, y_test):
    sentiment_fit = pipeline.fit(X_train, y_train)
    y_pred = sentiment_fit.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy score: {0:.2f}%".format(accuracy*100))
    return accuracy

cv = CountVectorizer()
rf = RandomForestClassifier(class_weight="balanced")
n_features = np.arange(10000,25001,5000)
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
        print("Test result for {} features".format(n))
        nfeature_accuracy = accuracy_summary(checker_pipeline, X_train, y_train, X_test, y_test)
        result.append((n,nfeature_accuracy))
    return result
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
print("Result for trigram with stop words (Tfidf)\n")
feature_result_tgt = nfeature_accuracy_checker(vectorizer=tfidf,ngram_range=(1, 3))

#Chi-Squared for Feature Selection
'''from sklearn.feature_selection import chi2
tfidf = TfidfVectorizer(max_features=30000,ngram_range=(1, 3))
X_tfidf = tfidf.fit_transform(result.Subject)
y = result.Positivity
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
plt.show();
'''
'''
df1['cat_id'] = df1['Category'].factorize()[0]
cat_id_df = df1[['Category', 'cat_id']].drop_duplicates().sort_values('cat_id')
cat_to_id = dict(cat_id_df.values)
id_to_cat = dict(cat_id_df[['cat_id', 'Category']].values)
df1.head()
'''

#LSTM FRAMEWORK
max_fatures = 80000
tokenizer = Tokenizer(nb_words=max_fatures, split=' ')
tokenizer.fit_on_texts(result['Subject'].values)
X1 = tokenizer.texts_to_sequences(result['Subject'].values)
X1 = pad_sequences(X1)
Y1 = pd.get_dummies(result['Positivity']).values
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1,Y1, random_state = 42)
print(X1_train.shape,Y1_train.shape)
print(X1_test.shape,Y1_test.shape)
embed_dim = 350
lstm_out = 500

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X1.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'mean_squared_error', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


#Train and evaluate our model

batch_size = 32
model.fit(X1_train, Y1_train, epochs = 15, batch_size=batch_size, verbose = 2)

score,acc = model.evaluate(X1_test, Y1_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
for x in range(len(X1_test)):

    result = model.predict(X1_test[x].reshape(1, X1_test.shape[1]), batch_size=1, verbose=2)[0]

    if np.argmax(result) == np.argmax(Y1_test[x]):
        if np.argmax(Y1_test[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1

    if np.argmax(Y1_test[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1

print("pos_acc", pos_correct / pos_cnt * 100, "%")
print("neg_acc", neg_correct / neg_cnt * 100, "%")
