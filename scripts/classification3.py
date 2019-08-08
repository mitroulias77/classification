'''
https://stackabuse.com/text-classification-with-python-and-scikit-learn/
ΥΛΟΠΟΙΗΣΗ RANDOM FOREST KATΗΓΟΡΙΟΠΟΙΗΤΗ
'''
import re
from os import path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
import greek_stemmer as gr_stemm
from classification.utils import remove_emphasis
import pickle
from sklearn.feature_extraction.text import CountVectorizer

file = path.join('data', 'nsk_scrape.xlsx')
xl = pd.ExcelFile(file)
df = xl.parse('Sheet1')
df.head()

corpus = []
STOPWORDS = set(stopwords.words('greek'))

print(df.shape[0])
from nltk.stem import WordNetLemmatizer

stemmer = gr_stemm.GreekStemmer()
lemmetizer = WordNetLemmatizer()
for i in range(0, df.shape[0]):
    # Remove all the special characters
    subject = re.sub(r'\W', ' ', df['Concultatory'][i],flags=re.I)
    # remove all single characters
    subject = re.sub(r'\s+[a-zA-Z]\s+', ' ', subject)
    # Remove single characters from the start
    subject = re.sub(r'\^[a-zA-Z]\s+', ' ' , subject)
    # Substituting multiple spaces with single space
    subject = re.sub(r'\s+',' ', subject)
    # Removing prefixed 'b'
    subject = re.sub(r'^b\s+' , '' , subject)
    # Lemmatization
    subject = subject.split()

    subject = [remove_emphasis(x) for x in subject]
    subject = [x.upper() for x in subject]
    subject = [stemmer.stem(word) for word in subject if not word in STOPWORDS and len(word)>=3]
    subject = [lemmetizer.lemmatize(word) for word in subject]
    subject = " ".join(subject)
    subject = subject.lower()
    corpus.append(subject)

X = pd.DataFrame(corpus, columns=['Concultatory'])

X.head()

'''vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=STOPWORDS)
X = vectorizer.fit_transform(X['Concultatory']).toarray()
'''
y = df['Status']


from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer(max_features=2500, min_df=15, max_df=0.7, stop_words=STOPWORDS)
X = tfidfconverter.fit_transform(X['Concultatory']).toarray()

from sklearn.model_selection import train_test_split
# Δημιουργία train και test συνόλου
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Δημιουργία μοντέλου Random Forest
from sklearn import model_selection
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)
# Προβλέψεις κατηγοριοποιητή
y_pred = classifier.predict(X_test)

#Αξιολογηση του μοντέλου

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

#Saving and Loading the Model
with open('data/text_classifier', 'wb') as picklefile:
    pickle.dump(classifier,picklefile)
#load the model
with open('data/text_classifier', 'rb') as training_model:
    model = pickle.load(training_model)
'''
 Let's predict the sentiment for the test set using our loaded model and 
 see if we can get the same results. 
'''
y_pred2 = model.predict(X_test)

print(confusion_matrix(y_test, y_pred2))
print(classification_report(y_test, y_pred2))
print(accuracy_score(y_test, y_pred2))
