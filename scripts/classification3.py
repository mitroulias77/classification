'''
https://medium.com/@hjhuney/implementing-a-random-forest-classification-model-in-python-583891c99652

ΥΛΟΠΟΙΗΣΗ RANDOM FOREST KATΗΓΟΡΙΟΠΟΙΗΤΗ
'''

import re
import json
from os import path
import nltk
import pandas as pd
from pandas import DataFrame,np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import seaborn as sns
from IPython.display import display
from nltk.corpus import stopwords
import greek_stemmer as gr_stemm
from classification.utils import remove_emphasis

file = path.join('data', 'nsk_scrape.xlsx')
xl = pd.ExcelFile(file)
df = xl.parse('Sheet1')
df.head()

corpus = []
STOPWORDS = set(stopwords.words('greek'))

print(df.shape[0])

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

X_stem = pd.DataFrame(corpus, columns=['Concultatory'])

X_stem.head()

result = X_stem.join(df[['Title']['Year']['Status']])
result.groupby(['Status']).size()
result.head()
result.columns = ['Concultatory','Title','Year','Status']
tfidf = TfidfVectorizer()
tfidf.fit(result['Concultatory'])

X = tfidf.transform(result['Concultatory'])
y = df['Status']


from sklearn.model_selection import train_test_split
# Δημιουργία train και test συνόλου
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=66)

#Δημιουργία μοντέλου Random Forest
from sklearn import model_selection
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
# Προβλέψεις κατηγοριοποιητή
rfc_predict = rfc.predict(X_test)

#Αξιολογηση του μοντέλου

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc')

#Tuning Hyperparameters
from sklearn.model_selection import RandomizedSearchCV
# number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# number of features at every split
max_features = ['auto', 'sqrt']

# max depth
max_depth = [int(x) for x in np.linspace(100, 500, num = 11)]
max_depth.append(None)
# create random grid
random_grid = {
 'n_estimators': n_estimators,
 'max_features': max_features,
 'max_depth': max_depth
 }
# Random search of parameters
rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the model
rfc_random.fit(X_train, y_train)
# print results
print(rfc_random.best_params_)


rfc = RandomForestClassifier(n_estimators=1200, max_depth=300, max_features='sqrt')
rfc.fit(X_train,y_train)
rfc_predict = rfc.predict(X_test)
rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc')
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rfc_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, rfc_predict))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())


