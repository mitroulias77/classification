import re
import os
from os import path
from keras.preprocessing.text import Tokenizer
from warnings import simplefilter
import greek_stemmer as gr_stemm
import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer , CountVectorizer
from sklearn.model_selection import train_test_split , cross_val_score
from classification.utils import remove_emphasis

file = path.join('data', 'nsk_multiclass.xlsx')
xl = pd.ExcelFile(file)
df = xl.parse('Sheet1')
df.head()

nsk_list= df['Category'].values.tolist()
nsk_list = df['Category'].astype(str)
nsk_list = [x.split(',')[0] for x in nsk_list]

import matplotlib.pyplot as plt
df['Label'] = pd.Series(nsk_list)

corpus = []
STOPWORDS = set(stopwords.words('greek'))

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

nsk=pd.DataFrame(corpus, columns=['Concultatory'])
nsk['Category'] = df['Label']
nsk.head()
#test

#fname_path = os.path.join('data','multi.xlsx')
#nsk.to_excel(fname_path, index=False)

value_counts = nsk['Category'].value_counts()
to_remove = value_counts[value_counts < 50].index
nsk = nsk[~nsk.Category.isin(to_remove)]
nsk = nsk.reset_index(drop=True)

fig = plt.figure(figsize=(8,6))
nsk.groupby('Category').Concultatory.count().plot.bar(ylim=0)
plt.show()

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8', ngram_range=(1, 2))
tfidf.fit(nsk['Concultatory'])

X = tfidf.transform(nsk['Concultatory'])
y = nsk['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 66)

#Δημιουργία μοντέλου Random Forest

#Αξιολογηση του μοντέλου

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix


rfc = RandomForestClassifier(n_estimators=1200, max_depth=100, max_features='sqrt')
rfc.fit(X_train,y_train)
rfc_predict = rfc.predict(X_test)
rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='accuracy')

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
