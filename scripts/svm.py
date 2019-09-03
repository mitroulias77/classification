from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import re
from os import path
import greek_stemmer as gr_stemm
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
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

value_counts = nsk['Category'].value_counts()
to_remove = value_counts[value_counts < 50].index
nsk = nsk[~nsk.Category.isin(to_remove)]
nsk = nsk.reset_index(drop=True)

fig = plt.figure(figsize=(8,6))
nsk.groupby('Category').Concultatory.count().plot.bar(ylim=0)
plt.show()

nsk['cat_id'] = nsk['Category'].factorize()[0]
cat_id_df = nsk[['Category', 'cat_id']].drop_duplicates().sort_values('cat_id')
cat_to_id = dict(cat_id_df.values)
id_to_cat = dict(cat_id_df[['cat_id', 'Category']].values)
nsk.head()

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8', ngram_range=(1, 2))
tfidf.fit(nsk['Concultatory'])

X = tfidf.transform(nsk['Concultatory'])
y = nsk['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 66)


svm_clf = LinearSVC().fit(X_train, y_train)
clf_predict = svm_clf.predict(X_test)

clf_cv_score = cross_val_score(svm_clf, X, y, cv=10, scoring='accuracy')
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, clf_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, clf_predict))
print('\n')
print("=== All AUC Scores ===")
print(clf_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Support Vector Machine: ", clf_cv_score.mean())


