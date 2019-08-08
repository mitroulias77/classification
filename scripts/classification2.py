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

file = path.join('data', 'nsk_all.xlsx')
xl = pd.ExcelFile(file)
df = xl.parse('nsk_prakseis')
df.head()

corpus = []
STOPWORDS = set(stopwords.words('greek'))

print(df.shape[0])

for i in range(0, df.shape[0]):
    subject = re.sub(r"\d+", '', df['Θέμα'][i],flags=re.I)
    subject = re.sub(r"[-,()/@\'?\.$%_+\d]", '', df['Θέμα'][i],flags=re.I)
    stemmer = gr_stemm.GreekStemmer()
    subject = subject.split()
    subject = [remove_emphasis(x) for x in subject]
    subject = [x.upper() for x in subject]
    subject = [stemmer.stem(word) for word in subject if not word in STOPWORDS and len(word)>=3]
    subject = [x.lower() for x in subject]
    subject = " ".join(subject)
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

nsk = nsk.reset_index(drop=True)
print(nsk)
'''
to_add = value_counts[value_counts <=300].index
nsk = nsk[~nsk.Category.isin(to_add)]
nsk = nsk.reset_index(drop=False)
'''

nsk['cat_id'] = nsk['Category'].factorize()[0]
cat_id_df = nsk[['Category', 'cat_id']].drop_duplicates().sort_values('cat_id')
cat_to_id = dict(cat_id_df.values)
id_to_cat = dict(cat_id_df[['cat_id', 'Category']].values)
nsk.head()

fig = plt.figure(figsize=(8,6))
nsk.groupby('Category').Subject.count().plot.bar(ylim=0)
plt.show()

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8', ngram_range=(1, 2), stop_words=STOPWORDS)
nsk.shape
vectorizer = CountVectorizer(analyzer = 'char_wb',
                         tokenizer = None,
                         preprocessor = None,
                         stop_words = STOPWORDS,
                         max_features = 30000)
features = vectorizer.fit_transform(nsk.Subject).toarray()
features = tfidf.fit_transform(nsk.Subject).toarray()
labels = nsk.cat_id
features.shape

N=3
for Category, cat_id in sorted(cat_to_id.items()):
  features_chi2 = chi2(features, labels == cat_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(Category))
  print("  . Unigrams καλύτερης συσχέτισης:\n       . {}".format('\n       . '.join(unigrams[-N:])))
  print("  . Βigrams καλύτερης συσχέτησης:\n       . {}".format('\n       . '.join(bigrams[-N:])))

'''
from sklearn.manifold import TSNE

# Sampling a subset of our dataset because t-SNE is computationally expensive
SAMPLE_SIZE = int(len(features) * 0.3)
np.random.seed(0)
indices = np.random.choice(range(len(features)), size=SAMPLE_SIZE, replace=False)
projected_features = TSNE(n_components=2, random_state=0).fit_transform(features[indices])

colors = ['pink', 'green', 'midnightblue', 'orange']#'cyan', 'darkmagenta', 'red'


for category, category_id in sorted(cat_to_id.items()):
    points = projected_features[(labels[indices] == category_id).values]
    plt.scatter(points[:, 0], points[:, 1], s=30, c=colors[category_id], label=category)
plt.title("tf-idf feature vector για κάθε θέμα, προβολή σε 2 διαστάσεις.",
          fontdict=dict(fontsize=15))
plt.legend()
'''
X_train, X_test, y_train, y_test = train_test_split(nsk['Subject'], nsk['Category'], random_state = 0)
count_vect = CountVectorizer(stop_words=STOPWORDS)
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

#print(clf.predict(vectorizer.transform(["Αν είναι επιτρεπτή κατά νόμο η ανάθεση αυτοδύναμης επίβλεψης διαλέξεων και διπλωματικών εργασιών σε Βοηθούς Διδάκτορες στη Σχολή Αρχιτεκτόνων Μηχανικών."])))

#print(clf.predict(vectorizer.transform(["Σχετικά με το εάν συντρέχει νόμιμη περίπτωση για την έγκριση του προϋπολογισμού του κληρ/τος Γ. Α. Β. ειδικώς ως προς τη δαπάνη καταβολής από το κληρ/μα ασφαλιστικών εισφορών."])))

#nsk[df['']== 'Αν, ενόψει του από την Υπηρεσία διδόμενου πραγματικού, το Νομικό Συμβούλιο του Κράτους, παρέχει, κατ’ άρθρο 66 παρ. 2 π.δ. 284/1988 τη θετική γνωμοδότησή του για το σχέδιο της 9ης Τροποποίησης της Σύμβασης 007Α/1999, η οποία αφορά στην προμήθεια και εγκατάσταση έξι (6) πυραυλικών συστημάτων μεσαίων και μεγάλων αποστάσεων PATRIOT, καθώς και του απαραίτητου εξοπλισμού και των υπηρεσιών εκπαίδευσης.']

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0, solver='lbfgs')
]

CV = 10
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df,
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()

cv_df.groupby('model_name').accuracy.mean()

#######################

model = LinearSVC()#Linear Support Vector Machine Classification.

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, nsk.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=cat_id_df.Category.values, yticklabels=cat_id_df.Category.values)

plt.ylabel('Πραγματικά')
plt.xlabel('Πρόβλεψη')
plt.show()

for predicted in cat_id_df.cat_id:
    for actual in cat_id_df.cat_id:
        if predicted != actual and conf_mat[actual, predicted] >= 0:
            print("'{}' Προβλέφθηκαν στην κατηγορία '{}' : {} παραδείγματα.".format(id_to_cat[actual], id_to_cat[predicted], conf_mat[actual, predicted]))
            display(nsk.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['Category', 'Subject']])
            print('')

model.fit(features, labels)

N = 3
for Category, cat_id in sorted(cat_to_id.items()):
  indices = np.argsort(model.coef_[cat_id])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
  print("# '{}':".format(Category))
  print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
  print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))


texts = ["Εάν η ιδιότητα του εσωτερικού μέλους του Συμβουλίου του Ε.Κ.Π.Α. είναι κατά νόμο συμβατή με την κατοχή της ιδιότητας άλλου μονοπρόσωπου     πανεπιστημιακού οργάνου διοίκησης, όπως του Αναπληρωτή Πρύτανη ( ή και Αντιπρύτανη) ή του Προέδρου Τμήματος του Ε.Κ.Π.Α.",
         "Εάν ο Δημοσιογράφος Τ.Χ. ο οποίος τέθηκε σε αυτοδίκαιη αργία λόγω παραπομπής στο Πειθαρχικό Συμβούλιο για το παράπτωμα της αδικαιολόγητης αποχής από τα καθήκοντά του, δικαιούται αποδοχές αργίας.",
         "Δυνατότητα συμμετοχής εξωτερικών μελών, ήδη συνταξιούχων καθηγητών στην Ειδική Επταμελή Επιτροπή για την εκλογή ή εξέλιξη Καθηγητών ΕΚΠΑ."]

text_features = tfidf.transform(texts)
predictions = model.predict(text_features)
for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Προβλέφθηκε ώς: '{}'".format(id_to_cat[predicted]))
  print("")

from sklearn import metrics
print(metrics.classification_report(y_test, y_pred))
