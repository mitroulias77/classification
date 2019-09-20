from os import path
import nltk
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import f1_score,precision_score,recall_score
from datetime import datetime


file = path.join('data', 'nsk_multiclass.xlsx')
xl = pd.ExcelFile(file)
df = xl.parse('Sheet1')
df.head()
df.describe()
#df.sort_values('Concultatory', inplace=True)

corpus = []
stop_words = nltk.corpus.stopwords.words('greek')
newStopWords = ['του','της','και','την','η','των','το','να', 'από', 'με', 'που', 'δεν', 'για', 'ν', 'σε','ή','α','αυτού',
                'όπως','αυτό','όμως','στούς','υπό','άνω','πλειοψ','κατ','αυτής','όχι','γ','οποίες','ούτε','οποίο','οποιο',
                'αυτές','πριν','από']
stop_words.extend(newStopWords)

for i in range(0, df.shape[0]):
    text = re.sub (r"\d+", '', df['Concultatory'][i], flags=re.I)
    text = re.sub ('[!@#$]', '', text)
    text = re.sub (r"[-,()/@\'?\.$%_+\d]", '', text, flags=re.I)
    text = text.split()
    text = [x.upper() for x in text]
    text = [word.strip() for word in text if word.strip() not in stop_words and len(word)>2]
    text = [x.lower() for x in text]
    text = " ".join(text)
    corpus.append(text)

decisions =pd.DataFrame(corpus, columns=['Cοncultatory'])
df['Concultatory'] = decisions

start = datetime.now()
df['lemma_count'] = df['Category'].apply(lambda text : len(str(text).split(",")))
df.lemma_count.value_counts()

lemma_data = pd.DataFrame(df['Category'])
#lemma_data = pd.DataFrame(df['Category'])

#Εκτύπωση τυχαίων δειγμάτων γνωμοδοτήσεων
decision_1 = df['Concultatory'].values[0]
print(decision_1)
print("\nΛήμματα: {}".format(df['Category'].values[0]))
print("="*215)

decision_2 = df['Concultatory'].values[1000]
print(decision_2)
print("\nCategory: {}".format(df['Category'].values[1000]))
print("="*215)

decision_3 = df['Concultatory'].values[1500]
print(decision_3)
print("\nCategory: {}".format(df['Category'].values[1500]))
print("="*215)

decision_4 = df['Concultatory'].values[4900]
print(decision_4)
print("\nΛήμματα: {}".format(df['Category'].values[4900]))
print("="*215)


#Συνολικός Αριθμός Μοναδικών Λημματών
def tokenize(x):
    x=x.split(',')
    lemmata=[i.strip() for i in x] #μερικά λήμματα περιέχουν κενά πρίν
    return lemmata

vectorizer = CountVectorizer(tokenizer = tokenize)
lemma_dtm = vectorizer.fit_transform(lemma_data['Category'])

print("Σύνολο Γνωμοδοτήσεων :", lemma_dtm.shape[0])
print("Μοναδικά Λήμματα :", lemma_dtm.shape[1])

#'get_feature_name()' μας επιστρέφει λεξικό.
lemmas = vectorizer.get_feature_names()
print("Τα μοναδικά λήμματα είναι τα εξής :\n\n", lemmas[:1922])
#Αριθμός εμφάνισης λημμάτων
#https://stackoverflow.com/questions/15115765/how-to-access-sparse-matrix-elements
#Αποθηκεύουμε τα λήμματα από το dotmatrix σε λεξικό
freqs = lemma_dtm.sum(axis=0).A1 #axis=0 στήλες. Που περιέχουν το πλήθος εμφάνισης των λημμάτων
result = dict(zip(lemmas, freqs))

lemma_df = pd.DataFrame ({'Lemma': lemmas, 'Counts': freqs})
#Ταξινόμηση ληματων σύμφωνα με τη συχνότητα εμφάνισης
lemma_df_sorted = lemma_df.sort_values(['Counts'], ascending=False)
lemma_counts = lemma_df_sorted['Counts'].values
#υπάλληλοι δημοσιοι , διορισμος πρόσλησψη, αρμοδιότητα, εταιρίες ανώνυμες, μεταταξη ...
#είναι τα πέντε λήμματα
lemma_df_sorted.head(10)
#o πίνακας lemma_counts απαρριθμεί τα λήμματα σε όλο το σύνολο
lemma_counts
#Κατανομή των λημμάτων
plt.figure(figsize=(12, 6))
plt.plot(lemma_counts)
plt.title("Σύνολο Λημμάτων: Κατανομή του πλήθους εμφάνισης σε κάθε γνωμοδότηση")
plt.grid()
plt.xlabel("Αριθμoί Λημμάτων. (Σύνολο 1922)")
plt.ylabel("Πλήθος εμφάνισης στις Γνωμοδοτήσεις")
plt.show()
#Στατιστική πληροφορία για τα λήμματα
lemma_df_sorted.describe()
'''
α.75% των λημμάτων εμφανίζεται λιγότερο από 26 φορές σε διαφορετικές γνωμ/τησεις
β.25% των λημμάτων εμφανίζεται λιγότερο από 3 φορές σε διαφορετικές γνωμ/τησεις
γ.Ο Μέγιστος αριθμός εμφάνισης ενός λήμματος είναι σε 598 φιαφορετικές γνωμ/τησεις
'''
#Quantile
plt.figure(figsize=(5, 8))
sns.boxplot(data = lemma_df_sorted)
plt.xlabel("Αριθμός Λημμάτων")
plt.ylabel("Αριθμός Γνωμοδοτήσεων")

#πάνω από 10
list_lemmas_grt_thn_10 = lemma_df_sorted[lemma_df_sorted.Counts>10].Lemma
#Εκτύπωση Λίστας
print ('{} Λήμματα εφανίζονται σε πάνω από 10 Γνωμοδοτήσεις'.format(len(list_lemmas_grt_thn_10)))

#πάνω από 50
list_lemmas_grt_thn_50 = lemma_df_sorted[lemma_df_sorted.Counts>50].Lemma
#Εκτύπωση Λίστας
print ('{} Λήμματα εφανίζονται σε πάνω από 50 Γνωμοδοτήσεις'.format(len(list_lemmas_grt_thn_50)))

#πάνω από 100
list_lemmas_grt_thn_100 = lemma_df_sorted[lemma_df_sorted.Counts>100].Lemma
#Εκτύπωση Λίστας
print ('{} Λήμματα εφανίζονται σε πάνω από 100 Γνωμοδοτήσεις'.format(len(list_lemmas_grt_thn_100)))

#πάνω από 200
list_lemmas_grt_thn_200 = lemma_df_sorted[lemma_df_sorted.Counts>200].Lemma
#Εκτύπωση Λίστας
print ('{} Λήμματα εφανίζονται σε πάνω από 200 Γνωμοδοτήσεις'.format(len(list_lemmas_grt_thn_200)))

#πάνω από 400
list_lemmas_grt_thn_500 = lemma_df_sorted[lemma_df_sorted.Counts>500].Lemma
#Εκτύπωση Λίστας
print ('{} Λήμματα εφανίζονται σε πάνω από 500 Γνωμοδοτήσεις'.format(len(list_lemmas_grt_thn_500)))

#Λήμμα με την συχνότερη εμφάνιση
print("Λήμμα(Κατηγορία) με τη συχνότερη εμφάνιση: {}".format(lemma_df_sorted.iloc[0][0]))
print("Το λήμμα [{}]: εμφανίζεται {} φορές".format(lemma_df_sorted.iloc[0][0],lemma_counts[0]))

#Λήμμα ανά Γνωμοδότηση
#Αποθήκευση του πλήθους λημμάτων για κάθε γνωμοδότηση στη λίστα 'lemma_count'
lemma_decision_count = lemma_dtm.sum(axis=1).tolist()

#Μετατροπή κάθε τιμής στο 'lemma_decision_count' σε int
lemma_decision_count=[int(j) for i in lemma_decision_count for j in i]
print ('Συνολικά έχουμε {} εγγραφές.'.format(len(lemma_decision_count)))
print(lemma_decision_count[:50])

print("Μέγιστος Αριθμός Λημμάτων ανά Γνωμοδόηση: %d"%max(lemma_decision_count))
print("Ελάχιστος Αριθμός Λημμάτων ανά Γνωμοδόηση: %d"%min(lemma_decision_count))
print("M.O. Αριθμός Λημμάτων ανά Γνωμοδόηση: %f"% ((sum(lemma_decision_count)*1.0)/len(lemma_decision_count)))

#Πόσες Γνωμοδοτήσεις έχουν μέχρι 3 λήματα
lemma_greater_than_avg_count = list(filter(lambda x: x<=3, lemma_decision_count))
len(lemma_greater_than_avg_count)

#Πόσες Γνωμοδοτήσεις έχουν μέχρι 4 λήματα
lemma_greater_than_avg_count = list(filter(lambda x: x<=4, lemma_decision_count))
len(lemma_greater_than_avg_count)

#Πόσες Γνωμοδοτήσεις έχουν μέχρι 5 λήματα
lemma_greater_than_avg_count = list(filter(lambda x: x<=5, lemma_decision_count))
len(lemma_greater_than_avg_count)

#Πόσες Γνωμοδοτήσεις έχουν μέχρι 6 λήματα
lemma_greater_than_avg_count = list(filter(lambda x: x<=6, lemma_decision_count))
len(lemma_greater_than_avg_count)

#Πόσες Γνωμοδοτήσεις έχουν μέχρι 7 λήματα
lemma_greater_than_avg_count = list(filter(lambda x: x<=7, lemma_decision_count))
len(lemma_greater_than_avg_count)

#Ιστόγραμμα κατανομής λημμάτων
plt.figure(figsize=(10,5))
sns.countplot(lemma_decision_count, palette = 'gist_rainbow')
plt.title("Κατανoμή λημμάτων ανά Γνωμοδότηση")
plt.xlabel("Λήμματα")
plt.ylabel("Γνωμοδοτήσεις")
plt.show()

#Εφαρμογή Word Cloud για τα πιο συχνά εμφανιζόμενα λήμματα γνωμοδοτήσεων
# Σχεδίαση από Word Cloud
#Μετατροπή 'result' λεξικού σε tuple
tup = dict(result.items())

#Αρχικοποίηση του WordCloud χρησιμοποιώντας συχνότητες εμφάνισης λημμάτων.
wordcloud = WordCloud(background_color='black',width=1600,height=800,).generate_from_frequencies(tup)

fig = plt.figure(figsize=(15,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
fig.savefig("data/tag.png")
plt.show()
#Κατανομή των συχνά εμφανιζόμενων λημμάτων από τη συχνότητά τους
i=np.arange(50)
lemma_df_sorted.head(50).plot(kind='bar', figsize=(15,10), rot=90, color='red')
plt.title('Συχνότητα εμφάνισης των top 50 ')
plt.xticks(i, lemma_df_sorted['Lemma'])
plt.xlabel('Λήμμα')
plt.ylabel('Αριθμός εμφάνισης')
plt.show()
'''
i=np.arange(1500)
lemma_df_sorted.tail(1500).plot(kind='bar', figsize=(15,10), rot=90, color='blue')
plt.title('Συχνότητα εμφάνισης των 1500 λιγότερο σημαντικών λημμάτων')
plt.xticks(i, lemma_df_sorted['Lemma'][-1500])
plt.xlabel('Λήμμα')
plt.ylabel('Αριθμός εμφάνισης')
plt.show()
'''
#Ελάχιστα Χρησιμοποιούμενα Λήμματα
print("Το 75% των λήμμάτων εφανίζεται ελάχιστα: \n")
print(list(lemma_df_sorted['Lemma'][-1440:]))
lemma_data_lower = pd.DataFrame([x.lower() for x in lemma_data.Category])
lemma_data_lower['Category'] = lemma_data_lower
del lemma_data_lower[0]

lemma_data.head()
lemma_data_lower.head()
#####################

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from datetime import datetime
# Elbow Μέθοδος για τον προσδιορισμό της καλύτερης τιμής του Κ στην ομαδοποίηση K-Means.
t_start = datetime.now()
def plot_elbow(sumOfSquaredErrors, n_clusters, vectorizationType):
    '''Αυτή η συνάρτηση χρησιμοποιείται για να σχεδιάσει την elbow καμπύλη για το άθροισμα των τετραγωνικών σφαλμάτων
    έναντι των τιμών των ομάδων και να αποκτήσει τη βέλτιστη
     τιμή της παραμέτρου Κ'''

    k_values = n_clusters
    loss = sumOfSquaredErrors

    # Plot K_Values vs Loss Values
    plt.figure (figsize=(15, 8))
    plt.plot (k_values, loss, color='red', linestyle='dashed', linewidth=5, marker='o', markerfacecolor='blue',
              markersize=10)
    for xy in zip (k_values, np.round (loss, 3)):
        plt.annotate ('(%s, %s)' % xy, xy=xy, textcoords='data')
    plt.title ('K vs Loss για {} μοντέλο'.format (vectorizationType))
    plt.xlabel ('Αριθμός Cluster')
    plt.ylabel ('Loss (Άθροισμα Τετραγώνων Σφαλμάτων)')
    plt.show ( )

    optimal_k = input ("Επέλεξε τον αριθμό ομάδων από την elbow καμπύλη και πάτα enter : ")
    print ("Επιλέχτηκε το {}".format (optimal_k))

    return optimal_k

# Function to perform KMeans Clustering.
def KMeansPlusPlus(lemmas_vectors):
    '''Αυτή η συνάρτηση χρησιμοποιείται για κλήσεις πολλαπλών μεθόδων οι οποίες θα καθορίζουν τη βέλτιστη τιμή του k.
    Υπολογίζεται η απώλεια και η βέλτιστη τιμή για κάθε ομάδα.
    Ο αριθμός των συστάδων αποκτάται με οπτική εξέταση της επιφάνειας της καμπύλης.
    Στο τέλος ο αλγόριθμος k-μέσου θα τρέξει με την καλύτερη τιμή του Κ που επιλέγεται από την καμπύλη'''
    t_start = datetime.now( )
    sumOfSquaredErrors = []
    n_clusters = range (1, 15)
    k_means = [KMeans (n_clusters=i, n_init=5, init='k-means++', n_jobs=8, random_state=0) for i in
               n_clusters]  # algorithm = elkan for dense data data, default: algorithm = auto
    k_means_centroids = [k_mean.fit (lemmas_vectors) for k_mean in k_means]
    sumOfSquaredErrors = [k_mean.inertia_ for k_mean in
                          k_means_centroids]  # Inertia: Sum of distances of samples to their closest cluster center
    optimal_k = int (plot_elbow (sumOfSquaredErrors, n_clusters, "BOW"))

    # Run k-medoids with the optimal number of clusters obtained from the elbow method
    kmeans = KMeans (n_clusters=optimal_k, init='k-means++', algorithm='auto', n_jobs=8, random_state=0).fit (
        lemmas_vectors)
    print ("Χρόνος για υλοποίηση K-Means clustering στα λήμματα...: ", datetime.now ( ) - t_start)

    return kmeans, optimal_k


# Function to draw word clouds for each clusters.
from wordcloud import WordCloud
def word_clouds(kmeans_object, lemmas_corpus):
    # Labels of each data point
    labels = kmeans_object.labels_
    clusters_dict = {i: np.where (labels == i) [0] for i in range (optimal_k)}
    # Transform this dictionary into list (if you need a list as result)
    clusters_list = []
    print ("Αριθμός datapoints σε κάθε cluster είναι ο εξής : ")
    for key, value in clusters_dict.items():
        temp = [key, value]
        clusters_list.append (temp)
        print ("Cluster = {}, Αριθμός από data points = {}".format (key + 1, len (value)))

    from wordcloud import WordCloud
    for cluster_number in range (optimal_k - 2):
        cluster = [clusters_dict [cluster_number] [i] for i in range (clusters_dict [cluster_number].size)]

        reviews_cluster = []
        for i in cluster:
            reviews_cluster.append (lemmas_corpus [i])

        review_corpus = ""
        for review in reviews_cluster:
            review_corpus = review_corpus + " " + review

        # lower max_font_size
        wordcloud = WordCloud (width=800, height=450, margin=2, prefer_horizontal=0.9, scale=1, max_words=75,
                               min_font_size=4, random_state=42, background_color='black',
                               contour_color='black', repeat=False).generate (str (review_corpus))
        plt.figure (figsize=(16, 9))
        plt.title ("Word Cloud για τα Clusters {}".format (cluster_number + 1))
        plt.imshow (wordcloud, interpolation="bilinear")
        plt.axis ("off")
        plt.show()


#####################
#Εξάγωγή όλων των λημμάτων
lemmas_corpus=lemma_data_lower['Category'].apply(lambda x: str(x)) #Aποφυγή προβλημάτων κωδικοποίησης στα λήμματα
cv_object = CountVectorizer(tokenizer = tokenize).fit(lemmas_corpus) #Αρχικοποιήση BOW constructor
lemmas_vectors = cv_object.transform(lemmas_corpus) #Δημιούργία BOW vectors για όλα τα λήμματα
kmeans_object, optimal_k = KMeansPlusPlus(lemmas_vectors) #KMeans++ αλγόριθμος συνάρτηση κλήσης ,εξαγωγή kmeans αντικειμένου και βέλτιστος αριθμός συστάδων

word_clouds(kmeans_object, lemmas_corpus)

##########################################################################
def clean_decisions(text):
    text = re.sub(r"\d+", '', text)
    text = re.sub(r"[-,()/@\'?\.$%_+\d]", '', text)
    #αφαιρεση κενών διαστημάτων
    text = ''.join(text)
    text = text.lower ( )
    return text

df['Clean_Concultatories'] = df['Concultatory'].apply(lambda k : clean_decisions(k))

def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

df['Clean_Concultatories'] = df['Clean_Concultatories'].apply(lambda k : remove_stopwords(k))


lemma_data_lower['Lemmata']=[x.split(',') for x in lemma_data_lower['Category']]
from sklearn.preprocessing import MultiLabelBinarizer
y =np.array([np.array(x) for x in lemma_data_lower['Lemmata'].values.tolist()])
multilabel_binarizer = MultiLabelBinarizer()
y_1 = multilabel_binarizer.fit_transform(y)
labels = multilabel_binarizer.classes_

#Χωρισμός Συνόλου (ΣΕΚ και ΣΕΛ)
#ML Μοντέλο με OneVsRest Ταξινομητή

X_train, X_test, y_train, y_test = train_test_split(df['Concultatory'], df['Category'], test_size=0.1,random_state=0)
print("Αριθμός ΣΕΚ Γνωμοδοτήσεων: ",X_train.shape[0])
print("Αριθμός ΣΕΛ Γνωμοδοτήσεων: ",X_test.shape[0])

#Μετατροπή Λημμάτων σε binary vectors

#Εισαγωγή-Αρχικοποίηση "CountVectorizer" αντικειμένου, το οποίο είναι
# scikit-learn's ΒΟW εργαλείο. Το 'split()' κάνει tokenize αρχικά κάθε λήμμα με χρήση του κενού διαστήματος.

# binary='true' δίνει δυαδικό vectorizer
vectorizer = CountVectorizer(tokenizer = tokenize, binary='true').fit(y_train)
y_train_multilabel = vectorizer.transform(y_train)
y_test_multilabel = vectorizer.transform(y_test)

#Εισαγωγή Χαρακτηριστικών στα δεδομένα  TF-IDF vectorizer (1-grams)
vectorizer = TfidfVectorizer(min_df=0.00009, smooth_idf=True, norm="l2", tokenizer = lambda x: x.split(" "),
                             sublinear_tf=False, ngram_range=(1,1))
X_train_multilabel = vectorizer.fit_transform(X_train)
X_test_multilabel = vectorizer.transform(X_test)

print("Διάσταση ΣΕΚ για το περιεχόμενο(Χαρακτηριστικά) των Γνωμοδοτήσεων:",X_train_multilabel.shape, "Y :",y_train_multilabel.shape)
print("Διάσταση ΣΕΚ για το περιεχόμενο(Χαρακτηριστικά) των Γνωμοδοτήσεων:",X_test_multilabel.shape,"Y:",y_test_multilabel.shape)

'''Εφαρμογή Λογιστικής Παλινδρόμησης με OneVsRest ταξινομητή'''
from sklearn.linear_model import LogisticRegression, SGDClassifier

start = datetime.now()

classifier1 = OneVsRestClassifier (LogisticRegression (penalty='l1', class_weight='balanced'), n_jobs=-1)
classifier1.fit (X_train_multilabel, y_train_multilabel)
predictions = classifier1.predict (X_test_multilabel)

print ("Ακρίβεια :", metrics.accuracy_score (y_test_multilabel, predictions))
print ("Hamming loss ", metrics.hamming_loss (y_test_multilabel, predictions))

precision = precision_score (y_test_multilabel, predictions, average='micro')
recall = recall_score (y_test_multilabel, predictions, average='micro')
f1 = f1_score (y_test_multilabel, predictions, average='micro')

print ("\nΜικρό-Μέσος όρος")
print ("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format (precision, recall, f1))

precision = precision_score (y_test_multilabel, predictions, average='macro')
recall = recall_score (y_test_multilabel, predictions, average='macro')
f1 = f1_score (y_test_multilabel, predictions, average='macro')

print ("\nΜάκρο-Μέσος όρος")
print ("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format (precision, recall, f1))

print ("\nΑναφορά Κατηγοριοποίησης")
print (metrics.classification_report (y_test_multilabel, predictions))
print ("Χρόνος Εκτέλεσης για τον Κατηγοριοποιητή 1 :", datetime.now ( ) - start)



from sklearn.externals import joblib
joblib.dump(classifier1, 'data/logisticR_clf1.pkl')

'''Εφαρμογή Λογιστικής Παλινδρόμησης με συνδυασμό OneVsRest και SGD ταξινομητή'''

start = datetime.now ( )

classifier2 = OneVsRestClassifier(SGDClassifier(loss='hinge',penalty='l1', class_weight='balanced'), n_jobs=-1)
classifier2.fit (X_train_multilabel, y_train_multilabel)
predictions = classifier1.predict (X_test_multilabel)

print ("Ακρίβεια :", metrics.accuracy_score (y_test_multilabel, predictions))
print ("Hamming loss ", metrics.hamming_loss (y_test_multilabel, predictions))

precision = precision_score (y_test_multilabel, predictions, average='micro')
recall = recall_score (y_test_multilabel, predictions, average='micro')
f1 = f1_score (y_test_multilabel, predictions, average='micro')

print ("\nΜικρό-Μέσος όρος")
print ("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format (precision, recall, f1))

precision = precision_score (y_test_multilabel, predictions, average='macro')
recall = recall_score (y_test_multilabel, predictions, average='macro')
f1 = f1_score (y_test_multilabel, predictions, average='macro')

print ("\nΜάκρο-Μέσος όρος")
print ("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format (precision, recall, f1))

print ("\nΑναφορά Κατηγοριοποίησης")
print (metrics.classification_report (y_test_multilabel, predictions))
print ("Χρόνος Εκτέλεσης για τον Κατηγοριοποιητή 2 :", datetime.now ( ) - start)
joblib.dump(classifier2, 'data/onevsrest_with_sgd_clf2_hinge_loss_bigrams.pkl')

#Featurizing data with TfIdf vectorizer (1-2 Grams)
start = datetime.now()
vectorizer = TfidfVectorizer(min_df=0.00009, smooth_idf=True, norm="l2", tokenizer = lambda x: x.split(" "), sublinear_tf=False, ngram_range=(1,2))
X_train_multilabel = vectorizer.fit_transform(X_train)
X_test_multilabel = vectorizer.transform(X_test)

print("Time taken to run this cell :", datetime.now() - start)

print("Διάσταση ΣΕΚ για το περιεχόμενο(Χαρακτηριστικά) των Γνωμοδοτήσεων:",X_train_multilabel.shape, "Y :",y_train_multilabel.shape)
print("Διάσταση ΣΕΚ για το περιεχόμενο(Χαρακτηριστικά) των Γνωμοδοτήσεων:",X_test_multilabel.shape,"Y:",y_test_multilabel.shape)

'''Εξαγωγή καταλληλότερου Εκτιμητή'''

from sklearn.model_selection import RandomizedSearchCV
from scipy import stats

st=datetime.now()

alpha = [0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
penalty=['l1','l2']

params  = {"estimator__C":alpha,
           "estimator__penalty":penalty}

base_estimator = OneVsRestClassifier(LogisticRegression(class_weight='balanced'), n_jobs=-1)
rsearch_cv = RandomizedSearchCV(estimator=base_estimator, param_distributions=params, n_iter=10, cv=5, scoring='f1_micro', n_jobs=-1, verbose=0)
rsearch_cv.fit(X_train_multilabel, y_train_multilabel)

print("Ο χρόνος που απαιτείται για την εκτέλεση της ρύθμισης υπερπαραμέτρων: ",datetime.now()-st)
print("Καλύτερος εκτιμητής: ",rsearch_cv.best_estimator_)
print("Βέλτιστη Βαθμολογία για Cross Validation : ",rsearch_cv.best_score_)

'''Προσαρμογή Εκτιμητή στα Δεδομένα'''

start = datetime.now ( )

classifier = rsearch_cv.best_estimator_
classifier.fit (X_train_multilabel, y_train_multilabel)
predictions = classifier.predict (X_test_multilabel)

print ("Ακρίβεια :", metrics.accuracy_score (y_test_multilabel, predictions))
print ("Hamming loss ", metrics.hamming_loss (y_test_multilabel, predictions))

precision = precision_score (y_test_multilabel, predictions, average='micro')
recall = recall_score (y_test_multilabel, predictions, average='micro')
f1 = f1_score (y_test_multilabel, predictions, average='micro')

print ("\nΜικρό-Μέσος όρος")
print ("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format (precision, recall, f1))

precision = precision_score (y_test_multilabel, predictions, average='macro')
recall = recall_score (y_test_multilabel, predictions, average='macro')
f1 = f1_score (y_test_multilabel, predictions, average='macro')

print ("\nΜάκρο-Μέσος όρος")
print ("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format (precision, recall, f1))

print ("\nΑναφορά Κατηγοριοποίησης")
print (metrics.classification_report (y_test_multilabel, predictions))
print ("Χρόνος Εκτέλεσης για τον Κατηγοριοποιητή :", datetime.now ( ) - start)
joblib.dump(classifier, 'data/LogisticR_tfidf_hyp_tuned_1gram.pkl')

#TFIDF με 1-2 Grams


start = datetime.now()

#Χρήση tf-idf vectorizer για να γίνουν διανύσματα οι γνωμοδοτήσεις
vectorizer = TfidfVectorizer(min_df=0.00009, max_features=100000, smooth_idf=True, norm="l2", tokenizer = lambda x: x.split(" "), sublinear_tf=False, ngram_range=(1,2))
X_train_multilabel = vectorizer.fit_transform(X_train)
X_test_multilabel = vectorizer.transform(X_test)

#Μετατροπή λημμάτων σε δυαδική κωδικοποίηση με χρήση sklearns Countvectorizer
vectorizer = CountVectorizer(tokenizer = tokenize, binary='true').fit(y_train)
y_train_multilabel = vectorizer.transform(y_train)
y_test_multilabel = vectorizer.transform(y_test)

print("Χρόνος Εκτέλεσης :", datetime.now() - start)

'''Εξαγωγή καταλληλότερου Εκτιμητή'''
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats

st=datetime.now()

alpha = [0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
penalty=['l1','l2']

params  = {"estimator__C":alpha,
           "estimator__penalty":penalty}

base_estimator = OneVsRestClassifier(LogisticRegression(class_weight='balanced'), n_jobs=-1)
rsearch_cv = RandomizedSearchCV(estimator=base_estimator, param_distributions=params, n_iter=10, cv=5, scoring='f1_micro', n_jobs=-1, verbose=0)
rsearch_cv.fit(X_train_multilabel, y_train_multilabel)

print("Ο χρόνος που απαιτείται για την εκτέλεση της ρύθμισης υπερπαραμέτρων: ",datetime.now()-st)
print("Καλύτερος εκτιμητής: ",rsearch_cv.best_estimator_)
print("Βέλτιστη Βαθμολογία για Cross Validation : ",rsearch_cv.best_score_)
#Βέλτιστη Βαθμολογία για Cross Validation :  0.47648317660981054


'''Τοποθέτηση-Προσαρμογή καλύτερου εκτιμητή στα δεδομένα'''
start = datetime.now ( )

classifier = rsearch_cv.best_estimator_
classifier.fit (X_train_multilabel, y_train_multilabel)
predictions = classifier.predict (X_test_multilabel)

print ("Ακρίβεια :", metrics.accuracy_score (y_test_multilabel, predictions))
print ("Hamming loss ", metrics.hamming_loss (y_test_multilabel, predictions))

precision = precision_score (y_test_multilabel, predictions, average='micro')
recall = recall_score (y_test_multilabel, predictions, average='micro')
f1 = f1_score (y_test_multilabel, predictions, average='micro')

print ("\nΜικρό-Μέσος όρος")
print ("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format (precision, recall, f1))

precision = precision_score (y_test_multilabel, predictions, average='macro')
recall = recall_score (y_test_multilabel, predictions, average='macro')
f1 = f1_score (y_test_multilabel, predictions, average='macro')

print ("\nΜάκρο-Μέσος όρος")
print ("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format (precision, recall, f1))

print ("\nΑναφορά Κατηγοριοποίησης")
print (metrics.classification_report (y_test_multilabel, predictions))
print ("Χρόνος Εκτέλεσης για τον Κατηγοριοποιητή :", datetime.now ( ) - start)
joblib.dump(classifier, 'data/logisticR_ovr_tfidf_hyp_tuned_1_2.pkl')

#Μέσος όρος αριθμού λημμάτων ~5

vectorizer = CountVectorizer(tokenizer = tokenize, binary='true', max_features=5).fit(y_train)
y_train_multilabel = vectorizer.transform(y_train)
y_test_multilabel = vectorizer.transform(y_test)

'''Δημιουργία Διανυσμάτων στις γνωμοδοτήσεις με TFIDF Unigrams'''
start = datetime.now()
vectorizer = TfidfVectorizer(min_df=0.00009, max_features=100000, smooth_idf=True, norm="l2", tokenizer = lambda x: x.split(" "), sublinear_tf=False, ngram_range=(1,1))
X_train_multilabel = vectorizer.fit_transform(X_train)
X_test_multilabel = vectorizer.transform(X_test)

print("Χρόνος Εκτέλεσης Δημιουργίας Διανυσμάτων :", datetime.now() - start)

'''Εξαγωγή καταλληλότερου Εκτιμητή'''
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats

st=datetime.now()

#alpha = [0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
alpha=stats.uniform(0,1000)
penalty=['l1','l2']

params  = {"estimator__C":alpha,
           "estimator__penalty":penalty}

base_estimator = OneVsRestClassifier(LogisticRegression(class_weight='balanced'), n_jobs=-1)
rsearch_cv = RandomizedSearchCV(estimator=base_estimator, param_distributions=params, n_iter=10, cv=5, scoring='f1_micro', n_jobs=-1, verbose=0)
rsearch_cv.fit(X_train_multilabel, y_train_multilabel)

print("Ο χρόνος που απαιτείται για την εκτέλεση της ρύθμισης υπερπαραμέτρων: ",datetime.now()-st)
print("Καλύτερος εκτιμητής: ",rsearch_cv.best_estimator_)
print("Βέλτιστη Βαθμολογία για Cross Validation : ",rsearch_cv.best_score_)
#Βέλτιστη Βαθμολογία για Cross Validation :  0.5644844881254699
'''Τοποθέτηση-Προσαρμογή καλύτερου εκτιμητή στα δεδομένα'''
start = datetime.now ( )

classifier = rsearch_cv.best_estimator_
classifier.fit (X_train_multilabel, y_train_multilabel)
predictions = classifier.predict (X_test_multilabel)

print ("Ακρίβεια :", metrics.accuracy_score (y_test_multilabel, predictions))
print ("Hamming loss ", metrics.hamming_loss (y_test_multilabel, predictions))

precision = precision_score (y_test_multilabel, predictions, average='micro')
recall = recall_score (y_test_multilabel, predictions, average='micro')
f1 = f1_score (y_test_multilabel, predictions, average='micro')

print ("\nΜικρό-Μέσος όρος")
print ("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format (precision, recall, f1))

precision = precision_score (y_test_multilabel, predictions, average='macro')
recall = recall_score (y_test_multilabel, predictions, average='macro')
f1 = f1_score (y_test_multilabel, predictions, average='macro')

print ("\nΜάκρο-Μέσος όρος")
print ("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format (precision, recall, f1))

print ("\nΑναφορά Κατηγοριοποίησης")
print (metrics.classification_report (y_test_multilabel, predictions))
print ("Χρόνος Εκτέλεσης για τον Κατηγοριοποιητή :", datetime.now ( ) - start)
from sklearn.externals import joblib
joblib.dump(classifier, 'data/5_lemma_unigram.pkl')

#Λήμματα ~6

vectorizer = CountVectorizer(tokenizer = tokenize, binary='true', max_features=6).fit(y_train)
y_train_multilabel = vectorizer.transform(y_train)
y_test_multilabel = vectorizer.transform(y_test)

'''Δημιουργία Διανυσμάτων στις γνωμοδοτήσεις με TFIDF Unigrams'''
start = datetime.now()
vectorizer = TfidfVectorizer(min_df=0.00009, max_features=100000, smooth_idf=True, norm="l2", tokenizer = lambda x: x.split(" "), sublinear_tf=False, ngram_range=(1,1))
X_train_multilabel = vectorizer.fit_transform(X_train)
X_test_multilabel = vectorizer.transform(X_test)

print("Χρόνος Εκτέλεσης Δημιουργίας Διανυσμάτων :", datetime.now() - start)

'''Εξαγωγή καταλληλότερου Εκτιμητή'''
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats

st=datetime.now()

#alpha = [0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
alpha=stats.uniform(0,1000)
penalty=['l1','l2']

params  = {"estimator__C":alpha,
           "estimator__penalty":penalty}

base_estimator = OneVsRestClassifier(LogisticRegression(class_weight='balanced'), n_jobs=-1)
rsearch_cv = RandomizedSearchCV(estimator=base_estimator, param_distributions=params, n_iter=10, cv=5, scoring='f1_micro', n_jobs=-1, verbose=0)
rsearch_cv.fit(X_train_multilabel, y_train_multilabel)

print("Ο χρόνος που απαιτείται για την εκτέλεση της ρύθμισης υπερπαραμέτρων: ",datetime.now()-st)
print("Καλύτερος εκτιμητής: ",rsearch_cv.best_estimator_)
print("Βέλτιστη Βαθμολογία για Cross Validation : ",rsearch_cv.best_score_)

#Βέλτιστη Βαθμολογία για Cross Validation :  0.5639008027186905
'''Τοποθέτηση-Προσαρμογή καλύτερου εκτιμητή στα δεδομένα'''
start = datetime.now ( )

classifier = rsearch_cv.best_estimator_
classifier.fit (X_train_multilabel, y_train_multilabel)
predictions = classifier.predict (X_test_multilabel)

print ("Ακρίβεια :", metrics.accuracy_score (y_test_multilabel, predictions))
print ("Hamming loss ", metrics.hamming_loss (y_test_multilabel, predictions))

precision = precision_score (y_test_multilabel, predictions, average='micro')
recall = recall_score (y_test_multilabel, predictions, average='micro')
f1 = f1_score (y_test_multilabel, predictions, average='micro')

print ("\nΜικρό-Μέσος όρος")
print ("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format (precision, recall, f1))

precision = precision_score (y_test_multilabel, predictions, average='macro')
recall = recall_score (y_test_multilabel, predictions, average='macro')
f1 = f1_score (y_test_multilabel, predictions, average='macro')

print ("\nΜάκρο-Μέσος όρος")
print ("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format (precision, recall, f1))

print ("\nΑναφορά Κατηγοριοποίησης")
print (metrics.classification_report (y_test_multilabel, predictions))
print ("Χρόνος Εκτέλεσης για τον Κατηγοριοποιητή :", datetime.now ( ) - start)
from sklearn.externals import joblib
joblib.dump(classifier, 'data/6_lemma_unigram.pkl')
#Ακρίβεια : 0.798989898989899
#Hamming loss  0.038047138047138045
########################################################

#Λήμματα ~7

vectorizer = CountVectorizer(tokenizer = tokenize, binary='true', max_features=7).fit(y_train)
y_train_multilabel = vectorizer.transform(y_train)
y_test_multilabel = vectorizer.transform(y_test)

'''Δημιουργία Διανυσμάτων στις γνωμοδοτήσεις με TFIDF Unigrams'''
start = datetime.now()
vectorizer = TfidfVectorizer(min_df=0.00009, max_features=100000, smooth_idf=True, norm="l2", tokenizer = lambda x: x.split(" "), sublinear_tf=False, ngram_range=(1,1))
X_train_multilabel = vectorizer.fit_transform(X_train)
X_test_multilabel = vectorizer.transform(X_test)

print("Χρόνος Εκτέλεσης Δημιουργίας Διανυσμάτων :", datetime.now() - start)

'''Εξαγωγή καταλληλότερου Εκτιμητή'''
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats

st=datetime.now()

#alpha = [0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
alpha=stats.uniform(0,1000)
penalty=['l1','l2']

params  = {"estimator__C":alpha,
           "estimator__penalty":penalty}

base_estimator = OneVsRestClassifier(LogisticRegression(class_weight='balanced'), n_jobs=-1)
rsearch_cv = RandomizedSearchCV(estimator=base_estimator, param_distributions=params, n_iter=10, cv=5, scoring='f1_micro', n_jobs=-1, verbose=0)
rsearch_cv.fit(X_train_multilabel, y_train_multilabel)

print("Ο χρόνος που απαιτείται για την εκτέλεση της ρύθμισης υπερπαραμέτρων: ",datetime.now()-st)
print("Καλύτερος εκτιμητής: ",rsearch_cv.best_estimator_)
print("Βέλτιστη Βαθμολογία για Cross Validation : ",rsearch_cv.best_score_)
#Βέλτιστη Βαθμολογία για Cross Validation :  0.5806151344670608

'''Τοποθέτηση-Προσαρμογή καλύτερου εκτιμητή στα δεδομένα'''
start = datetime.now ( )

classifier = rsearch_cv.best_estimator_
classifier.fit (X_train_multilabel, y_train_multilabel)
predictions = classifier.predict (X_test_multilabel)

print ("Ακρίβεια :", metrics.accuracy_score (y_test_multilabel, predictions))
print ("Hamming loss ", metrics.hamming_loss (y_test_multilabel, predictions))

precision = precision_score (y_test_multilabel, predictions, average='micro')
recall = recall_score (y_test_multilabel, predictions, average='micro')
f1 = f1_score (y_test_multilabel, predictions, average='micro')

print ("\nΜικρό-Μέσος όρος")
print ("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format (precision, recall, f1))

precision = precision_score (y_test_multilabel, predictions, average='macro')
recall = recall_score (y_test_multilabel, predictions, average='macro')
f1 = f1_score (y_test_multilabel, predictions, average='macro')

print ("\nΜάκρο-Μέσος όρος")
print ("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format (precision, recall, f1))

print ("\nΑναφορά Κατηγοριοποίησης")
print (metrics.classification_report (y_test_multilabel, predictions))
print ("Χρόνος Εκτέλεσης για τον Κατηγοριοποιητή :", datetime.now ( ) - start)
from sklearn.externals import joblib
joblib.dump(classifier, 'data/7_lemma_unigram.pkl')

##########################
#Διανυσματοποίηση γνωμοδοτήσεων με TFIDF NGrams(1,2)

vectorizer = TfidfVectorizer(min_df=0.00009, max_features=100000, smooth_idf=True, norm="l2", tokenizer = lambda x: x.split(" "), sublinear_tf=False, ngram_range=(1,2))
X_train_multilabel = vectorizer.fit_transform(X_train)
X_test_multilabel = vectorizer.transform(X_test)

print("Χρόνος Εκτέλεσης :", datetime.now() - start)
#Βέλτιστη Βαθμολογία για Cross Validation :  0.5792082537632464

'''Τοποθέτηση-Προσαρμογή καλύτερου εκτιμητή στα δεδομένα'''
start = datetime.now ( )

classifier = rsearch_cv.best_estimator_
classifier.fit (X_train_multilabel, y_train_multilabel)
predictions = classifier.predict (X_test_multilabel)

print ("Ακρίβεια :", metrics.accuracy_score (y_test_multilabel, predictions))
print ("Hamming loss ", metrics.hamming_loss (y_test_multilabel, predictions))

precision = precision_score (y_test_multilabel, predictions, average='micro')
recall = recall_score (y_test_multilabel, predictions, average='micro')
f1 = f1_score (y_test_multilabel, predictions, average='micro')

print ("\nΜικρό-Μέσος όρος")
print ("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format (precision, recall, f1))

precision = precision_score (y_test_multilabel, predictions, average='macro')
recall = recall_score (y_test_multilabel, predictions, average='macro')
f1 = f1_score (y_test_multilabel, predictions, average='macro')

print ("\nΜάκρο-Μέσος όρος")
print ("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format (precision, recall, f1))

print ("\nΑναφορά Κατηγοριοποίησης")
print (metrics.classification_report (y_test_multilabel, predictions))
print ("Χρόνος Εκτέλεσης για τον Κατηγοριοποιητή :", datetime.now ( ) - start)
from sklearn.externals import joblib

from sklearn.externals import joblib
joblib.dump(classifier, '6_lemma_bigram_tfidf.pkl')

