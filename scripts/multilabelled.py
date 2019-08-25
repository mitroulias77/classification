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

file = path.join('data/files', '2016_1.xlsx')
xl = pd.ExcelFile(file)
df = xl.parse('Sheet1')
df.head()

nsk_list= df['Category'].values.tolist()