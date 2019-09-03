import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
stop_words = set(stopwords.words('greek'))
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import seaborn as sns

file = path.join('data', 'nsk_multiclass.xlsx')
xl = pd.ExcelFile(file)
df = xl.parse('Sheet1')
df.head()

nsk_list = df['Category'].values.tolist()
nsk_list = df['Category'].astype(str)


