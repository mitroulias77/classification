import re
from os import path
from keras import Sequential, optimizers
from keras.layers import LSTM, Embedding, Dense
from keras.preprocessing.text import Tokenizer
import greek_stemmer as gr_stemm
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from classification.utils import remove_emphasis
file = path.join('data', 'nsk_multiclass.xlsx')
xl = pd.ExcelFile(file)
df = xl.parse('Sheet1')
df.head()
df[df.isnull().any(axis=1)]


df = df.fillna({"Category": "ΑΛΛΟ"})
categories = df["Category"].value_counts()
nsk_list = df['Category'].values.tolist()
nsk_series= df['Category'].astype(str)
nsk_listToStr = ' '.join(map(str, nsk_list))
lemmas = nsk_listToStr.split(",")


import re
import string


def countX(lst, x):
    return lst.count (x)

x = 'ΔΙΑΖΥΓΙΟ ΔΗΜΟΣΙΑ ΚΤΗΜΑΤΑ'
print ('{} has occured {} times'.format (x, countX (nsk_list, x)))
