# Download necessary libraries

!pip install stanza
!pip install gensim
!pip install adjustText
from tqdm import tqdm
import os
import re
import pandas as pd
import stanza
stanza.download("ja")
nlp_stanza = stanza.Pipeline(lang="ja", processors="tokenize, pos, lemma, depparse, ner")
import gensim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
import json
from gensim.models import LdaModel
from gensim import corpora

# Create function for preprocessing

def corpus_read (path):
  corpus = []
  for filename in tqdm(os.listdir(path)):
      with open(path + filename, encoding='utf-8') as txt:
          text = txt.read()
          corpus.append(text)
  return corpus

def cleanizer(corpus):
    a1 = ''.join(corpus)
    a1 = re.sub(r'\n', ' ', a1)
    a1 = re.sub(r'[A-Za-z]', '', a1)
    a1 = re.sub(r'\d', '', a1)
    a1 = re.sub(r'\.', '', a1)
    a1 = re.sub(r'\（', '', a1)
    a1 = re.sub(r'\）', '', a1)
    a1 = re.sub(r'…', '', a1)
    a1 = re.sub(r'？', '', a1)
    a1 = re.sub(r'！', '', a1)
    a1 = re.sub(r'「', '', a1)
    a1 = re.sub(r'」', '', a1)
    a1 = re.sub(r'\u3000', '', a1)
    a1 = re.sub(r'\／', '', a1)
    a1 = re.sub(r'\＼', '', a1)
    a1 = re.sub(r'―', '', a1)
    a1 = re.sub(r'『', '', a1)
    a1 = re.sub(r'』', '', a1)
    a1 = re.sub(r'\［\S*\］', '', a1)
    a1 = re.sub(r'＊', '', a1)
    clean_corpus = re.sub(r'※', '', a1)
    return clean_corpus

# Use

corpus = corpus_read('name_of_writers/')

clean_corpus = cleanizer(corpus)
clean_corpus[:100]

# Annotate corpus with stanza and transform result to dataframe

def stanza_to_df(corp_doc):
    list_of_rows = []
    counter = 0
    for sentence in corp_doc.sentences:
        counter += 1
        for word in sentence.words:
            list_of_rows.append([counter, word.id, word.text, word.upos, word.deprel, word.head, sentence.words[word.head-1].text])

    df_sentence = pd.DataFrame(list_of_rows, columns=['sent_id', 'id', 'token', 'pos', 'synt_tag', 'head_id', 'head_tok'])
    return df_sentence

corp_doc = nlp_stanza(clean_corpus)

df = stanza_to_df(corp_doc)
df[:10]

df.info()

df.to_csv('name_of_writer_tagged.csv', index=False)
