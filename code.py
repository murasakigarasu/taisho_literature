# Download necessary libraries

!pip install stanza
!pip install gensim
from tqdm import tqdm
import os
import re
import pandas as pd
from collections import Counter
import stanza
stanza.download("ja")
nlp_stanza = stanza.Pipeline(lang="ja", processors="tokenize, pos, lemma, depparse, ner")
import gensim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier

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

# Example on multiple writers (can be varied for one or many person)

df_0 = pd.read_csv('/akutagawa_tagged.csv')
df_1 = pd.read_csv('/hagiwara_tagged.csv')
df_2 = pd.read_csv('/hiroshi_tagged.csv')
df_3 = pd.read_csv('/izumi_tagged.csv')
df_4 = pd.read_csv('/kikuti_tagged.csv')
df_5 = pd.read_csv('/miyamoto_tagged.csv')
df_6 = pd.read_csv('/miyazawa_tagged.csv')
df_7 = pd.read_csv('/tanizaki_tagged.csv')
df_8 = pd.read_csv('/tayama_tagged.csv')
df_9 = pd.read_csv('/tokutomi_tagged.csv')
df_10 = pd.read_csv('/yosano_tagged.csv')

combined_df = pd.concat([df_0, df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8, df_9, df_10])
print(combined_df.head())
print(combined_df.info())

df_clean = combined_df.loc[(combined_df['synt_tag'] != 'punct')]
df_clean.info()

# Analyzing colocate

def skipgrammer(tokens, window_size):
    skip_grams = []
    for index in range(len(tokens)):
        target_word = tokens[index]
        start = max(0, index - window_size)
        end = min(len(tokens), index + window_size + 1)

        for s_index in range(start, end):
            if index != s_index:
                context = tokens[s_index]
                skip_grams.append((target_word, context))
    return skip_grams

tokens = df_clean.get('token')
tokens_list = tokens.to_list()
print(len(set(tokens_list)))

skip_grams_w = skipgrammer(tokens_list, 5)

target = '女'
target_skipgrams = []
for skipgram in skip_grams_w:
    if skipgram[0] == target or skipgram[1] == target:
        target_skipgrams.append(skipgram)

target_skipgram_counts = Counter(target_skipgrams)
print(target_skipgram_counts.most_common())

# Take only noun

df_clean_n = df_clean.loc[(df_clean['pos'] == 'NOUN')]
df_clean_n.info()

tokens_n = df_clean_n.get('token')
tokens_list_n = tokens_n.to_list()
print(len(set(tokens_list_n)))

skip_grams_w_n = skipgrammer(tokens_list_n, 5)

target = '女'
target_skipgrams = []
for skipgram in skip_grams_w_n:
    if skipgram[0] == target or skipgram[1] == target:
        target_skipgrams.append(skipgram)

target_skipgram_counts = Counter(target_skipgrams)
print(target_skipgram_counts.most_common())

# Create vector model 

grouped_tokens = df_clean.groupby('sent_id')['token'].apply(list)
sentences = grouped_tokens.tolist()

w2v = gensim.models.Word2Vec(sentences, vector_size=300, window=5, min_count=2, sg=0, epochs=5)

w2v.wv.save_word2vec_format('taisho.bin', binary=True)

print('Для корпуса: ', w2v.wv.most_similar('愛', topn=10))
print("---")
print('Для корпуса: ', w2v.wv.most_similar('恋', topn=10))
print("---")
print('Для корпуса: ', w2v.wv.most_similar('女', topn=10))

# Prepare data for binary classification

taisho_speech = pd.read_csv("/taisho_speech.csv")

taisho_speech.info()

taisho_speech.head()

# Visualize the data

plt.hist(taisho_speech['gender'], bins = 2)
plt.title('gender of speakers')
plt.xlabel('gender')
plt.ylabel('Number of replics')

plt.hist(taisho_speech['author'], bins = 6)
plt.title('author')
plt.xlabel('author')

# Begin

le = LabelEncoder()
taisho_speech['gender_encoded'] = le.fit_transform(taisho_speech['gender'])

X = taisho_speech['text']
y = taisho_speech['gender_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set size (X_train):", X_train.shape[0])
print("Testing set size (X_test):", X_test.shape[0])
print("Training set gender distribution:\n", y_train.value_counts(normalize=True))
print("Testing set gender distribution:\n", y_test.value_counts(normalize=True))

tf_idf = TfidfVectorizer()
X_train_tfidf = tf_idf.fit_transform(X_train)
X_test_tfidf = tf_idf.transform(X_test)

print("Shape of X_train_tfidf:", X_train_tfidf.shape)
print("Shape of X_test_tfidf:", X_test_tfidf.shape)

# Logistic regression

model = LogisticRegression(random_state=42, solver='liblinear')
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# KNN

knn_model = KNeighborsClassifier(n_neighbors=5) # Using 5 neighbors as a common starting point
knn_model.fit(X_train_tfidf, y_train)

y_pred_knn = knn_model.predict(X_test_tfidf)

print("KNN Classification Report:")
print(classification_report(y_test, y_pred_knn))

cm_knn = confusion_matrix(y_test, y_pred_knn, labels=knn_model.classes_)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=le.classes_)
disp_knn.plot(cmap=plt.cm.Blues)
plt.title("KNN Confusion Matrix")
plt.show()

# SVM

svm_model = LinearSVC(random_state=42, dual=False) # dual=False is recommended for n_samples > n_features
svm_model.fit(X_train_tfidf, y_train)

y_pred_svm = svm_model.predict(X_test_tfidf)

print("SVM (LinearSVC) Classification Report:")
print(classification_report(y_test, y_pred_svm))

cm_svm = confusion_matrix(y_test, y_pred_svm, labels=svm_model.classes_)

# Naive Bayes

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

y_pred_nb = nb_model.predict(X_test_tfidf)

print("Naive Bayes (MultinomialNB) Classification Report:")
print(classification_report(y_test, y_pred_nb))

cm_nb = confusion_matrix(y_test, y_pred_nb, labels=nb_model.classes_)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=le.classes_)
disp_nb.plot(cmap=plt.cm.Blues)
plt.title("Naive Bayes (MultinomialNB) Confusion Matrix")
plt.show()
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=le.classes_)
disp_svm.plot(cmap=plt.cm.Blues)
plt.title("SVM (LinearSVC) Confusion Matrix")
plt.show()

# Decision Tree

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_tfidf, y_train)

y_pred_dt = dt_model.predict(X_test_tfidf)

print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))

cm_dt = confusion_matrix(y_test, y_pred_dt, labels=dt_model.classes_)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=le.classes_)
disp_dt.plot(cmap=plt.cm.Blues)
plt.title("Decision Tree Confusion Matrix")
plt.show()

# Bagging decision tree

bagging_model = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), random_state=42)
bagging_model.fit(X_train_tfidf, y_train)

y_pred_bagging = bagging_model.predict(X_test_tfidf)

print("Bagging Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_bagging))

cm_bagging = confusion_matrix(y_test, y_pred_bagging, labels=bagging_model.classes_)
disp_bagging = ConfusionMatrixDisplay(confusion_matrix=cm_bagging, display_labels=le.classes_)
disp_bagging.plot(cmap=plt.cm.Blues)
plt.title("Bagging Decision Tree Confusion Matrix")
plt.show()

# Gradient boosting

gbc_model = GradientBoostingClassifier(random_state=42)
gbc_model.fit(X_train_tfidf, y_train)

y_pred_gbc = gbc_model.predict(X_test_tfidf)

print("Gradient Boosting Classifier Report:")
print(classification_report(y_test, y_pred_gbc))

cm_gbc = confusion_matrix(y_test, y_pred_gbc, labels=gbc_model.classes_)
disp_gbc = ConfusionMatrixDisplay(confusion_matrix=cm_gbc, display_labels=le.classes_)
disp_gbc.plot(cmap=plt.cm.Blues)
plt.title("Gradient Boosting Classifier Confusion Matrix")
plt.show()

# Random Forest

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_tfidf, y_train)

y_pred_rf = rf_model.predict(X_test_tfidf)

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf, labels=rf_model.classes_)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=le.classes_)
disp_rf.plot(cmap=plt.cm.Blues)
plt.title("Random Forest Confusion Matrix")
plt.show()

# Voting

estimators = [
    ('lr', LogisticRegression(random_state=42, solver='liblinear')),
    ('mnb', MultinomialNB()),
    ('svc', LinearSVC(random_state=42, dual=False))
]

voting_model = VotingClassifier(estimators=estimators, voting='hard') # You can change to 'soft' if models support predict_proba
voting_model.fit(X_train_tfidf, y_train)

y_pred_voting = voting_model.predict(X_test_tfidf)

print("Voting Classifier Classification Report:")
print(classification_report(y_test, y_pred_voting))

cm_voting = confusion_matrix(y_test, y_pred_voting, labels=voting_model.classes_)
disp_voting = ConfusionMatrixDisplay(confusion_matrix=cm_voting, display_labels=le.classes_)
disp_voting.plot(cmap=plt.cm.Blues)
plt.title("Voting Classifier Confusion Matrix")
plt.show()

# Neural Network

mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1)
mlp_model.fit(X_train_tfidf, y_train)

y_pred_mlp = mlp_model.predict(X_test_tfidf)

print("Neural Network (MLP) Classification Report:")
print(classification_report(y_test, y_pred_mlp))

cm_mlp = confusion_matrix(y_test, y_pred_mlp, labels=mlp_model.classes_)
disp_mlp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp, display_labels=le.classes_)
disp_mlp.plot(cmap=plt.cm.Blues)
plt.title("Neural Network (MLP) Confusion Matrix")
plt.show()

# Clean dataset

taisho_speech['text'] = taisho_speech['text'].str.replace('わ。', '。')
taisho_speech['text'] = taisho_speech['text'].str.replace('よ。', '。')
taisho_speech['text'] = taisho_speech['text'].str.replace('ワ。', '。')
taisho_speech['text'] = taisho_speech['text'].str.replace('ヨ。', '。')
taisho_speech['text'] = taisho_speech['text'].str.replace('わ…', '…')
taisho_speech['text'] = taisho_speech['text'].str.replace('よ…', '…')

taisho_speech.head()

taisho_speech_new = taisho_speech

# New Binary Classification

le = LabelEncoder()
taisho_speech_new['gender_encoded'] = le.fit_transform(taisho_speech_new['gender'])

X = taisho_speech_new['text']
y = taisho_speech_new['gender_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set size (X_train):", X_train.shape[0])
print("Testing set size (X_test):", X_test.shape[0])
print("Training set gender distribution:\n", y_train.value_counts(normalize=True))
print("Testing set gender distribution:\n", y_test.value_counts(normalize=True))

tf_idf = TfidfVectorizer()
X_train_tfidf = tf_idf.fit_transform(X_train)
X_test_tfidf = tf_idf.transform(X_test)

# Logistic Regression

y_pred = model.predict(X_test_tfidf)

print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# KNN

knn_model = KNeighborsClassifier(n_neighbors=5) # Using 5 neighbors as a common starting point
knn_model.fit(X_train_tfidf, y_train)

y_pred_knn = knn_model.predict(X_test_tfidf)

print("KNN Classification Report:")
print(classification_report(y_test, y_pred_knn))

cm_knn = confusion_matrix(y_test, y_pred_knn, labels=knn_model.classes_)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=le.classes_)
disp_knn.plot(cmap=plt.cm.Blues)
plt.title("KNN Confusion Matrix")
plt.show()

# SVM

svm_model = LinearSVC(random_state=42, dual=False) # dual=False is recommended for n_samples > n_features
svm_model.fit(X_train_tfidf, y_train)

y_pred_svm = svm_model.predict(X_test_tfidf)

print("SVM (LinearSVC) Classification Report:")
print(classification_report(y_test, y_pred_svm))

cm_svm = confusion_matrix(y_test, y_pred_svm, labels=svm_model.classes_)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=le.classes_)
disp_svm.plot(cmap=plt.cm.Blues)
plt.title("SVM (LinearSVC) Confusion Matrix")
plt.show()

# Naive Bayes

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

y_pred_nb = nb_model.predict(X_test_tfidf)

print("Naive Bayes (MultinomialNB) Classification Report:")
print(classification_report(y_test, y_pred_nb))

cm_nb = confusion_matrix(y_test, y_pred_nb, labels=nb_model.classes_)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=le.classes_)
disp_nb.plot(cmap=plt.cm.Blues)
plt.title("Naive Bayes (MultinomialNB) Confusion Matrix")
plt.show()

# Decision Tree

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_tfidf, y_train)

y_pred_dt = dt_model.predict(X_test_tfidf)

print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))

cm_dt = confusion_matrix(y_test, y_pred_dt, labels=dt_model.classes_)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=le.classes_)
disp_dt.plot(cmap=plt.cm.Blues)
plt.title("Decision Tree Confusion Matrix")
plt.show()

# Bagging Decision Tree

bagging_model = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), random_state=42)
bagging_model.fit(X_train_tfidf, y_train)

y_pred_bagging = bagging_model.predict(X_test_tfidf)

print("Bagging Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_bagging))

cm_bagging = confusion_matrix(y_test, y_pred_bagging, labels=bagging_model.classes_)
disp_bagging = ConfusionMatrixDisplay(confusion_matrix=cm_bagging, display_labels=le.classes_)
disp_bagging.plot(cmap=plt.cm.Blues)
plt.title("Bagging Decision Tree Confusion Matrix")
plt.show()

# Gradient Boosting

gbc_model = GradientBoostingClassifier(random_state=42)
gbc_model.fit(X_train_tfidf, y_train)

y_pred_gbc = gbc_model.predict(X_test_tfidf)

print("Gradient Boosting Classifier Report:")
print(classification_report(y_test, y_pred_gbc))

cm_gbc = confusion_matrix(y_test, y_pred_gbc, labels=gbc_model.classes_)
disp_gbc = ConfusionMatrixDisplay(confusion_matrix=cm_gbc, display_labels=le.classes_)
disp_gbc.plot(cmap=plt.cm.Blues)
plt.title("Gradient Boosting Classifier Confusion Matrix")
plt.show()

# Random Forest

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_tfidf, y_train)

y_pred_rf = rf_model.predict(X_test_tfidf)

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf, labels=rf_model.classes_)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=le.classes_)
disp_rf.plot(cmap=plt.cm.Blues)
plt.title("Random Forest Confusion Matrix")
plt.show()

# Voting

estimators = [
    ('lr', LogisticRegression(random_state=42, solver='liblinear')),
    ('mnb', MultinomialNB()),
    ('svc', LinearSVC(random_state=42, dual=False))
]
voting_model = VotingClassifier(estimators=estimators, voting='hard') # You can change to 'soft' if models support predict_proba
voting_model.fit(X_train_tfidf, y_train)

y_pred_voting = voting_model.predict(X_test_tfidf)

print("Voting Classifier Classification Report:")
print(classification_report(y_test, y_pred_voting))

cm_voting = confusion_matrix(y_test, y_pred_voting, labels=voting_model.classes_)
disp_voting = ConfusionMatrixDisplay(confusion_matrix=cm_voting, display_labels=le.classes_)
disp_voting.plot(cmap=plt.cm.Blues)
plt.title("Voting Classifier Confusion Matrix")
plt.show()

# Neural Network

mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1)
mlp_model.fit(X_train_tfidf, y_train)

y_pred_mlp = mlp_model.predict(X_test_tfidf)

print("Neural Network (MLP) Classification Report:")
print(classification_report(y_test, y_pred_mlp))

cm_mlp = confusion_matrix(y_test, y_pred_mlp, labels=mlp_model.classes_)
disp_mlp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp, display_labels=le.classes_)
disp_mlp.plot(cmap=plt.cm.Blues)
plt.title("Neural Network (MLP) Confusion Matrix")
plt.show()




