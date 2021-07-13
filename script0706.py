# import library 
import numpy as np
from numpy.lib.function_base import _cov_dispatcher
import scipy as sp
import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import train_test_split


# train & label data load
'''
- train_data: index, title, topic_idx
- topic_dict: topic, topic_idx
- test_data: index, title
'''
PATH = "./datasets/"
df_train = pd.read_csv(PATH + "train_data.csv")
df_test = pd.read_csv(PATH + "test_data.csv")
topic_dict = pd.read_csv(PATH + "topic_dict.csv")
kr_stopwords_list = pd.read_csv(PATH + "korean_stopwords.csv").columns.values.reshape(-1,).tolist()

# preprocessing
# html tag 및 숫자, 영어, 특수문자 등 제거

# punctuation remove
def clean_punc(text):

    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&' 
    punct_mapping = {
        "‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ",
        "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', 
        '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha',
        '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

    for p in punct_mapping:
        text = text.replace(p, punct_mapping[p])
    
    for p in punct:
        text = text.replace(p, f'{p}')

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}

    for s in specials:
        text = text.replace(s, specials[s])

    return text.strip()

# using re to preprocess: remove punct, html tag, number, and lowering, spacing
def clean_text(texts):

    corpus = []
    for i in range(0, len(texts)):

        review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"]', '',str(texts[i])) #remove punctuation
        review = re.sub(r'\d+','',str(texts[i])) # remove number
        review = review.lower() #lower case
        review = re.sub(r'\s+',' ', review)
        review = re.sub(r'<[^>]+>','',review) #remove Html tags 
        review = re.sub(r'\s+', ' ', review) #remove spaces 
        review = re.sub(r"^\s+", '', review) #remove space from start 
        review = re.sub(r'\s+$', '', review) #remove space from the end corpus.append(review) return corpus

        corpus.append(review)

    return corpus

# dacon sample function
def clean_text_dacon(text):
    text_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", text)
    return text_clean

# 맞춤법, 문장 분리, 단어 분리(외부 라이브러리 이용)
    
'''

try:
    from pykospacing import spacing
    from hanspell import spell_checker
    import kss
except:
    pip install git+https://github.com/haven-jeon/PyKoSpacing.git
    pip install kss
    pip install py-hanspell
    from pykospacing import spacing
    from hanspell import spell_checker
    import kss
    
'''

from pykospacing import Spacing
   
spacing = Spacing()

def text_spacing(text):
    text_spacing = spacing(text)
    return text_spacing

# stemming: 동사를 원형으로 복원
# 불용어 제거, nltk
# 불용어 제거는 먼저 토큰화가 이루어져야 한다. 
    
import nltk
#nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

def stem_stopwords(text):
    corpus = []
    for word in text.split(' '):
        if word not in kr_stopwords_list:
            corpus.append(wnl.lemmatize(word))
    return corpus

# preprocessing: stopword, spacing, stemming, remove punctuation

def preprocessing(df):

    df_copy = df.copy()
    texts = df_copy["title"].values

    text_inputs = []

    for text in texts:

        text_clean = clean_punc(text)
        text_clean = clean_text_dacon(text_clean)
        text_sc = text_spacing(text_clean)
        corpus_stem_stopwords = stem_stopwords(text_sc)
        text_inputs.append(corpus_stem_stopwords)

    return text_inputs

# tokenized

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

text_preprocessed = preprocessing(df_train)
tokenizer = Tokenizer(filters = "", num_words = 10000)

datasets_token = tokenizer.fit_on_texts(text_preprocessed)
datasets = pad_sequences(datasets_token, padding = "post")
print(datasets.shape)


# train - test data split
labels = df_train["topic_idx"].values.reshape(-1,)
x_train, x_test, y_train, y_test = train_test_split(datasets, labels, test_size = 0.2, shuffle = True, random_state = 42)


# model architecture(word embedding 포함)
'''
(모델)
- 1) attention model
- 2) RNN model
(설명)
- 라벨은 총 7개
- 단일분류 모델
- vocab_size, maxlen 필요
'''


# Transformer block

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate = 0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads = num_heads, key_dim = embed_dim)
        self.ffn = tf.keras.models.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation = "relu"), tf.keras.layers.Dense(embed_dim),]
        )

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions    


embed_dim = 32
num_heads = 2
ff_dim = 32

inputs = tf.keras.layers.Input(shape = ())
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(20, activation="relu")(x)
x = tf.keras.layers.Dropout(0.1)(x)
outputs = tf.keras.layers.Dense(7, activation="softmax")(x)

model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer = tf.keras.optimizers.Adam(lr = 1e-3),
    loss = tf.keras.losses.SparseCategoricalCrossentropy,
    metrics = ["accuracy"]
)

# training

batch_size = 64
epochs = 64

tf.debugging.set_log_device_placement(True)
with tf.device("./GPU:0"):
    hist = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = epochs, batch_size = batch_size, verbose = 1)
    loss, acc = hist.history["loss"], hist.history["acc"]

    epoch_axis = range(1, epochs + 1)
    plt.plot(epoch_axis, loss, "r", label = "train loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.show() 

# submission
