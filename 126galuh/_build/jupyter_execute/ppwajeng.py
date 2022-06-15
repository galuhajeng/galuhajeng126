#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install scrapy')


# Melakukan Crawling Data Berita
# 

# In[2]:


import scrapy
from scrapy.crawler import CrawlerProcess

class Spider(scrapy.Spider):
    name = "sindonews"
    keyword = 'international'
    start_urls = [
        'https://international.sindonews.com/'+keyword
        ]
    custom_settings = {
        'FEED_FORMAT': 'csv',
        'FEED_URI': 'internasional.csv'
        }

    def parse(self, response):
        for data in response.css('div.homelist-box'):
            yield {
                'Kategori': data.css('div.homelist-channel::text').get(),
                'Tanggal': data.css('div.homelist-date::text').get(),
                'Judul': data.css('div.homelist-title a::text').get(),
                'Deskripsi': data.css('div.homelist-desc::text').get()
                }
proses = CrawlerProcess()
proses.crawl(Spider)
proses.start()


# Import Library

# In[3]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')
get_ipython().system('pip install nltk')
get_ipython().system('pip install swifter')
get_ipython().system('pip install sastrawi')
get_ipython().system('pip install Stemmer')
get_ipython().system('pip install sklearn')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn')


# In[4]:


import nltk
nltk.download('stopwords')


# Import Modul

# In[5]:


# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#import nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize

#preprocessing
from nltk.corpus import stopwords  #stopwords
from nltk import word_tokenize,sent_tokenize # tokenizing
from nltk.stem import PorterStemmer,LancasterStemmer  # using the Porter Stemmer and Lancaster Stemmer and others
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet

# for named entity recognition (NER)
from nltk import ne_chunk

# vectorizers for creating the document-term-matrix (DTM)
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

#stop-words
stop_words=set(nltk.corpus.stopwords.words('english'))


# Melakukan Load Data

# In[6]:


df=pd.read_csv('internasional.csv')
df.head()


# Menghapus kolom 'Kategori', 'Tanggal', 'Judul', karena tidak diperlukan

# In[7]:


#drop the publish date
df.drop(['Kategori'],axis=1,inplace=True)
df.drop(['Tanggal'],axis=1,inplace=True)
df.drop(['Judul'],axis=1,inplace=True)


# In[8]:


df.head()


# **Data Cleaning & Pre-Processing**

# Di sini saya telah melakukan pra-pengolahan data. Saya sudah menggunakan lemmatizer dan bisa juga menggunakan stemmer. Juga kata-kata berhenti telah digunakan bersama dengan kata-kata dengan panjang lebih pendek dari 3 karakter untuk mengurangi beberapa kata yang menyimpang.

# In[9]:


def clean_text(Deskripsi):
  le=WordNetLemmatizer()
  word_tokens=word_tokenize(Deskripsi)
  tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
  cleaned_text=" ".join(tokens)
  return cleaned_text


# In[10]:


import nltk
nltk.download('wordnet')


# In[11]:


import nltk
nltk.download('omw-1.4')


# In[12]:


import nltk
nltk.download('punkt')


# Lalu hapus kolom yang belum di lakukan cleaning data.

# In[13]:


# time taking
df['Deskripsi_cleaned_text']=df['Deskripsi'].apply(clean_text)



# In[14]:


df.head()


# In[15]:


df.drop(['Deskripsi'],axis=1,inplace=True)


# In[16]:


df.head()


# Dapat dilihat deskripsi dari sebuah dokumen tertentu

# In[17]:


df['Deskripsi_cleaned_text'][0]


# **EXTRACTING THE FEATURES AND CREATING THE DOCUMENT-TERM-MATRIX ( DTM )**
# **MENGEKSTRAK FITUR DAN MEMBUAT DOCUMENT-TERM-MATRIX ( DTM ) **

# Dalam DTM nilainya adalah nilai TFidf.
# 
# Saya juga telah menentukan beberapa parameter dari vectorizer Tfidf.
# 
# Beberapa poin penting:
# 
# 1) LSA umumnya diimplementasikan dengan nilai Tfidf di mana-mana dan tidak dengan Count Vectorizer.
# 
# 2) max_features tergantung pada daya komputasi Anda dan juga pada eval. metrik (skor koherensi adalah metrik untuk model topik). Coba nilai yang memberikan evaluasi terbaik. metrik dan tidak membatasi kekuatan pemrosesan.
# 
# 3) Nilai default untuk min_df & max_df bekerja dengan baik.
# 
# 4) Dapat mencoba nilai yang berbeda untuk ngram_range.

# In[18]:


vect =TfidfVectorizer(stop_words=stop_words,max_features=1000) # to play with. min_df,max_df,max_features etc...


# In[19]:


vect_text=vect.fit_transform(df['Deskripsi_cleaned_text'])


# Kita sekarang dapat melihat kata-kata yang paling sering dan langka di berita utama berdasarkan skor idf. Semakin kecil nilainya; lebih umum adalah kata dalam berita utama.

# In[20]:


print(vect_text.shape)
print(vect_text)


# In[21]:


idf=vect.idf_


# In[22]:


dd=dict(zip(vect.get_feature_names(), idf))
l=sorted(dd, key=(dd).get)
# print(l)
print(l[0],l[-1])
print(dd['baru'])
print(dd['rusia'])


# # TOPIC MODELLING

# ## **Latent Semantic Analysis (LSA)**

# Pendekatan pertama yang saya gunakan adalah LSA. LSA pada dasarnya adalah dekomposisi nilai tunggal.
# 
# SVD menguraikan DTM asli menjadi tiga matriks S=U.(sigma).(V.T). Di sini matriks U menunjukkan matriks dokumen-topik sementara (V) adalah matriks topik-term.
# 
# Setiap baris dari matriks U (matriks istilah dokumen) adalah representasi vektor dari dokumen yang sesuai. Panjang vektor ini adalah jumlah topik yang diinginkan. Representasi vektor untuk suku-suku dalam data kami dapat ditemukan dalam matriks V (matriks istilah-topik).
# 
# Jadi, SVD memberi kita vektor untuk setiap dokumen dan istilah dalam data kita. Panjang setiap vektor adalah k. Kami kemudian dapat menggunakan vektor-vektor ini untuk menemukan kata-kata dan dokumen serupa menggunakan metode kesamaan kosinus.
# 
# Kita dapat menggunakan fungsi truncatedSVD untuk mengimplementasikan LSA. Parameter n_components adalah jumlah topik yang ingin kita ekstrak. Model tersebut kemudian di fit dan ditransformasikan pada hasil yang diberikan oleh vectorizer.

# **Terakhir perhatikan bahwa LSA dan LSI (I untuk pengindeksan) adalah sama dan yang terakhir kadang-kadang digunakan dalam konteks pencarian informasi.**

# In[23]:


from sklearn.decomposition import TruncatedSVD
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(vect_text)


# In[24]:


print(lsa_top)
print(lsa_top.shape)  # (no_of_doc*no_of_topics)


# In[25]:


l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
  print("Topic ",i," : ",topic*100)


# Mirip dengan dokumen lain kita bisa melakukan ini. Namun perhatikan bahwa nilai tidak menambah 1 seperti di LSA itu bukan kemungkinan topik dalam dokumen.

# In[26]:


print(lsa_model.components_.shape) # (no_of_topics*no_of_words)
print(lsa_model.components_)


# Sekarang kita bisa mendapatkan daftar kata-kata penting untuk masing-masing dari 10 topik seperti yang ditunjukkan. Untuk kesederhanaan di sini saya telah menunjukkan 10 kata untuk setiap topik.

# In[27]:


# most important words for each topic
vocab = vect.get_feature_names()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")

