#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 09:19:28 2019

@modified : anton
@author   : paulhuynh
"""

#----------------------------------------------------------------------------
#                            IMPORTS 
#----------------------------------------------------------------------------

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

import re
import string
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nltk import FreqDist
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score
from sklearn.manifold import MDS
from sklearn.ensemble import RandomForestClassifier

from gensim import corpora
from gensim.models import Word2Vec,LdaMulticore, TfidfModel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

#----------------------------------------------------------------------------
#                             GLOBALS
#----------------------------------------------------------------------------

#set working Directory to where class corpus is saved.
os.chdir('/Users/anton/Northwestern/MSDS453')

K = 8
RANDOM_STATE = 42
CORPUS_DIR = 'articles/'
CLEAN_CSV = 'processed.csv'
LDA_NUM_TOPICS = 3
CLUSTER_KEYWORDS = 10

#----------------------------------------------------------------------------
#                            FUNCTIONS 
#----------------------------------------------------------------------------

"""Print information about the status of a corpus."""
def corpus_info(corpus):

    fids   = len(corpus.fileids())
    paras  = len(corpus.paras())
    sents  = len(corpus.sents())
    sperp  = sum(len(para) for para in corpus.paras()) / float(paras)
    tokens = FreqDist(corpus.words())
    count  = sum(tokens.values())
    vocab  = len(tokens)
    lexdiv = float(count) / float(vocab)

    print((
        "\n[+] Text corpus contains {} files composed of:\n"
        "  > {} paragraphs and {} sentences\n"
        "  > {:0.3f} sentences per paragraph\n"
        "  > word count of {} with a vocabulary of {}\n"
        "  > lexical diversity is {:0.3f}"
    ).format(
        fids, paras, sents, sperp, count, vocab, lexdiv
    ))

'''Save processed data to csv - Format: filename, text'''
def clean_to_csv(data):
    df = pd.DataFrame(data)
    df.to_csv(CLEAN_CSV, index=False)

'''Remove stop words, punctuation, and small words'''
def clean_data(corpus):
    safe_words = ['5G', '4G', '3G', '3D', '2D', '50mm', 'ai', 'gun', 'fly', 'uk', 'mod', 'us',
                  'key', 'bot', 'oil', 'gas', 'ag', '3d', '5g', '4g', '3g', '2d', 'gps', 'pbs',
                  'led', 'mit', 'pc', 'map', 'car', 'uav', 'lg', 'cpu', 'lab', 'app', 'usa',
                  'gps', 'ask', 'rpa', 'hub', 'job', 'dod']

    bad_words = ['used', 'said', 'also', 'need', 'using', 'make', 'however', 'made', 
                 'making', 'want', 'according', 'say']

    # Initialize empty list to hold values
    titles = []
    processed_strings = []
    processed_save = []
    processed_words = []
    
    # For each file in the directory, clean, parse and save to lists
    for infile in sorted(corpus.fileids()):
        cleaned = {}
        raw_text = corpus.raw(infile)
        
        # Tokenize using TreebankWordTokenizer
        tokenizer = TreebankWordTokenizer()
        tokens = tokenizer.tokenize(raw_text)
        
        # remove punctuation from each word
        re_punc = re.compile('[%s]' % re.escape(string.punctuation))
        tokens = [re_punc.sub(' ', token) for token in tokens]
        
        # Fix a few smaller words that have significant meaning
        tokens = [token.replace('3 D', '3D') for token in tokens]
        tokens = [token.replace('2 D', '2D') for token in tokens]

        # Convert text to lower-alpha
        tokens=[token.lower() for token in tokens if token.isalpha() or token in safe_words]
        #tokens = [token.lower() for token in tokens if token.isalnum()] 
        
        # Strip stop words
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if not token in stop_words and token not in bad_words]        
        
        # strip words with size <= 3, do not include safe_words
        tokens = [token for token in tokens if len(token) > 3 or token in safe_words]
        
        # Lemmatize robot, robotic, robotics, robots
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        # Append tokens to token list
        processed_words.append(tokens)
        
        # Rebuild the tokens into a single string
        single = TreebankWordDetokenizer().detokenize(tokens)
        processed_strings.append(single)
        
        # Append the title and clean data to a list - for saving to csv
        cleaned['titles'] = infile
        cleaned['data'] = tokens   
        processed_save.append(cleaned)
        
        # Save the titles
        titles.append(infile)
    
    # Save to csv for later processing
    clean_to_csv(processed_save)
    
    # Return lists
    return processed_strings, processed_words, titles

#----------------------------------------------------------------------------
#                              GET DATA 
#----------------------------------------------------------------------------

raw_corpus = PlaintextCorpusReader(CORPUS_DIR, '.*\.txt')
corpus_info(raw_corpus)
processed_strings, processed_words, titles = clean_data(raw_corpus)

#----------------------------------------------------------------------------
#                            WORD COUNTS 
#----------------------------------------------------------------------------

# Min_df ignores terms that have a document frequency (presence in % of documents) 
# strictly lower than the given threshold. For example, Min_df=0.66 requires that 
# a term appear in 66% of the docuemnts for it to be considered part of the vocabulary.
count_vec = CountVectorizer(stop_words='english', analyzer='word', 
                            ngram_range=(1, 3), max_df=1.0, min_df=0.65, max_features=None)

count_train = count_vec.fit(processed_strings)
bag_of_words = count_vec.transform(processed_strings)

print('\n[+] Word Frequency across documents: N-Grams -> N=3')
print('  > ', count_vec.get_feature_names())

# The goal of using tf-idf is to scale down the impact of tokens that occur very 
# frequently in a given corpus and that are hence empirically less informative 
# than features that occur in a small fraction of the training corpus.
tf = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word')
txt_fitted = tf.fit(processed_strings)
txt_transformed = txt_fitted.transform(processed_strings)

# The learned corpus vocabulary
cv = tf.vocabulary_

# IDF: The inverse document frequency
idf = tf.idf_

# get feature names
feature_names = np.array(tf.get_feature_names())
sorted_by_idf = np.argsort(tf.idf_)
print('\n[+] Features with lowest idf:\n  > {}'.format(feature_names[sorted_by_idf[:5]]))
print('\n[+] Features with highest idf:\n  > {}'.format(feature_names[sorted_by_idf[-5:]]))

# TF-IDF - Maximum token value throughout the whole dataset
new1 = tf.transform(processed_strings)

# find maximum value for each of the features over all of dataset:
max_val = new1.max(axis=0).toarray().ravel()

#sort weights from smallest to biggest and extract their indices 
sort_by_tfidf = max_val.argsort()
print('\n[+] Features with lowest tfidf:\n  > {}'.format(feature_names[sort_by_tfidf[:5]]))
print('\n[+] Features with highest tfidf: \n  > {}'.format(feature_names[sort_by_tfidf[-5:]]))

#----------------------------------------------------------------------------
#                            SKLEARN TFIDF 
#----------------------------------------------------------------------------

# Call Tfidf Vectorizer - range 3 
Tfidf=TfidfVectorizer(ngram_range=(1,3))

# Fit the vectorizer using final processed documents.
TFIDF_matrix=Tfidf.fit_transform(processed_strings)     

# Creating datafram from TFIDF Matrix
matrix=pd.DataFrame(TFIDF_matrix.toarray(), columns=Tfidf.get_feature_names(), index=titles)

average_TFIDF={}
for i in matrix.columns:
    average_TFIDF[i]=np.mean(matrix[i])

average_TFIDF_DF=pd.DataFrame(average_TFIDF,index=[0]).transpose()

average_TFIDF_DF.columns=['TFIDF']

#calculate Q1 and Q3 range
Q1=np.percentile(average_TFIDF_DF, 25)
Q3=np.percentile(average_TFIDF_DF, 75)
IQR = Q3 - Q1
outlier=Q3+(1.5*IQR)

# words that exceed the Q3+IQR*1.5
outlier_list=average_TFIDF_DF[average_TFIDF_DF['TFIDF']>=outlier]
keep_list=average_TFIDF_DF[average_TFIDF_DF['TFIDF']<outlier]

# print(outlier_list)
# print(keep_list)

#----------------------------------------------------------------------------
#                                DOC2VEC 
#----------------------------------------------------------------------------

# Initialize and train model
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(processed_strings)]
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

# Infer vector to dataframe
doc2vec_df=pd.DataFrame()
for i in range(0,len(processed_words)):
    vector=pd.DataFrame(model.infer_vector(processed_words[i])).transpose()
    doc2vec_df=pd.concat([doc2vec_df,vector], axis=0)

doc2vec_df=doc2vec_df.reset_index()

doc_titles={'title': titles}
t=pd.DataFrame(doc_titles)

doc2vec_df=pd.concat([doc2vec_df,t], axis=1)

doc2vec_df=doc2vec_df.drop('index', axis=1)

#----------------------------------------------------------------------------
#                               WORD2VEC 
#----------------------------------------------------------------------------

#Note, there are opportunities to use the word2vec matrix to determine words 
#which are similar.  Similar words can be used to create equivalent classes.  
#k-means is not used to group individual words using the Word2Vec output.

# word to vec
model_w2v = Word2Vec(processed_words, size=100, window=5, min_count=0.75, workers=4)

# join all processed DSI words into single list
processed_text_w2v=[]
for i in processed_words:
    for k in i:
        processed_text_w2v.append(k)

# obtian all the unique words from DSI
w2v_words=list(set(processed_text_w2v))

#can also use the get_feature_names() from TFIDF to get the list of words
#w2v_words=Tfidf.get_feature_names()

# empty dictionary to store words with vectors
w2v_vectors={}

# for loop to obtain weights for each word
for i in w2v_words:
    temp_vec=model_w2v.wv[i]
    w2v_vectors[i]=temp_vec

# create a final dataframe to view word vectors
w2v_df=pd.DataFrame(w2v_vectors).transpose()

''' EXPERIMENTATION
from sklearn.manifold import TSNE
import plotly.plotly as py
import plotly.graph_objs as go

# load the data into X for t-SNE
vocab = model_w2v.wv.vocab
X = model_w2v[vocab]

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

# put X_tsne together with the vocab list
df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

# plot with matplotlib - so messy with a big vocab
fig = plt.figure()
ax = fig.add_subplot(figsize=(20, 20))

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos)
    
plt.show()

# Create a trace and plot with plotly - interactive
trace = go.Scatter(
    name = 'Vocab',
    x = df['x'],
    y = df['y'],
    mode = 'markers',
    text = list(df.index)
)

data = [trace]

# Plot and embed in ipython notebook!
py.iplot(data, filename='basic-scatter')
'''

#----------------------------------------------------------------------------
#                       K-MEANS CLUSTERING w/TFIDF 
#----------------------------------------------------------------------------

# Initialize and fit
km = KMeans(n_clusters=K, random_state=RANDOM_STATE)
km.fit(TFIDF_matrix)
clusters = list(km.labels_)

terms = Tfidf.get_feature_names()
Dictionary={'Doc Name': titles, 'Cluster': clusters,  'Text': processed_strings}
frame=pd.DataFrame(Dictionary, columns=['Cluster', 'Doc Name', 'Text'])

print("\n[+] Top terms per cluster:")

#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

terms_dict=[]

#save the terms for each cluster and document to dictionaries.  To be used later
#for plotting output.

#dictionary to store terms and titles
cluster_terms={}
cluster_title={}

for i in range(K):
    print("\nCluster %d:" % i),
    temp_terms=[]
    temp_titles=[]
    for ind in order_centroids[i, :CLUSTER_KEYWORDS]:
        print('  > %s' % terms[ind])
        terms_dict.append(terms[ind])
        temp_terms.append(terms[ind])
    cluster_terms[i]=temp_terms
    
    print("\nCluster %d titles:" % i, end='\n')
    temp=frame[frame['Cluster']==i]
    for title in temp['Doc Name']:
        print('  > %s' % title, end='\n')
        temp_titles.append(title)
    cluster_title[i]=temp_titles

#----------------------------------------------------------------------------
#                            PLOTTING - TFIDF
#----------------------------------------------------------------------------

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=RANDOM_STATE)

dist = 1 - cosine_similarity(TFIDF_matrix)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]

#set up cluster names using a dict.  
cluster_dict=cluster_title

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=range(0,len(clusters)))) 

#group by cluster
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(12, 12)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
            label=cluster_dict[name], mec='none')    
    ax.set_aspect('auto')
    ax.title.set_text('TFIDF')
    ax.tick_params(\
        axis= 'x',         # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True)
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left=False,        # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelleft=True)
    
ax.legend(title='Legend', loc='center left', bbox_to_anchor=(1, 0.5))   #show legend with only 1 point

#The following section of code is to run the k-means algorithm on the doc2vec outputs.
#note the differences in document clusters compared to the TFIDF matrix.
#----------------------------------------------------------------------------
#                    K-MEANS CLUSTERING w/DOC2VEX
#----------------------------------------------------------------------------

doc2vec_k_means=doc2vec_df.drop('title', axis=1)

km = KMeans(n_clusters=K, random_state=RANDOM_STATE)
km.fit(doc2vec_k_means)
clusters_d2v = km.labels_.tolist()

Dictionary={'Doc Name':titles, 'Cluster':clusters_d2v,  'Text': processed_strings}
frame=pd.DataFrame(Dictionary, columns=['Cluster', 'Doc Name', 'Text'])

#dictionary to store clusters and respective titles
cluster_title={}

#note doc2vec clusters will not have individual words due to the vector representation
#is based on the entire document not indvidual words. As a result, there won't be individual
#word outputs from each cluster.   
for i in range(K):
    temp=frame[frame['Cluster']==i]
    temp_title_list=[]
    for title in temp['Doc Name']:
        temp_title_list.append(title)
    cluster_title[i]=temp_title_list

#----------------------------------------------------------------------------
#                           PLOTTING - DOC2VEC
#----------------------------------------------------------------------------
###############################################################################
### Plotting Doc2vec
###############################################################################
# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=RANDOM_STATE)

dist = 1 - cosine_similarity(doc2vec_k_means)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]

#set up cluster names using a dict.  
cluster_dict=cluster_title         

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=range(0,len(clusters)))) 

#group by cluster
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(12, 12)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
            label=cluster_dict[name], mec='none')
    ax.set_aspect('auto')
    ax.title.set_text('K-MEANS')
    ax.tick_params(\
        axis= 'x',         # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True)
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left=False,        # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelleft=True)

ax.legend(title='Clusters', loc='center left', bbox_to_anchor=(1, 0.5))      #show legend with only 1 point

#The following section is used to create a model to predict the clusters labels 
#based on the the TFIDF matrix and the doc2vec vectors.  Note the model performance 
#using the two different vectorization methods.
#----------------------------------------------------------------------------
#                      CLASSIFICATION - VARIOUS METHODS
#----------------------------------------------------------------------------

# Instantiate RF 
model_RF=RandomForestClassifier()

# TFIDF
Y=clusters
X=TFIDF_matrix

# cross validation
cv_score=cross_val_score(model_RF, X, Y)

#mean CV score
np.mean(cv_score)

print('\n[+] Performance: ')
print('  > Random Forest mean_cv_score: {}'.format(round(np.mean(cv_score), 4)))

#Doc2Vec
Y=clusters_d2v
X=doc2vec_k_means

#cross validation
cv_score=cross_val_score(model_RF, X, Y)

#mean CV score
np.mean(cv_score)

print('  > Doc2Vec mean_cv_score: {}'.format(round(np.mean(cv_score), 4)))

#----------------------------------------------------------------------------
#                                  LDA 
#----------------------------------------------------------------------------

#LDA using bag of words
dictionary = corpora.Dictionary(processed_words)
corpus = [dictionary.doc2bow(doc) for doc in processed_words]

ldamodel = LdaMulticore(corpus, num_topics=LDA_NUM_TOPICS, id2word=dictionary, passes=2, workers=2)    

print('\n[+] LDA Bag of Words')
for idx, topic in ldamodel.print_topics(-1):
    print('  > Topic: {}'.format(idx))
    print('  > Words: {}'.format(topic))

print('\n[+] LDA TFIDF')
tfidf = TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
ldamodel = LdaMulticore(corpus_tfidf, num_topics=LDA_NUM_TOPICS, id2word=dictionary, passes=2, workers=2)    

for idx, topic in ldamodel.print_topics(-1):
    print('  > Topic: {}'.format(idx))
    print('  > Words: {}'.format(topic))

'''
#the following section is example code to create ECs within the corpus.  A dictionary
#will need to be created for every EC.  Each EC will need to be applied to the corpus.
#Below is an example of how the function works.
#----------------------------------------------------------------------------
#                            EC CLEAN-UP CODE
#----------------------------------------------------------------------------
def create_ec(dictionary, corpus):
    for key, values in dictionary.items():
        for value in values:
            corpus= corpus.replace(value, key)
    return corpus


corpus='i like swiss.  i like cheddar.  i like provolone.'
cheese_dic={'cheese': ['swiss', 'cheddar', 'provolone']}

corpus_new=create_ec(cheese_dic, corpus)

'''
