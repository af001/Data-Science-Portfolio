#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@author      : Anton
@date        : 05/14/2018
@description : A Python 3.6 script that reads a csv file of Amazon reviews
               and determines sentiment using Afinn and Google NLP. The output 
               is stored as a pickle for further analysis.
'''

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                              IMPORTS
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import re, os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                              FUNCTIONS
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Function used to get column headers throughout the application
def get_columns(df):
    # Get column names --> aka variables
    headers = [col.encode('ascii', 'ignore') for col in df]
    return headers

# Function to perform regex to remove html and chars from author names.  
def replace_bad_author(author):
    auth_regex = re.compile(r'<.*?>')
    author = auth_regex.sub('', author)
    author = author.replace('?>', '')
    return author.replace('?', '')

# Function to perform regex to remove html and chars from content.
def replace_bad_content(content):
    content = re.sub(r'[^\x00-\x7F]+',' ', content)
    content = re.sub(r'[^A-Za-z]+', ' ', content)            
    content = content.lower()
    content = content.strip()
    return content

# Functoin to get the afinn sentiment score for the content (aka review)
def score(comment):
    """
    Returns a float for sentiment strength based on the input text.
    Positive values are positive valence, negative value are negative valence. 
    """
    words = pattern_split.split(comment.lower())
    sentiments = list(map(lambda word: afinn.get(word, 0), words))
    if sentiments:
        # Weight the individual word sentiments using sqrt 
        sentiment = float(sum(sentiments))/math.sqrt(len(sentiments))
    else:
        sentiment = 0
    return sentiment

# Function to take a review, package it as a text file, upload to Google GCP, 
# and receive the result from the NLP analysis. Store the values as a list. In
# some cases, there is a NLP error. Use np.nan for those values. 
def google_nlp(text):
    global counter
    global score
    global magnitude
    global df
    
    # The text to analyze
    document = types.Document(content=text, 
                              type=enums.Document.Type.PLAIN_TEXT)

    # Detects the sentiment of the text. Print the first 10 reviews to stdout
    try:
        sentiment = client.analyze_sentiment(document=document).document_sentiment
        if counter < 10:
            print('\n[+] Google analysis preview:')
            print('Text: {}'.format(text))
            print('\nSentiment: {}, {}'.format(sentiment.score, sentiment.magnitude))
            counter+=1
    
        sent_score.append(sentiment.score)
        sent_magnitude.append(sentiment.magnitude)
    except:
        sent_score.append(np.nan)
        sent_magnitude.append(np.nan)

# Polarity to string mapping for Google score. Values are from -1 to 1.      
def google_sentiment(row):
    sentiment = np.nan
    
    if row < 0.1:
        sentiment = 'Negative'
    elif row > 0.1:
        sentiment = 'Positive'
    else:
        sentiment = 'Neutral'
    
    return sentiment

# Polarity to string mapping for Afinn score. Values are from -5 to 5.    
def afinn_sentiment(row):
    sentiment = np.nan
    
    if row < 0.5:
        sentiment = 'Negative'
    elif row > 0.5:
        sentiment = 'Positive'
    else:
        sentiment = 'Neutral'
    
    return sentiment

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                           SCRIPT START
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

print('\n[+] Script running...\n')

# Define the application path that contains the google credentials and amazon
# reviews.csv file
path = '/Users/anton/sentiment'

# Open the JSON file and read the data to a variable. Normalize the Reviews data 
# and map the values to columns
df = pd.read_csv('{}/wifi-201805131402.csv'.format(path)) 

# Data cleaning and type casting
df['review'] = df['review'].astype('str')
df['title'] = df['title'].astype('str')
df['author'] = df['author'].astype('str')

# Remove possible bad characters
df['author'] = df['author'].apply(lambda x: replace_bad_author(x))
df['review'] = df['review'].apply(lambda x: replace_bad_content(x))
df['title'] = df['title'].apply(lambda x: replace_bad_content(x))

# Word splitter pattern for afinn
pattern_split = re.compile(r'\W+')

# Read AFINN-165 wordlist and map to a dict. Preferred over the afinn Python
# package...this method forces scores to be under -5-+5, and aligns better with
# Google scoring
afinn_file = '{}/AFINN-en-165.txt'.format(path)
afinn = dict(map(lambda w: (w[0], int(w[1])), 
                 [ ws.strip().split('\t') for ws in open(afinn_file) ]))

# Use the pandas apply method to perform sentiment analysis against each row.
# Return the results to a new column named 'afinn'. 
df['afinn'] = df['review'].apply(lambda x: score(x))

# Print the first 4 records
print('\n[+] First 5 Records:')
print(df[['author','afinn']].head(5))

# Print the descriptive statistics for the afin values. 
print('\n[+] Descriptive Statistics:')
print(df['afinn'].describe())

# Plot afinn score as a function of rating
fig1 = plt.gcf()
x = pd.to_numeric(df['rating'])
y = df['afinn']
plt.scatter(x, y)
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
plt.xlabel('Guest Rating (Rating)')
plt.ylabel('Afinn Score (Sentiment)')
plt.title('Afinn Score as a function of Rating')
plt.show()
fig1.savefig('{}/afinn_rating_scat.pdf'.format(path), format='pdf', dpi=100)

# Histogram of afinn scores.
fig2 = plt.gcf()
plt.hist(df['afinn'].dropna(), 70, density=1)
plt.title('Histogram (Afinn Score of Reviews)')
plt.xlabel('Afinn Score')
plt.show()
fig2.savefig('{}/afinn_rating_hist.pdf'.format(path), format='pdf', dpi=100)

# Use Google NLP credentials for API calls
os.environ['GOOGLE_APPLICATION_CREDENTIALS']='{}/GCPKey-4ce3daf1f394.json'.format(path)

# Instantiates a client
client = language.LanguageServiceClient()

# Holder list for Google NLP values
sent_score = []
sent_magnitude = []

# Perform Google NLP on each review. This requires writing the review text
# to a document, uploading to GCP, analyzing, and receiving input. Store 
# returned values in a list. Counter is used to track the print function of
# the first 10 reviews
counter = 0
df['review'].apply(lambda x: google_nlp(x))

# Write values received from google into the primary dataframe
values1 = pd.Series(sent_score)
values2 = pd.Series(sent_magnitude)

df['google_score'] = values1.values
df['google_magnitude'] = values2.values

# Histogram of Google scores
fig3 = plt.gcf()
plt.hist(df['google_score'].dropna(), 10, density=1)
plt.title('Histogram (Google Score of Reviews)')
plt.xlabel('Google Score')
plt.show()
fig3.savefig('{}/google_score_hist.pdf'.format(path), format='pdf', dpi=100)

# Show a plot of magnitude and score 
fig4 = plt.gcf()
plt.scatter(df['google_score'], df['google_magnitude'])
plt.xlabel('Guest Sentiment (Google Score)')
plt.ylabel('Guest Magnitude (Google Magnitude)')
plt.title('Magnitude as a function of Score')
plt.show()
fig4.savefig('{}/google_mag_scat.pdf'.format(path), format='pdf', dpi=100)

# Show a histogram of magnitude
fig5 = plt.gcf()
plt.hist(df['google_magnitude'].dropna(), 50, density=1)
plt.title('Histogram (Magnitude of Reviews)')
plt.xlabel('Google Magnitude')
plt.show()
fig5.savefig('{}/google_mag_hist.pdf'.format(path), format='pdf', dpi=100)

# Save the output to a pickle file. This is great for testing and dev because
# the time to analyze all reviews from Google is time consuming. 
df.to_pickle('{}/wifi_reviews.pkl'.format(path))
print('\n[+] Wrote pickle to: {}/wifi_reviews.pkl'.format(path))

# Print the descriptive statistics for the Google values. 
print('\n[+] Descriptive Statistics:')
print(df['google_score'].describe())

# Print the first 5 records
print('\n[+] First 5 Records:')
print(df[['author','google_score','google_magnitude']].head(5))

print('\n[+] Comparative analytics:')
print('\nGoogle Mean Sentiment: {}'.format(round(df['google_score'].mean(skipna=True)), 4))
print('Afinn Sentiment: {}'.format(round(df['afinn'].mean(skipna=True)), 4))

# Map the values to a classification of positive, negative, or neutral
df['google_sentiment'] = df['google_score'].apply(lambda x: google_sentiment(x))
df['afinn_sentiment'] = df['afinn'].apply(lambda x: afinn_sentiment(x))

# View the count of classifications for comparison
print('\nGoogle counts by sentiment classification:')
df_1 = df['google_sentiment'].value_counts()
print(df_1)

# View the count of classifications for comparison
print('\nAfinn counts by sentiment classification:')
df_2 = df['afinn_sentiment'].value_counts()
print(df_2)

# Verify data imported. Show first 10 rows to view comparisons.
print('\n[+] First 10 rows: \n', df.head(10))
