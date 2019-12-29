#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import requests
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from bs4 import BeautifulSoup, SoupStrainer
from nltk.corpus import stopwords
%matplotlib inline

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                 FUNCTIONS
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

'''
url_builder : Generate a url based on the page and product number. Note: the html for pages
              that only have one page is missing the ?page= arg so we use full_list=true.
              
    args:
        url        : The base url
        page       : The page to scrape
        product_id : The product to scrape
'''
def url_builder(url, page, product_id):
    if product_id == '168797' or product_id == '170397':
        link = '{}/gdm_reviews?full_list=true&product_id={}'.format(url, product_id)
    else:
        link = '{}gdm_reviews?page={}&product_id={}'.format(url, page, product_id)

    print('GET: {}'.format(link))
    return link

'''
url_builder : Remove strong tags from html. This is necessary for the value_score since it
              it's tag is not the same as the others. 
              
    args:
        text : A string of reviews
'''
def remove_strong(text):
    text_regex = re.compile(r'<strong>.*?</strong>')
    text = text_regex.sub('', text)
    text_regex = re.compile(r'<.*?>')  
    return text_regex.sub('', text)

# Simple function to strip non-ascii characters from text
def remove_non_ascii(text):
    return ' '.join(i for i in text if ord(i)<128)

# Last check. Remove all special characters and convert to lowercase
def normalize_data(content):
    # Remove stopwords from string of reviews
    extra_words = ['google', 'google', 'platform', 'cloud', 'product', 'use', 'log', 'sheet', 
                   'spreadsheet', 'excel', 'drive','file', 'document', 'name', 'storage', 
                   'iot', 'sheet', 'sheets', 'pro', 'con', 'overall', 'doc', 'documents',
                   'software', 'files', 'sometime', 'docs', 'pros', 'cons', 'sometimes']
    
    stop_words = set(stopwords.words('english'))
    
    content = re.sub(r'[^\x00-\x7F]+',' ', content)
    content = re.sub(r'[^A-Za-z]+', ' ', content)            
    content = content.lower()
    #content = content.strip()
    
    clean = ''
    words = content.split()
    
    # words = list(filter(None, words))
    for r in words:
        if not r in stop_words and not r in extra_words:
            clean = '{} {}'.format(clean, r)
    
    return clean
    
def create_wordcloud(wordlist):    
    return WordCloud().generate(wordlist)

def create_wordcloud_2(wordlist):
    return WordCloud(max_font_size=40).generate(wordlist)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               START SCRIPT
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
nltk.download('stopwords')

product_map = {'123836': 'GCP', '174303': 'GCP_Storage', '168797': 'GCP_IOT', 
               '170397': 'GCP_ML_Engine', '160756': 'Google_Docs', '168389': 'GSuite',
               '169718': 'Google_Sheets', '161425': 'Google_Drive'}  

product_ids = ['123836', '174303', '168797', '168389', '169718', '161425', '170397', '160756']

url = 'https://www.capterra.com/'

csv = '/Users/anton/finalproject/out_google.csv'
pkl = '/Users/anton/finalproject/out_google.pkl'

reviews = []
for product in product_ids:
    page = 1
    run = True
    while(run):
        _url = url_builder(url, page, product)
        result = requests.get(_url)
        if result.status_code == 200:
            c = result.content
            soup = BeautifulSoup(c, 'html.parser', parse_only=SoupStrainer('div', attrs={'class': 'cell-review'}))
            
            if len(soup) > 0:
                for rv in soup:
                    review = {}
                    
                    review['product_id'] = product
                    
                    if rv.find('q') is not None:
                        title = rv.find('q')
                        review['title'] = title.text.strip()
                    else:
                        review['title'] = np.nan
                    
                    date = rv.find('div', attrs={'class': 'quarter-margin-bottom'})
                    review['date'] = date.string.strip()
                    
                    name = rv.find('div', attrs={'class': 'epsilon'})
                    review['author'] = name.string.strip()
                    
                    overall_rating = rv.find('div', attrs={'class': 'overall-rating-container'})
                    span = overall_rating.findAll('span')
                    review['overall_score'] = span[1].string.strip()
                    
                    count=0
                    misc_ratings = rv.findAll('span', attrs={'class': 'rating-decimal'})
                    for rate in misc_ratings:
                        span = rate.findAll('span')
                        score = span[0].string.strip()
                        if count == 0:
                            review['usability_score'] = score
                        elif count == 1:
                            review['feature_score'] = score
                        elif count == 2:
                            review['customer_service_score'] = score
                        elif count == 3:
                            review['value_score'] = score
                        count+=1

                    count = 0
                    comment_text = rv.find('div', attrs={'class': 'review-comments'})
                    comment = comment_text.findAll('p')
                    for p in comment:
                        p = remove_strong(str(p))
                        p = remove_non_ascii(p)
                        
                        if count == 0:
                            review['comment_pros'] = p
                        elif count == 1:
                            review['comment_cons'] = p
                        elif count == 2:
                            review['comment_overall'] = p
                        count+=1
                    
                    recommend = rv.findAll('div', attrs={'class': 'gauge-wrapper'})                     
                    for result in recommend:
                        score = result.attrs['data-rating']
                        review['recommend'] = score
                    
                    reviews.append(review)
                    
                    if product == '168797' or product == '170397':
                        run = False
            else:
                run = False
        else:
            run = False
        
        page+=1

# Merge the list of dicts to a dataframe
reviews_df = pd.DataFrame(reviews)

# Save the file as a pickle and csv for additional post processing
reviews_df.to_csv(csv, index=False)
reviews_df.to_pickle(pkl)

print('\nSAVE: {}'.format(csv))
print('SAVE: {}'.format(pkl))

# Testing pickle import for Jupyter
# reviews_df = pd.read_pickle(pkl)

pros = []
cons = []
overall = []
title = []

print()
for product in product_ids:
    print('STRINGIFY: {}'.format(product))
    df = reviews_df.loc[reviews_df['product_id'] == product]
    
    df1 = df.filter(['title'], axis=1)
    df1 = df1.dropna(subset = ['title'])
    dump = ' '.join(df1['title'].values)
    title.append(dump)
    #df1.apply(lambda x: stringify_reviews(x))
    
    df1 = df.filter(['comment_pros'], axis=1)
    df1 = df1.dropna(subset = ['comment_pros'])
    dump = ' '.join(df1['comment_pros'].values)
    pros.append(dump)
    #df1.apply(lambda x: stringify_reviews(x))
  
    df1 = df.filter(['comment_cons'], axis=1)
    df1 = df1.dropna(subset = ['comment_cons'])
    dump = ' '.join(df1['comment_cons'].values)
    cons.append(dump)
    #df1.apply(lambda x: stringify_reviews(x))

    df1 = df.filter(['comment_overall'], axis=1)
    df1 = df1.dropna(subset = ['comment_overall'])
    dump = ' '.join(df1['comment_overall'].values)
    overall.append(dump)
    #df1.apply(lambda x: stringify_reviews(x))

'''
Create wordclouds for each user piece of user provided data. This should be wrapped up in a function,
an is currnetly only testing. 

print('\nWORDCLOUD: Title')
title = ' '.join(title)
title = normalize_data(title)
wc = create_wordcloud(title)

fig = plt.gcf()
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
fig.set_figheight(12)
fig.set_figwidth(15)
plt.show()
fig.savefig('Title_Wordcloud_1.png', format='png', dpi=300)
    
# lower max_font_size
wc = create_wordcloud_2(title)
fig1 = plt.gcf()
fig1.set_figheight(12)
fig1.set_figwidth(15)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
fig1.savefig('Title_Wordcloud_2.png', format='png', dpi=300)

print('WORDCLOUD: Pros')
pros = ' '.join(pros)
pros = normalize_data(pros)
wc = create_wordcloud(pros)

fig2 = plt.gcf()
fig2.set_figheight(12)
fig2.set_figwidth(15)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
fig2.savefig('Pros_Wordcloud_1.png', format='png', dpi=300)
    
# lower max_font_size
wc = create_wordcloud_2(pros)
fig3 = plt.gcf()
fig3.set_figheight(12)
fig3.set_figwidth(15)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
fig3.savefig('Pros_Wordcloud_2.png', format='png', dpi=300)

print('WORDCLOUD: Cons')
cons = ' '.join(cons)
cons = normalize_data(cons)
wc = create_wordcloud(cons)

fig4 = plt.gcf()
fig4.set_figheight(12)
fig4.set_figwidth(15)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
fig4.savefig('Cons_Wordcloud_1.png', format='png', dpi=300)
    
# lower max_font_size
wc = create_wordcloud_2(cons)
fig5 = plt.gcf()
fig5.set_figheight(12)
fig5.set_figwidth(15)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
fig5.savefig('Cons_Wordcloud_2.png', format='png', dpi=300)

print('WORDCLOUD: Overall')
overall = ' '.join(overall)
overall = normalize_data(overall)
wc = create_wordcloud(overall)

fig6 = plt.gcf()
fig6.set_figheight(12)
fig6.set_figwidth(15)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
fig6.savefig('Title_Wordcloud_1.png', format='png', dpi=300)
    
# lower max_font_size
wc = create_wordcloud_2(overall)
fig7 = plt.gcf()
fig7.set_figheight(12)
fig7.set_figwidth(15)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
fig7.savefig('Title_Wordcloud_2.png', format='png', dpi=300)

print('WORDCLOUD: All')
merged = title + ' ' + pros + ' ' + cons + ' ' + overall
final = normalize_data(merged)
wc = create_wordcloud(final)

fig8 = plt.gcf()
fig8.set_figheight(12)
fig8.set_figwidth(15)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
fig8.savefig('All_Wordcloud_1.png', format='png', dpi=300)
    
# lower max_font_size
wc = create_wordcloud_2(final)
fig9 = plt.gcf()
fig9.set_figheight(12)
fig9.set_figwidth(15)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
fig9.savefig('All_Wordcloud_2.png', format='png', dpi=300)

'''

# Additinal testing
from collections import Counter

freq = final.split()
Counter(freq).most_common(20)

tmp = reviews_df.filter(['feature_score', 'value_score', 'customer_service_score', 'feature_score', 'recommend'])
tmp = tmp.dropna(axis='rows')
tmp['feature_score'] = tmp['feature_score'].astype(int)
tmp['value_score'] = tmp['value_score'].astype(int)
tmp['customer_service_score'] = tmp['customer_service_score'].astype(int)
tmp['feature_score'] = tmp['feature_score'].astype(int)
tmp['recommend'] = pd.to_numeric(tmp['recommend'], errors='coerce')
'''
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import numpy as np

trace1 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=12,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)
data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
plotly.tools.set_credentials_file(username='af001', api_key='SmDNnZnbnaA9Y4yZJAdw')
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='simple-3d-scatter')
'''
import seaborn as sns

sns.set(style="white")

# Generate a large random dataset
rs = np.random.RandomState(33)
d = pd.DataFrame(data=tmp)

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Additional testing
import plotly.plotly as py
import plotly.graph_objs as go
tmp = reviews_df.filter(['feature_score', 'value_score', 'customer_service_score', 'recommend'])
tmp = tmp.dropna(axis='rows')
x = tmp['feature_score'].values.astype(int)
y = tmp['value_score'].values.astype(int)
tmp['customer_service_score'] = tmp['customer_service_score'].astype(int)
tmp['recommend'] = pd.to_numeric(tmp['recommend'], errors='coerce')

y = tmp['value_score'].values
z = tmp['customer_service_score'].values
w = tmp['feature_score'].values
v = tmp['recommend'].values

data = [go.Histogram(x=x)]
py.iplot(data, filename='basic histogram')
