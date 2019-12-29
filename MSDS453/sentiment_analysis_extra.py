#################################################################
# Author: Anton, Extra Credit, 08/20/2018
#################################################################

import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from afinn import Afinn

# Define the path to the JSON files
path = '/Users/Anton/assignment3'

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

# Open the JSON file and read the data to a variable. Normalize the Reviews data 
# and map the values to columns
with open('%s/100506.json' % path, 'r') as input_file:
    jsondata=json.load(input_file)
    df = pd.io.json.json_normalize(jsondata['Reviews'])
    df.columns = df.columns.map(lambda x: x.split(".")[-1])
    
    # Headers contain special characters and are messy. Reformat each column header by
    # first getting the headers, and then running them through a series of regex.    
    headers = get_columns(df)
    for head in headers:
        name = re.sub(r'[^\x00-\x7F]+',' ', head)
        name = re.sub(r'[^A-Za-z]+', ' ', name)            
        name = name.lower()
        name = name.strip()
        name = re.sub(r'\s+', '_', name)
        df.rename(columns={head:name}, inplace=True)
        
    # Remove the html tags in some of the author and content fields
    df['author'] = df['author'].apply(lambda x: replace_bad_author(x))
    df['content'] = df['content'].apply(lambda x: replace_bad_content(x))

# Functoin to get the affinn sentiment score for the content (aka review)
def score(comment):
    return afinn.score(comment)

# Initialize affin
afinn = Afinn()

# Use the pandas apply method to perform sentiment analysis against each row.
# Return the results to a new column named 'affin'
df['affin'] = df['content'].apply(lambda x: score(x))

# Print the first 4 records
print '[+] First 4 Records:'
print df[['author','affin']].head(4)

'''
RESULT:

[+] First 4 Records:
          author  affin
0  luvsroadtrips  -30.0
1      estelle e   11.0
2     RobertEddy   -9.0
3        James R   -7.0
'''

# Print the descriptive statistics for the affin values. 
print '\n[+] Descriptive Statistics:'
print df['affin'].describe()

'''
RESULT:

[+] Descriptive Statistics:
count    48.000000
mean      2.375000
std      12.885131
min     -30.000000
25%      -4.000000
50%       1.000000
75%       8.000000
max      46.000000
Name: affin, dtype: float64
'''

# Idea from Stephen to include a scatter plot that depitcts the relationship
# between the affin score and overall rating. Graph shows that user rating
# is not necessarily related to sentiment based on the number of positive 
# sentiment values that user's rated as 1.0. 
# %matplotlib inline <-- Uncomment for jupyter notebook
x = pd.to_numeric(df['overall'])
y = df['affin']
plt.scatter(x, y)
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
plt.xlabel('Guest Overall Score (Rating)')
plt.ylabel('Affin Score (Sentiment)')
plt.title('Affin Score as a function of Overall Rating');
