# -*- coding: utf-8 -*-

################################################################################
#                                                                              #
# Assignment: GrEx3                                                            #
# Author: Anton                                                                #
# Date: 20170727                                                               #                                                       #
#                                                                              #
################################################################################
#                                   PART I                                     #
################################################################################

import json
import re
import os
import pandas as pd
import numpy as np
import sys
from pathlib2 import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# Function used to get column headers throughout the application
def get_columns(df):
    # Get column names --> aka variables
    headers = [col.encode('ascii', 'ignore') for col in df]
    return headers

# Set variables that are used throughout the application
path = '/Users/Anton/assignment3'

# Set two variables in the beginning, used to hold values of the dataframes 
# and comments when parsing each file
comments = {}
hotel_dfs = []

# Look for .json files in our path, and iterate over each file for processing
pathlist = Path(path).glob('**/*.json')
for item in pathlist:

    # Because path is object not string
    json_file = str(item)
    
    # Open each .json file and process and load the data from the input file
    with open(json_file, 'r') as input_file:
        jsondat=json.load(input_file)
        
        # Store the hotel name as a variable. Since the name exists in the URL, extract the
        # name from the URL and set the variable that way. Use regex to extract and format.
        try:
            hotel_name = jsondat['HotelInfo']['Name']
        except KeyError:
            try:
                url = jsondat['HotelInfo']['HotelURL']
                p = re.compile(r'(?<=\-)([A-Z].*?)(?=\-)')
                search = p.search(url).group(1)
                hotel_name = str(search.replace('_',' '))
            except KeyError:
                print '\n[!] Parsing error generated from %s' % json_file
                print '[!] This file is a post-processed or non-hotel review file!'
                sys.exit(1)
        
        # Set the hotelID. This is needed for creating a dict of dicts of our comments.
        # Use map to extract each review and create a new column to hold each value.
        # Lastly, extract the content and append the comments list using the id as a key.
        try:
            hotel_id = jsondat['HotelInfo']['HotelID']
            df = pd.io.json.json_normalize(jsondat['Reviews'])
            df.columns = df.columns.map(lambda x: x.split(".")[-1])
            comments[int(hotel_id)] = str(df['Content'])
        except KeyError:
            print '\n[!] Parsing error generated from %s' % json_file
            print '[!] This file is a post-processed or non-hotel review file!'
            sys.exit(1)
        
        # Get rid of the unnecessary columns, if they exist. Use try/except for those that 
        # don't have the columns we are looking for. 
        try:
            df.drop(['AuthorLocation','Content','Title'], axis=1, inplace=True)
        except ValueError:
            df.drop(['Content'], axis=1, inplace=True)
        
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
        
        # Append the dataframe to our list so we can concat them once all files are processed.
        hotel_dfs.append(df)

        # Extract the number of reviews by counting the number of reviewIDs in the dataframe.
        top = df['reviewid'].describe()
        print '[+] %s Review Count: %s' % (hotel_name, int(top['count']))

'''
Results:

[+] Hotel Seattle Review Count: 48
[+] Deca Hotel Review Count: 1
[+] Riu Bambu Review Count: 710
[+] Park Shore Waikiki Review Count: 174
[+] Kendall Hotel and Suites Review Count: 52
[+] San Diego Marriott Mission Valley Review Count: 96
[+] Hotel Banys Orientals Review Count: 188
[+] Meninas Hotel Review Count: 167
[+] Balisandy Cottages Review Count: 1
[+] Plaza Madrid Review Count: 1
[+] BEST WESTERN PLUS Pioneer Square Hotel Review Count: 233
[+] BEST WESTERN Loyal Inn Review Count: 113
[+] BEST WESTERN PLUS Executive Inn Review Count: 137
[+] Comfort Inn & Suites Seattle Review Count: 36
[+] Christopher's Inn Review Count: 93
[+] BEST WESTERN Market Center Review Count: 54
[+] BEST WESTERN Airport Inn Review Count: 45
[+] Super 8 Phoenix Review Count: 25
[+] Lexington Hotel Central Phoenix Review Count: 57
[+] Grace Inn Phoenix Review Count: 44
[+] BEST WESTERN PLUS InnSuites Phoenix Hotel & Suites Review Count: 60
[+] A Victory Inn & Suites Phoenix North Review Count: 32
[+] Courtyard by Marriott Phoenix North Review Count: 19
[+] Renaissance Phoenix Downtown Review Count: 36
[+] Days Inn Camelback Phoenix and Conference Center Review Count: 39
[+] Days Inn I-17 & Thomas Review Count: 24
'''       

# Combine the dataframes into a signle dataframe. Merge on column names.
final = pd.concat(hotel_dfs, ignore_index=True)

# Convert the files to strings and numeric values. Strings include the date, reviewid, and
# author. All other values are floats. We need numeric values for retrieving statistics. 
# Keep all missing values as NaN for ease of describe() statistic more accurate since it 
# does not count NaN values. Change -1 values to NaN. Key: 0-5, -1 values equal missing values.
for col in get_columns(final):
    try:
        if col == 'overall' or col == 'value':
            final[col] = pd.to_numeric(final[col], downcast='float')
        else:
            final[col] = final[col].apply(lambda x : x.encode("utf-8"))
            final[col] = final[col].astype('str')
    except AttributeError:
        final[col] = pd.to_numeric(final[col], downcast='float')

final.replace(-1, np.nan, inplace=True)        
# Get the combined statistics of the overall column using describe().
print '\n[+] Overall Hotel Info (Combined):\n%s' % final.describe()['overall']

'''
Result:

[+] Overall Hotel Info (Combined):
count    2485.000000
mean        3.728370
std         1.300262
min         1.000000
25%         3.000000
50%         4.000000
75%         5.000000
max         5.000000
Name: overall, dtype: float64
'''

# Write the dataframe to a pickle in our path
final.to_pickle('%s/foltz_grr3.p' % path)

# Verify the pickle <- For debugging only, and to verify data was wrote to pickle.
# test = pd.read_pickle('%s/foltz_grr3.p' % path)
# print test

if os.path.isfile('%s/foltz_grr3.p' % path):
    print '\n[+] Pickle file wrote to: %s' % path
else:
    print '[!] Error writing pickle file to %s' % path
    
'''
Result:

[+] Pickle file wrote to: /Users/Anton/assignment3
'''

################################################################################
#                                   PART II                                    #
################################################################################

# On first run, you may need to download the stopwords. Uncomment the next three lines.
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

# Work regex magic again. Create a dict to hold parsed/formatted comments. Set stopwords.
filtered = {}
stop = stopwords.words('english')
for comment in comments:
    temp = []
    # Regex time!! Strip non-ascii chars, numbers, special chars, and convert to lower
    # Remove short words <-- <2 chars
    com = re.sub(r'[^\x00-\x7F]+',' ', comments[comment])
    com = re.sub(r'\d+', '', com)
    com = re.sub(r'[^A-Za-z]+', ' ', com)
    com = re.sub(r'^.$',' ', com)
    shortword = re.compile(r'\W*\b\w{1,2}\b')
    com = shortword.sub('', com)
    com = com.lower()
    
    # Tokenize the comments and remove all stopwords. Store the info to a list. Once done,
    # assign the list to the dict with the hotelid as the key.
    word_tokens = word_tokenize(com)
    for w in word_tokens:
        if w not in stop:
            temp.append(w)
    filtered[comment] = temp

# Create a final dict that will be our dict of dicts. Use the Counter library to get the
# frequency of each word. Store the dict from Counter using the hotelid key.
final_dict = {}
for item in filtered:
    final_dict[item] = dict(Counter(filtered[item]))

# Write the dict of dicts to a json file in our path. 
with open('%s/foltz_json.json' % path, 'w') as dump:
    json.dump(final_dict, dump, sort_keys=True, indent=4)

if os.path.isfile('%s/foltz_json.json' % path):
    print '\n[+] JSON file wrote to: %s' % path
else:
    print '[!] Error writing JSON file to %s' % path

'''
Result:

[+] JSON file wrote to: /Users/Anton/assignment3
'''
print
