################################################################################
#                                                                              #                                                         #
# Author: Anton                                                                #
# Date: 20170811                                                               #                                                       #
#                                                                              #
################################################################################

import re
import pandas as pd 
import numpy as np
from elasticsearch import Elasticsearch, helpers 
from pandas.io.json import json_normalize
from __future__ import division

# Create an elastic search client connector to query the database over the 
# Northwestern VPN. Pull down all of the emails and store them in selectdocs. 
es=Elasticsearch('http://enron:spsdata@129.105.88.91:9200')
query={"query" : {"match_all" : {}}}
scanResp=helpers.scan(client= es, query=query, scroll="10m", index="enron", 
                      doc_type="email", timeout="10m")
selectdocs = [resp['_source'] for resp in scanResp]
print(len(selectdocs))

# Create a dataframe from the email data stored in selectdocs. This dataframe 
# will be used to query later on, and will serve as a reference for validating 
# output
df = pd.io.json.json_normalize(selectdocs)
df.columns = df.columns.map(lambda x: x.split(".")[-1])
print df

# Create two separate lists, one for to email addresses and one for from 
# email addresses. These will hold the msgId to email mapping as a dict. 
to_list=[]
from_list=[]

# Create a list of dicts that contain a mapping of msgIDs to addresses. 
# Since the to field has multiple addresses, split the addresses and map 
# each email address to the msgID. If an error occurs, pass the error and 
# continue. A list for to and from mappings will be generated for processing. 
for msg in selectdocs:
    try:
        headers = msg['headers']
        msg_id = headers['Message-ID']
        msg_from = headers['From']
        from_list.append({'msgID': msg_id, 'address': msg_from})
    except KeyError:
        pass
    try:
        msg_to=headers['To'].split()
        for tmsg in msg_to:
            to_list.append({'msgID': msg_id, 'address': tmsg.strip()})
    except KeyError:
        pass
    
# Get possible combinations of ken's email using regex. If it is unique, 
# remove any trailing comma that may exist and if it is unique, store 
# it in the list. Note: This regex will need lots of work. It will match 
# mark.lay, so we have to account for this in our processing
match = ('[s]{0,3}k+[enth]{0,6}[lL_\.]{0,3}lay.*@enron[\.]+[comnet]{3}|ken[.comuniatios]+@enron[\.]+[comnet]{3}|chairman[\._]+ken@enron[\.]+[comnet]{3}|.+chairman@enron[\.]+[comnet]{3}|lay[._]{0,1}k.+@enron[\.]+[comnet]{3}|ken[.skilng]+@enron[\.]+[comnet]{3}|ken[.board]+@enron[\.]+[comnet]{3}')
         
comp = re.compile(match)
ken = []

for email in to_list:
    result = comp.match(email['address'])
    if result is not None:
        result = result.group(0).replace(',', '')
        if result not in ken:
            ken.append(result)

# Transform the to and from list into dataframes that contain emails ONLY 
# assocoated with Ken. The dataframe is created from the list of dicts, and 
# applys the ken regex to extract the email and messages associated to
# ken lay. Remove false positive matches for 'mark.lay@enron.com', and remove 
# any trailing commas from the emails. Future fix is to ajust the ken regex 
# to prevent the 'mark.lay' mathes. 
def find_ken(addrs):
    df = pd.io.json.json_normalize(addrs)
    df.columns = df.columns.map(lambda x: x.split(".")[-1])
    mask = np.column_stack([df[col].str.contains(r"(?=("+'|'.join(ken)+r"))", 
                                                 na=False) for col in df])
    result = df.loc[mask.any(axis=1)]
    result = result[result.address.str.contains("mark") == False]
    result['address'] = result['address'].str.replace(',','')
    # result.to_csv("test3.csv", sep=',', encoding='utf-8')
    return result

# Call the find_ken functions and pass it the appropriate to/from list. 
# Reset the indices for each dataframe and create a merged dataframe. This 
# gives us a to, from, and combined dataframe to manipulate / query.
to_df = find_ken(to_list)
to_df.reset_index(drop=True, inplace=True)

from_df = find_ken(from_list)
from_df.reset_index(drop=True, inplace=True)

combined_df = to_df.append(from_df, ignore_index=True)

# Get counts for unique emails. Since Ken often includes his other emails 
# in the to line, count the number of unique msgIds for each to/from dataframe. 
# Instead of using groupby again for the combined, just add the two results for 
# a total. Get the frequency of each email used overall, and print the results
to_gr = to_df.groupby(['msgID'])
from_gr = from_df.groupby(['msgID'])
freq = combined_df['address'].value_counts()

print '[+] Email stats for Ken Lay:'
print 'Count of unique email addresses used by Ken Lay: %s' % len(freq)
print 'Count of unique emails to Ken Lay: %s' % len(to_gr)
print 'Count of unique emails from Ken Lay: %s' % len(from_gr)
print 'Count of unique emails to/from Ken Lay combined: %s' % (len(to_gr)+len(from_gr))
print '\n[+] Frequency of emails used by Ken Lay:'
print freq

'''
RESULT:
[+] Email stats for Ken Lay:
Count of unique email addresses used by Ken Lay: 18
Count of unique emails to Ken Lay: 3107
Count of unique emails from Ken Lay: 598
Count of unique emails to/from Ken Lay combined: 3705

[+] Frequency of emails used by Ken Lay:
kenneth.lay@enron.com           2130
klay@enron.com                   967
office.chairman@enron.com        432
chairman.ken@enron.com           157
kenneth_lay@enron.com             23
ken.communications@enron.com      11
ken.lay@enron.com                  3
lay.kenneth@enron.com              3
kenneth_lay@enron.net              3
ken_lay@enron.com                  2
k_lay@enron.com                    2
ssskenneth.lay@enron.com           2
kenneth.l.lay@enron.com            2
ken_lay@enron.net                  1
kenlay@enron.com                   1
k.lay@enron.com                    1
k.l.lay@enron.com                  1
kennethlay@enron.com               1
Name: address, dtype: int64
'''

# Similar to the earlier implementation of get_ken, but in reverse. This time take 
# each list and create a dataframe of addresses that are not associated with Ken Lay. 
# Add 'mark.lay' addresses back to the dataframe because they match our regex. Strip 
# the trailing comma from each email, if it exists. 
def get_addrs(addrs):
    df = pd.io.json.json_normalize(addrs)
    df.columns = df.columns.map(lambda x: x.split(".")[-1])
    mask = np.column_stack([~df[col].str.contains(r"(?=("+'|'.join(ken)+r"))", 
                                                  na=False) for col in df])
    result = df.loc[mask.any(axis=1)]
    #mask = np.column_stack([df[col].str.contains(r"mark.lay@enron.com|mark.lay@enron.net", 
    #                                             na=False) for col in df])
    result = result.append(df.loc[mask.any(axis=1)], ignore_index=True)
    result['address'] = result['address'].str.replace(',','')
    return result

# Return a dataframe that contains emails from the from column that do not include Ken's 
# emails. Reset the the index and merge the dataframe that includes the messages sent by 
# Ken to the messages that were received by other people. The merge will be on the msgId 
# that was previously mapped to email addresses. Find the frequency of each email address, 
# and print the top 5 emails that sent emails to Ken
addrs = get_addrs(from_list)
addrs.reset_index(drop=True, inplace=True)
merge1 = pd.merge(to_df, addrs, how='inner', on='msgID')
freq = merge1['address_y'].value_counts()
print '[+] Top 5 email addresses sending mail to Ken Lay:'
print freq.head(5)

'''
RESULT:

[+] Top 5 email addresses sending mail to Ken Lay:
leonardo.pacheco@enron.com      374
kenneth.thibodeaux@enron.com    164
rosalee.fleming@enron.com       160
simone.rose@enron.com           116
karen.denne@enron.com           116
Name: address_y, dtype: int64
'''

addrs = get_addrs(to_list)
addrs.reset_index(drop=True, inplace=True)
merge2 = pd.merge(from_df, addrs, how='inner', on='msgID')
freq = merge2['address_y'].value_counts()
print '\n[+] Top 5 email addresses Ken Lay sends mail to:'
print freq.head(5)

'''
RESULT:

[+] Top 5 email addresses Ken Lay sends mail to:
all.worldwide@enron.com                 638
dl-ga-all_enron_worldwide1@enron.com     94
dl-ga-all_enron_worldwide2@enron.com     84
dl-ga-all_enron_worldwide@enron.com      62
l..wells@enron.com                       56
Name: address_y, dtype: int64
'''

# Date parser function is used to parse the date found in the emails to a pandas 
# readable date. The basic functionality splits the date string, and then uses string 
# formatting to re-arrange the date field. For any empy cells, assign the value of NaN
def date_parser(date):
    # Reference Format: Mon, 13 Nov 2000 10:46:00 -0800 (PST)
    month={'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,
           'Oct':10,'Nov':11,'Dec':12}
    
    split = str(date).split()   
    try:
        return "%s-%s-%s %s" % (split[3],month[split[2]],split[1],split[4])
    except IndexError:
        return np.nan

# Merge emails from Ken to the main dataframe, and match on msgId. Once the new dataframe 
# is created, apply the date_parse function to the 'Date' column. This will return a series 
# of strings that are formatted as dates for emails associated with Ken.
from_ken = pd.merge(from_df, df, how='inner', left_on='msgID', right_on='Message-ID')
parsed_df = from_ken['Date'].apply(date_parser)

# Attempt to assign the date strings to a pandas datetime type. This breaks on NaN values 
# that we assigned earlier, so pass over them
try:
    parsed_df = pd.to_datetime(parsed_df)
except:
    pass

# General statistics to show how many emails are being analyzed before vs after the banksuptsy.
bankrupt = '2001-12-02'
before = parsed_df[(parsed_df < bankrupt)].count()
before_percent = before / len(from_ken.index)
after = parsed_df[(parsed_df >= bankrupt)].count()
after_percent = after / len(from_ken.index)

print '[+] Email Analysis:'
print '[*] Before bankruptcy: %s emails or %s percent' % (before, round(before_percent, 3))
print '[*] After bankruptcy: %s emails or %s percent' % (after, round(after_percent, 3))

# Identify which emails were sent before and after enron declared bankruptsy on Dec. 2, 2001. 
# Since we cast the date to type datetime64, we can leverage the .dt class to perform operations. 
# Group the emails by year and month, and then get the median of emails sent by Ken per month. 
# Compare these two values against each other to determine if the rate of emails sent by Ken increased 
# or decreased based on the historical trends. Don't use mean...the last month had a spike in emails 
# and skewed the data. 
before = parsed_df[(parsed_df < bankrupt)]
after = parsed_df[(parsed_df >= bankrupt)]
res_before = before.groupby([before.dt.year, before.dt.month]).count().median()
res_after = after.groupby([after.dt.year, after.dt.month]).count().median()

print '[*] Median of emails sent by Ken before bankruptsy: %s' % res_before
print '[*] Median of emails sent by Ken after bankruptsy: %s' % res_after

if (res_before > res_after):
    print ('[*] Analysis of the medians identified that email activity decreased by %s emails' 
           % (res_before-res_after))
elif (res_before < res_after):
    print ('[*]Analysis of the medians identified that email activity increased by %s emails'
           % (res_after-res_before))
else:
    '[*] There was no change in email activity after bankruptcy'
    
'''
RESULT:

[+] Email Analysis:
[*] Before bankruptcy: 589 emails or 0.985 percent
[*] After bankruptcy: 9 emails or 0.015 percent
[*] Median of emails sent by Ken before bankruptsy: 12.5
[*] Median of emails sent by Ken after bankruptsy: 9.0
[*] Analysis of the medians identified that email activity decreased by 3.5 emails
'''

# Use matplotlib to get a visualization of how often Ken normally sends emails. Group the 
# emails by year and month, and display to stdout
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')
ax = parsed_df.groupby([parsed_df.dt.year, parsed_df.dt.month]).count().plot(
    kind="bar", title="Ken's Sent Email History")
ax.set_xlabel("Date")
ax.set_ylabel("Count")
plt.show()

# The objective of this part is to find references of Aurthur Andersen in emails 
# to or from Ken. Since we created a dataframe of the elastic search results, merge 
# the dataframes we have for from_df and to_df with the main df. This will allow access 
# to the body of the emails
from_ken_to_greg = pd.merge(from_df, df, how='inner', left_on='msgID', right_on='Message-ID')
to_ken_from_greg = pd.merge(to_df, df, how='inner', left_on='msgID', right_on='Message-ID')

# Use regex in str.contains to search for references. Store matching values to a dataframe. 
ref_to = to_ken_from_greg[to_ken_from_greg.body.str.contains(
    r'Aurthur|Andersen|andersen|aurthur', na=False)]
ref_from = from_ken_to_greg[from_ken_to_greg.body.str.contains(
    r'Aurthur|Andersen|andersen|aurthur', na=False)]

# Check to see if the dataframe is empy. If not, count the number of emails and display the 
# emails that contain a reference to Arthur Andersen to stdout.
print '[+] Arthur Andersen References:'
if len(ref_to.index) is not 0:
    print ('[*] Count of emails referencing Aurthur Andersen to Ken: %s' 
           % ref_to['msgID'].count())
    for msg in ref_to['msgID']:
        print '\t', msg
else:
    print '[*] Did not find any reference to Aurthur Anderen in emails to Ken!'

if len(ref_from.index) is not 0:
    print ('[*] Count of emails referencing Aurthur Andersen to Ken: %s'
           % ref_from['msgID'].count())
    for msg in ref_to['msgID']:
        print '\t', msg
else:
    print '[*] Did not find any reference to Aurthur Anderen in emails to Ken!'
    
'''
RESULT:

[+] Arthur Andersen References:
[*] Count of emails referencing Aurthur Andersen to Ken: 7
	<2065318.1075861418012.JavaMail.evans@thyme>
	<21293428.1075862346091.JavaMail.evans@thyme>
	<21685709.1075840233850.JavaMail.evans@thyme>
	<855413.1075840257504.JavaMail.evans@thyme>
	<12560137.1075840214584.JavaMail.evans@thyme>
	<24571255.1075840265068.JavaMail.evans@thyme>
	<14699860.1075861375899.JavaMail.evans@thyme>
[*] Did not find any reference to Aurthur Anderen in emails to Ken!
'''
