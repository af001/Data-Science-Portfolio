################################################################################
#                                                                              #
# Assignment: GrEx2                                                            #
# Author: Anton                                                                #
# Date: 20170709                                                               #                                                       #
#                                                                              #
################################################################################
#                                   PART I                                     #
################################################################################

import pandas as pd
import numpy as np
import shelve as sh
from sqlalchemy import create_engine

path = '/Users/Anton/assignment2'

def parse_files():
    # Read the csv files into dataframes and return the data
    customers = pd.read_csv('%s/afg4627customer.csv' % path)
    items = pd.read_csv('%s/afg4627item.csv' % path)
    mail = pd.read_csv('%s/afg4627mail.csv' % path)
    return customers, items, mail
    
def get_columns(df):
    # Get column names --> aka variables
    headers = [col.encode('ascii', 'ignore') for col in df]
    return headers

 # Parse files and store them into dataframes   
customers, items, mail = parse_files()

# Get columns from 'items' dataframe and print the column names
# as well as first 4 records. This only asked for the item dataframe.
item_columns = get_columns(items)
print '[+] Item Columns: '
for col in item_columns:
    print '%s, ' % col,

print '\n\n[+] First 4 records from \'items\':\n', items.head(n=4)

# Decribe the data types of the columns in the DataFrame. Strip column info and only display
# the relevant info.
pd.set_option('max_info_columns', 0)
pd.set_option('memory_usage', False)
print '\n[+] Customers: \n', customers.info()
print '\n[+] Items: \n', items.info()
print '\n[+] Mail: \n', mail.info()

# RESULTS
#[+] Item Columns: 
#acctno,  qty,  trandate,  tran_channel,  price,  totamt,  orderno,  deptdescr,  
#
#[+] First 4 records from 'items':
#   acctno  qty    trandate tran_channel   price  totamt       orderno  \
#0   GGGSD    1  2009-11-18           RT   93.60   93.60  CCXUKCIXXXKI   
#1   GGGSD    1  2009-11-19           RT  213.60  213.60  CCXURVCXXXKI   
#2   GGGSD    1  2009-11-19           RT   93.60   93.60  CCXURVCXXXKI   
#3  WGDQLA    1  2009-06-09           RT  599.85  599.85  CCXXNNXXXXUX   
#
#              deptdescr  
#0  Portable Electronics  
#1  Portable Electronics  
#2  Portable Electronics  
#3            Home Audio  
#
#[+] Customers: 
#<class 'pandas.core.frame.DataFrame'>
#RangeIndex: 50000 entries, 0 to 49999
#Columns: 451 entries, acctno to endofline
#dtypes: float64(133), int64(126), object(192)None
#
#[+] Items: 
#<class 'pandas.core.frame.DataFrame'>
#RangeIndex: 77121 entries, 0 to 77120
#Columns: 8 entries, acctno to deptdescr
#dtypes: float64(2), int64(1), object(5)None
#
#[+] Mail: 
#<class 'pandas.core.frame.DataFrame'>
#RangeIndex: 30946 entries, 0 to 30945
#Columns: 17 entries, acctno to mail_16
#dtypes: int64(16), object(1)None

################################################################################
#                                   PART II                                    #
################################################################################

# Create a dataframe that only contains 'ACTIVE' buyers from customers and reset the index
active_customers = customers.loc[customers['buyer_status'] == 'ACTIVE']
active_customers.reset_index(drop=True, inplace=True)

# Create a Sqlalchemy engine used for mapping python objects to the database   
engine = create_engine('sqlite:////Users/Anton/assignment2/xyz.db', echo=False)

# Get the 'acctno' from the 'ACTIVE' customers
# Used as a filter for the active users in 'mail' and 'items'
customers_acctno_filter = active_customers.filter(['acctno'], axis=1)

# Filter 'mail' and 'items' based on 'ACTIVE' accounts using the 'acctno' filter
active_items = pd.merge(customers_acctno_filter, items, on='acctno', how='inner') 
active_mail = pd.merge(customers_acctno_filter, mail, on='acctno', how='inner') 

# Calculate 'heavy buyers' using rank, ignore warning 
pd.set_option('mode.chained_assignment',None)
active_customers['percentile'] = active_customers['ytd_sales_2009'].rank(pct=True)

# Write 'customers,' 'items,' and 'mail' to the sql database using sqlalchemy
active_customers.to_sql(name='customers', con=engine, if_exists = 'append', index=False)
active_items.to_sql(name='items', con=engine, if_exists = 'append', index=False)
active_mail.to_sql(name='mail', con=engine, if_exists = 'append', index=False)

# Debugging Code --> Used to verify data exists in the database
# print pd.read_sql_table(table_name='customers', con=engine)
# print pd.read_sql_table(table_name='items', con=engine)
# print pd.read_sql_table(table_name='mail', con=engine)

# Debugging SQL Statement to verify percentiles
# sqlstatement = 'SELECT * FROM customers WHERE percentile>=0.9'
# query = pd.read_sql_query(sqlstatement, con=engine)

# RESULTS
# Writes data to tables (customers, items, mail) in xyz.db

################################################################################
#                                   PART III                                   #
################################################################################

# Create the cust_sum dataframe and select customers above the 90th percentile
cust_sum = pd.DataFrame
cust_sum = customers_acctno_filter
cust_sum['heavy_buyer'] = active_customers['percentile']>=0.9

# Debugging code to verify customer is in the 90th percentile using numpy
# print query_customers[query_customers['ytd_sales_2009'] >= np.percentile(query_customers \
#       ['ytd_sales_2009'], 90)]

# Get all cc types and write bool if user has prem or reg version of the cc
cc = active_customers[['amex_prem','amex_reg','visa_prem','visa_reg','disc_prem',
                       'disc_reg','mc_prem','mc_reg']]>='Y'
cust_sum['amex'] = cc.amex_prem | cc.amex_reg
cust_sum['visa'] = cc.visa_prem | cc.visa_reg
cust_sum['disc'] = cc.disc_prem | cc.disc_reg
cust_sum['mc'] = cc.mc_prem | cc.mc_reg

print '\n[!] Credit card columns are \'Y\' or \'N\' values based on if a customer has a \
premium or regular credit card for that type (i.e. amex_prem OR amex_reg)\n'

# Replance bool values with 'Y' and 'N'; any 'U' values are considered 'N'
cust_sum.replace(False, 'N', inplace=True)
cust_sum.replace(True, 'Y', inplace=True)

# Add estimated hh income to cust_sum; Multiply by 1000 to get actual est hh income
# If value is zero, get the median income, as per experian key
cust_sum['hh_est_income'] = active_customers['inc_scs_amt_v4'] * 1000
cust_sum['hh_est_income'].replace(0, active_customers['med_inc'], inplace=True)

# Add gender for adult_1 and adult_2 to cust_sum; Replace NaN values with 'U' as per 
# the experian key
gender = active_customers[['adult1_g','adult2_g']]
cust_sum['adult1_g'] = gender['adult1_g']
cust_sum['adult2_g'] = gender['adult2_g']
cust_sum['adult1_g'].fillna('U', inplace=True)
cust_sum['adult2_g'].fillna('U', inplace=True)

# Add zip code and zip code 4 to cust_sum; Modify type to elimiate float
# Change 0 values in zip 4 or values that don't have 4 digits to 0000
zipcode = active_customers[['zip', 'zip4']]
cust_sum['zip'] = zipcode['zip']
cust_sum['zip4'] = zipcode['zip4']
cust_sum['zip'] = cust_sum['zip'].astype(int)
cust_sum['zip4'] = cust_sum['zip4'].fillna(0)
cust_sum['zip4'] = cust_sum['zip4'].astype(int)
cust_sum['zip4'][cust_sum['zip4'] < 999] = '0000'

# Write the new dataframe to the database as a table
cust_sum.to_sql(name='cust_sum', con=engine, if_exists = 'append', index=False)

# Print a count of rows in each table. Use a list and iterate each table
tables = ['customers','items','mail','cust_sum']
print '[+] Table Counts: '
for table in tables:
    sql = 'SELECT COUNT(acctno) FROM %s' % table
    query = pd.read_sql_query(sql, con=engine)
    print '\t%s: %s' % (table, str(query['COUNT(acctno)'][0]))

# Debugging Code --> Used to verify data exists in the new database
# print pd.read_sql_table(table_name='cust_sum', con=engine)

# RESULTS
#[!] Credit card columns are 'Y' or 'N' values based on if a customer has a 
# premium or regular credit card for that type (i.e. amex_prem OR amex_reg)

#[+] Table Counts: 
#	customers: 17490
#	items: 77115
#	mail: 13713
#	cust_sum: 17490

################################################################################
#                                   PART IV                                    #
################################################################################

# Get LAPSED customers from main customer dataframe and concat LAPSED and ACTIVE
# Changed from filtering and concatinating for accuracy
active_lapse_cust = customers.loc[(customers['buyer_status'] == 'ACTIVE') | 
                                  (customers['buyer_status'] == 'LAPSED')]

# Create a new filter to apply to the items dataframe. Extract 'acctno'
marketing = pd.DataFrame
marketing = active_lapse_cust.filter(['acctno'], axis=1)

# Create a variable of items that are contain ACTIVE and LAPSED accounts using the filter
marketing_items = pd.merge(marketing, items, on='acctno', how='inner') 

#Adjust index to be acctno --> easier to set_value and eliminate duplicates
marketing.set_index('acctno')

# Super inefficient having 3 for loops, but better than some prevous implementations
# Modified multiple times using built in functions. Apply() with a function was a little 
# faster, but slowed down on set_value again. Start with grouping by deptdescr and setting 
# a value of 'Y' for acctnos that have purchsed this type of item.
for descp, detail in marketing_items.groupby('deptdescr'):
    for index, row in detail.iterrows():
        marketing.set_value(row['acctno'], descp, 'Y')

# Calculate total purchases based on acctno. Group by acctno and get a sum of totamt        
for descp, detail in marketing_items.groupby('acctno'):
    marketing.set_value(descp, 'total_purchases', detail['totamt'].sum())

# Add buyer status, takes a bit longer here than adding it in the beginning,
# but it is easier to parse. Use interrow, which is slow, but accurate compared
# to other methods used.
for descp, detail in active_lapse_cust.groupby('buyer_status'):  
    for index, row in detail.iterrows():
        marketing.set_value(row['acctno'], 'buyer_status', descp)

# Do cleanup ops, drop acctno column since it is nan, drop nan rows   
marketing.drop('acctno', axis=1, inplace=True)     
marketing.dropna(axis=0, how='all', inplace=True)  

# Re-create 'acctno' column and reset the index. Replace nan values with 'N'
# When done, write to a csv in our path
marketing.insert(0, 'acctno', marketing.index)
marketing.reset_index(drop=True, inplace=True)
marketing.replace(np.nan, 'N', inplace=True)
marketing.to_csv('%s/xyz_marketing.csv' % path, sep=',', encoding='utf-8')

# Create a shelve for marketing object in our path. Used dbm as file extention, but can
# be anything.
db = sh.open('%s/xyz_shelve.dbm' % path)
db['df'] = marketing
db.close()

# RESULTS
# Writes xyz_shelve.dbm and xyz_marketing.csv to path

################################################################################
#                                   PART V                                     #
################################################################################

counter = 0
genders={'B':'Both','U':'Unknown','M':'Male','F':'Female'}
top_6 = {}

# This is the function that is used with apply(). Calculates values for each group and prints
# to stdout
def calc(value):
    global counter
    global top_6
    
    # Apply() has built in optimization, so the first group occurs twice. Prevent that by 
    # skipping the first round. 
    if counter == 0:
        counter+=1
    else:
        # Get a list of our columns. Kind of a hack, but the columns in the marketing df 
        # were capital letters. Look for that capital letter as a key, and create bool logic
        # to determine the number of 'Y' values. Count the True statements and get a sum of
        # the purchases. 
        cols = get_columns(value)
        print '\n[+] Gender: %s' % (genders[value['adult1_g'].iloc[0]])
        print '\tTotal Purchases: $%s' % value['total_purchases'].sum()
        for col in cols:
            if col[0].isupper():
                isY=value[col]=='Y'
                top_6[str(col)] = sum(isY)
        
        # To print the top 6, we take the list top_6 and reverse sort them and take the top 6
        # Print the result to stdout. Also reset for the next group
        top_6 = sorted(top_6.iteritems(), key=lambda (k, v): (-v, k))[:6]
        for item in top_6:
            print '\t%s: %s' % (item[0], item[1])
        top_6 = {}

# Using marketing dataframe because we can calculate 'Y' values easier. Also, the sum for each
# user is already calculated. Get the gender of active customers and merge them with marketing
# to add a column 'adult1_g' and to get rid of the LAPSED customers
gender = active_customers.filter(['acctno', 'adult1_g'])
top_categories = pd.merge(marketing, gender, on='acctno', how='inner')

# Apply our function to the merged dataframe. 
top_categories.groupby('adult1_g').apply(calc)

print '\n[+] DONE!!!'

# RESULTS
#[+] Gender: Both
#	Total Purchases: $3451.23
#	Mobile Electronics: 6
#	Small Appliances: 6
#	Mobile Electronic Accessories: 5
#	Home Audio: 2
#	Portable Electronics: 2
#	Cameras & Camcorder Accessori: 1
#
#[+] Gender: Female
#	Total Purchases: $5756596.83
#	Small Appliances: 6206
#	Mobile Electronic Accessories: 5827
#	Mobile Electronics: 4696
#	Home Audio: 3853
#	Portable Electronics: 2920
#	Cameras & Camcorder Accessori: 1389
#
#[+] Gender: Male
#	Total Purchases: $1485159.54
#	Small Appliances: 1609
#	Mobile Electronic Accessories: 1303
#	Home Audio: 1086
#	Mobile Electronics: 920
#	Portable Electronics: 736
#	Cameras & Camcorder Accessori: 332
#
#[+] Gender: Unknown
#	Total Purchases: $1132439.10
#	Small Appliances: 802
#	Mobile Electronic Accessories: 705
#	Mobile Electronics: 560
#	Home Audio: 508
#	Portable Electronics: 383
#	Cameras & Camcorder Accessori: 174
#
#[+] DONE!!!
