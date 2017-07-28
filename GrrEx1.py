################################################################################
#
# Assignment: GrEx1
# Author: Anton
# Date: 20170630
#
################################################################################

import pandas as pd
import numpy as np
from platform import system as system_name 
from os import system as system_call 
from prettytable import PrettyTable

# Define global variables
renamed_columns = []
col_names_2016 = []
col_names_2015  = []
col_names_2014 = []
com_df_2016 = pd.DataFrame()
com_df_2015 = pd.DataFrame()
user_df_2016 = pd.DataFrame()
user_df_2015 = pd.DataFrame()

def parse_files():
    # Read the excel files to a dataframe
    xl1_2016 = pd.ExcelFile('/Users/Anton/assignment1/2016_SFO_Customer_Survey.xlsx')
    xl2_2014 = pd.ExcelFile('/Users/Anton/assignment1/2014_SFO_Customer_Survey.xlsx')
    
    # Read csv files, take all files and create a dataframe of the data
    df_2015 = pd.read_csv('/Users/Anton/assignment1/2015_SFO_Customer_survey.csv')
    df_2016 = xl1_2016.parse('Data')
    df_2014 = xl2_2014.parse('Sheet 1')
    
    # Parse the required columns and store them in separate frames
    df_2016 = parse_2016(df_2016)
    df_2015 = parse_2015(df_2015)
    df_2014 = parse_2014(df_2014)
    
    # Use append to combine dataframes for each year of SFO data
    final_df = df_2016.append(df_2015, ignore_index=True)
    final_df = final_df.append(df_2014, ignore_index=True)
    
    # Fill 'Blank' and 0 values with NaN for consistency
    final_df.fillna(np.nan, inplace=True)
    final_df.replace('Blank', np.nan, inplace=True)
    
    # Display integers, not floats 
    pd.options.display.float_format = '{:,.0f}'.format
    
    return final_df
    
def get_columns(df):
    # Get column names --> aka variables
    headers = [col.encode('ascii', 'ignore') for col in df]
    return headers
          
def parse_2016(df_2016):
    global renamed_columns
    global com_df_2016
    global user_df_2016
    global col_names_2016
    
    # Define column names to filter from main dataframe
    col_names_2016 = ['*RESPNUM','Q7ART','Q7FOOD','Q7STORE','Q7SIGN',
    'Q7WALKWAYS','Q7SCREENS','Q7INFODOWN','Q7INFOUP','Q7WIFI','Q7ROADS',
    'Q7PARK','Q7AIRTRAIN','Q7LTPARKING','Q7RENTAL','Q7ALL','Q9BOARDING',
    'Q9AIRTRAIN','Q9RENTAL','Q9FOOD','Q9RESTROOM','Q9ALL','Q10SAFE',
    'Q12PRECHECKRATE','Q13GETRATE','Q14FIND','Q14PASSTHRU','Q16LIVE']
    
    # Filter the variables, replace improperly named variables, and append
    # the year to the column list
    df = df_2016.filter(col_names_2016, axis=1)
    df.rename(columns = {'*RESPNUM':'RESPNUM'}, inplace = True)
    df['YEAR'] = 2016
    renamed_columns.append('2016  :  \'*RESPNUM\' --> \'RESPNUM\'')
    
    # Parse Part II data and store it
    com_columns = ['Q8COM', 'Q8COM2', 'Q8COM3', 'Q8COM4', 'Q8COM5']
    com_df_2016 = df_2016.filter(com_columns, axis=1)
    
    # Replace non-integer variables for filtering Part IV data
    df_2016.replace({'Male':1,'Female':2,'Other':3}, inplace=True)
    df_2016.replace({'Under 18':1,'18-24': 2,'25-34':3,'35-44':4,'45-54':5,
    '55-64':6,'65-Over':7,'Don\'t Know or Refused':8,'Under 32':0,'Under 30':0,
    'Under 29':0,'Under 28':0,'Under 27':0,'Under 26':0,'Under 25':0,
    'Under 24':0,'Under 23':0,'Under 22':0,'Under 21':0,'Under 20':0,
    'Under 19':0,'Under 31':0}, inplace=True)
    
    # Parse Part IV data and store it. Add year column
    user_columns = ['*RESPNUM','INTDATE','DESTGEO','DESTMARK','Q2PURP1','Q2PURP2',
    'Q2PURP3','Q3GETTO1','Q3GETTO2','Q3GETTO3','Q3PARK','Q4BAGS','Q4STORE',
    'Q4FOOD','Q4WIFI','Q5TIMESFLOWN','Q5FIRSTTIME','Q6LONGUSE','Q16LIVE',
    'Q17CITY','Q17STATE','Q17ZIP','Q17COUNTRY','HOME','Q18PET','Q19AGE',
    'Q20GENDER','Q21INCME','Q22FLY','Q23SJC','Q23OAK','LANG']
   
    user_df_2016 = df_2016.filter(user_columns, axis=1)
    user_df_2016.rename(columns = {'*RESPNUM':'RESPNUM'}, inplace = True)
    user_df_2016['YEAR'] = 2016
        
    return df
    
def parse_2015(df_2015):
    global renamed_columns
    global col_names_2015
    global user_df_2015
    global com_df_2015
    
    # Define column names to filter from main dataframe
    col_names_2015 = ['RESPNUM','Q7ART','Q7FOOD','Q7STORE','Q7SIGN',
    'Q7WALKWAYS','Q7SCREENS','Q7INFODOWN','Q7INFOUP','Q7WIFI','Q7ROADS',
    'Q7PARK','Q7AIRTRAIN','Q7LTPARKING','Q7RENTAL','Q7ALL','Q9BOARDING',
    'Q9AIRTRAIN','Q9RENTAL','Q9FOOD','Q9RESTROOM','Q9ALL','Q10SAFE',
    'Q12PRECHEKCRATE','Q13GETRATE','Q14FIND','Q14PASSTHRU','Q16LIVE']

    # Filter the variables, replace improperly named variables, and append
    # the year to the column list
    df = df_2015.filter(col_names_2015, axis=1)
    df.rename(columns = {'Q12PRECHEKCRATE':'Q12PRECHECKRATE'}, inplace = True)
    df['YEAR'] = 2015
    renamed_columns.append('2015  :  \'Q12PRECHEKCRATE\' --> \'Q12PRECHECKRATE\'')
    
    # Parse Part II data and store it
    com_columns = ['Q8COM1', 'Q8COM2', 'Q8COM3']
    com_df_2015 = df_2015.filter(com_columns, axis=1)
    
    # Define Part IV data, filter, rename, and store
    user_columns = ['RESPNUM','INTDATE','DESTGEO','DESTMARK','Q2PURP1',
    'Q2PURP2','Q2PURP3','Q3GETTO1','Q3GETTO2','Q3GETTO3','Q3PARK','Q4BAGS',
    'Q4STORE','Q4FOOD','Q4WIFI','Q5TIMESFLOWN','Q5FIRSTTIME','Q6LONGUSE',
    'Q16LIVE','HOME','Q17CITY','Q17STATE','Q17ZIP','Q17COUNTRY','Q18AGE',
    'Q19GENDER','Q20INCOME','Q21FLY','Q22SJC','Q22OAK','LANG']
    user_df_2015 = df_2015.filter(user_columns, axis=1)
    user_df_2015.rename(columns = {'Q18AGE':'Q19AGE','Q19GENDER':'Q20GENDER',
    'Q20INCOME':'Q21INCME','Q21FLY':'Q22FLY','Q22SJC':'Q23SJC',
    'Q22OAK':'Q23OAK'}, inplace = True)
    user_df_2015['Q18PET'] = 5
    user_df_2015['YEAR'] = 2015
    
    return df

def parse_2014(df_2014):
    global col_names_2014
    col_names_2014 = ['RESPNUM','Q7ART','Q7FOOD','Q7STORE','Q7SIGN',
    'Q7WALKWAYS','Q7SCREENS','Q7INFODOWN','Q7INFOUP','Q7WIFI','Q7ROADS',
    'Q7PARK','Q7AIRTRAIN','Q7LTPARKING','Q7RENTAL','Q7ALL','Q9BOARDING',
    'Q9AIRTRAIN','Q9RENTAL','Q9FOOD','Q9RESTROOM','Q9ALL','Q10SAFE',
    'Q12PRECHECKRATE','Q13GETRATE','Q14FIND','Q14PASSTHRU','Q16LIVE']
    
    df = df_2014.filter(col_names_2014, axis=1)
    df['YEAR'] = 2015
    
    return df
    
def organize_data(df):
    # Organize the columns before writing to file. 
    df = df[['RESPNUM', 'YEAR', 'Q16LIVE', 'Q7ART', 'Q7FOOD', 'Q7STORE', 'Q7SIGN', 
             'Q7WALKWAYS', 'Q7SCREENS', 'Q7INFODOWN', 'Q7INFOUP', 'Q7WIFI', 'Q7ROADS', 
             'Q7PARK', 'Q7AIRTRAIN', 'Q7LTPARKING', 'Q7RENTAL', 'Q7ALL', 'Q9BOARDING', 
             'Q9AIRTRAIN', 'Q9RENTAL', 'Q9FOOD', 'Q9RESTROOM', 'Q9ALL', 'Q10SAFE', 
             'Q12PRECHECKRATE', 'Q13GETRATE', 'Q14FIND', 'Q14PASSTHRU']]  
    return df
    
def print_summary(section, df):
    count = 4
    cols = {2016:col_names_2016,2015:col_names_2015,2014:col_names_2014}
    add_cols = ['2016','2015','2014']
    
    # Statements for displaying data, show renamed columns
    if section == 1:
        print '[+] Renamed Columns:'
        for col in renamed_columns:
            print '\t%s' % col
    # Show original variable names 
    elif section == 2:  
        for year, col in cols.iteritems():
            print '[+] %s Variable Names: [Count = %s]' % (year,len(col))
            for ix in col:
                if count > 0:
                    print '\t%-13s|' % ix,
                    count-=1
                else:
                    print '\n\t%-13s|' % ix,
                    count=3
            count = 4
            print '\n'
    # Show table to show the meaning of codes used for SFO data. Pretty print
    # the data in a table.
    elif section == 3:
        describe_dataset(df)
        legend1={5:['Outstanding', 'Clean', 'Safe', 'Much Better', 'Easy'],
        4:['', '', '', 'Somewhat Better', ''],3:['', 'Average', 'Neutral', 
        'Same', 'Average'],2:['', '', '', 'Somewhat Worse', ''],
        1:['Unacceptable', 'Dirty', 'Not Safe', 'Much Worse', 
        'Difficult'],6:['NA', 'NA', 'Don\'t Know', 'NA', 'NA']}
        legend2={1:'County Bay Area',2:'Northern California outside the Bay Area',
        3:'In another region'}
        
        # Prety print the tables
        table = PrettyTable(['Rating', 'Q7', 'Q9', 'Q10', 'Q12', 'Q13:Q14'])       
        for rating, desc in legend1.iteritems(): 
            table.add_row([rating,desc[0],desc[1],desc[2],desc[3],desc[4]])         
        
        print '\n[+] Coding:\n', table, '\n'
              
        table = PrettyTable(['Code', 'Live in...'])
        for rating,desc in legend2.iteritems():
            table.add_row([rating,desc[0:]]) 
        print table  
    # Show years that had columns added (2016,2015,2014 data)
    elif section == 4:
        print '\n[+] Added Columns: '
        for col in add_cols:
            print '\t%s  :  \'YEAR\'' % col      
    # Show variables used for part IV
    elif section == 5:
        describe_dataset(df)
        print '\n[+] Combined Variables:' 
        for ix in get_columns(df):
                if count > 0:
                    print '\t%-13s|' % ix,
                    count-=1
                else:
                    print '\n\t%-13s|' % ix,
                    count=3  
        print '\n'
        
def describe_dataset(df):
    # Print description of dataframe. Could use df.describe, but result doesn't
    # describe the physical characteristics of the data, as requested. 
    print '\n[+] Dataframe Description:'
    print '\tDimentions: %s x %s (Rows x Columns)' % (str(len(df.index)), 
                                                          str(len(get_columns(df))))
    print '\tEntries: %s' % (len(df.index) * len(get_columns(df)))
    print '\tMissing/NA/Blank Responses: %s' % sum(df.isnull().sum(axis=0))

def start_program():
    # Clear command as function of OS. Shows fresh screen on start
    command = "-cls" if system_name().lower()=="windows" else "clear"
    system_call(command)
    
    print '[+] GrrExercise #1: Anton Foltz'
    print '[+] Predict420 - Northwestern'
    print '[+] June 28, 2017\n'

def do_partI():
    # Part I functions. Used to parse, display, pickle, and store data to file
    parsed_data = parse_files()
    parsed_data = organize_data(parsed_data)
    print_summary(2, None)
    print_summary(1, None)
    print_summary(4, None)
    print_summary(3, parsed_data)
    parsed_data.to_csv('/Users/Anton/assignment1/out.csv', encoding='utf-8')
    parsed_data.to_pickle('/Users/Anton/assignment1/foltz_pickle.p')
    return parsed_data

def get_top_comments(df):
    # Get top 3 comments
    codes = {103:'Going through security takes too long/add more checkpoints',
    202:'Need more places to eat/drink/more variety in types of restaurants', 
    999:'Good experience/keep up the good work/other positive comment'}
    
    # Unstack the dataframe to make one column; then count
    unstack = df.unstack(level=0)
    counts = unstack.value_counts()
    print '\n[+] Top Comments: [Format = Count - [Code] Comment]'
    counts = counts.head(n=3)

    # Iterate over list and pretty print the output to stdout
    for row in counts.index.tolist():
        print '\t%s - [%s] %s' % (counts[row], int(row), codes[int(row)])

def do_partII():
    # Part II functions. Used to parse, display, and pickle the dataframe
    df = com_df_2016.append(com_df_2015, ignore_index=True)
    df.replace(0, np.nan, inplace=True)
    get_top_comments(df)
    df.to_pickle('/Users/Anton/assignment1/foltz_pickle.p')
    
def find_location_trend(df):
    # Get count of overall assessment by respondent residence
    codes = {1:'County Bay Area', 2:'Northern California outside the Bay Arera', 
    3:'In another region', 0:'Blank/Multiple Responses'}
    
    # Count the frequency, and pretty print the results to the screen.
    print ('\n[+] Assessment By Respondent Residence: [Format = Count - [Code] Residence]')
    counts = df.groupby(["Q16LIVE"]).count()    
    for row,ix in zip(counts['Q7ALL'],range(3)):
        print '\t%s - [%s] %s' % (int(row), ix+1, codes[ix+1])
        
def get_targeted_users(df):
    # Read selected respondents file and extract respondents from main dataframe
    targets = pd.read_csv('/Users/Anton/assignment1/select_resps.csv')
    targets.rename(columns = {'year':'YEAR'}, inplace = True)
    df = pd.merge(df, targets, on=['YEAR', 'RESPNUM'], how='inner')    
    return df
    
def count_frequency(df):
    # Part IV frequency counter. Define data to count
    columns = ['Q3PARK','Q5TIMESFLOWN','Q6LONGUSE']
    titles = ['[+] Parking:','\n[+] Times Flown:','\n[+] SFO Usage']
    parking = {1:'Domestic (hourly) garage',2:'International garage',
    3:'SFO long term parking',4: 'Off-airport parking'}
    flown = {1:'1 time',2:'2 times',3:'3-6 times',4:'7-12 times',
    5:'13-24 times',6:'More than 24 times'}
    use = {1:'Less than 1 year [0.5]',2:'1-5 years [3]',
    3: '6-10 years [8]',4:'10+ years [15]'}
    
    # Do some magic using nested for statements. Display results of the frequency
    # counter
    df.replace(0, np.nan, inplace=True)
    for item,ix in zip(columns,range(3)):
        counts = df[item].value_counts()
        shouldPrint = True
        for iz,iw in counts.iteritems():
            if shouldPrint:
                print '%s' % titles[ix]
                shouldPrint = False
            if ix == 0:
                print '\t%s - [%s] %s' % (iw,int(iz),parking[iz])
            elif ix == 1:
                print '\t%s - [%s] %s' % (iw,int(iz),flown[iz])
            elif ix == 2:
                print '\t%s - [%s] %s' % (iw,int(iz),use[iz])
        shouldPrint = True
    
def do_partIII(df):
    # Main functions used to display part III data. Filter, parse, display, pickle.
    parsed_data = df.filter(['Q7ALL','Q16LIVE'], axis=1)
    parsed_data.replace(4, 3, inplace=True)
    find_location_trend(parsed_data)
    parsed_data.to_pickle('/Users/Anton/assignment1/foltz_pickle.p')
    
def do_partIV():
    # Main functions to filter, parse, display, pickle, and store part IV data
    df = user_df_2016.append(user_df_2015)
    df.replace(0, np.nan, inplace=True)
    parsed_data = get_targeted_users(df)
    print_summary(5, parsed_data)
    parsed_data.to_csv('/Users/Anton/assignment1/foltz_partIV_out.csv', encoding='utf-8')
    parsed_data = parsed_data.filter(['Q3PARK','Q5TIMESFLOWN','Q6LONGUSE'], axis=1)
    count_frequency(parsed_data)
    parsed_data.to_pickle('/Users/Anton/assignment1/foltz_pickle.p')
    
def main():
    # Main function. Launch functions for part I, II, II, IV
    start_program()
    partI_data = do_partI()
    do_partII()
    do_partIII(partI_data)
    do_partIV()
    print partI_data.info()
    
if __name__ == "__main__":
    main()
