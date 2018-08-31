#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author      : Anton
@date        : August 23, 2018
@description : A Python script that can be used to optimize a driving route
             : by leverage Gurobipy and the Google Distance Matrix API. This 
             : script can be used to solve TSP problems that use addresses
             : as the input. 
"""

import json
import os
import datetime
import time
import itertools
import argparse
import requests
import pandas as pd
from bs4 import BeautifulSoup
from gurobipy import Model, GRB, tupledict, tuplelist, quicksum

# Set a Google API key to increase queries to unlimited/day.
# Note: This key will be disabled at the end of the course. 
API_KEY = 'YOUR_API_KEY'

# Wait time before calling the Google API if the rate limit is reached
BACKOFF_TIME = 2

# Define a static column name for the infile
ADDRESS_COLUMN_NAME = 'Address'

# Return Full Google Results? If True, full JSON results from Google are included in output
RETURN_FULL_RESULTS = False

# Time, in hours, to estimate traffic based on Google 'duraton_in_traffic' parameter
ROUTE_TIME_FUTURE = 2

# Chunk size for querying Google to avoid rate limit. Set default to allow use
# without API_KEY
CHUNK_SIZE = 9

# Track the position in a list to create a dictionary of points for Gurobi
counter = 0   

# Number of routes, as calculated by the number of addresses from infile
n = 0

# BASE DIRECTORY - WORKING DIRECTORY
BASE = os.path.dirname(os.path.abspath(__file__))

# Given a tuplelist of edges, find the shortest subtour
def subtour(edges):
    unvisited = list(range(n))
    cycle = range(n+1) # initial length has 1 more city
    while unvisited: # true if list is non-empty
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i,j in edges.select(current,'*') if j in unvisited]
        if len(cycle) > len(thiscycle):
            cycle = thiscycle
    return cycle

# Subtour function to prevent subtours by using lazy constraints during
# optimization
def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        # make a list of edges selected in the solution
        vals = model.cbGetSolution(model._vars)
        selected = tuplelist((i,j) for i,j in model._vars.keys() if vals[i,j] > 0.5)
        # find the shortest cycle in the selected edge list
        tour = subtour(selected)
        if len(tour) < n:
            # add subtour elimination constraint for every pair of cities in tour
            model.cbLazy(quicksum(model._vars[i,j]
                                  for i,j in itertools.combinations(tour, 2))
                         <= len(tour)-2)
    
def get_google_matrix(addresses):
    
    # Create an empty list to hold dict values that contain distance and duration
    # information. This is extracted from the JSON response
    data = []
    
    # This can be used for looking into future routes (say 2 hours from now)
    orig_time = datetime.datetime.now()
    new_time = orig_time + datetime.timedelta(hours=ROUTE_TIME_FUTURE)
    stamp = int(new_time.timestamp())
    
    # Google Distance Matrix base URL to which all other parameters are attached
    base_url = 'https://maps.googleapis.com/maps/api/distancematrix/json?'
    
    # Chunk the list of addresses to avoid a rate_limit error
    #chunks = list(divide_chunks(addresses, CHUNK_SIZE))
    
    # Chunks can be larger with an API_KEY, set to n if under 20
    chunks = list(divide_chunks(addresses, n))
    
    # For each address in the list, query the distance and duration from 
    # the address to point x.
    for address in addresses:
        for chunk in chunks:
            print('[+] Working\n')
            
            # Prepare the request details for the assembly into a request URL
            payload = {'units': 'imperial', 
                       'origins' : address,
                       'destinations' : '|'.join(chunk), 
                       'mode' : 'driving',
                       'departure_time' : stamp,
                       'key' : API_KEY}
            
            # Assemble the URL and query the web service
            r = requests.get(base_url, params = payload)
        
            # Check the HTTP status code returned by the server. Only process the response, 
            # if the status code is 200 (OK in HTTP terms).
            if r.status_code != 200:
                print('[!] Received HTTP status code {}'.format(r.status_code))
                if r.status_code == 500:
                    while True:
                        print('[!] Ratelimit exceeded. Backing off!')
                        time.sleep(BACKOFF_TIME+5)
                        r = requests.get(base_url, params = payload)
                        if r.status_code != 500:
                            print()
                            break
                        
            if r.status_code == 200:
                try:
                    x = json.loads(r.text)
                    
                    # Extract the elements from JSON and store them in a dict
                    # Once done, append to the data list
                    for isrc, src in enumerate(x['origin_addresses']):
                        for idst, dst in enumerate(x['destination_addresses']):
                            obj = {}
                            row = x['rows'][isrc]
                            cell = row['elements'][idst]
                            if cell['status'] == 'OK':
                                obj['origin'] = src
                                obj['destination'] = dst
                                obj['distance'] = cell['distance']['text']
                                obj['distance_meters'] = cell['distance']['value']
                                obj['duration'] = cell['duration']['text']
                                obj['duration_seconds'] = cell['duration']['value']
                                try:
                                    obj['duration_traffic'] = cell['duration_in_traffic']['text']
                                    obj['duration_traffic_seconds'] = cell['duration_in_traffic']['value']
                                except:
                                    obj['duration_traffic'] = cell['duration']['text']
                                    obj['duration_traffic_seconds'] = cell['duration']['value']
                                    
                                data.append(obj)
                                print('{} to {}: {}, {}.'.format(src, dst, cell['distance']['text'], cell['duration']['text']))
                            else:
                                print('{} to {}: status = {}'.format(src, dst, cell['status']))                    
                except ValueError:
                    print('[!] Error while parsing JSON response, program terminated.')
            print('\n[+] Sleeping')
            
            # Avoid the rate limit by setting a backoff (sleep) time between queries
            time.sleep(BACKOFF_TIME)
    
    # Store the list of dicts to a pandas dataframe and write to a XLSX for additional
    # processing
    df = pd.DataFrame(data)
    df['id'] = pd.Categorical(df['origin'], categories=df['origin'].unique()).codes
    df.to_excel(os.path.join(BASE,'full_dataframe_{}.xlsx'.format(n)), index=False)
    
    print('[+] Done fetching locations')
    print('\n[+] Wrote full_dataframe_{}.xlsx file to disk!'.format(n))
    
    return df

# Takes a list and divides into m chunks
def divide_chunks(l, m):
    for i in range(0, len(l), m):
        yield l[i:i + m]

# Takes multiple lists and merges into a single list
def combine_lists(arr):
    return(list(itertools.chain.from_iterable(arr)))

# Start the optimization process to determine the best route based on 
# distance or duration
def start_tsp(data):
    global counter
    
    print('\n[OPTIMIZING ROUTE FOR {} POINTS]\n'.format(n))
    
    # Create a dictionary containing each pair of points
    sched = {(i,j): get_val(data) for i in range(n) for j in range(n)}
    
    # Instantiate a model
    m = Model('ROUTE')
    
    # Create variables and store as a tuple dict.  
    vars = tupledict()
    for i,j in sched.keys():
        vars[i,j] = m.addVar(obj=sched[i,j], vtype=GRB.BINARY, 
            name='z({},{})'.format(i,j))
        
        # Set the upper bounds to zero for all points where i == j
        if i == j:
            vars[i,j].ub = 0

    m.update()

    # Add constraints. Instead of lazy constraints and the original subtourlim,
    # define a constraint where all rows must == 1 and all columns must also 
    # == 1. 
    m.addConstrs(vars.sum(i,'*') == 1 for i in range(n))
    m.addConstrs(vars.sum('*',j) == 1 for j in range(n))
    
    # Additional constraint to prevent subtours. The inverse of a point 
    # plus a point should be 1 or 0.
    for i,j in sched.keys():
        m.addConstr(vars[i,j] + vars[j,i] <= 1)
    
    # Update the model constraints...just in case. 
    m.update()
    
    # Assign vars and optimize the model. Once done, extract the values and 
    # store as vals. Use a lazy constraint limit subtours
    m._vars = vars
    m.Params.lazyConstraints = 1
    m.optimize(subtourelim)
    vals = m.getAttr('x', vars)    

    # Extract the selected points that are greater than 0.5
    selected = tuplelist((i,j) for i,j in vals.keys() if vals[i,j] > 0.5)
    
    # Once complete, display the values and move on to the next n
    tour = subtour(selected)
    assert len(tour) == n
    
    print('\n[SENSITIVITY ANALYSIS]\n')
    sensitivity_analysis(m)
    
    print('\n[OPTIMIZATION COMPLETE FOR {} POINTS]\n'.format(n))
    
    print('[+] Optimal route: {}'.format(str(tour)))
    print('[+] Optimal time: {} seconds'.format(int(m.objVal)))
    
    return tour,int(m.objVal)

# Take a list send back the value for that position in the list. Used to 
# create variables (tupledict) in Gurobi
def get_val(data):
    global counter
    val = data[counter]
    counter+=1
    return val

def get_data(infile):
    # Read the data from infile to a pandas dataframe
    if os.path.isfile(infile):
        data = pd.read_excel(infile)
        if ADDRESS_COLUMN_NAME not in data.columns:
            raise ValueError('[!] Missing Address column in input data')
        else:
            addresses = data[ADDRESS_COLUMN_NAME].tolist()
    else:
        raise ValueError('[!] Cannot find file {} in path'.format(infile))
    
    return addresses

# Generate an HTML page with the data to view on a map
def generate_html(data, final_route):
    
    # Open the template html file and read with BS. 
    # NOTE: Make sure you modify with your API key. In the future, just modify
    # with code as done below
    with open(os.path.join(BASE, 'template.html')) as html:
        soup = BeautifulSoup(html, 'html.parser')

    # Find the start tag and replace the contents
    for i in soup.find('select', {'id': 'start'}).findChildren():
        origin = data[data.id == 0].iloc[0]
        option = BeautifulSoup('<option value=\"{}\">{}</option>'.format(origin['origin'],origin['origin']), 'lxml')
        i.replace_with(option.option)
     
    # Find the end tag and replace the contents
    for i in soup.find('select', {'id': 'end'}).findChildren():
        origin = data[data.id == 0].iloc[0]
        option = BeautifulSoup('<option value=\"{}\">{}</option>'.format(origin['origin'],origin['origin']), 'lxml')
        i.replace_with(option.option)
    
    # Find the waypoints and replace their contents
    waypoints = []
    s = soup.find('select', {'id': 'waypoints'})
    
    for point in final_route:
        p = int(point)
        if not p == 0:               
            origin = data[data.id == point].iloc[0]
            option = '<option value=\"{}\">{}</option>'.format(origin['origin'],origin['origin'])
            waypoints.append(option)
    
    wps = ' '.join(waypoints)   
    wps = BeautifulSoup('<select multiple id=\"waypoints\">{}</select>'.format(wps),'html.parser')
    s.replace_with(wps)
        
    # Save the new html file as route.html               
    with open(os.path.join(BASE, 'route_{}.html'.format(n)), 'wb') as f_output:
        f_output.write(soup.prettify('utf-8'))
        
    print('\n[+] Wrote route_{}.html file to disk!'.format(n))
    
def sensitivity_analysis(model):
    # Store the optimal solution
    origObjVal = model.ObjVal
    for v in model.getVars():
        v._origX = v.X
        
    # Disable solver output for subsequent solves
    model.Params.outputFlag = 0
    
    # Iterate through unfixed, binary variables in model
    for v in model.getVars():
        if (v.LB == 0 and v.UB == 1 \
            and (v.VType == GRB.BINARY or v.VType == GRB.INTEGER)):
            
            # Set variable to 1-X, where X is its value in optimal solution
            if v._origX < 0.5:
                v.LB = v.Start = 1
            else:
                v.UB = v.Start = 0

            # Update MIP start for the other variables
            for vv in model.getVars():
                if not vv.sameAs(v):
                    vv.Start = vv._origX

            # Solve for new value and capture sensitivity information
            model.optimize()

            if model.status == GRB.Status.OPTIMAL:
                print('Objective sensitivity for variable %s is %g' % \
                      (v.VarName, model.ObjVal - origObjVal))
            else:
                print('Objective sensitivity for variable %s is infinite' % \
                      v.VarName)

            # Restore the original variable bounds
            v.LB = 0
            v.UB = 1
     
def main(infile):
    global n
    
    print('[STARTING SCRIPT]')
    
    # Get a list of addresses for geocoding
    addresses = get_data(infile)
    n = len(addresses)
    
    print('\n[+] Read {} addresses from file {}'.format(n, infile))
    
    # Fetch routes from Google. If run at least once, you can comment this out
    # and uncomment the next commented Python function call to read_excel
    #data = get_google_matrix(addresses)
    
    # If run at least once, and full_dataframe exists, uncomment to prevent
    # pulling data from Google
    data = pd.read_excel(os.path.join(BASE,'full_dataframe_100.xlsx'))

    #arr = data.set_index(['origin','destination'])['duration_seconds'].unstack().values
    arr = data.set_index(['origin','destination'])['duration_traffic_seconds'].unstack().values
    
    # In some cases, the horizontal zeros can be > 0 when traveling from A to A
    # using the duration_traffic_seconds variable. This does not happen with 
    # duration_seconds. Account for that and set them to zero.
    count = 0
    for a in arr:
        a[count] = 0
        count+=1
        
    df = pd.DataFrame(arr)
    df.to_excel(os.path.join(BASE,'dataframe_matrix_{}.xlsx'.format(n)), index=True)
    print('[+] Wrote dataframe_matrix_{}.xlsx file to disk!'.format(n))

    print(arr[:n,:n])      # DEBUG ONLY
    combined = combine_lists(arr)
    
    # Start the timer to see how long it takes to run    
    start = time.time()
    
    # Perform optimization
    final_route, duration = start_tsp(combined)
    
    # Stop the timer 
    stop = time.time()
    
    # Generate an html map
    generate_html(data, final_route)
    
    print('\n[ROUTE DETAILS]\n')
    meters = 0
    for index,point in enumerate(final_route):
        
        if index <= n-2:
            origin = data[data.id == point]
            destination = data[data.id == final_route[index+1]].iloc[0]
            route = origin[origin.destination == destination['origin']].iloc[0]
            meters+=int(route['distance_meters'])
            print('Leg {}:'.format(index+1))
            print('{} to {} is {} in {}'.format(route['origin'],
                  route['destination'], route['distance'],
                  route['duration_traffic']))
        else:
            origin = data[data.id == point]
            destination = data[data.id == final_route[0]].iloc[0]
            route = origin[origin.destination == destination['origin']].iloc[0]
            meters+=int(route['distance_meters'])
            print('Leg {}:'.format(n))
            print('{} to {} is {} in {}'.format(route['origin'],
                  route['destination'], route['distance'],
                  route['duration_traffic']))
            break
    
    # Convert total meters to miles.
    km = meters/1000
    mi = round(km * 0.62137,1)
    
    # Convert total seconds to human readable
    sec = datetime.timedelta(seconds=int(duration))
    d = datetime.datetime(1,1,1) + sec
    
    print('\nTotal Distance: {} mi'.format(mi))
    print('Total Duration: {} days {} hours {} minutes'.format(d.day-1, d.hour, d.minute))
    
    print('\n[COMPUTATION TIME]\n')
    total = stop-start
    print('Exection Time: {} seconds'.format(total))
    print('\n[+] Done!')
    
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', help='XLSX file containing addresses')
    
    try:
        args = parser.parse_args()
    except:
        print('\n[!] Invalid command line arguments. Run with -h\n')
        
    if args.infile:
        main(args.infile)
    else:
        print('\n[!] Invalid command line arguments. Run with -h\n') 
