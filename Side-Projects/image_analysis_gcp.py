#!/usr/bin/python

# To authenticate, download your json keyfile from your account.
# Once done, run the following command in a terminal, or set the
# export in your ~/.bashrc or ~/.profile
# export GOOGLE_APPLICATION_CREDENTIALS="/path/to/keyfile.json"

from __future__ import print_function

import os
import re
from google.cloud.vision_v1 import ImageAnnotatorClient

def analyze_image(URL,client):
    request = {'image': {'source': {'image_uri': URL},},}
    response = client.annotate_image(request)
    return response

def print_face_details(response):
    print('\n++++++++++++++++++++++++++++++++++++++++++')
    print('# FACE DETECTION')
    print('++++++++++++++++++++++++++++++++++++++++++\n')

    if len(response.face_annotations) > 0:
        print('[+] Detected %s face annotations...' % len(response.face_annotations))

        counter = 0
        for face in response.face_annotations:
            print('\n[+] Face %s detected with %s confidence...' % (counter+1, face.detection_confidence))
            print('\tJoy:\t\t%s' % category[face.joy_likelihood])
            print('\tSorrow:\t\t%s' % category[face.sorrow_likelihood])
            print('\tAnger:\t\t%s' % category[face.anger_likelihood])
            print('\tSurprise:\t%s' % category[face.surprise_likelihood])
            print('\tHeadwear:\t%s' % category[face.headwear_likelihood])
            counter+=1
    else:
        print('[!] No faces found...')

def print_logo_details(response):
    print('\n++++++++++++++++++++++++++++++++++++++++++')
    print('# LOGO DETECTION')
    print('++++++++++++++++++++++++++++++++++++++++++\n')

    if len(response.logo_annotations) > 0:
        print('[+] Detected %s logo annotations...' % len(response.logo_annotations))

        counter = 0
        for logo in response.logo_annotations:
            print('\n[+] Logo %s detected with %s confidence...' % (counter+1, logo.score))
            print('\tDescription:\t%s' % logo.description)       
    	    counter+=1
    else:
        print('[!] No logos found...')

def print_text_details(response):
    print('\n++++++++++++++++++++++++++++++++++++++++++')
    print('# TEXT DETECTION')
    print('++++++++++++++++++++++++++++++++++++++++++\n')

    output = []
    seen = set()
    if len(response.text_annotations) > 0:
        print('[+] Detected %s text annotations...' % len(response.text_annotations))
	print('\n[+] Extracted, unique text:')
        for text in response.text_annotations:
            text = text.description.strip()
	    text = text.lower()
	    text = re.sub('[^A-Za-z0-9]+', ' ', text)
	    words = text.split()
            for word in words:
 		if word not in seen:
		    seen.add(word)
		    output.append(word)
	for word in output: 
            print(word, end=" ")
	print('')
    else:
        print('[!] No text found...')

def print_label_details(response):
    print('\n++++++++++++++++++++++++++++++++++++++++++')
    print('# LABEL DETECTION')
    print('++++++++++++++++++++++++++++++++++++++++++\n')
    output = []
    seen = set()
    if len(response.label_annotations) > 0:
        print('[+] Detected %s label annotations...' % len(response.label_annotations))
        print('\n[+] Extracted, unique labels:')
        for label in response.label_annotations:
            label = label.description
            if label not in seen:
                seen.add(label)
                output.append(label)
        for label in output:
            print(label, end=" ")
        print('')
    else:
        print('[!] No labels found...')

def print_landmark_details(response):
    print('\n++++++++++++++++++++++++++++++++++++++++++')
    print('# LANDMARK DETECTION')
    print('++++++++++++++++++++++++++++++++++++++++++\n')

    if len(response.landmark_annotations) > 0:
        print('[+] Detected %s landmark annotations...' % len(response.landmark_annotations))

        counter = 0
        for landmark in response.landmark_annotations:
            print('\n[+] Landmark %s detected with %s confidence...' % (counter+1, landmark.score))
            print('\tDescription:\t%s' % landmark.description)
	    lat = response.landmark_annotations._values[0].locations[0].lat_lng.latitude
	    lon = response.landmark_annotations._values[0].locations[0].lat_lng.longitude
	    print('\tCoordinates:\t%s,%s' % (lat, lon))
            counter+=1
    else:
        print('[!] No landmarks found...')

def print_web_details(response):
    print('\n++++++++++++++++++++++++++++++++++++++++++')
    print('# WEB DETECTION')
    print('++++++++++++++++++++++++++++++++++++++++++\n')

    entities = False
    full = False
    pages = False

    if len(response.web_detection.web_entities) > 0:
        print('[+] Identified %s web entities...' % len(response.web_detection.web_entities))

        counter = 0
        for web in response.web_detection.web_entities:
            print('\n[+] Web %s detected with %s confidence...' % (counter+1, web.score))
            print('\tDescription:\t%s' % web.description)
            entities = True
            counter+=1
    
    if len(response.web_detection.full_matching_images) > 0:
        print('\n[+] Full matching images:')

        for web in response.web_detection.full_matching_images:
            print(web.url)
            full = True
           
    if len(response.web_detection.pages_with_matching_images) > 0:
        print('\n[+] Pages with matching images:')

        for web in response.web_detection.pages_with_matching_images:
            print(web.url)
            pages = True

    if not entities:
        print('[!] No web entities found...')
    elif not full:
        print('[!] No full matching images found...')
    elif not pages:
        print('[!] No pages with matching images found...')

def print_safe_search_details(response):
    print('\n++++++++++++++++++++++++++++++++++++++++++')
    print('# SAFE SEARCH DETECTION')
    print('++++++++++++++++++++++++++++++++++++++++++\n')

    if response.safe_search_annotation is not None:
        print('[+] Image content summary:')
        print('\tAdult:\t\t%s' % category[response.safe_search_annotation.adult])
        print('\tSpoof:\t\t%s' % category[response.safe_search_annotation.spoof])
        print('\tMedical:\t%s' % category[response.safe_search_annotation.medical])
        print('\tViolence:\t%s' % category[response.safe_search_annotation.violence])
    else:
        print('[!] No safe search attributes found...') 

# Start script
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/Anton/data-666680c8f996.json'
client = ImageAnnotatorClient()
category = {0: 'Unknown', 1: 'Very Unlikely', 2: 'Unlikely', 3: 'Possible', 4: 'Likely', 5: 'Very Likely'}

while True:
    print('\n++++++++++++++++++++++++++++++++++++++++++\n')
    url = raw_input('[+] Enter image URL: ')

    resp = analyze_image(url,client)

    # Print data about the image
    print_face_details(resp)
    print_logo_details(resp)
    print_text_details(resp)
    print_label_details(resp)
    print_landmark_details(resp)
    print_web_details(resp)
    print_safe_search_details(resp)
