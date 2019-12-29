# -*- coding: utf-8 -*-
import glob
from os.path import basename
import os
import csv

print('\n[+] Starting file parser...')

read_files = glob.glob(r"reviews/train/pos/*.txt")
with open(r"reviews/train/traindata.tsv", "w", encoding='utf-8', newline='\n') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t', quoting=csv.QUOTE_ALL)
    writer.writerow(["id", "sentiment", "review"])
    for f in read_files:
        with open(f, "r", encoding='utf-8') as infile:
            data=infile.read().replace('\n', '')
            b = basename(f)
            filename = os.path.splitext(b)[0]
            writer.writerow([filename, 1, data])

read_files = glob.glob(r"reviews/train/neg/*.txt")
with open(r"reviews/train/traindata.tsv", "a", encoding='utf-8', newline='\n') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t', quoting=csv.QUOTE_ALL)
    for f in read_files:
        with open(f, "r", encoding='utf-8') as infile:
            data=infile.read().replace('\n', '')
            b = basename(f)
            filename = os.path.splitext(b)[0]
            writer.writerow([filename, 0, data])

read_files = glob.glob(r"reviews/test/pos/*.txt")
with open(r"reviews/test/testdata.tsv", "w", encoding='utf-8', newline='\n') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t', quoting=csv.QUOTE_ALL)
    writer.writerow(["id", "sentiment", "review"])
    for f in read_files:
        with open(f, "r", encoding='utf-8') as infile:
            data=infile.read().replace('\n', '')
            b = basename(f)
            filename = os.path.splitext(b)[0]
            writer.writerow([filename, 1, data])

read_files = glob.glob(r"reviews/test/neg/*.txt")
with open(r"reviews/test/testdata.tsv", "a", encoding='utf-8', newline='\n') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t', quoting=csv.QUOTE_ALL)
    for f in read_files:
        with open(f, "r", encoding='utf-8') as infile:
            data=infile.read().replace('\n', '')
            b = basename(f)
            filename = os.path.splitext(b)[0]
            writer.writerow([filename, 0, data])

read_files = glob.glob(r"reviews/train/unsup/*.txt")
with open(r"reviews/train/unsupdata.tsv", "w", encoding='utf-8', newline='\n') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t', quoting=csv.QUOTE_ALL)
    writer.writerow(["id", "sentiment", "review"])
    for f in read_files:
        with open(f, "r", encoding='utf-8') as infile:
            data=infile.read().replace('\n', '')
            b = basename(f)
            filename = os.path.splitext(b)[0]
            writer.writerow([filename, 0, data])

# For now, comment out until batch sizes are fixed when restoring to a tensorflow checkpoint.
# this currently errors out due to an improper shape.       

#read_files = glob.glob(r"reviews/test/tom/*.txt")
#with open(r"reviews/test/unsupdata.tsv", "a", encoding='utf-8', newline='\n') as tsvfile:
#    writer = csv.writer(tsvfile, delimiter='\t', quoting=csv.QUOTE_ALL)
#    writer.writerow(["id", "sentiment", "review"])
#    for f in read_files:
#        with open(f, "r", encoding='utf-8') as infile:
#            data=infile.read().replace('\n', '')
#            b = basename(f)
#            filename = os.path.splitext(b)[0]
#            writer.writerow([filename, 0, data])
         
print('[+] Done cleaning and making files!')
