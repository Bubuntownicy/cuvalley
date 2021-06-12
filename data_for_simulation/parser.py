import sys
import csv
import pickle
import numpy as np
from time import strptime
import time

date = sys.argv[1]
folder_name = sys.argv[2]
file = sys.argv[3]

file_name = '../data/' + folder_name + '/' + date + '_merged_' + file + '.csv'
data = []

with open(file_name, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['Czas'] != 'Czas':
            ts = strptime(row['Czas'], '%Y-%m-%d %H:%M:%S')
            row['Czas'] = time.mktime(ts)

wynik_name_pickle = date + '_'+ file + '_timestamp.pickle'
pickle_out = open(wynik_name_pickle, "wb")
pickle.dump(reader, pickle_out)