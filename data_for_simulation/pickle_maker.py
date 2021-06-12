import numpy as np
import csv
import sys
import pickle


date = sys.argv[1]
folder_name = sys.argv[2]

x_file_name = '../data/' + folder_name + '/' + date + '_merged_mani.csv'
x = []

with open(x_file_name, newline='') as x_csvfile:
    x_reader = csv.DictReader(x_csvfile)
    for row in x_reader:
        if (row['001FCx00285_SPPV.PV'] != '001FCx00285_SPPV.PV'):
            x.append(row['001FCx00285_SPPV.PV'])

x_name_pickle = 'X_' + date + '.pickle'
pickle_out = open(x_name_pickle, "wb")
pickle.dump(x, pickle_out)

y_file_name = '../data/' + folder_name + '/'+ date + '_merged_straty.csv'
y = []

with open(y_file_name, newline='') as y_csvfile:
    y_reader = csv.DictReader(y_csvfile)
    for row in y_reader:
        if (row['001NIR0SZR0.daca.pv'] != '001NIR0SZR0.daca.pv'):
            y.append(row['001NIR0SZR0.daca.pv'])

y_name_pickle = 'Y_' + date + '.pickle'
pickle_out = open(y_name_pickle, "wb")
pickle.dump(y, pickle_out)