from tqdm import tqdm
import numpy as np
import sys
import csv
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pickle

date = sys.argv[1]
folder_name = sys.argv[2]

file_name = '../data/' + folder_name + '/' + date + '_merged_zaklocane.csv'
data = []

with open(file_name, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['temp_zuz'] != 'temp_zuz':
            data.append(row['temp_zuz'])

x = []
y = []
newx = []


for i in range(0, len(data), 3600):
    x.append(i)
    y.append(data[i])

x.append(len(data))
y.append(data[len(data) - 1])

newx = [i for i in range(len(data))]

f2 = interp1d(x, y, kind='cubic')

#plt.plot(f2(newx))
#plt.savefig(date + '_temperature.png')

data2 = []
for i in tqdm(range(len(data))):
    data2.append(f2(i))

wynik_name_pickle = date + '_temperature.pickle'
pickle_out = open(wynik_name_pickle, "wb")
pickle.dump(data2, pickle_out)
