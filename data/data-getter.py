import numpy as np
import pickle
from os.path import join
from tqdm import tqdm as tqdm
import random

# path - ścieżka do pliku *.pickle z którego pobiera dane
# folder - folder do którego zapisuje pliki wyjściowe
# fq - co który pomiar ma pobierać
# n - liczba pomiarów w pojedyńczym pliku
# name - nazwa plików wyjściowych (z doklejonym indeksem)
def data_getter(path, folder, fq, n, name="foobar"):
  X = pickle.load(open(path, "rb"))
  X = np.asarray(X)

  for i in tqdm(range(0, len(X)-(n*fq), fq)):
    pickle_out = open(join(folder, name+str(i)+".pickle"),"wb")
    qpa = []
    for ent in range(i, i+(n*fq), fq):
      qpa.append(X[ent])
    pickle.dump(qpa, pickle_out)
    pickle_out.close()


def data_getter2(path_X, path_y, folder, fq, n, name="foobar"):
  X = pickle.load(open(path_X, "rb"))
  X = np.asarray(X)
  y = pickle.load(open(path_y, "rb"))
  y = np.asarray(X)

  i = random.randint(0, len(X)-(n*fq))
  pickle_out = open(join(folder, "X"+name+".pickle"),"wb")
  qpa = []
  for ent in range(i, i+(n*fq), fq):
    qpa.append(X[ent])
  pickle.dump(qpa, pickle_out)
  pickle_out.close()

  pickle_out = open(join(folder, "y"+name+".pickle"),"wb")
  qpa = []
  for ent in range(i, i+(n*fq), fq):
    qpa.append(y[ent])
  pickle.dump(qpa, pickle_out)
  pickle_out.close()
