import numpy as np
import pickle
from os.path import join
from tqdm import tqdm as tqdm

# path - ścieżka do pliku *.pickle z którego pobiera dane
# folder - folder do którego zapisuje pliki wyjściowe
# fq - co który pomiar ma pobierać
# n - liczba pomiarów w pojedyńczym pliku
# name - nazwa plików wyjściowych (z doklejonym indeksem)
def data_getter(path, folder, fq, n, name="foobar"):
  X = pickle.load(open(path, "rb"))
  X = np.asarray(X)

  for i in tqdm(range(0, len(X), fq)):
    pickle_out = open(join(folder, name+str(i)+".pickle"),"wb")
    qpa = []
    for ent in range(0, len(X)-(n*fq), fq):
      qpa.append(ent)
    pickle.dump(qpa, pickle_out)
    pickle_out.close()

data_getter("X_08052021.pickle", "./lodziarnia", 100, 1000)