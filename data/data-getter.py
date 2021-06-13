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

    for i in tqdm(range(0, len(X) - (n * fq), fq)):
        pickle_out = open(join(folder, name + str(i) + ".pickle"), "wb")
        qpa = []
        for ent in range(i, i + (n * fq), fq):
            qpa.append(X[ent])
        pickle.dump(qpa, pickle_out)
        pickle_out.close()


def data_getter2(path_X, path_y, folder, fq, n, name="foobar"):
    X = pickle.load(open(path_X, "rb"))
    X = np.asarray(X)
    y = pickle.load(open(path_y, "rb"))
    y = np.asarray(X)

    i = random.randint(0, len(X) - (n * fq))
    pickle_out = open(join(folder, "X" + name + ".pickle"), "wb")
    qpa = []
    for ent in range(i, i + (n * fq), fq):
        qpa.append(X[ent])
    pickle.dump(qpa, pickle_out)
    pickle_out.close()

    pickle_out = open(join(folder, "y" + name + ".pickle"), "wb")
    qpa = []
    for ent in range(i, i + (n * fq), fq):
        qpa.append(y[ent])
    pickle.dump(qpa, pickle_out)
    pickle_out.close()


# path - z którego pliku
# start - od którego momentu
# fq - co który wpis
# n - ile wpisów
def data_getter3(path, start, fq, n):
    X = pickle.load(open(path, "rb"))
    X = np.asarray(X)

    out = []

    for i in range(start, start + (fq * n), fq):
        out.append(X[i])

    return out


# there is one impostor among us
def data_getter4(frequency=10,batch_size=200):
    paths_loss = [r"2021-04-19\19042021_merged_straty.csv", r"2021-05-08\08052021_merged_straty.csv",
             r"2021-05-27\27052021_merged_straty.csv"]
    paths_action = [r"2021-04-19\19042021_merged_mani.csv", r"2021-05-08\08052021_merged_mani.csv",
             r"2021-05-27\27052021_merged_mani.csv"]
    X = []
    Y = []
    for path_loss,path_action in zip(paths_loss,paths_action):
        loss = [float(line.split()[1].split(',')[1]) for line in open(path_loss) if
                 len(line.split()) > 1]
        actions = [float(line.split()[1].split(',')[1]) for line in open(path_action) if
                    len(line.split()) > 1]
        l_loss = len(loss)
        l_actions = len(actions)
        loss = loss[:min(l_loss,l_actions)]
        actions = actions[:min(l_loss,l_actions)]
        loss_cropped = [element for i, element in enumerate(loss) if i%frequency==0]
        actions_cropped = [element for i, element in enumerate(actions) if i%frequency==0]
        temp_X = []
        temp_Y = []
        for i in range(len(loss_cropped)-1):
            if len(temp_X)<batch_size:
                temp_X.append([loss_cropped[i],actions_cropped[i]])
                temp_Y.append([loss_cropped[i]])
            else:
                X.append(temp_X)
                Y.append(temp_Y)
                temp_Y = []
                temp_X = []
    # print(np.array(X).shape)
    print(X)

if __name__ == "__main__":
    data_getter4()
