import pandas as pd
import pickle

result = pd.read_excel('../data/Dane_sejsmika/2016-2020.xlsx')
pickle.dump(result, open('../data/parsed_2016_2020_excel.pickle', 'wb'))
