import pickle
import pandas as pd
import sys
import os

result = pd.read_csv(os.path.join(sys.argv[1], sys.argv[2]))
pickle.dump(result, open(os.path.join(sys.argv[1], sys.argv[2].split(".")[0]+".pickle"), 'wb'))
