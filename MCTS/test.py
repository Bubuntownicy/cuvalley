import numpy as np
import random

if __name__ == '__main__':
    to_rollout = 100
    prev = [10,10,10,10]
    end = [np.random.randint(0,9) for _ in range(to_rollout)]
    print(prev+end)