import numpy as np 

def combine(data,actions,losses):
    return np.vstack([data,np.hstack([np.array(actions),np.array(losses)])])

if __name__=="__main__":
    data = np.array([[1,2],[2,3],[3,4],[4,5]])
    actions = [10,11,12]
    loss = [91,92,93]
    print(combinte(data,actions,losses))