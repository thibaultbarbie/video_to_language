import numpy as np
from scipy import misc
import chainer.links as L

def import_data():
    n_data=12
    X=np.zeros((n_data,3,224,224))
    Y=np.zeros((n_data,4))
    
    for i in range(n_data):
        X[i] = L.model.vision.googlenet.prepare(misc.imread("dataset/"+str(i+1)+".jpg"))
        Y[i] = 2
    X=X.astype(np.float32)
    Y=Y.astype(np.int32)
    return X,Y
