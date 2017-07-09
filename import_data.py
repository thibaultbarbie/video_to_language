import numpy as np
from scipy import misc
import chainer.links as L

def import_data():
    n_data=400
    X=np.zeros((n_data,3,224,224))
    Y=np.zeros(n_data)

    # English
    video_n_data=[n_data]
    for i in range(video_n_data[0]):
        X[i] = L.model.vision.vgg.prepare(misc.imread("dataset/EN/1/"+str(i+1)+".jpg"))
        Y[i] = 0

    X=np.asarray(X.astype(np.float32))
    Y=np.asarray(Y.astype(np.int32))
    return X,Y
