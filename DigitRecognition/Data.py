import numpy as np
import os

def loadData(name):
    if not os.path.isfile(name + ".npy"):
        print "Loading " + name
        train = np.genfromtxt(name,  delimiter=",", skip_header=1)
        np.save(name + ".npy", train)
    else:
        print "Loading " + name + ".npy"
        train = np.load(name + ".npy")

    Y = train[:,0]
    Y=Y.reshape(Y.shape[0], 1)
    X = train[:,1:]

    return X,Y
