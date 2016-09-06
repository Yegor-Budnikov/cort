import pickle
from os import path
import os

__author__ = 'smartschat'


def pDump(data, dataname):
    filename = '/home/redll/cort/my_test/pickle_files/' + dataname + '.pickle'
    os.makedirs(path.dirname(filename), exist_ok=True)

    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()

def pLoad(dataname):
    filename = '/home/redll/cort/my_test/pickle_files/' + dataname + '.pickle'
    with open(filename, 'rb') as f:
        data_new = pickle.load(f)
    return data_new



