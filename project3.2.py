
import numpy
import sklearn.datasets 
import lib
import matplotlib
import matplotlib.pyplot as plt

import scipy.linalg

if __name__ == '__main__':

    D, L = lib.load('./trainData.txt')
    U = lib.compute_lda_geig(D, L, m=1)
    DP_geig=lib.apply_lda(U,D)
    print(DP_geig)
    print("Dimensioni di DP_geig:", DP_geig.shape)
    lib.plot_lda_histogram(DP_geig, L)