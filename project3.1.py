
import numpy
import sklearn.datasets 

import matplotlib
import matplotlib.pyplot as plt

import scipy.linalg

def mcol(v):
    return v.reshape((v.size, 1))


def load(fname):
    DList = []
    labelsList = []
    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:-1]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                label = line.split(',')[-1].strip()
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)


def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def compute_pca(D, m):

    mu, C = compute_mu_C(D)
    U, s, Vh = numpy.linalg.svd(C)
    P = U[:, 0:m]
    return P

def apply_pca(P, D):
    return P.T @ D
    
def plot_pca_histograms(DP, L):
    
    
    hFea = {
        0: 'Prima direzione',
        1: 'Seconda direzione',
        2: 'Terza direzione',
        3: 'Quarta direzione',
        4: 'Quinta direzione',
        5: 'Sesta direzione'
    }
    
    plt.figure(figsize=(15, 10))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.hist(DP[i, L==0], bins=10, density=True, alpha=0.4, label='False')
        plt.hist(DP[i, L==1], bins=10, density=True, alpha=0.4, label='True')
        plt.xlabel(hFea[i])
        plt.ylabel('Frequenza')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
def plot_pca_scatter(DP, L):
    
    
    hFea = {
        0: 'Prima direzione',
        1: 'Seconda direzione',
        2: 'Terza direzione',
        3: 'Quarta direzione',
        4: 'Quinta direzione',
        5: 'Sesta direzione'
    }
    
    plt.figure(figsize=(15, 10))
    for i in range(5):
        plt.subplot(2, 3, i + 1)
        plt.scatter(DP[i, L==0], DP[i+1, L==0], label='False')
        plt.scatter(DP[i, L==1], DP[i+1, L==1], label='True')
        plt.xlabel(hFea[i])
        plt.ylabel(hFea[i+1])
        plt.legend()
    
    plt.tight_layout()
    plt.show()
def calculate_stats(D, L):
    features = ['first feature', 'second feature', 'third feature', 'fourth feature', 'fifth feature', 'sixth feature']

    for cls in [0, 1]:
        DCls = D[:, L == cls]
        print(f"Class {cls} Statistics:")
        for i, feature in enumerate(features):
            mean = DCls[i, :].mean()
            variance = DCls[i, :].var()
            print(f"{feature} - Mean: {mean:.6f}, Variance: {variance:.6f}")
        print()

if __name__ == '__main__':

    D, L = load('./trainData.txt')
    P = compute_pca(D, m = 6)
    DP = apply_pca(P, D)
    plot_pca_histograms(DP, L)
    plot_pca_scatter(DP, L)
    calculate_stats(DP, L)