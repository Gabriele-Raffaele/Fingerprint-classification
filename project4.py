import numpy

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

def mcol(v):
    return v.reshape((v.size, 1))

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

# Compute log-density for a single sample x (column vector). The result is a 1-D array with 1 element
def logpdf_GAU_ND_singleSample(x, mu, C):
    P = numpy.linalg.inv(C)
    return -0.5*x.shape[0]*numpy.log(numpy.pi*2) - 0.5*numpy.linalg.slogdet(C)[1] - 0.5 * ((x-mu).T @ P @ (x-mu)).ravel()

# Compute log-densities for N samples, arranged as a MxN matrix X (N stacked column vectors). The result is a 1-D array with N elements, corresponding to the N log-densities
def logpdf_GAU_ND_slow(X, mu, C):
    ll = [logpdf_GAU_ND_singleSample(X[:, i:i+1], mu, C) for i in range(X.shape[1])]
    return numpy.array(ll).ravel()


# Compute log-densities for N samples, arranged as a MxN matrix X (N stacked column vectors). The result is a 1-D array with N elements, corresponding to the N log-densities
def logpdf_GAU_ND_fast(x, mu, C):
    P = numpy.linalg.inv(C)
    return -0.5*x.shape[0]*numpy.log(numpy.pi*2) - 0.5*numpy.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)

logpdf_GAU_ND = logpdf_GAU_ND_slow

def compute_ll(X, mu, C):
    return logpdf_GAU_ND(X, mu, C).sum()




if __name__ == '__main__':

    import matplotlib.pyplot as plt

    D, L = load('trainData.txt')
    for feature_index in range(D.shape[0]):
        DF = D[feature_index, L == 0]
        DT = D[feature_index, L == 1]
        
        plt.figure()
        muF = mcol(DF.mean())
        muT = mcol(DT.mean())
        CT = ((DT-muT) @ (DT-muT).T) / float(len(DT))
        CF = ((DF-muF) @ (DF-muF).T) / float(len(DF))
        # Plot dell'istogramma e della gaussiana per False
        plt.subplot(2, 1, 1)
        plt.hist(DF.ravel(), bins=50, density=True, alpha=0.5, color='red', label='False')
        XPlotF = numpy.linspace(DF.min(), DF.max(), 1000)
        YPlotF = numpy.exp(logpdf_GAU_ND(vrow(XPlotF), muF, CF))
        plt.plot(XPlotF, YPlotF, color='red')
        plt.title('Feature {} - False'.format(feature_index + 1))
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)

        # Plot dell'istogramma e della gaussiana per True
        plt.subplot(2, 1, 2)
        plt.hist(DT.ravel(), bins=50, density=True, alpha=0.5, color='blue', label='True')
        XPlotT = numpy.linspace(DT.min(), DT.max(), 1000)
        YPlotT = numpy.exp(logpdf_GAU_ND(vrow(XPlotT), muT, CT))
        plt.plot(XPlotT, YPlotT, color='blue')
        plt.title('Feature {} - True'.format(feature_index + 1))
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()