import numpy
import scipy.special
import sklearn.datasets
import scipy.optimize
import matplotlib.pyplot as plt
import lib
def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))


# Optimize SVM
def train_dual_SVM_linear(DTR, LTR, C, K = 1):
    
    ZTR = LTR * 2.0 - 1.0 # Convert labels to +1/-1
    DTR_EXT = numpy.vstack([DTR, numpy.ones((1,DTR.shape[1])) * K])
    H = numpy.dot(DTR_EXT.T, DTR_EXT) * vcol(ZTR) * vrow(ZTR)

    # Dual objective with gradient
    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - numpy.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, numpy.zeros(DTR_EXT.shape[1]), bounds = [(0, C) for i in LTR], factr=1.0)
    
    # Primal loss
    def primalLoss(w_hat):
        S = (vrow(w_hat) @ DTR_EXT).ravel()
        return 0.5 * numpy.linalg.norm(w_hat)**2 + C * numpy.maximum(0, 1 - ZTR * S).sum()

    # Compute primal solution for extended data matrix
    w_hat = (vrow(alphaStar) * vrow(ZTR) * DTR_EXT).sum(1)
    
    # Extract w and b - alternatively, we could construct the extended matrix for the samples to score and use directly v
    w, b = w_hat[0:DTR.shape[0]], w_hat[-1] * K # b must be rescaled in case K != 1, since we want to compute w'x + b * K

    primalLoss, dualLoss = primalLoss(w_hat), -fOpt(alphaStar)[0]
    print ('SVM - C %e - K %e - primal loss %e - dual loss %e - duality gap %e' % (C, K, primalLoss, dualLoss, primalLoss - dualLoss))
    
    return w, b

# We create the kernel function. Since the kernel function may need additional parameters, we create a function that creates on the fly the required kernel function
# The inner function will be able to access the arguments of the outer function
def polyKernel(degree, c):
    
    def polyKernelFunc(D1, D2):
        return (numpy.dot(D1.T, D2) + c) ** degree

    return polyKernelFunc

def rbfKernel(gamma):

    def rbfKernelFunc(D1, D2):
        # Fast method to compute all pair-wise distances. Exploit the fact that |x-y|^2 = |x|^2 + |y|^2 - 2 x^T y, combined with broadcasting
        D1Norms = (D1**2).sum(0)
        D2Norms = (D2**2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * numpy.dot(D1.T, D2)
        return numpy.exp(-gamma * Z)

    return rbfKernelFunc

# kernelFunc: function that computes the kernel matrix from two data matrices
def train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps = 1.0):

    ZTR = LTR * 2.0 - 1.0 # Convert labels to +1/-1
    K = kernelFunc(DTR, DTR) + eps
    H = vcol(ZTR) * vrow(ZTR) * K

    # Dual objective with gradient
    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - numpy.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, numpy.zeros(DTR.shape[1]), bounds = [(0, C) for i in LTR], factr=1.0)

    print ('SVM (kernel) - C %e - dual loss %e' % (C, -fOpt(alphaStar)[0]))

    # Function to compute the scores for samples in DTE
    def fScore(DTE):
        
        K = kernelFunc(DTR, DTE) + eps
        H = vcol(alphaStar) * vcol(ZTR) * K
        return H.sum(0)

    return fScore # we directly return the function to score a matrix of test samples

import bayesRisk

if __name__ == '__main__':

    D, L = lib.load('./trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = lib.split_db_2to1(D, L)
    
    C_array = numpy.logspace(-5, 0, 11)
    minDCF_absolute = float('inf')
    #k = 1 K =10
    minDCF_values = []
    actDCF_values = []
    for K in [1]:

        for C in C_array:
            w, b = train_dual_SVM_linear(DTR, LTR, C, K)
            SVAL = (vrow(w) @ DVAL + b).ravel()
            PVAL = (SVAL > 0) * 1
            err = (PVAL != LVAL).sum() / float(LVAL.size)
            minDCF = bayesRisk.compute_minDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0) 
            actDCF = bayesRisk.compute_actDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
            print ('Error rate: %.1f' % (err*100))
            print ('minDCF - pT = 0.1: %.4f' % minDCF )
            print ('actDCF - pT = 0.1: %.4f' % actDCF)
            print ()
            minDCF_values.append(minDCF)
            actDCF_values.append(actDCF)
            if minDCF < minDCF_absolute:
                minDCF_absolute = minDCF
                C_best = C
                actDCF_best = actDCF

     # Plot the results
    plt.figure()
    plt.semilogx(C_array, minDCF_values, label='minDCF')
    plt.semilogx(C_array, actDCF_values, label='actDCF')
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.legend()
    plt.title('minDCF and actDCF vs. C')
    plt.grid(True)
    plt.show()
  
    print ('---------------centered data----------------')
    D_train_mean = numpy.mean(DTR, axis=1)
    DTR_centered = lib.center_data(DTR, D_train_mean)
    DVAL_centered = lib.center_data(DVAL, D_train_mean)
    minDCF_values_c = []
    actDCF_values_c = []
    minDCF_absolute_c = float('inf')
    for K in [1]:

        for C in C_array:
            w, b = train_dual_SVM_linear(DTR_centered, LTR, C, K)
            SVAL = (vrow(w) @ DVAL_centered + b).ravel()
            PVAL = (SVAL > 0) * 1
            err = (PVAL != LVAL).sum() / float(LVAL.size)
            minDCF = bayesRisk.compute_minDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0) 
            actDCF = bayesRisk.compute_actDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
            print ('Error rate: %.1f' % (err*100))
            print ('minDCF - pT = 0.1: %.4f' % minDCF )
            print ('actDCF - pT = 0.1: %.4f' % actDCF)
            print ()
            minDCF_values_c.append(minDCF)
            actDCF_values_c.append(actDCF)
            if minDCF < minDCF_absolute_c:
                minDCF_absolute_c = minDCF
                C_best_c = C
                actDCF_best_c = actDCF
    

    # Plot the results
    plt.figure()
    plt.semilogx(C_array, minDCF_values, label='minDCF (non-centered)', color='blue')
    plt.semilogx(C_array, actDCF_values, label='actDCF (non-centered)', color='cyan')
    plt.semilogx(C_array, minDCF_values_c, label='minDCF (centered)', color='red')
    plt.semilogx(C_array, actDCF_values_c, label='actDCF (centered)', color='orange')
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.legend()
    plt.grid(True)
    plt.title('minDCF and actDCF vs. C (Centered vs Non-Centered Data)')
    plt.show()
 
    print('---------------kernel----------------')
    minDCF_absolute_k = float('inf')
    # Polynomial kernel with d=2 and c=1
    kernelFunc = polyKernel(2, 1)

    minDCF_values_kernel = []
    actDCF_values_kernel = []

    for C in C_array:
        fScore = train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps=0.0)
        SVAL = fScore(DVAL)
        minDCF = bayesRisk.compute_minDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
        actDCF = bayesRisk.compute_actDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
        print ('minDCF - C %e: %.4f' % (C, minDCF))
        print ('actDCF - C %e: %.4f' % (C, actDCF))
        minDCF_values_kernel.append(minDCF)
        actDCF_values_kernel.append(actDCF)
        if minDCF < minDCF_absolute_k:
            minDCF_absolute_k = minDCF
            C_best_k = C
            actDCF_best_k = actDCF

    # Plot the results
    plt.figure()
    plt.semilogx(C_array, minDCF_values_kernel, label='minDCF (poly kernel)', color='blue')
    plt.semilogx(C_array, actDCF_values_kernel, label='actDCF (poly kernel)', color='cyan')
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.legend()
    plt.grid(True)
    plt.title('minDCF and actDCF vs. C (Polynomial Kernel)')
    plt.show()
    
    print('---------------RBF----------------')
    minDCF_absolute_rbf = float('inf')
    C_array = numpy.logspace(-3, 2, 11)
    gamma_array = numpy.exp(numpy.array([-4, -3, -2, -1]))

    minDCF_values = {gamma: [] for gamma in gamma_array}
    actDCF_values = {gamma: [] for gamma in gamma_array}

    for gamma in gamma_array:
        kernelFunc = rbfKernel(gamma)
        for C in C_array:
            fScore = train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps=1.0)
            SVAL = fScore(DVAL)
            minDCF = bayesRisk.compute_minDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
            actDCF = bayesRisk.compute_actDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
            minDCF_values[gamma].append(minDCF)
            actDCF_values[gamma].append(actDCF)
            print(f'Gamma: {gamma}, C: {C}, minDCF: {minDCF:.4f}, actDCF: {actDCF:.4f}')
            if minDCF < minDCF_absolute_rbf:
                minDCF_absolute_rbf = minDCF
                gamma_best_rbf = gamma
                C_bes_rbft = C
                actDCF_best_rbf = actDCF
    # Plot the results
    plt.figure()
    for gamma in gamma_array:
        plt.semilogx(C_array, minDCF_values[gamma], label=f'minDCF (gamma={gamma})')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.legend()
    plt.grid(True)
    plt.title('minDCF vs. C for different gamma values (RBF Kernel)')
    plt.show()
    
    plt.figure()
    for gamma in gamma_array:
        plt.semilogx(C_array, actDCF_values[gamma], label=f'actDCF (gamma={gamma})')
    plt.xlabel('C')
    plt.ylabel('actDCF')
    plt.legend()
    plt.grid(True)
    plt.title('actDCF vs. C for different gamma values (RBF Kernel)')
    plt.show()
          
        
    print ('Best C: %e, minDCF: %.4f, actDCF: %.4f' % (C_best, minDCF_absolute, actDCF_best))
    print ('Best C (centered): %e, minDCF: %.4f, actDCF: %.4f' % (C_best_c, minDCF_absolute_c, actDCF_best_c))
    print ('Best C (kernel): %e, minDCF: %.4f, actDCF: %.4f' % (C_best_k, minDCF_absolute_k, actDCF_best_k))
    print ('Best C (RBF): %e, minDCF: %.4f, actDCF: %.4f gamma: %e' % (C_bes_rbft, minDCF_absolute_rbf, actDCF_best_rbf, gamma_best_rbf))
           
