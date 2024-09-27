
import numpy
import scipy.special
import sklearn.datasets
import lib 
import matplotlib.pyplot as plt

def center_data(D, mean):
    return D - mean[:, None]
def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def split_db_2to1(D, L, seed=0):
    
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)
def quadratic_feature_expansion(D):
    n = D.shape[1]
    D_expanded = numpy.zeros((D.shape[0] + D.shape[0]*(D.shape[0]+1)//2, n))
    
    idx = D.shape[0]
    for i in range(D.shape[0]):
        D_expanded[i, :] = D[i, :]
        for j in range(i, D.shape[0]):
            D_expanded[idx, :] = D[i, :] * D[j, :]
            idx += 1
    return D_expanded

# Optimize the logistic regression loss
def trainLogRegBinary(DTR, LTR, l):

    ZTR = LTR * 2.0 - 1.0 # We do it outside the objective function, since we only need to do it once

    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]
        s = numpy.dot(vcol(w).T, DTR).ravel() + b

        loss = numpy.logaddexp(0, -ZTR * s)

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        GW = (vrow(G) * DTR).mean(1) + l * w.ravel()
        Gb = G.mean()
        return loss.mean() + l / 2 * numpy.linalg.norm(w)**2, numpy.hstack([GW, numpy.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = numpy.zeros(DTR.shape[0]+1))[0]
    print ("Log-reg - lambda = %e - J*(w, b) = %e" % (l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]

# Optimize the weighted logistic regression loss
def trainWeightedLogRegBinary(DTR, LTR, l, pT):

    ZTR = LTR * 2.0 - 1.0 # We do it outside the objective function, since we only need to do it once
    
    wTrue = pT / (ZTR>0).sum() # Compute the weights for the two classes
    wFalse = (1-pT) / (ZTR<0).sum()

    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]
        s = numpy.dot(vcol(w).T, DTR).ravel() + b

        loss = numpy.logaddexp(0, -ZTR * s)
        loss[ZTR>0] *= wTrue # Apply the weights to the loss computations
        loss[ZTR<0] *= wFalse

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        G[ZTR > 0] *= wTrue # Apply the weights to the gradient computations
        G[ZTR < 0] *= wFalse
        
        GW = (vrow(G) * DTR).sum(1) + l * w.ravel()
        Gb = G.sum()
        return loss.sum() + l / 2 * numpy.linalg.norm(w)**2, numpy.hstack([GW, numpy.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = numpy.zeros(DTR.shape[0]+1))[0]
    print ("Weighted Log-reg (pT %e) - lambda = %e - J*(w, b) = %e" % (pT, l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]

import bayesRisk # Laboratory 7
if __name__ == '__main__':
    D, L = lib.load('./trainData.txt')
    print("-----------------first part-----------------")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    lambdas = numpy.logspace(-4, 2, 13)
    act_DCF_01 = []
    min_DCF_01 = []
    min_DCF1= float('inf')
    for l in lambdas:
        w, b = trainLogRegBinary(DTR, LTR, l) # Train model
        sVal = numpy.dot(w.T, DVAL) + b  # Compute validation scores
        PVAL = (sVal > 0) * 1 # Predict validation labels - sVal > 0 returns a boolean array, multiplying by 1 (integer) we get an integer array with 0's and 1's corresponding to the original True and False values
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        print('Error rate: %.1f' % (err * 100))
        # Compute empirical prior
        pEmp = (LTR == 1).sum() / LTR.size
        # Compute LLR-like scores
        sValLLR = sVal - numpy.log(pEmp / (1 - pEmp))
        act = bayesRisk.compute_empirical_Bayes_risk_binary_llr_optimal_decisions(sValLLR, LVAL, 0.1, 1.0, 1.0)
        min=bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, 0.1, 1.0, 1.0)
        print ('minDCF - pT = 0.1: %.4f' % min )
        print ('actDCF - pT = 0.1: %.4f' % act)
        act_DCF_01.append(act)
        min_DCF_01.append(min)
        if (min < min_DCF1):
            min_DCF1 = min
            actDCF1 = act
            l1 = l
     # Plotting the results
    plt.figure()
    plt.plot(lambdas, min_DCF_01, label='minDCF (pT=0.1)')
    plt.plot(lambdas, act_DCF_01, label='actDCF (pT=0.1)')
    plt.xscale('log', base=10)
    plt.xlabel('Lambda')
    plt.ylabel('DCF')
    plt.title('Detection Cost Function (DCF) vs Lambda')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("-----------------second part-----------------")
    act_DCF_01_sub = []
    min_DCF_01_sub = []
    min_DCF2= float('inf')
    for l in lambdas:
        w, b = trainLogRegBinary(DTR[:, ::50], LTR[::50], l) # Train model
        sVal = numpy.dot(w.T, DVAL) + b  # Compute validation scores
        PVAL = (sVal > 0) * 1 # Predict validation labels - sVal > 0 returns a boolean array, multiplying by 1 (integer) we get an integer array with 0's and 1's corresponding to the original True and False values
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        print('Error rate: %.1f' % (err * 100))
        # Compute empirical prior
        pEmp = (LTR[::50] == 1).sum() / LTR[::50].size
        # Compute LLR-like scores
        sValLLR = sVal - numpy.log(pEmp / (1 - pEmp))
        act = bayesRisk.compute_empirical_Bayes_risk_binary_llr_optimal_decisions(sValLLR, LVAL, 0.1, 1.0, 1.0)
        min=bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, 0.1, 1.0, 1.0)
        print ('minDCF - pT = 0.1: %.4f' % min )
        print ('actDCF - pT = 0.1: %.4f' % act)
        act_DCF_01_sub.append(act)
        min_DCF_01_sub.append(min)
        if (min < min_DCF2):
            min_DCF2 = min
            actDCF2 = act
            l2 = l
    # Plotting the results
    plt.figure()
    plt.plot(lambdas, min_DCF_01_sub, label='minDCF (pT=0.1)')
    plt.plot(lambdas, act_DCF_01_sub, label='actDCF (pT=0.1)')
    plt.xscale('log', base=10)
    plt.xlabel('Lambda')
    plt.ylabel('DCF')
    plt.title('Detection Cost Function (DCF) vs Lambda (Downsampled Data)')
    plt.legend()
    plt.grid(True)
    plt.show()

    
    print("-----------------third part-----------------")
    act_DCF_01_w = []
    min_DCF_01_w = []
    prior = 0.1
    min_DCF3= float('inf')

    for l in lambdas:
        w, b = trainWeightedLogRegBinary(DTR, LTR, l, prior)  # Train weighted model
        sVal = numpy.dot(w.T, DVAL) + b  # Compute validation scores
        PVAL = (sVal > 0) * 1  # Predict validation labels
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        print('Error rate: %.1f' % (err * 100))

        # Transform scores to LLRs by removing the log-odds of the prior
        sValLLR = sVal - numpy.log(prior / (1 - prior))
        act = bayesRisk.compute_empirical_Bayes_risk_binary_llr_optimal_decisions(sValLLR, LVAL, 0.1, 1.0, 1.0)
        min=bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, 0.1, 1.0, 1.0)
        print ('minDCF - pT = 0.1: %.4f' % min )
        print ('actDCF - pT = 0.1: %.4f' % act)
        act_DCF_01_w.append(act)
        min_DCF_01_w.append(min)
        if (min < min_DCF3):
            min_DCF3 = min
            actDCF3 = act
            l3 = l
    # Plotting the results of the weighted logistic regression and comparing with the unweighted logistic regression
    plt.figure()
    plt.plot(lambdas, min_DCF_01, label='minDCF (pT=0.1) - Unweighted')
    plt.plot(lambdas, act_DCF_01, label='actDCF (pT=0.1) - Unweighted')
    plt.plot(lambdas, min_DCF_01_w, label='minDCF (pT=0.1) - Weighted')
    plt.plot(lambdas, act_DCF_01_w, label='actDCF (pT=0.1) - Weighted')
    plt.xscale('log', base=10)
    plt.xlabel('Lambda')
    plt.ylabel('DCF')
    plt.title('Detection Cost Function (DCF) vs Lambda (Weighted Data)')
    plt.legend()
    plt.grid(True)
    plt.show()

    

    print("-----------------fourth part-----------------")

# Expand features to quadratic terms
    DTR_expanded = quadratic_feature_expansion(DTR)
    DVAL_expanded = quadratic_feature_expansion(DVAL)


    # Plot for quadratic logistic regression on the full dataset
    act_DCF_quadratic = []
    min_DCF_quadratic = []
    min_DCF4= float('inf')
    for l in lambdas:
        w, b = trainLogRegBinary(DTR_expanded, LTR, l)
        sVal = numpy.dot(w.T, DVAL_expanded) + b
        pEmp = (LTR == 1).sum() / LTR.size
        sValLLR = sVal - numpy.log(pEmp / (1 - pEmp))
        
        act = bayesRisk.compute_empirical_Bayes_risk_binary_llr_optimal_decisions(sValLLR, LVAL, 0.1, 1.0, 1.0)
        min=bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, 0.1, 1.0, 1.0)
        print ('minDCF - pT = 0.1: %.4f' % min )
        print ('actDCF - pT = 0.1: %.4f' % act)
        act_DCF_quadratic.append(act)
        min_DCF_quadratic.append(min)
        if (min < min_DCF4):
            min_DCF4 = min
            actDCF4 = act
            l4 = l
    # Plot for quadratic logistic regression on the full dataset
    plt.figure()
    plt.plot(lambdas, min_DCF_quadratic, label='minDCF (pT=0.1)')
    plt.plot(lambdas, act_DCF_quadratic, label='actDCF (pT=0.1)')
    plt.xscale('log', base=10)
    plt.xlabel('Lambda')
    plt.ylabel('DCF')
    plt.title('Detection Cost Function (DCF) vs Lambda (Quadratic Features)')
    plt.legend()
    plt.grid(True)
    plt.show()


    # Plot all four graphs in a single figure with subplots
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.xscale('log', base=10)
    plt.plot(lambdas, act_DCF_01, label='Actual DCF, πT=0.1')
    plt.plot(lambdas, min_DCF_01, label='Minimum DCF, πT=0.1')
    plt.xlabel('λ')
    plt.ylabel('DCF')
    plt.title('Unweighted Logistic Regression (Full Data)')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.xscale('log', base=10)
    plt.plot(lambdas, act_DCF_01_sub, label='Actual DCF, πT=0.1')
    plt.plot(lambdas, min_DCF_01_sub, label='Minimum DCF, πT=0.1')
    plt.xlabel('λ')
    plt.ylabel('DCF')
    plt.title('Unweighted Logistic Regression (Downsampled Data)')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.xscale('log', base=10)
    plt.plot(lambdas, act_DCF_01_w, label='Actual DCF, πT=0.1')
    plt.plot(lambdas, min_DCF_01_w, label='Minimum DCF, πT=0.1')
    plt.xlabel('λ')
    plt.ylabel('DCF')
    plt.title('Weighted Logistic Regression (Full Data)')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.xscale('log', base=10)
    plt.plot(lambdas, act_DCF_quadratic, label='Actual DCF, πT=0.1')
    plt.plot(lambdas, min_DCF_quadratic, label='Minimum DCF, πT=0.1')
    plt.xlabel('λ')
    plt.ylabel('DCF')
    plt.title('Quadratic Logistic Regression (Full Data)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("-----------------fifth part-----------------")
    D_train_mean = numpy.mean(DTR, axis=1)
    DTR_centered = center_data(DTR, D_train_mean)
    DVAL_centered = center_data(DVAL, D_train_mean)
    UPCA = lib.compute_pca(DTR, 6) # Stima solo sui dati di addestramento
    DTR_pca = lib.apply_pca(UPCA, DTR_centered)   # Applicazione sui dati di addestramento
    DVAL_pca = lib.apply_pca(UPCA, DVAL_centered) # Applicazione sui dati di validazione
    act_DCF_01_centered = []
    min_DCF_01_centered = []
    min_DCF5= float('inf')

    for l in lambdas:
        w, b = trainLogRegBinary(DTR_pca, LTR, l) # Train model
        sVal = numpy.dot(w.T, DVAL_pca) + b  # Compute validation scores
        PVAL = (sVal > 0) * 1 # Predict validation labels - sVal > 0 returns a boolean array, multiplying by 1 (integer) we get an integer array with 0's and 1's corresponding to the original True and False values
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        print('Error rate: %.1f' % (err * 100))
        # Compute empirical prior
        pEmp = (LTR == 1).sum() / LTR.size
        # Compute LLR-like scores
        sValLLR = sVal - numpy.log(pEmp / (1 - pEmp))
        
        act = bayesRisk.compute_empirical_Bayes_risk_binary_llr_optimal_decisions(sValLLR, LVAL, 0.1, 1.0, 1.0)
        min=bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, 0.1, 1.0, 1.0)
        print ('minDCF - pT = 0.1: %.4f' % min )
        print ('actDCF - pT = 0.1: %.4f' % act)
        act_DCF_01_centered.append(act)
        min_DCF_01_centered.append(min)
        if (min < min_DCF5):
            min_DCF5 = min
            actDCF5 = act
            l5 = l
    # Plot the results

    plt.figure(figsize=(8, 4))

    plt.xscale('log', base=10)
    plt.plot(lambdas, act_DCF_01, label='Actual DCF, πT=0.1 (Original)', color='blue', linestyle='-')
    plt.plot(lambdas, min_DCF_01, label='Minimum DCF, πT=0.1 (Original)', color='blue', linestyle='--')
    plt.plot(lambdas, act_DCF_01_centered, label='Actual DCF, πT=0.1 (Centered + pca)', color='red', linestyle='-')
    plt.plot(lambdas, min_DCF_01_centered, label='Minimum DCF, πT=0.1 (Centered + pca)', color='red', linestyle='--')
    plt.xlabel('λ')
    plt.ylabel('DCF')
    plt.title('Comparison of Centered and Uncentered Data')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print('')
    print('Best results:')
    print(f'Original data: minDCF={min_DCF1:.4f}, actDCF={actDCF1:.4f}, λ={l1:.4f}')
    print(f'Downsampled data: minDCF={min_DCF2:.4f}, actDCF={actDCF2:.4f}, λ={l2:.4f}')     
    print(f'Weighted data: minDCF={min_DCF3:.4f}, actDCF={actDCF3:.4f}, λ={l3:.4f}')
    print(f'Quadratic features: minDCF={min_DCF4:.4f}, actDCF={actDCF4:.4f}, λ={l4:.4f}')
    print(f'Centered + PCA: minDCF={min_DCF5:.4f}, actDCF={actDCF5:.4f}, λ={l5:.4f}')
    print('')






