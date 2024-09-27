import numpy
import scipy
import scipy.special
import matplotlib.pyplot as plt
import bayesRisk
import lib


def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def logpdf_GAU_ND(x, mu, C): # Fast version from Lab 4
    P = numpy.linalg.inv(C)
    return -0.5*x.shape[0]*numpy.log(numpy.pi*2) - 0.5*numpy.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)

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

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

######
# from GMM_load.py
import json

def save_gmm(gmm, filename):
    gmmJson = [(i, j.tolist(), k.tolist()) for i, j, k in gmm]
    with open(filename, 'w') as f:
        json.dump(gmmJson, f)
    
def load_gmm(filename):
    with open(filename, 'r') as f:
        gmm = json.load(f)
    return [(i, numpy.asarray(j), numpy.asarray(k)) for i, j, k in gmm]
######

def logpdf_GMM(X, gmm):

    S = []
    
    for w, mu, C in gmm:
        logpdf_conditional = logpdf_GAU_ND(X, mu, C)
        logpdf_joint = logpdf_conditional + numpy.log(w)
        S.append(logpdf_joint)
        
    S = numpy.vstack(S)
    logdens = scipy.special.logsumexp(S, axis=0)
    return logdens

def smooth_covariance_matrix(C, psi):

    U, s, Vh = numpy.linalg.svd(C)
    s[s<psi]=psi
    CUpd = U @ (vcol(s) * U.T)
    return CUpd

# X: Data matrix
# gmm: input gmm
# covType: 'Full' | 'Diagonal' | 'Tied'
# psiEig: factor for eignvalue thresholding
#
# return: updated gmm
def train_GMM_EM_Iteration(X, gmm, covType = 'Full', psiEig = None): 

    assert (covType.lower() in ['full', 'diagonal', 'tied'])
    
    # E-step
    S = []
    
    for w, mu, C in gmm:
        logpdf_conditional = logpdf_GAU_ND(X, mu, C)
        logpdf_joint = logpdf_conditional + numpy.log(w)
        S.append(logpdf_joint)
        
    S = numpy.vstack(S) # Compute joint densities f(x_i, c), i=1...n, c=1...G
    logdens = scipy.special.logsumexp(S, axis=0) # Compute marginal for samples f(x_i)

    # Compute posterior for all clusters - log P(C=c|X=x_i) = log f(x_i, c) - log f(x_i)) - i=1...n, c=1...G
    # Each row for gammaAllComponents corresponds to a Gaussian component
    # Each column corresponds to a sample (similar to the matrix of class posterior probabilities in Lab 5, but here the rows are associated to clusters rather than to classes
    gammaAllComponents = numpy.exp(S - logdens)

    # M-step
    gmmUpd = []
    for gIdx in range(len(gmm)): 
    # Compute statistics:
        gamma = gammaAllComponents[gIdx] # Extract the responsibilities for component gIdx
        Z = gamma.sum()
        F = vcol((vrow(gamma) * X).sum(1)) # Exploit broadcasting to compute the sum
        S = (vrow(gamma) * X) @ X.T
        muUpd = F/Z
        CUpd = S/Z - muUpd @ muUpd.T
        wUpd = Z / X.shape[1]
        if covType.lower() == 'diagonal':
            CUpd  = CUpd * numpy.eye(X.shape[0]) # An efficient implementation would store and employ only the diagonal terms, but is out of the scope of this script
        gmmUpd.append((wUpd, muUpd, CUpd))

    if covType.lower() == 'tied':
        CTied = 0
        for w, mu, C in gmmUpd:
            CTied += w * C
        gmmUpd = [(w, mu, CTied) for w, mu, C in gmmUpd]

    if psiEig is not None:
        gmmUpd = [(w, mu, smooth_covariance_matrix(C, psiEig)) for w, mu, C in gmmUpd]
        
    return gmmUpd

# Train a GMM until the average dela log-likelihood becomes <= epsLLAverage
def train_GMM_EM(X, gmm, covType = 'Full', psiEig = None, epsLLAverage = 1e-6, verbose=True):

    llOld = logpdf_GMM(X, gmm).mean()
    llDelta = None
    #if verbose:
        #print('GMM - it %3d - average ll %.8e' % (0, llOld))
    it = 1
    while (llDelta is None or llDelta > epsLLAverage):
        gmmUpd = train_GMM_EM_Iteration(X, gmm, covType = covType, psiEig = psiEig)
        llUpd = logpdf_GMM(X, gmmUpd).mean()
        llDelta = llUpd - llOld
        #if verbose:
            #print('GMM - it %3d - average ll %.8e' % (it, llUpd))
        gmm = gmmUpd
        llOld = llUpd
        it = it + 1

    #if verbose:
        #print('GMM - it %3d - average ll %.8e (eps = %e)' % (it, llUpd, epsLLAverage))        
    return gmm
    
def split_GMM_LBG(gmm, alpha = 0.1, verbose=True):

    gmmOut = []
    #if verbose:
        #print ('LBG - going from %d to %d components' % (len(gmm), len(gmm)*2))
    for (w, mu, C) in gmm:
        U, s, Vh = numpy.linalg.svd(C)
        d = U[:, 0:1] * s[0]**0.5 * alpha
        gmmOut.append((0.5 * w, mu - d, C))
        gmmOut.append((0.5 * w, mu + d, C))
    return gmmOut

# Train a full model using LBG + EM, starting from a single Gaussian model, until we have numComponents components. lbgAlpha is the value 'alpha' used for LBG, the otehr parameters are the same as in the EM functions above
def train_GMM_LBG_EM(X, numComponents, covType = 'Full', psiEig = None, epsLLAverage = 1e-6, lbgAlpha = 0.1, verbose=True):

    mu, C = compute_mu_C(X)

    if covType.lower() == 'diagonal':
        C = C * numpy.eye(X.shape[0]) # We need an initial diagonal GMM to train a diagonal GMM
    
    if psiEig is not None:
        gmm = [(1.0, mu, smooth_covariance_matrix(C, psiEig))] # 1-component model - if we impose the eignevalus constraint, we must do it for the initial 1-component GMM as well
    else:
        gmm = [(1.0, mu, C)] # 1-component model
    
    while len(gmm) < numComponents:
        # Split the components
        #if verbose:
            #print ('Average ll before LBG: %.8e' % logpdf_GMM(X, gmm).mean())
        gmm = split_GMM_LBG(gmm, lbgAlpha, verbose=verbose)
        #if verbose:
            #print ('Average ll after LBG: %.8e' % logpdf_GMM(X, gmm).mean()) # NOTE: just after LBG the ll CAN be lower than before the LBG - LBG does not optimize the ll, it just increases the number of components
        # Run the EM for the new GMM
        gmm = train_GMM_EM(X, gmm, covType = covType, psiEig = psiEig, verbose=verbose, epsLLAverage = epsLLAverage)
    return gmm
def compute_prior(log_odds):
    return 1 / (1 + numpy.exp(-log_odds))

def bayes_error_plot(llr, labels, log_odds_range, model_name):
    minDCFs = []
    actDCFs = []
    for log_odds in log_odds_range:
        prior = compute_prior(log_odds)
        minDCF, threshold = lib.compute_minDCF_binary_fast(llr, labels, prior, 1, 1, returnThreshold=True)
        actDCF = lib.compute_empirical_Bayes_risk_binary_llr_optimal_decisions(llr, labels, prior, 1, 1, normalize=True)
        minDCFs.append(minDCF)
        actDCFs.append(actDCF)
    
    plt.plot(log_odds_range, minDCFs, label=f'{model_name} minDCF')
    plt.plot(log_odds_range, actDCFs, label=f'{model_name} actDCF', linestyle='dashed')
    plt.xlabel('Prior Log Odds')
    plt.ylabel('DCF')
    plt.title('Bayes Error Plot')
    plt.legend()

    
if __name__ == '__main__':
    D, L = lib.load('./trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    # Configurazione dei modelli
    numComponentsList = [1, 2, 4, 8, 16, 32]
    covTypes = ['Full', 'Diagonal']
    psiEig = 1e-2

    minDCF_results = float('inf')

    for covType in covTypes:
        for class_fixed in [0, 1]:
            for numComponents_fixed in numComponentsList:
                for numComponents_variable in numComponentsList:
                    print(f'Training {covType} GMM with class {class_fixed} fixed ({numComponents_fixed} components) and class {1 - class_fixed} varying ({numComponents_variable} components)')

                    if class_fixed == 0:
                        gmm_fixed = train_GMM_LBG_EM(DTR[:, LTR == 0], numComponents_fixed, covType=covType, psiEig=psiEig)
                        gmm_variable = train_GMM_LBG_EM(DTR[:, LTR == 1], numComponents_variable, covType=covType, psiEig=psiEig)
                    else:
                        gmm_fixed = train_GMM_LBG_EM(DTR[:, LTR == 1], numComponents_fixed, covType=covType, psiEig=psiEig)
                        gmm_variable = train_GMM_LBG_EM(DTR[:, LTR == 0], numComponents_variable, covType=covType, psiEig=psiEig)

                    ll_fixed = logpdf_GMM(DVAL, gmm_fixed)
                    ll_variable = logpdf_GMM(DVAL, gmm_variable)

                    if class_fixed == 0:
                        S = ll_variable - ll_fixed
                    else:
                        S = ll_fixed - ll_variable

                    minDCF = bayesRisk.compute_minDCF_binary_fast(S, LVAL, 0.1, 1.0, 1.0)
                    actDCF = bayesRisk.compute_actDCF_binary_fast(S, LVAL, 0.1, 1.0, 1.0)
                    if (minDCF < minDCF_results):
                        minDCF_results = minDCF
                        actDCF_results = actDCF
                        covTypes_best = covType
                        class_fixed_best = class_fixed
                        numComponents_fixed_best = numComponents_fixed
                        numComponents_variable_best = numComponents_variable
                    print(f'{covType} GMM with class {class_fixed} fixed ({numComponents_fixed} components) and class {1 - class_fixed} varying ({numComponents_variable} components): minDCF={minDCF:.4f}, actDCF={actDCF:.4f}')

                    
    
    print('')
    print(f'Best minDCF: {minDCF_results:.4f}, bestactDCF: {actDCF_results},  with {covTypes_best} GMM with class {class_fixed_best} fixed ({numComponents_fixed_best} components) and class {1 - class_fixed_best} varying ({numComponents_variable_best} components)')
    '''
    best GMM - diagonal, class 0: 8, class 1: 32, minDCF: 0.1312, actDCF: 0.1516
    best SVM - RBF minDCF: 0.1755, actDCF: 0.4216, gamma: 0.1353=e^-2, C: 31.6228=10^1.5
    best LR -  Quadratic features: minDCF=0.2436, actDCF=0.4952, λ=0.0316=10^(-1.5) 
    '''

    log_odds = numpy.linspace(-4, 4, 20)

    # Expand features to quadratic terms
    DTR_expanded = lib.quadratic_feature_expansion(DTR)
    DVAL_expanded = lib.quadratic_feature_expansion(DVAL)
    # Plot for quadratic logistic regression on the full dataset
    act_DCF_quadratic = []
    min_DCF_quadratic = []
    min_DCF_RBF = []
    act_DCF_RBF = [] 
    min_DCF_GMM = []
    act_DCF_GMM = []
    kernelFunc = lib.rbfKernel(numpy.exp(-2))
    for lo in log_odds:
        # Logistic regression
        w, b = lib.trainLogRegBinary(DTR_expanded, LTR, 10**-1.5   )
        sVal = numpy.dot(w.T, DVAL_expanded) + b
        pEmp = (LTR == 1).sum() / LTR.size
        sValLLR = sVal - numpy.log(pEmp / (1 - pEmp))
        act = bayesRisk.compute_empirical_Bayes_risk_binary_llr_optimal_decisions(sValLLR, LVAL, compute_prior(lo), 1.0, 1.0)
        min=bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, compute_prior(lo), 1.0, 1.0)
        act_DCF_quadratic.append(act)
        min_DCF_quadratic.append(min)

        # RBF SVM
        fScore = lib.train_dual_SVM_kernel(DTR, LTR, 10**1.5, kernelFunc, eps=1.0)
        SVAL = fScore(DVAL)
        minDCF = bayesRisk.compute_minDCF_binary_fast(SVAL, LVAL, compute_prior(lo), 1.0, 1.0)
        actDCF = bayesRisk.compute_actDCF_binary_fast(SVAL, LVAL, compute_prior(lo), 1.0, 1.0)
        min_DCF_RBF.append(minDCF)
        act_DCF_RBF.append(actDCF)

        gmm_0 = train_GMM_LBG_EM(DTR[:, LTR == 0], 8, covType='Diagonal', psiEig=1e-2)
        gmm_1 = train_GMM_LBG_EM(DTR[:, LTR == 1], 32, covType='Diagonal', psiEig=1e-2)
        ll_0 = logpdf_GMM(DVAL, gmm_0)
        ll_1 = logpdf_GMM(DVAL, gmm_1)
        S = ll_1 - ll_0
        minDCF = bayesRisk.compute_minDCF_binary_fast(S, LVAL, compute_prior(lo), 1.0, 1.0)
        actDCF = bayesRisk.compute_actDCF_binary_fast(S, LVAL, compute_prior(lo), 1.0, 1.0)
        min_DCF_GMM.append(minDCF)
        act_DCF_GMM.append(actDCF)


        # Plot the results
    plt.figure(figsize=(15, 5))

    # Subplot for quadratic logistic regression
    plt.subplot(1, 3, 1)
    plt.plot(log_odds, act_DCF_quadratic, label='Actual DCF')
    plt.plot(log_odds, min_DCF_quadratic, label='Min DCF')
    plt.xlabel('Log-odds')
    plt.ylabel('DCF')
    plt.title('Quadratic Logistic Regression')
    plt.legend()

    # Subplot for RBF SVM
    plt.subplot(1, 3, 2)
    plt.plot(log_odds, act_DCF_RBF, label='Actual DCF')
    plt.plot(log_odds, min_DCF_RBF, label='Min DCF')
    plt.xlabel('Log-odds')
    plt.ylabel('DCF')
    plt.title('RBF SVM')
    plt.legend()

    # Subplot for GMM
    plt.subplot(1, 3, 3)
    plt.plot(log_odds, act_DCF_GMM, label='Actual DCF')
    plt.plot(log_odds, min_DCF_GMM, label='Min DCF')
    plt.xlabel('Log-odds')
    plt.ylabel('DCF')
    plt.title('GMM')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig('comparison.pdf')




