import numpy
import sklearn.datasets 
import scipy.special
import matplotlib
import matplotlib.pyplot as plt

import scipy.linalg


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
#-----------------LOGISTIC REGRESSION----------------


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




#----------------------------------------------------


#----------------SVM----------------
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
#----------------------------------------------------

#--------------------GMM-----------------------------

def logpdf_GAU_ND(x, mu, C): # Fast version from Lab 4
    P = numpy.linalg.inv(C)
    return -0.5*x.shape[0]*numpy.log(numpy.pi*2) - 0.5*numpy.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)

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


#----------------------------------------------------

# Funzione per plottare i singoli grafici
def plot_double(x, y_min_t, y_act_t,y_min_e, y_act_e, title):
    plt.figure()
    plt.plot(x, y_min_t, label='minDCF (train)')
    plt.plot(x, y_act_t, label='actDCF (train)')
    plt.plot(x, y_min_e, label='minDCF (eval)')
    plt.plot(x, y_act_e, label='actDCF (eval)')
    plt.xlabel('log_odds')
    plt.ylabel('DCF')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
def plot_individual(x,y_min, y_act, title):
    plt.figure()
    plt.plot(x, y_min, label='minDCF ')
    plt.plot(x, y_act, label='actDCF ')

    plt.xlabel('log_odds')
    plt.ylabel('DCF')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
#----------------------Bayes--------------------------
def compute_prior(log_odds):
    return 1 / (1 + numpy.exp(-log_odds))

def bayes_error_plot(llr, labels, log_odds_range, model_name):
    minDCFs = []
    actDCFs = []
    for log_odds in log_odds_range:
        prior = compute_prior(log_odds)
        minDCF, threshold = compute_minDCF_binary_fast(llr, labels, prior, 1, 1, returnThreshold=True)
        actDCF = compute_empirical_Bayes_risk_binary_llr_optimal_decisions(llr, labels, prior, 1, 1, normalize=True)
        minDCFs.append(minDCF)
        actDCFs.append(actDCF)
    
    plt.plot(log_odds_range, minDCFs, label=f'{model_name} minDCF')
    plt.plot(log_odds_range, actDCFs, label=f'{model_name} actDCF', linestyle='dashed')
    plt.xlabel('Prior Log Odds')
    plt.ylabel('DCF')
    plt.title('Bayes Error Plot')
    plt.legend()

def compute_prior(log_odds):
    return 1 / (1 + numpy.exp(-log_odds))

# compute matrix of posteriors from class-conditional log-likelihoods (each column represents a sample) and prior array
def compute_posteriors(log_clas_conditional_ll, prior_array):
    logJoint = log_clas_conditional_ll + vcol(numpy.log(prior_array))
    logPost = logJoint - scipy.special.logsumexp(logJoint, 0)
    return numpy.exp(logPost)

# Compute optimal Bayes decisions for the matrix of class posterior (each column refers to a sample)
def compute_optimal_Bayes(posterior, costMatrix):
    expectedCosts = costMatrix @ posterior
    return numpy.argmin(expectedCosts, 0)

# Build uniform cost matrix with cost 1 for all kinds of error, and cost 0 for correct assignments
#Crea una matrice dei costi uniformi dove ogni errore ha un costo di 1 e le assegnazioni corrette hanno un costo di 0.
def uniform_cost_matrix(nClasses):
    return numpy.ones((nClasses, nClasses)) - numpy.eye(nClasses)

# Assume that classes are labeled 0, 1, 2 ... (nClasses - 1)
#Costruisce una matrice di confusione confrontando le etichette predette con quelle reali.
def compute_confusion_matrix(predictedLabels, classLabels):
    nClasses = classLabels.max() + 1
    M = numpy.zeros((nClasses, nClasses), dtype=numpy.int32)
    for i in range(classLabels.size):
        M[predictedLabels[i], classLabels[i]] += 1
    return M

# Optimal Bayes deicsions for binary tasks with log-likelihood-ratio scores
#Calcola decisioni ottimali di Bayes per compiti binari basati su log-likelihood ratio (llr). La soglia è determinata dai costi e dai priori.
def compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp):
    th = -numpy.log( (prior * Cfn) / ((1 - prior) * Cfp) )
    return numpy.int32(llr > th)

# Multiclass solution that works also for binary problems
#Calcola il rischio di Bayes empirico data una matrice di confusione, una matrice dei costi e un array di priori. Normalizza il rischio se specificato.
def compute_empirical_Bayes_risk(predictedLabels, classLabels, prior_array, costMatrix, normalize=True):
    M = compute_confusion_matrix(predictedLabels, classLabels) # Confusion matrix
    errorRates = M / vrow(M.sum(0))
    bayesError = ((errorRates * costMatrix).sum(0) * prior_array.ravel()).sum()
    if normalize:
        return bayesError / numpy.min(costMatrix @ vcol(prior_array))
    return bayesError

# Specialized function for binary problems (empirical_Bayes_risk is also called DCF or actDCF)
#Versione specifica per problemi binari del rischio di Bayes empirico. Normalizza il rischio se specificato.
def compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=True):
    M = compute_confusion_matrix(predictedLabels, classLabels) # Confusion matrix
    Pfn = M[0,1] / (M[0,1] + M[1,1])
    Pfp = M[1,0] / (M[0,0] + M[1,0])
    bayesError = prior * Cfn * Pfn + (1-prior) * Cfp * Pfp
    if normalize:
        return bayesError / numpy.minimum(prior * Cfn, (1-prior)*Cfp)
    return bayesError

# Compute empirical Bayes (DCF or actDCF) risk from llr with optimal Bayes decisions
#Calcola il rischio di Bayes empirico dato un llr e etichette di classe. Normalizza il rischio se specificato.
def compute_empirical_Bayes_risk_binary_llr_optimal_decisions(llr, classLabels, prior, Cfn, Cfp, normalize=True):
    predictedLabels = compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp)
    return compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=normalize)

# Compute all combinations of Pfn, Pfp for all thresholds (sorted)
#Calcola tutte le combinazioni di Pfn e Pfp per tutte le soglie. Versione lenta.
def compute_Pfn_Pfp_allThresholds_slow(llr, classLabels):
    llrSorter = numpy.argsort(llr)
    llrSorted = llr[llrSorter] # We sort the llrs

    Pfn = []
    Pfp = []
    thresholds = numpy.concatenate([numpy.array([-numpy.inf]), llrSorted, numpy.array([numpy.inf])]) #The function returns a slightly different array than the fast version, which does not include -numpy.inf as threshold - see the fast function comment
    for th in thresholds:
        M = compute_confusion_matrix(predictedLabels, classLabels) # type: ignore # Confusion matrix
        Pfn.append(M[0,1] / (M[0,1] + M[1,1]))
        Pfp.append(M[1,0] / (M[0,0] + M[1,0]))
    return Pfn, Pfp, thresholds
        
    
    
# Compute minDCF (slow version, loop over all thresholds recomputing the costs)
# Note: for minDCF llrs can be arbitrary scores, since we are optimizing the threshold
# We can therefore directly pass the logistic regression scores, or the SVM scores
#Calcola il minDCF (Decision Cost Function) ottimale passando in rassegna tutte le soglie. Versione lenta
def compute_minDCF_binary_slow(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):
    # llrSorter = numpy.argsort(llr) 
    # llrSorted = llr[llrSorter] # We sort the llrs
    # classLabelsSorted = classLabels[llrSorter] # we sort the labels so that they are aligned to the llrs
    # We can remove this part
    llrSorted = llr # In this function (slow version) sorting is not really necessary, since we re-compute the predictions and confusion matrices everytime
    
    thresholds = numpy.concatenate([numpy.array([-numpy.inf]), llrSorted, numpy.array([numpy.inf])])
    dcfMin = None
    dcfTh = None
    for th in thresholds:
        predictedLabels = numpy.int32(llr > th)
        dcf = compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp)
        if dcfMin is None or dcf < dcfMin:
            dcfMin = dcf
            dcfTh = th
    if returnThreshold:
        return dcfMin, dcfTh
    else:
        return dcfMin

# Compute minDCF (fast version)
# If we sort the scores, then, as we sweep the scores, we can have that at most one prediction changes everytime. We can then keep a running confusion matrix (or simply the number of false positives and false negatives) that is updated everytime we move the threshold

# Auxiliary function, returns all combinations of Pfp, Pfn corresponding to all possible thresholds
# We do not consider -inf as threshld, since we use as assignment llr > th, so the left-most score corresponds to all samples assigned to class 1 already
#Versione veloce per calcolare Pfn e Pfp, considerando tutte le soglie possibili.
def compute_Pfn_Pfp_allThresholds_fast(llr, classLabels):
    llrSorter = numpy.argsort(llr)
    llrSorted = llr[llrSorter] # We sort the llrs
    classLabelsSorted = classLabels[llrSorter] # we sort the labels so that they are aligned to the llrs

    Pfp = []
    Pfn = []
    
    nTrue = (classLabelsSorted==1).sum()
    nFalse = (classLabelsSorted==0).sum()
    nFalseNegative = 0 # With the left-most theshold all samples are assigned to class 1
    nFalsePositive = nFalse
    
    Pfn.append(nFalseNegative / nTrue)
    Pfp.append(nFalsePositive / nFalse)
    
    for idx in range(len(llrSorted)):
        if classLabelsSorted[idx] == 1:
            nFalseNegative += 1 # Increasing the threshold we change the assignment for this llr from 1 to 0, so we increase the error rate
        if classLabelsSorted[idx] == 0:
            nFalsePositive -= 1 # Increasing the threshold we change the assignment for this llr from 1 to 0, so we decrease the error rate
        Pfn.append(nFalseNegative / nTrue)
        Pfp.append(nFalsePositive / nFalse)

    #The last values of Pfn and Pfp should be 1.0 and 0.0, respectively
    #Pfn.append(1.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    #Pfp.append(0.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    llrSorted = numpy.concatenate([-numpy.array([numpy.inf]), llrSorted])

    # In case of repeated scores, we need to "compact" the Pfn and Pfp arrays (i.e., we need to keep only the value that corresponds to an actual change of the threshold
    PfnOut = []
    PfpOut = []
    thresholdsOut = []
    for idx in range(len(llrSorted)):
        if idx == len(llrSorted) - 1 or llrSorted[idx+1] != llrSorted[idx]: # We are indeed changing the threshold, or we have reached the end of the array of sorted scores
            PfnOut.append(Pfn[idx])
            PfpOut.append(Pfp[idx])
            thresholdsOut.append(llrSorted[idx])
            
    return numpy.array(PfnOut), numpy.array(PfpOut), numpy.array(thresholdsOut) # we return also the corresponding thresholds
    
# Note: for minDCF llrs can be arbitrary scores, since we are optimizing the threshold
# We can therefore directly pass the logistic regression scores, or the SVM scores
#Versione veloce per calcolare il minDCF.
def compute_minDCF_binary_fast(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):

    Pfn, Pfp, th = compute_Pfn_Pfp_allThresholds_fast(llr, classLabels)
    minDCF = (prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp) / numpy.minimum(prior * Cfn, (1-prior)*Cfp) # We exploit broadcasting to compute all DCFs for all thresholds
    idx = numpy.argmin(minDCF)
    if returnThreshold:
        return minDCF[idx], th[idx]
    else:
        return minDCF[idx]

compute_actDCF_binary_fast = compute_empirical_Bayes_risk_binary_llr_optimal_decisions # To have a function with a similar name to the minDCF one
#-----------------------------------------------------


def center_data(D, mean):
    return D - mean[:, None]



def mcol(v):
    return v.reshape((v.size, 1))
# Compute a dictionary of ML parameters for each class
def Gau_MVG_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L==lab]
        hParams[lab] = compute_mu_C(DX)
    return hParams

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
def logpdf_GAU_ND(x, mu, C):
    P = numpy.linalg.inv(C)
    return -0.5*x.shape[0]*numpy.log(numpy.pi*2) - 0.5*numpy.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)

# Compute per-class log-densities. We assume classes are labeled from 0 to C-1. The parameters of each class are in hParams (for class i, hParams[i] -> (mean, cov))
def compute_log_likelihood_Gau(D, hParams):

    S = numpy.zeros((len(hParams), D.shape[1]))
    for lab in range(S.shape[0]):
        S[lab, :] = logpdf_GAU_ND(D, hParams[lab][0], hParams[lab][1])
    return S

def compute_llr(S_logLikelihood):
    return S_logLikelihood[1, :] - S_logLikelihood[0, :]


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

def plot_lda_histogram(DP, L):
    print("Dimensioni di DP:", DP.shape)
    plt.figure()
    plt.hist(DP[0,L==0], bins=10, alpha=0.4, label='False')
    plt.hist(DP[0, L==1], bins=10, alpha=0.4, label='True')
    plt.xlabel('LD1')
    plt.ylabel('Frequenza')
    plt.legend()
    plt.title('Istogramma dei campioni LDA proiettati')
    plt.show()

def compute_Sb_Sw(D, L):
    Sb = 0
    Sw = 0
    muGlobal = vcol(D.mean(1))
    for i in numpy.unique(L):
        DCls = D[:, L == i]
        mu = vcol(DCls.mean(1))
        Sb += (mu - muGlobal) @ (mu - muGlobal).T * DCls.shape[1]
        Sw += (DCls - mu) @ (DCls - mu).T
    return Sb / D.shape[1], Sw / D.shape[1]

def compute_lda_geig(D, L, m):
    
    Sb, Sw = compute_Sb_Sw(D, L)
    s, U = scipy.linalg.eigh(Sb, Sw)
    return U[:, ::-1][:, 0:m]

def compute_lda_JointDiag(D, L, m):

    Sb, Sw = compute_Sb_Sw(D, L)

    U, s, _ = numpy.linalg.svd(Sw)
    P = numpy.dot(U * vrow(1.0/(s**0.5)), U.T)

    Sb2 = numpy.dot(P, numpy.dot(Sb, P.T))
    U2, s2, _ = numpy.linalg.svd(Sb2)

    P2 = U2[:, 0:m]
    return numpy.dot(P2.T, P).T

def apply_lda(U, D):
    return U.T @ D

# Compute a dictionary of ML parameters for each class - Tied Gaussian model
# We exploit the fact that the within-class covairance matrix is a weighted mean of the covraince matrices of the different classes
def Gau_Tied_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    hMeans = {}
    CGlobal = 0
    for lab in labelSet:
        DX = D[:, L==lab]
        mu, C_class = compute_mu_C(DX)
        CGlobal += C_class * DX.shape[1]
        hMeans[lab] = mu
    CGlobal = CGlobal / D.shape[1]
    for lab in labelSet:
        hParams[lab] = (hMeans[lab], CGlobal)
    return hParams
# Compute a dictionary of ML parameters for each class - Naive Bayes version of the model
# We compute the full covariance matrix and then extract the diagonal. Efficient implementations would work directly with just the vector of variances (diagonal of the covariance matrix)
def Gau_Naive_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L==lab]
        mu, C = compute_mu_C(DX)
        hParams[lab] = (mu, C * numpy.eye(D.shape[0]))
    return hParams
