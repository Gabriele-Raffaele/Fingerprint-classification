import numpy
import bayesRisk
import logReg
import matplotlib
import matplotlib.pyplot as plt
import scipy.special
import lib
def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))
def bayes_error_plot(llr, labels, log_odds_range, model_name):
    minDCFs = []
    actDCFs = []
    for log_odds in log_odds_range:
        prior = lib.compute_prior(log_odds)
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

def bayesPlot(S, L, left = -3, right = 3, npts = 21):
    
    effPriorLogOdds = numpy.linspace(left, right, npts)
    effPriors = 1.0 / (1.0 + numpy.exp(-effPriorLogOdds))
    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        actDCF.append(bayesRisk.compute_actDCF_binary_fast(S, L, effPrior, 1.0, 1.0))
        minDCF.append(bayesRisk.compute_minDCF_binary_fast(S, L, effPrior, 1.0, 1.0))
    return effPriorLogOdds, actDCF, minDCF

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
def uniform_cost_matrix(nClasses):
    return numpy.ones((nClasses, nClasses)) - numpy.eye(nClasses)

# Assume that classes are labeled 0, 1, 2 ... (nClasses - 1)
def compute_confusion_matrix(predictedLabels, classLabels):
    nClasses = classLabels.max() + 1
    M = numpy.zeros((nClasses, nClasses), dtype=numpy.int32)
    for i in range(classLabels.size):
        M[predictedLabels[i], classLabels[i]] += 1
    return M

# Optimal Bayes deicsions for binary tasks with log-likelihood-ratio scores
def compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp):
    th = -numpy.log( (prior * Cfn) / ((1 - prior) * Cfp) )
    return numpy.int32(llr > th)

# Multiclass solution that works also for binary problems
def compute_empirical_Bayes_risk(predictedLabels, classLabels, prior_array, costMatrix, normalize=True):
    M = compute_confusion_matrix(predictedLabels, classLabels) # Confusion matrix
    errorRates = M / vrow(M.sum(0))
    bayesError = ((errorRates * costMatrix).sum(0) * prior_array.ravel()).sum()
    if normalize:
        return bayesError / numpy.min(costMatrix @ vcol(prior_array))
    return bayesError

# Specialized function for binary problems (empirical_Bayes_risk is also called DCF or actDCF)
def compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=True):
    M = compute_confusion_matrix(predictedLabels, classLabels) # Confusion matrix
    Pfn = M[0,1] / (M[0,1] + M[1,1])
    Pfp = M[1,0] / (M[0,0] + M[1,0])
    bayesError = prior * Cfn * Pfn + (1-prior) * Cfp * Pfp
    if normalize:
        return bayesError / numpy.minimum(prior * Cfn, (1-prior)*Cfp)
    return bayesError

# Compute empirical Bayes (DCF or actDCF) risk from llr with optimal Bayes decisions
def compute_empirical_Bayes_risk_binary_llr_optimal_decisions(llr, classLabels, prior, Cfn, Cfp, normalize=True):
    predictedLabels = compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp)
    return compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=normalize)


    

# Compute minDCF (fast version)
# If we sort the scores, then, as we sweep the scores, we can have that at most one prediction changes everytime. We can then keep a running confusion matrix (or simply the number of false positives and false negatives) that is updated everytime we move the threshold

# Auxiliary function, returns all combinations of Pfp, Pfn corresponding to all possible thresholds
# We do not consider -inf as threshld, since we use as assignment llr > th, so the left-most score corresponds to all samples assigned to class 1 already
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
def compute_minDCF_binary_fast(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):

    Pfn, Pfp, th = compute_Pfn_Pfp_allThresholds_fast(llr, classLabels)
    minDCF = (prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp) / numpy.minimum(prior * Cfn, (1-prior)*Cfp) # We exploit broadcasting to compute all DCFs for all thresholds
    idx = numpy.argmin(minDCF)
    if returnThreshold:
        return minDCF[idx], th[idx]
    else:
        return minDCF[idx]

compute_actDCF_binary_fast = compute_empirical_Bayes_risk_binary_llr_optimal_decisions # To have a function with a similar name to the minDCF one

KFOLD = 5

# Extract i-th fold from a 1-D numpy array (as for the single fold case, we do not need to shuffle scores in this case, but it may be necessary if samples are sorted in peculiar ways to ensure that validation and calibration sets are independent and with similar characteristics   
def extract_train_val_folds_from_ary(X, idx):
    return numpy.hstack([X[jdx::KFOLD] for jdx in range(KFOLD) if jdx != idx]), X[idx::KFOLD]



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
    
    wTar = pT / (ZTR>0).sum() # Compute the weights for the two classes
    wNon = (1-pT) / (ZTR<0).sum()

    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]
        s = numpy.dot(vcol(w).T, DTR).ravel() + b

        loss = numpy.logaddexp(0, -ZTR * s)
        loss[ZTR>0] *= wTar # Apply the weights to the loss computations
        loss[ZTR<0] *= wNon

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        G[ZTR > 0] *= wTar # Apply the weights to the gradient computations
        G[ZTR < 0] *= wNon
        
        GW = (vrow(G) * DTR).sum(1) + l * w.ravel()
        Gb = G.sum()
        return loss.sum() + l / 2 * numpy.linalg.norm(w)**2, numpy.hstack([GW, numpy.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = numpy.zeros(DTR.shape[0]+1))[0]
    #print ("Weighted Log-reg (pT %e) - lambda = %e - J*(w, b) = %e" % (pT, l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]
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

    mu, C = lib.compute_mu_C(X)

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


if __name__ == "__main__":
    # Load dataset
    D, L = lib.load('./trainData.txt')
    #importing the evalData, is in the same form as the trainData
    D_eval, L_eval = lib.load('./evalData.txt')

    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    # LOGISTIC REGRESSION
    lambda_reg = 1e-3
    DTR_expanded = lib.quadratic_feature_expansion(DTR)
    DVAL_expanded = lib.quadratic_feature_expansion(DVAL)
    # Addestramento del modello
    w_lr, b_lr = trainLogRegBinary(DTR_expanded, LTR, 10**-1.5 )
    # Compute empirical prior
    pEmp = (LTR == 1).sum() / LTR.size
    # Calcolo degli score sul validation set
    scores_logreg = (numpy.dot(w_lr.T, DVAL_expanded) + b_lr) - numpy.log(pEmp / (1 - pEmp))
    numpy.random.seed(0)
    idx = numpy.random.permutation(DVAL.shape[1])
    scores_logreg = scores_logreg[idx]
    LVAL_permuted = LVAL[idx]
  
    actDCF_logreg = []
    minDCF_logreg = []
    actDCF_absoluteLR = float('inf')
    
    priors = numpy.linspace(0.01, 0.99, 99)

    KFOLD = 5
    for prior in priors:    
        calibrated_scores_LR = []
        labels_LR = []
        for foldIdx in range(KFOLD):
            SCAL_lr, SVAL_lr = extract_train_val_folds_from_ary(scores_logreg, foldIdx)
            LCAL_lr, LVAL_lr = extract_train_val_folds_from_ary(LVAL_permuted, foldIdx)
            # Calibrazione degli score per logistic regression
            w_cal, b_cal = trainWeightedLogRegBinary(vrow(SCAL_lr), LCAL_lr, 0, prior)
            calibrated_SVAL_LR =  (w_cal.T @ vrow(SVAL_lr) + b_cal - numpy.log(prior / (1-prior))).ravel()
            calibrated_scores_LR.append(calibrated_SVAL_LR)
            labels_LR.append(LVAL_lr)
        calibrated_scores_LRHS = numpy.hstack(calibrated_scores_LR)
        labels_LRHS = numpy.hstack(labels_LR)

        actDCF_LR = bayesRisk.compute_actDCF_binary_fast(calibrated_scores_LRHS, labels_LRHS, 0.1, 1, 1)
        minDCF_LR = compute_minDCF_binary_fast(calibrated_scores_LRHS, labels_LRHS,0.1 , 1,1)
        # Valutazione del logistic regression
        actDCF_logreg.append(actDCF_LR)
        minDCF_logreg.append(minDCF_LR)
        if actDCF_LR < actDCF_absoluteLR:
            actDCF_absoluteLR = actDCF_LR
            best_minDCFLR = minDCF_LR
            best_priorLR = prior
    


    l_lr, a_lr, m_lr = bayesPlot(calibrated_scores_LRHS, labels_LRHS)
    lib.plot_individual(l_lr, a_lr, m_lr, 'Logistic Regression')

    print('Il miglior modello LR ha minDCF = %f, actDCF = %f, prior = %f' % (best_minDCFLR, actDCF_absoluteLR, best_priorLR))
    #---------------------------SVM---------------------------

    actDCF_absoluteSVM  = float('inf')
    actDCF_svm = []
    minDCF_svm = []
    
    # RBF SVM
    kernelFunc = lib.rbfKernel(numpy.exp(-2))
    
    # Addestramento del modello
    fScore = lib.train_dual_SVM_kernel(DTR, LTR, 10**1.5, kernelFunc, eps=1.0)
    scores_svm = fScore(DVAL)
    
    numpy.random.seed(0)
    idx = numpy.random.permutation(DVAL.shape[1])
    scores_svm = scores_svm[idx]
    LVAL_permutedSVM = LVAL[idx]
    priors = numpy.linspace(0.01, 0.99, 99)
    for prior in priors:
        calibrated_scores_SVM = []
        labels_SVM = []
        for foldIdx in range(KFOLD):
            SCAL_svm, SVAL_svm = extract_train_val_folds_from_ary(scores_svm, foldIdx)
            LCAL_svm, LVAL_svm = extract_train_val_folds_from_ary(LVAL_permutedSVM, foldIdx)
            
            # Calibrazione degli score per SVM
            w_cal_svm, b_cal_svm = lib.trainWeightedLogRegBinary(vrow(SCAL_svm), LCAL_svm, 0, prior)
            calibrated_SVAL_SVM = (w_cal_svm.T @ vrow(SVAL_svm) + b_cal_svm - numpy.log(prior / (1-prior))).ravel()
            
            calibrated_scores_SVM.append(calibrated_SVAL_SVM)
            labels_SVM.append(LVAL_svm)
        
        calibrated_scores_SVMHS = numpy.hstack(calibrated_scores_SVM)
        labels_SVMHS = numpy.hstack(labels_SVM)

        actDCF_SVM = bayesRisk.compute_actDCF_binary_fast(calibrated_scores_SVMHS, labels_SVMHS, 0.1, 1, 1)
        minDCF_SVM = lib.compute_minDCF_binary_fast(calibrated_scores_SVMHS, labels_SVMHS, 0.1, 1, 1)
        
        actDCF_svm.append(actDCF_SVM)
        minDCF_svm.append(minDCF_SVM)
        
        if actDCF_SVM < actDCF_absoluteSVM:
            actDCF_absoluteSVM = actDCF_SVM
            best_minDCFSVM = minDCF_SVM
            best_priorSVM = prior
    
    l_svm, a_svm, m_svm = bayesPlot(calibrated_scores_SVMHS, labels_SVMHS)
    lib.plot_individual(l_svm, a_svm, m_svm, 'SVM')

    print(f'Il miglior modello SVM ha minDCF = {best_minDCFSVM:.6f}, actDCF = {actDCF_absoluteSVM:.6f}, prior = {best_priorSVM:.2f}')

    #---------------------------GMM---------------------------
     
    # GMM
    # Addestramento del modello GMM
    gmm_0 = train_GMM_LBG_EM(DTR[:, LTR == 0], 8, covType='Diagonal', psiEig=1e-2)
    gmm_1 = train_GMM_LBG_EM(DTR[:, LTR == 1], 32, covType='Diagonal', psiEig=1e-2)
    # Calcolo degli score sul validation set
    scores_gmm = logpdf_GMM(DVAL, gmm_1) - logpdf_GMM(DVAL, gmm_0)
    numpy.random.seed(0)
    idx = numpy.random.permutation(DVAL.shape[1])
    scores_gmm= scores_gmm[idx]
    LVAL_permutedGMM= LVAL[idx]
  
    actDCF_gmm =[]
    minDCF_gmm = []
    actDCF_absoluteGMM = float('inf')

    priors = numpy.linspace(0.01, 0.99, 99)
    for prior in priors:
        calibrated_scores_GMM = []
        labels_GMM = []
        for foldIdx in range(KFOLD):
            SCAL_gmm, SVAL_gmm = extract_train_val_folds_from_ary(scores_gmm, foldIdx)
            LCAL_gmm, LVAL_gmm = extract_train_val_folds_from_ary(LVAL_permutedGMM, foldIdx)
            
            # Calibrazione degli score per GMM
            w_cal_gmm, b_cal_gmm = lib.trainWeightedLogRegBinary(vrow(SCAL_gmm), LCAL_gmm, 0, prior)
            calibrated_SVAL_GMM = (w_cal_gmm.T @ vrow(SVAL_gmm) + b_cal_gmm - numpy.log(prior / (1-prior))).ravel()
            
            calibrated_scores_GMM.append(calibrated_SVAL_GMM)
            labels_GMM.append(LVAL_gmm)
        
        calibrated_scores_GMMHS = numpy.hstack(calibrated_scores_GMM)
        labels_GMMHS = numpy.hstack(labels_GMM)

        actDCF_GMM = bayesRisk.compute_actDCF_binary_fast(calibrated_scores_GMMHS, labels_GMMHS, 0.1, 1, 1)
        minDCF_GMM = lib.compute_minDCF_binary_fast(calibrated_scores_GMMHS, labels_GMMHS, 0.1, 1, 1)
        
        actDCF_gmm.append(actDCF_GMM)
        minDCF_gmm.append(minDCF_GMM)
        
        if actDCF_GMM < actDCF_absoluteGMM:
            actDCF_absoluteGMM = actDCF_GMM
            best_minDCFGMM = minDCF_GMM
            best_priorGMM= prior

    l_gmm, a_gmm, m_gmm = bayesPlot(calibrated_scores_GMMHS, labels_GMMHS)
    lib.plot_individual(l_gmm, a_gmm, m_gmm, 'GMM')
    print(f'Il miglior modello GMM ha minDCF = {best_minDCFGMM:.6f}, actDCF = {actDCF_absoluteGMM:.6f}, prior = {best_priorGMM:.2f}')

    
    '''   
    MODELLI DEL LABORATORIO 10
    best GMM - diagonal, class 0: 8, class 1: 32, minDCF: 0.1312, actDCF: 0.1516
    best SVM - RBF minDCF: 0.1755, actDCF: 0.4216, gamma: 0.1353=e^-2, C: 31.6228=10^1.5
    best LR -  Quadratic features: minDCF=0.2436, actDCF=0.4952, λ=0.0316=10^(-1.5) 
    
    '''

    #fusion
    actDCF_absoluteFusion = float('inf')   
    priors = numpy.linspace(0.01, 0.99, 99)
    for prior in priors:
        calibrated_scores_fusion = []
        labels_fusion = []
        for foldIdx in range(KFOLD):
            SCAL_lr, SVAL_lr = extract_train_val_folds_from_ary(scores_logreg, foldIdx)
            SCAL_svm, SVAL_svm = extract_train_val_folds_from_ary(scores_svm, foldIdx)
            SCAL_gmm, SVAL_gmm = extract_train_val_folds_from_ary(scores_gmm, foldIdx)
            LCAL_fusion, LVAL_fusion = extract_train_val_folds_from_ary(LVAL_permuted, foldIdx)
            SCAL_fusion = numpy.vstack([SCAL_lr, SCAL_svm, SCAL_gmm])

       
            w_fusion, b_fusion = lib.trainWeightedLogRegBinary(SCAL_fusion, LCAL_fusion, 0, prior)
            SVAL_fusion = numpy.vstack([SVAL_lr, SVAL_svm, SVAL_gmm])
            calibrated_SVAL_fusion = (w_fusion.T @ SVAL_fusion + b_fusion - numpy.log(prior / (1-prior))).ravel()
            calibrated_scores_fusion.append(calibrated_SVAL_fusion)
            labels_fusion.append(LVAL_fusion)
            
        calibrated_scores_fusionHS = numpy.hstack(calibrated_scores_fusion)
        labels_fusionHS = numpy.hstack(labels_fusion)
        actDCF_fusion = bayesRisk.compute_actDCF_binary_fast(calibrated_scores_fusionHS, labels_fusionHS, 0.1, 1, 1)
        minDCF_fusion = lib.compute_minDCF_binary_fast(calibrated_scores_fusionHS, labels_fusionHS, 0.1, 1, 1)
        
        if actDCF_fusion < actDCF_absoluteFusion:
            actDCF_absoluteFusion = actDCF_fusion
            best_minDCFFusion = minDCF_fusion
            best_priorFusion = prior

    l_fusion, a_fusion, m_fusion = bayesPlot(calibrated_scores_fusionHS, labels_fusionHS)
    lib.plot_individual(l_fusion, a_fusion, m_fusion, 'Fusion')
    #here i print minDCF and actDCF with reference to the log-odds prior in this form (minDCF, actDCF, log-odds prior)
    print("minDCF and actDCF with reference to the log-odds prior")   
    print("(minDCF, actDCF, log-odds prior)")
    for i in range(len(l_fusion)):
        print(f'{m_fusion[i]:.6f}, {a_fusion[i]:.6f}, {l_fusion[i]:.2f}')
    #--------------------------------------------------------------------
    print(f'Il miglior modello fusion ha minDCF = {best_minDCFFusion:.6f}, actDCF = {actDCF_absoluteFusion:.6f}, prior = {best_priorFusion:.2f}')


    '''
    Il miglior modello LR ha minDCF = 0.243920, actDCF = 0.247600, prior = 0.90
    Il miglior modello SVM ha minDCF = 0.179291, actDCF = 0.179291, prior = 0.21
    Il miglior modello GMM ha minDCF = 0.145993, actDCF = 0.146985, prior = 0.99
    Il miglior modello fusion ha minDCF = 0.119624, actDCF = 0.119624, prior = 0.14
    '''


    #----------------------------evaluation------------------------------
    
    #lr

    D_eval_expanded = lib.quadratic_feature_expansion(D_eval)
    scores_logreg_ev = (numpy.dot(w_lr.T, D_eval_expanded) + b_lr) - numpy.log(pEmp / (1 - pEmp))
    w_lr_ev, b_lr_ev = lib.trainWeightedLogRegBinary(vrow(scores_logreg),LVAL_permuted , 0, 0.9)

    calibrated_eval_score_lr = (w_lr_ev @ vrow(scores_logreg_ev) + b_lr_ev - numpy.log(0.9 / (1-0.9))).ravel()
    #svm
    w_svm_ev, b_svm_ev = lib.trainWeightedLogRegBinary(vrow(scores_svm),LVAL_permutedSVM , 0, 0.21)

    scores_svm_eval = fScore(D_eval)
    calibrated_eval_score_svm = (w_svm_ev @ vrow(scores_svm_eval) + b_svm_ev - numpy.log(0.21 / (1-0.21))).ravel()
    #gmm
    w_gmm_ev, b_gmm_ev = lib.trainWeightedLogRegBinary(vrow(scores_gmm),LVAL_permutedGMM , 0, 0.99)

    scores_gmm_eval = logpdf_GMM(D_eval, gmm_1) - logpdf_GMM(D_eval, gmm_0)

    calibrated_eval_score_gmm = (w_gmm_ev @ vrow(scores_gmm_eval) + b_gmm_ev - numpy.log(0.99 / (1-0.99))).ravel()
    

    #fusion

    w_fusion_ev, b_fusion_ev = lib.trainWeightedLogRegBinary(numpy.vstack([scores_logreg, scores_svm, scores_gmm]), LVAL_permuted, 0, 0.14)
    scores_fusion_eval = numpy.vstack([scores_logreg_ev, scores_svm_eval, scores_gmm_eval])
    calibrated_eval_score_fusion = (w_fusion_ev @ scores_fusion_eval + b_fusion_ev - numpy.log(0.14 / (1-0.14))).ravel()

    

    #bayes error plot
    min_lr_ev=[]
    min_lr_t = []
    act_lr_ev=[]
    act_lr_t = []
    min_svm_ev=[]
    min_svm_t = []
    act_svm_ev=[]
    act_svm_t = []
    min_gmm_ev=[]
    min_gmm_t = []
    act_gmm_ev=[]
    act_gmm_t = []
    min_fusion_ev=[]
    act_fusion_ev=[]


    log_odds = numpy.linspace(-3, 3, 21)
    for lo in log_odds:
        prior = lib.compute_prior(lo)
        #lr
        minDCF_lr_ev = lib.compute_minDCF_binary_fast(calibrated_eval_score_lr, L_eval, prior, 1,1)
        min_lr_ev.append(minDCF_lr_ev)
        min_lr_t.append(lib.compute_minDCF_binary_fast(scores_logreg, LVAL_permuted, prior, 1, 1))
        actDCF_lr_ev = bayesRisk.compute_actDCF_binary_fast(calibrated_eval_score_lr, L_eval, prior , 1, 1) 
        act_lr_ev.append(actDCF_lr_ev)
        act_lr_t.append(bayesRisk.compute_actDCF_binary_fast(scores_logreg, LVAL_permuted, prior, 1, 1))
        #svm
        minDCF_svm_ev = lib.compute_minDCF_binary_fast(calibrated_eval_score_svm, L_eval, prior, 1, 1)
        min_svm_ev.append(minDCF_svm_ev)
        min_svm_t.append(lib.compute_minDCF_binary_fast(scores_svm, LVAL_permutedSVM, prior, 1, 1))
        actDCF_svm_ev = bayesRisk.compute_actDCF_binary_fast(calibrated_eval_score_svm, L_eval,prior, 1, 1)
        act_svm_ev.append(actDCF_svm_ev)
        act_svm_t.append(bayesRisk.compute_actDCF_binary_fast(scores_svm, LVAL_permutedSVM, prior, 1, 1))
        #gmm
        minDCF_gmm_ev = lib.compute_minDCF_binary_fast(calibrated_eval_score_gmm, L_eval, prior, 1, 1)
        min_gmm_ev.append(minDCF_gmm_ev)
        min_gmm_t.append(lib.compute_minDCF_binary_fast(scores_gmm, LVAL_permutedGMM, prior, 1, 1))
        actDCF_gmm_ev = bayesRisk.compute_actDCF_binary_fast(calibrated_eval_score_gmm, L_eval, prior, 1, 1)
        act_gmm_ev.append(actDCF_gmm_ev)
        act_gmm_t.append(bayesRisk.compute_actDCF_binary_fast(scores_gmm, LVAL_permutedGMM, prior, 1, 1))
        #fusion
        minDCF_fusion_ev = lib.compute_minDCF_binary_fast(calibrated_eval_score_fusion, L_eval, prior, 1, 1)
        min_fusion_ev.append(minDCF_fusion_ev)
        actDCF_fusion_ev = bayesRisk.compute_actDCF_binary_fast(calibrated_eval_score_fusion, L_eval, prior, 1, 1)
        act_fusion_ev.append(actDCF_fusion_ev)


    
    # Plotta i singoli grafici
    lib.plot_double(log_odds, min_lr_t ,act_lr_t , min_lr_ev, act_lr_ev, 'Logistic Regression')
    lib.plot_double(log_odds ,min_svm_t ,act_svm_t , min_svm_ev, act_svm_ev, 'SVM')
    lib.plot_double(log_odds, min_gmm_t ,act_gmm_t  , min_gmm_ev, act_gmm_ev, 'GMM')
    lib.plot_individual(log_odds, min_fusion_ev, act_fusion_ev, 'Fusion')

    # Plotta tutti i grafici in un'unica schermata
    plt.figure(figsize=(12, 8))
    plt.plot(log_odds, min_lr_ev, label='minDCF LR')
    plt.plot(log_odds, act_lr_ev, label='actDCF LR')
    plt.plot(log_odds, min_svm_ev, label='minDCF SVM')
    plt.plot(log_odds, act_svm_ev, label='actDCF SVM')
    plt.plot(log_odds, min_gmm_ev, label='minDCF GMM')
    plt.plot(log_odds, act_gmm_ev, label='actDCF GMM')
    plt.plot(log_odds, min_fusion_ev, label='minDCF Fusion')
    plt.plot(log_odds, act_fusion_ev, label='actDCF Fusion')
    plt.xlabel('log_odds')
    plt.ylabel('DCF')
    plt.title('DCF for Different Models')
    plt.legend()
    plt.grid(True)
    plt.show()






    #----------------------------------------------------------------------
    print()
    print("-------------------------------------------------")
    print("-------------------logistic regression on evalData.txt-------------------")
    D_e, L_e = lib.load('./evalData.txt')
    print("-----------------first part-----------------")
    lambdas = numpy.logspace(-4, 2, 13)
    min_DCF_01 = []
    min_DCF1= float('inf')
    for l in lambdas:
        w, b = trainLogRegBinary(DTR, LTR, l) # Train model
        sVal = numpy.dot(w.T, D_e) + b  # Compute validation scores
        PVAL = (sVal > 0) * 1 # Predict validation labels - sVal > 0 returns a boolean array, multiplying by 1 (integer) we get an integer array with 0's and 1's corresponding to the original True and False values
        err = (PVAL != L_e).sum() / float(L_e.size)
        print('Error rate: %.1f' % (err * 100))
        # Compute empirical prior
        pEmp = (L_e == 1).sum() / L_e.size
        # Compute LLR-like scores
        sValLLR = sVal - numpy.log(pEmp / (1 - pEmp))
        min=bayesRisk.compute_minDCF_binary_fast(sValLLR, L_e, 0.1, 1.0, 1.0)
        print ('minDCF - pT = 0.1: %.4f' % min )
        min_DCF_01.append(min)
        if (min < min_DCF1):
            min_DCF1 = min
            l1 = l
     # Plotting the results
    plt.figure()
    plt.plot(lambdas, min_DCF_01, label='minDCF (pT=0.1)')
    plt.xscale('log', base=10)
    plt.xlabel('Lambda')
    plt.ylabel('DCF')
    plt.title('Detection Cost Function (DCF) vs Lambda')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("-----------------second part-----------------")
    min_DCF_01_sub = []
    min_DCF2= float('inf')
    for l in lambdas:
        w, b = trainLogRegBinary(DTR[:, ::50], LTR[::50], l) # Train model
        sVal = numpy.dot(w.T, D_e) + b  # Compute validation scores
        PVAL = (sVal > 0) * 1 # Predict validation labels - sVal > 0 returns a boolean array, multiplying by 1 (integer) we get an integer array with 0's and 1's corresponding to the original True and False values
        err = (PVAL != L_e).sum() / float(L_e.size)
        print('Error rate: %.1f' % (err * 100))
        # Compute empirical prior
        pEmp = (L_e[::50] == 1).sum() / L_e[::50].size
        # Compute LLR-like scores
        sValLLR = sVal - numpy.log(pEmp / (1 - pEmp))
        min=bayesRisk.compute_minDCF_binary_fast(sValLLR, L_e, 0.1, 1.0, 1.0)
        print ('minDCF - pT = 0.1: %.4f' % min )
        min_DCF_01_sub.append(min)
        if (min < min_DCF2):
            min_DCF2 = min
            l2 = l
    # Plotting the results
    plt.figure()
    plt.plot(lambdas, min_DCF_01_sub, label='minDCF (pT=0.1)')
    plt.xscale('log', base=10)
    plt.xlabel('Lambda')
    plt.ylabel('DCF')
    plt.title('Detection Cost Function (DCF) vs Lambda (Downsampled Data)')
    plt.legend()
    plt.grid(True)
    plt.show()

    
    print("-----------------third part-----------------")
    min_DCF_01_w = []
    prior = 0.1
    min_DCF3= float('inf')

    for l in lambdas:
        w, b = trainWeightedLogRegBinary(DTR, LTR, l, prior)  # Train weighted model
        sVal = numpy.dot(w.T, D_e) + b  # Compute validation scores
        PVAL = (sVal > 0) * 1  # Predict validation labels
        err = (PVAL != L_e).sum() / float(L_e.size)
        print('Error rate: %.1f' % (err * 100))

        # Transform scores to LLRs by removing the log-odds of the prior
        sValLLR = sVal - numpy.log(prior / (1 - prior))
        min=bayesRisk.compute_minDCF_binary_fast(sValLLR, L_e, 0.1, 1.0, 1.0)
        print ('minDCF - pT = 0.1: %.4f' % min )
        min_DCF_01_w.append(min)
        if (min < min_DCF3):
            min_DCF3 = min
            l3 = l
    # Plotting the results of the weighted logistic regression and comparing with the unweighted logistic regression
    plt.figure()
    plt.plot(lambdas, min_DCF_01, label='minDCF (pT=0.1) - Unweighted')
    plt.plot(lambdas, min_DCF_01_w, label='minDCF (pT=0.1) - Weighted')
    plt.xscale('log', base=10)
    plt.xlabel('Lambda')
    plt.ylabel('DCF')
    plt.title('Detection Cost Function (DCF) vs Lambda (Weighted Data)')
    plt.legend()
    plt.grid(True)
    plt.show()

    

    print("-----------------fourth part-----------------")

# Expand features to quadratic terms
    D_e_expanded = quadratic_feature_expansion(D_e)
    DTR_expanded = quadratic_feature_expansion(DTR)
    # Plot for quadratic logistic regression on the full dataset
    min_DCF_quadratic = []
    min_DCF4= float('inf')
    for l in lambdas:
        w, b = trainLogRegBinary(DTR_expanded, LTR, l)
        sVal = numpy.dot(w.T, D_e_expanded) + b
        pEmp = (L_e == 1).sum() / L_e.size
        sValLLR = sVal - numpy.log(pEmp / (1 - pEmp))
        
        min=bayesRisk.compute_minDCF_binary_fast(sValLLR, L_e, 0.1, 1.0, 1.0)
        print ('minDCF - pT = 0.1: %.4f' % min )
        min_DCF_quadratic.append(min)
        if (min < min_DCF4):
            min_DCF4 = min
            l4 = l
    # Plot for quadratic logistic regression on the full dataset
    plt.figure()
    plt.plot(lambdas, min_DCF_quadratic, label='minDCF (pT=0.1)')
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
    plt.plot(lambdas, min_DCF_01, label='Minimum DCF, πT=0.1')
    plt.xlabel('λ')
    plt.ylabel('DCF')
    plt.title('Unweighted Logistic Regression (Full Data)')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.xscale('log', base=10)
    plt.plot(lambdas, min_DCF_01_sub, label='Minimum DCF, πT=0.1')
    plt.xlabel('λ')
    plt.ylabel('DCF')
    plt.title('Unweighted Logistic Regression (Downsampled Data)')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.xscale('log', base=10)
    plt.plot(lambdas, min_DCF_01_w, label='Minimum DCF, πT=0.1')
    plt.xlabel('λ')
    plt.ylabel('DCF')
    plt.title('Weighted Logistic Regression (Full Data)')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.xscale('log', base=10)
    plt.plot(lambdas, min_DCF_quadratic, label='Minimum DCF, πT=0.1')
    plt.xlabel('λ')
    plt.ylabel('DCF')
    plt.title('Quadratic Logistic Regression (Full Data)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("-----------------fifth part-----------------")
    D_e_mean = numpy.mean(D_e, axis=1)
    D_e_centered = lib.center_data(D_e, D_e_mean)
    DTR_mean = numpy.mean(DTR, axis=1)
    DTR_centered = lib.center_data(DTR, DTR_mean)
    
    UPCA = lib.compute_pca(D_e, 6) # Stima solo sui dati di addestramento
    DTR_pca = lib.apply_pca(UPCA, DTR_centered)   # Applicazione sui dati di addestramento
    D_e_pca = lib.apply_pca(UPCA, D_e_centered) # Applicazione sui dati di validazione
    min_DCF_01_centered = []
    min_DCF5= float('inf')

    for l in lambdas:
        w, b = trainLogRegBinary(DTR_pca, LTR, l) # Train model
        sVal = numpy.dot(w.T, D_e_pca) + b  # Compute validation scores
        PVAL = (sVal > 0) * 1 # Predict validation labels - sVal > 0 returns a boolean array, multiplying by 1 (integer) we get an integer array with 0's and 1's corresponding to the original True and False values
        err = (PVAL != L_e).sum() / float(L_e.size)
        print('Error rate: %.1f' % (err * 100))
        # Compute empirical prior
        pEmp = (L_e == 1).sum() / L_e.size
        # Compute LLR-like scores
        sValLLR = sVal - numpy.log(pEmp / (1 - pEmp))
        
        min=bayesRisk.compute_minDCF_binary_fast(sValLLR, L_e, 0.1, 1.0, 1.0)
        print ('minDCF - pT = 0.1: %.4f' % min )
        min_DCF_01_centered.append(min)
        if (min < min_DCF5):
            min_DCF5 = min
            l5 = l
    # Plot the results

    plt.figure(figsize=(8, 4))

    plt.xscale('log', base=10)
    plt.plot(lambdas, min_DCF_01, label='Minimum DCF, πT=0.1 (Original)', color='blue', linestyle='--')
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
    print(f'Original data: minDCF={min_DCF1:.4f}, λ={l1:.4f}')
    print(f'Downsampled data: minDCF={min_DCF2:.4f}, λ={l2:.4f}')     
    print(f'Weighted data: minDCF={min_DCF3:.4f}, λ={l3:.4f}')
    print(f'Quadratic features: minDCF={min_DCF4:.4f}, λ={l4:.4f}')
    print(f'Centered + PCA: minDCF={min_DCF5:.4f}, λ={l5:.4f}')
    print('')



