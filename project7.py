import numpy
import scipy.special
import matplotlib
import matplotlib.pyplot as plt
import lib

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

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
    
if __name__ == '__main__':
    D, L = lib.load('./trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = lib.split_db_2to1(D, L)

    hParams_MVG = lib.Gau_MVG_ML_estimates(DTR, LTR)
    S_logLikelihood_MVG = lib.compute_log_likelihood_Gau(DVAL, hParams_MVG)
    llr = lib.compute_llr(S_logLikelihood_MVG)
    
    applications = [
        (0.5, 1, 1),  # Uniform prior and costs
        (0.9, 1, 1),  # Higher prior probability of a genuine sample
        (0.1, 1, 1),  # Higher prior probability of a fake sample
        (0.5, 1, 9),  # Higher cost of accepting a fake image
        (0.5, 9, 1)   # Higher cost of rejecting a legit image
    ]
    
    for prior, Cfn, Cfp in applications:
        minDCF, threshold = compute_minDCF_binary_fast(llr, LVAL, prior, Cfn, Cfp, returnThreshold=True)
        actDCF = compute_empirical_Bayes_risk_binary_llr_optimal_decisions(llr, LVAL, prior, Cfn, Cfp, normalize=True)
        
        effective_prior = prior * Cfn / (prior * Cfn + (1 - prior) * Cfp)
        calibration_loss = (actDCF - minDCF) / minDCF
    
        print(f'  Prior (π1={prior}):')
        print(f'    minDCF: {minDCF:.3f}')
        print(f'    actDCF: {actDCF:.3f}')
        print(f'    Calibration Loss: {calibration_loss * 100:.2f}%')
    
    #--------------------------------------------------------------------------
    priors = [0.1, 0.5, 0.9]
    # Calcolo per il modello MVG senza PCA
    for prior in priors:
        minDCF, threshold = compute_minDCF_binary_fast(llr, LVAL, prior, 1, 1, returnThreshold=True)
        actDCF = compute_empirical_Bayes_risk_binary_llr_optimal_decisions(llr, LVAL, prior, 1, 1, normalize=True)
        
        calibration_loss = (actDCF - minDCF) / minDCF
        
        print(f'  Prior (π1={prior}):')
        print(f'    minDCF: {minDCF:.3f}')
        print(f'    actDCF: {actDCF:.3f}')
        print(f'    Calibration Loss: {calibration_loss * 100:.2f}%')

    # Calcolo per il modello MVG con PCA per diversi valori di m
    for m in [1,2,3,4,5, 6]:  # esempio di valori di m
        UPCA = lib.compute_pca(DTR, m) # Stima solo sui dati di addestramento
        DTR_pca = lib.apply_pca(UPCA, DTR)   # Applicazione sui dati di addestramento
        DVAL_pca = lib.apply_pca(UPCA, DVAL) # Applicazione sui dati di validazione
        hParams_MVG_pca = lib.Gau_MVG_ML_estimates(DTR_pca, LTR)
        S_logLikelihood_MVG_pca = lib.compute_log_likelihood_Gau(DVAL_pca, hParams_MVG_pca)
        llr_MVG_pca = lib.compute_llr(S_logLikelihood_MVG_pca)

        print(f'PCA m={m}:')
        for prior in priors:
            minDCF_PCA, threshold_PCA = compute_minDCF_binary_fast(llr_MVG_pca, LVAL, prior, 1, 1, returnThreshold=True)
            actDCF_PCA = compute_empirical_Bayes_risk_binary_llr_optimal_decisions(llr_MVG_pca, LVAL, prior, 1, 1, normalize=True)
            
            calibration_loss_PCA = (actDCF_PCA - minDCF_PCA) / minDCF_PCA
            
            print(f'  Prior (π1={prior}):')
            print(f'    minDCF: {minDCF_PCA:.3f}')
            print(f'    actDCF: {actDCF_PCA:.3f}')
            print(f'    Calibration Loss: {calibration_loss_PCA * 100:.2f}%')


    #--------------------------------------------------------------------------
    
    UPCA = lib.compute_pca(DTR, 6) # Stima solo sui dati di addestramento
    DTR_pca = lib.apply_pca(UPCA, DTR)   # Applicazione sui dati di addestramento
    DVAL_pca = lib.apply_pca(UPCA, DVAL) # Applicazione sui dati di validazione

    hParams_MVG_pca = lib.Gau_MVG_ML_estimates(DTR_pca, LTR)
    hParams_Tied_pca= lib.Gau_Tied_ML_estimates(DTR_pca, LTR)
    hParams_Naive_pca = lib.Gau_Naive_ML_estimates(DTR_pca, LTR)
    # Calcolo per il modello MVG
    S_logLikelihood_MVG_pca = lib.compute_log_likelihood_Gau(DVAL_pca, hParams_MVG_pca)
    llr_MVG_pca = lib.compute_llr(S_logLikelihood_MVG_pca)
    # Calcolo per il modello Naive
    S_logLikelihood_Naive_pca = lib.compute_log_likelihood_Gau(DVAL_pca, hParams_Naive_pca)
    llr_Naive_pca = lib.compute_llr(S_logLikelihood_Naive_pca)
    # Calcolo per il modello Tied
    S_logLikelihood_T_pca = lib.compute_log_likelihood_Gau(DVAL_pca, hParams_Tied_pca)
    llr_T_pca = lib.compute_llr(S_logLikelihood_T_pca)
    
    # Range dei prior log odds
    log_odds_range = numpy.linspace(-4, 4, 100)

    # Plot dei Bayes error plots per i tre modelli
    plt.figure(figsize=(12, 8))

    bayes_error_plot(llr_MVG_pca, LVAL, log_odds_range, 'MVG')
    bayes_error_plot(llr_T_pca, LVAL, log_odds_range, 'Tied MVG')
    bayes_error_plot(llr_Naive_pca, LVAL, log_odds_range, 'Naive MVG')

    plt.show()