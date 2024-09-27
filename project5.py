
import numpy 
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import lib
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


# Funzione per visualizzare gli istogrammi delle feature rispetto alle distribuzioni gaussiane stimate
def plot_feature_histograms(DTR, LTR, DVAL, LVAL, gaussianParams):
    plt.figure(figsize=(12, 6))

    for phase, D, L, title in [('Training', DTR, LTR, 'Training Data'), ('Validation', DVAL, LVAL, 'Validation Data')]:
        plt.subplot(1, 2, 1 if phase == 'Training' else 2)
        for lab in set(L):
            DX = D[:, L==lab]
            for i in range(D.shape[0]):
                mu, sigma = gaussianParams[(lab, i)]
                plt.hist(DX[i], bins=20, alpha=0.5, label=f'Class {lab} Feature {i+1} (mu={mu:.2f}, sigma={sigma:.2f})')
        plt.title(f'{phase} Data - Feature Histograms')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        
    plt.tight_layout()
    plt.show()

# Funzione per analizzare l'adeguatezza dell'assunzione gaussiana per ciascuna feature
def analyze_gaussian_assumption(D, L, gaussianParams):
    labelSet = set(L)
    num_features = D.shape[0]
    for i in range(num_features):
        print(f'Feature {i+1}:')
        for lab in labelSet:
            DX = D[:, L==lab]
            mu, sigma = gaussianParams[(lab, i)]
            actual_mean = np.mean(DX[i])
            actual_std = np.std(DX[i])
            print(f'  Class {lab} - Gaussian Params: (mu={mu:.2f}, sigma={sigma:.2f}), Actual Mean: {actual_mean:.2f}, Actual Std: {actual_std:.2f}')
        print()

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def logpdf_GAU_ND(x, mu, C):
    P = numpy.linalg.inv(C)
    return -0.5*x.shape[0]*numpy.log(numpy.pi*2) - 0.5*numpy.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)

# Compute a dictionary of ML parameters for each class
def Gau_MVG_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L==lab]
        hParams[lab] = compute_mu_C(DX)
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

# Compute per-class log-densities. We assume classes are labeled from 0 to C-1. The parameters of each class are in hParams (for class i, hParams[i] -> (mean, cov))
def compute_log_likelihood_Gau(D, hParams):

    S = numpy.zeros((len(hParams), D.shape[1]))
    for lab in range(S.shape[0]):
        S[lab, :] = logpdf_GAU_ND(D, hParams[lab][0], hParams[lab][1])
    return S

# compute log-postorior matrix from log-likelihood matrix and prior array
def compute_logPosterior(S_logLikelihood, v_prior):
    SJoint = S_logLikelihood + vcol(numpy.log(v_prior))
    SMarginal = vrow(scipy.special.logsumexp(SJoint, axis=0))
    SPost = SJoint - SMarginal
    return SPost
def plot_histograms(DTR_lda, DVAL_lda, title):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.hist(DTR_lda[0, LTR==1], color='blue', alpha=0.5, label='Class 1 (Training)')
    plt.hist(DTR_lda[0, LTR==0], color='red', alpha=0.5, label='Class 2 (Training)')
    plt.title('LDA Projection (Training)')
    plt.xlabel('Projection on 1st LDA Component')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(DVAL_lda[0, LVAL==1], color='blue', alpha=0.5, label='Class 1 (Validation)')
    plt.hist(DVAL_lda[0, LVAL==0], color='red', alpha=0.5, label='Class 2 (Validation)')
    plt.title('LDA Projection (Validation)')
    plt.xlabel('Projection on 1st LDA Component')
    plt.ylabel('Frequency')
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show() 
# Funzione per calcolare i parametri della distribuzione gaussiana per ogni feature e classe
def compute_gaussian_parameters(D, L):
    labelSet = set(L)
    gaussianParams = {}
    for lab in labelSet:
        DX = D[:, L==lab]
        for i in range(D.shape[0]):  # Per ciascuna feature
            mu = np.mean(DX[i])
            sigma = np.std(DX[i])
            gaussianParams[(lab, i)] = (mu, sigma)
    return gaussianParams

def compute_llr(S_logLikelihood):
    return S_logLikelihood[1, :] - S_logLikelihood[0, :]
def predict_from_llr(llr, threshold=0):
    return (llr > threshold).astype(int)
def error_rate(PRED, LVAL):
    return (PRED != LVAL).sum() / float(LVAL.size) * 100

    
if __name__ == '__main__':
    D, L = load('./trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    ##-----------------MVG---------------------
    hParams_MVG = Gau_MVG_ML_estimates(DTR, LTR)
    S_logLikelihood_MVG = compute_log_likelihood_Gau(DVAL, hParams_MVG)
    S_logPost_MVG = compute_logPosterior(S_logLikelihood_MVG, numpy.ones(2)/2.)
    llr_MVG= logpdf_GAU_ND(DVAL, hParams_MVG[1][0], hParams_MVG[1][1]) - logpdf_GAU_ND(DVAL, hParams_MVG[0][0], hParams_MVG[0][1])
    PVAL_MVG = predict_from_llr(llr_MVG)   

    ##-----------------Tied---------------------
    hParams_Tied= Gau_Tied_ML_estimates(DTR, LTR)
    S_logLikelihood_T = compute_log_likelihood_Gau(DVAL, hParams_Tied)
    S_logPost_T = compute_logPosterior(S_logLikelihood_T, numpy.ones(2)/2.)
    llr_Tied = compute_llr(S_logLikelihood_T)
    PVAL_Tied = predict_from_llr(llr_Tied)

    #--------------------------LDA--------------------------------
    ULDA = lib.compute_lda_JointDiag(DTR, LTR, m=1)
    DTR_lda = lib.apply_lda(ULDA, DTR)
    if DTR_lda[0, LTR==1].mean() > DTR_lda[0, LTR==0].mean():
        ULDA = -ULDA
        DTR_lda = lib.apply_lda(ULDA, DTR)
    DVAL_lda  = lib.apply_lda(ULDA, DVAL)
    threshold = (DTR_lda[0, LTR==0].mean() + DTR_lda[0, LTR==1].mean()) / 2.0 # Estimated only on model training data
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_lda[0] >= threshold] = 0
    PVAL[DVAL_lda[0] < threshold] = 1
  
    ##-----------------Naive Bayes--------------------
    hParams_Naive = Gau_Naive_ML_estimates(DTR, LTR)
    S_logLikelihood_Naive = compute_log_likelihood_Gau(DVAL, hParams_Naive)
    S_logPost_Naive = compute_logPosterior(S_logLikelihood_Naive, numpy.ones(2)/2.)
    llr_Naive = compute_llr(S_logLikelihood_Naive)
    PVAL_Naive = predict_from_llr(llr_Naive)
    ##-----------------------------------------
    #------------------------Error rate---------------------
    print("------------------------Error rate---------------------")
    print('Error rate LDA: %.1f%%' % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))
    print('')
    print("MultiVariate Gaussian - Error rate: %.1f%%" % ((S_logPost_MVG.argmax(0) != LVAL).sum() / float(LVAL.size) * 100))  
    print('')
    print("Tied Gaussian - Error rate: %.1f%%" % ((S_logPost_T.argmax(0) != LVAL).sum() / float(LVAL.size) * 100))    
    print('')
    print("Naive Bayes Gaussian - Error rate: %.1f%%" % ((S_logPost_Naive.argmax(0) != LVAL).sum() / float(LVAL.size) * 100))
    print('')
    print("------------------------------------------------------")
    #-----------------------Pearson---------------------
    print("class false MVG covariance matrix:")
    print(np.array2string(hParams_MVG[0][1], precision=3, max_line_width=np.inf)) 
    print("class true MVG covariance matrix:")
    print(np.array2string(hParams_MVG[1][1], precision=3, max_line_width=np.inf)) 
    CorrF = hParams_MVG[0][1] / ( vcol(hParams_MVG[0][1].diagonal()**0.5) * vrow(hParams_MVG[0][1].diagonal()**0.5) )
    print("Corr false: ", np.array2string(CorrF, precision=3, max_line_width=np.inf)) 
    CorrT = hParams_MVG[1][1] / ( vcol(hParams_MVG[1][1].diagonal()**0.5) * vrow(hParams_MVG[1][1].diagonal()**0.5) )
    print("Corr true: ", np.array2string(CorrT, precision=3, max_line_width=np.inf)) 
    ##-----------------Analisi---------------------
    print('')
    print('analisi')
    gaussianParams = compute_gaussian_parameters(DTR, LTR)
    analyze_gaussian_assumption(D, L, gaussianParams)
    for i in range(6):
        print(f"------------------------------------------------------")
        print(f"Feature {i+1}")
    
        D_feature = D[i:i+1, :]
        (DTR_f1, LTR_f1), (DVAL_f1, LVAL_f1) = split_db_2to1(D_feature, L)
        hParams_MVG_f1 = Gau_MVG_ML_estimates(DTR_f1, LTR_f1)
        S_logLikelihood_MVG_f1 = compute_log_likelihood_Gau(DVAL_f1, hParams_MVG_f1)
        S_logPost_MVG_f1 = compute_logPosterior(S_logLikelihood_MVG_f1, np.ones(2)/2.)
        llr_MVG_f1 = compute_llr(S_logLikelihood_MVG_f1)
        PVAL_MVG_f1 = predict_from_llr(llr_MVG_f1)
        print("MultiVariate Gaussian - Error rate: %.1f%%" % error_rate(PVAL_MVG_f1, LVAL_f1))

    #-------------------------feature 1 a 4-----------------------
    D_features = D[:4,:]
    print("------------------------------------------------------")
    print("feature 1 a 4")
    (DTR_f, LTR_f), (DVAL_f, LVAL_f) = split_db_2to1(D_features, L)
    hParams_MVG_f = Gau_MVG_ML_estimates(DTR_f, LTR_f)
    hParams_Tied_f= Gau_Tied_ML_estimates(DTR_f, LTR_f)
    hParams_Naive_f = Gau_Naive_ML_estimates(DTR_f, LTR_f)
    #-Tied
    S_logLikelihood_T_f = compute_log_likelihood_Gau(DVAL_f, hParams_Tied_f)
    S_logPost_T_f = compute_logPosterior(S_logLikelihood_T_f, numpy.ones(2)/2.)
    llr_Tied_f = compute_llr(S_logLikelihood_T_f)
    PVAL_Tied_f = predict_from_llr(llr_Tied_f)
    print("Error rate Tied: %.1f%%" % ((PVAL_Tied_f != LVAL_f).sum() / float(LVAL_f.size) * 100))    
    print("Tied Gaussian - Error rate: %.1f%%" % ((S_logPost_T_f.argmax(0) != LVAL_f).sum() / float(LVAL_f.size) * 100))
    #-Naive

    S_logLikelihood_Naive_f = compute_log_likelihood_Gau(DVAL_f, hParams_Naive_f)
    S_logPost_Naive_f = compute_logPosterior(S_logLikelihood_Naive_f, numpy.ones(2)/2.)
    llr_Naive_f = compute_llr(S_logLikelihood_Naive_f)
    PVAL_Naive_f = predict_from_llr(llr_Naive_f)
    print("Error rate Naive: %.1f%%" % ((PVAL_Naive_f != LVAL_f).sum() / float(LVAL_f.size) * 100))
    print("Naive Bayes Gaussian - Error rate: %.1f%%" % ((S_logPost_Naive_f.argmax(0) != LVAL_f).sum() / float(LVAL_f.size) * 100))
    #-MVG
    S_logLikelihood_MVG_f = compute_log_likelihood_Gau(DVAL_f, hParams_MVG_f)
    S_logPost_MVG_f = compute_logPosterior(S_logLikelihood_MVG_f, numpy.ones(2)/2.)
    llr_MVG_f = compute_llr(S_logLikelihood_MVG_f)
    PVAL_MVG_f = predict_from_llr(llr_MVG_f)
    print("Error rate MVG: %.1f%%" % ((PVAL_MVG_f != LVAL_f).sum() / float(LVAL_f.size) * 100))
    print("MultiVariate Gaussian - Error rate: %.1f%%" % ((S_logPost_MVG_f.argmax(0) != LVAL_f).sum() / float(LVAL_f.size) * 100))
    #--------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------
    #-----------------------feature 1 e 2--------------------------------------------
    D_features_12 = D[:2,:]
    print("------------------------------------------------------")
    print("feature 1 e 2")
    (DTR_12, LTR_12), (DVAL_12, LVAL_12) = split_db_2to1(D_features_12, L)
    hParams_MVG_12 = Gau_MVG_ML_estimates(DTR_12, LTR_12)
    hParams_Tied_12= Gau_Tied_ML_estimates(DTR_12, LTR_12)
    #-Tied
    S_logLikelihood_T_12 = compute_log_likelihood_Gau(DVAL_12, hParams_Tied_12)
    S_logPost_T_12 = compute_logPosterior(S_logLikelihood_T_12, numpy.ones(2)/2.)
    llr_Tied_12 = compute_llr(S_logLikelihood_T_12)
    PVAL_Tied_12 = predict_from_llr(llr_Tied_12)
    print("Error rate Tied (1,2): %.1f%%" % ((PVAL_Tied_12 != LVAL_12).sum() / float(LVAL_12.size) * 100))    
    print("Tied Gaussian - Error rate (1,2): %.1f%%" % ((S_logPost_T_12.argmax(0) != LVAL_f).sum() / float(LVAL_12.size) * 100))
    #-MVG
    S_logLikelihood_MVG_12 = compute_log_likelihood_Gau(DVAL_12, hParams_MVG_12)
    S_logPost_MVG_12 = compute_logPosterior(S_logLikelihood_MVG_12, numpy.ones(2)/2.)
    llr_MVG_12 = compute_llr(S_logLikelihood_MVG_12)
    PVAL_MVG_12 = predict_from_llr(llr_MVG_12)
    print("Error rate MVG (1,2): %.1f%%" % ((PVAL_MVG_12 != LVAL_12).sum() / float(LVAL_12.size) * 100))
    print("MultiVariate Gaussian - Error rate (1,2): %.1f%%" % ((S_logPost_MVG_12.argmax(0) != LVAL_12).sum() / float(LVAL_12.size) * 100))

    #-----------------------feature 3 e 4----------------------------------------
    print("------------------------------------------------------")
    print("feature 3 e 4")
    D_features_34 = D[2:4,:]
    (DTR_34, LTR_34), (DVAL_34, LVAL_34) = split_db_2to1(D_features_34, L)
    hParams_MVG_34 = Gau_MVG_ML_estimates(DTR_34, LTR_34)
    hParams_Tied_34= Gau_Tied_ML_estimates(DTR_34, LTR_34)
    #-Tied
    S_logLikelihood_T_34 = compute_log_likelihood_Gau(DVAL_34, hParams_Tied_34)
    S_logPost_T_34 = compute_logPosterior(S_logLikelihood_T_34, numpy.ones(2)/2.)
    llr_Tied_34 = compute_llr(S_logLikelihood_T_34)
    PVAL_Tied_34 = predict_from_llr(llr_Tied_34)
    print("Error rate Tied (3,4): %.1f%%" % ((PVAL_Tied_34 != LVAL_34).sum() / float(LVAL_34.size) * 100))
    print("Tied Gaussian - Error rate (3,4): %.1f%%" % ((S_logPost_T_34.argmax(0) != LVAL_34).sum() / float(LVAL_34.size) * 100))
    #-MVG
    S_logLikelihood_MVG_34 = compute_log_likelihood_Gau(DVAL_34, hParams_MVG_34)
    S_logPost_MVG_34 = compute_logPosterior(S_logLikelihood_MVG_34, numpy.ones(2)/2.)
    llr_MVG_34 = compute_llr(S_logLikelihood_MVG_34)
    PVAL_MVG_34 = predict_from_llr(llr_MVG_34)
    print("Error rate MVG (3,4): %.1f%%" % ((PVAL_MVG_34 != LVAL_34).sum() / float(LVAL_34.size) * 100))
    print("MultiVariate Gaussian - Error rate (3,4): %.1f%%" % ((S_logPost_MVG_34.argmax(0) != LVAL_34).sum() / float(LVAL_34.size) * 100))
    #--------------------------------------------------------------------------------
    #-------------------------------PCA---------------------------------------------
    UPCA = lib.compute_pca(DTR, 6) # Estimated only on model training data
    DTR_pca = lib.apply_pca(UPCA, DTR)   # Applied to original model training data
    DVAL_pca = lib.apply_pca(UPCA, DVAL) # Applied to original validation data
    hParams_MVG_pca = Gau_MVG_ML_estimates(DTR_pca, LTR)
    hParams_Tied_pca= Gau_Tied_ML_estimates(DTR_pca, LTR)
    hParams_Naive_pca = Gau_Naive_ML_estimates(DTR_pca, LTR)
    #-Tied
    S_logLikelihood_T_pca = compute_log_likelihood_Gau(DVAL_pca, hParams_Tied_pca)
    S_logPost_T_pca = compute_logPosterior(S_logLikelihood_T_pca, numpy.ones(2)/2.)
    llr_Tied_pca = compute_llr(S_logLikelihood_T_pca)
    PVAL_Tied_pca = predict_from_llr(llr_Tied_pca)
    print("Error rate Tied (PCA): %.1f%%" % ((PVAL_Tied_pca != LVAL).sum() / float(LVAL.size) * 100))    
    print("Tied Gaussian - Error rate (PCA): %.1f%%" % ((S_logPost_T_pca.argmax(0) != LVAL).sum() / float(LVAL.size) * 100))
    #-Naive

    S_logLikelihood_Naive_pca = compute_log_likelihood_Gau(DVAL_pca, hParams_Naive_pca)
    S_logPost_Naive_pca = compute_logPosterior(S_logLikelihood_Naive_pca, numpy.ones(2)/2.)
    llr_Naive_pca = compute_llr(S_logLikelihood_Naive_pca)
    PVAL_Naive_pca = predict_from_llr(llr_Naive_pca)
    print("Error rate Naive (PCA): %.1f%%" % ((PVAL_Naive_pca != LVAL).sum() / float(LVAL.size) * 100))
    print("Naive Bayes Gaussian - Error rate (PCA): %.1f%%" % ((S_logPost_Naive_pca.argmax(0) != LVAL).sum() / float(LVAL.size) * 100))
    #-MVG
    S_logLikelihood_MVG_pca = compute_log_likelihood_Gau(DVAL_pca, hParams_MVG_pca)
    S_logPost_MVG_pca = compute_logPosterior(S_logLikelihood_MVG_pca, numpy.ones(2)/2.)
    llr_MVG_pca = compute_llr(S_logLikelihood_MVG_pca)
    PVAL_MVG_pca = predict_from_llr(llr_MVG_pca)
    print("Error rate MVG (PCA): %.1f%%" % ((PVAL_MVG_pca != LVAL).sum() / float(LVAL.size) * 100))
    print("MultiVariate Gaussian - Error rate (PCA): %.1f%%" % ((S_logPost_MVG_pca.argmax(0) != LVAL).sum() / float(LVAL.size) * 100))




   







