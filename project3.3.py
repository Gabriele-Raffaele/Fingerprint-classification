import numpy as np
import matplotlib.pyplot as plt
import lib

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

def split_db_2to1(D, L, seed=0):
    
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    
    return (DTR, LTR), (DVAL, LVAL)


if __name__ == '__main__':

    D, L = lib.load('./trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # Solution without PCA pre-processing and threshold selection. The threshold is chosen half-way between the two classes
    ULDA = lib.compute_lda_JointDiag(DTR, LTR, m=1)

    DTR_lda = lib.apply_lda(ULDA, DTR)

    if DTR_lda[0, LTR==1].mean() > DTR_lda[0, LTR==0].mean():
        ULDA = -ULDA
        DTR_lda = lib.apply_lda(ULDA, DTR)

    DVAL_lda  = lib.apply_lda(ULDA, DVAL)

    threshold = (DTR_lda[0, LTR==0].mean() + DTR_lda[0, LTR==1].mean()) / 2.0 # Estimated only on model training data
    print('threshold: ',threshold)

    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_lda[0] >= threshold] = 0
    PVAL[DVAL_lda[0] < threshold] = 1
    print('LDA Only:')
    print('Labels:     ', LVAL)
    print('Predictions:', PVAL)
    print('Number of errors:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    print('Error rate: %.1f%%' % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))
    # Plot LDA only
    plot_histograms(DTR_lda, DVAL_lda, 'LDA Only')
    #-----------------------------------------------------------
    threshold = (DTR_lda[0, LTR==0].mean() + DTR_lda[0, LTR==1].mean()) / 10 # Estimated only on model training data
    print('threshold: ',threshold)

    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_lda[0] >= threshold] = 0
    PVAL[DVAL_lda[0] < threshold] = 1
    print('LDA changed threshold:')
    print('Labels:     ', LVAL)
    print('Predictions:', PVAL)
    print('Number of errors:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    print('Error rate: %.1f%%' % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))
    # Plot LDA only
    plot_histograms(DTR_lda, DVAL_lda, 'LDA changed threshold')
    #-----------------------------------------------------------
    # Solution with PCA pre-processing with dimension m.
    for m in range(1,7):
        UPCA = lib.compute_pca(DTR, m) # Estimated only on model training data
        DTR_pca = lib.apply_pca(UPCA, DTR)   # Applied to original model training data
        DVAL_pca = lib.apply_pca(UPCA, DVAL) # Applied to original validation data

        ULDA_pca = lib.compute_lda_JointDiag(DTR_pca, LTR, m=1) # Estimated only on model training data, after PCA has been applied

        DTR_lda_pca = lib.apply_lda(ULDA_pca, DTR_pca)   # Applied to PCA-transformed model training data, the projected training samples are required to check the orientation of the direction and to compute the threshold
        if DTR_lda_pca[0, LTR==1].mean() > DTR_lda_pca[0, LTR==0].mean():
            ULDA_pca = -ULDA_pca
            DTR_lda_pca = lib.apply_lda(ULDA_pca, DTR_pca)

        DVAL_lda_pca = lib.apply_lda(ULDA_pca, DVAL_pca) # Applied to PCA-transformed validation data

        threshold_pca = (DTR_lda_pca[0, LTR==1].mean() + DTR_lda_pca[0, LTR==0].mean()) / 2.0 # Estimated only on model training data

        PVAL_pca = np.zeros(shape=LVAL.shape, dtype=np.int32)
        PVAL_pca[DVAL_lda_pca[0] >= threshold_pca] = 0
        PVAL_pca[DVAL_lda_pca[0] < threshold_pca] = 1
        print('\nPCA + LDA, with', m,' features :')
        print('Labels:     ', LVAL)
        print('Predictions:', PVAL_pca)
        print('Number of errors:', (PVAL_pca != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
        print('Error rate: %.2f%%' % ((PVAL_pca != LVAL).sum() / float(LVAL.size) * 100))
        # Plot PCA+LDA
        plot_histograms(DTR_lda_pca, DVAL_lda_pca, 'PCA + LDA m = %d' % m)
   

   
