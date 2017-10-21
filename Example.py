import PCA.PCA as P
import numpy as np

# matrix phi has a constant variable -> needs to be preprocessed before using PCA
PHI = np.array([[1.1, 1., 2., 2.], [2.5, 3., 6., 2.], [4, 8, 1., 2.], [5., 3., 9., 2.]])
varnames = ['v1', 'v2', 'v3']  # variable names v1-v3 for plotting

print('PHI before preprocessing:')
print(PHI)

# preprocessing
preproc = P.preprocess(PHI)
PHI = preproc.manipulated

print('PHI after preprocessing:')
print(PHI)

# performing PCA
scaling = 'range'
pca = P.PCA(PHI, scaling)

# calculating the principal components
eta = pca.x2eta(PHI)
print('Principal Components using ' + pca.scaling + ' scaling:')
print(eta)

# plotting the absolute values of the eigenvectors
pca.eig_bar_plot_maker(len(varnames), varnames, plotABS=True)

# Running PCA regression test
P.test()
