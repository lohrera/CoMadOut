import pandas as pd
import numpy as np
from src.utils.stats_utils import zscore
from itertools import combinations_with_replacement
from fast_map import fast_map_async
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math

class ComadPCA(object):
    
    def __init__(self, n_components=2, center_by='median', centermethod='offset', sign_flip=False, random_state=0, verbose=0, fast=False):
        
        self.n_components=n_components
        self.n_components_=n_components
        self.eigenvalues_lambda = None
        self.eigenvectors_v = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.centering_offset = None
        self.centering_by = center_by # 'median' -> comadPCA, 'mean' -> Standard PCA
        self.centermethod = centermethod # 'offset' or 'zscore'
        self.sign_flip = sign_flip
        self.random_state=random_state
        self.verbose=verbose
        self.fast=fast
        
        np.random.seed(random_state)
        
    def coMAD(self, X):
        
        if type(X) != pd.DataFrame: X = pd.DataFrame(X, dtype=np.float64)
        X = X.apply(pd.to_numeric)
        
        # Eq.3:
        # A_i - med(A_i)
        X_centered = X - self.centering_offset # # 'median' -> comadPCA, 'mean' -> Standard PCA
                
        # Eq.1: 
        # [comad(A_1, A_1)...comad(A_d, A_d)]
        
        coMAD = np.zeros((X_centered.shape[1],X_centered.shape[1]), dtype=np.float64)
        median_center = (self.centering_by=='median')
        for i in range(X_centered.shape[1]): 
            for j in range(X_centered.shape[1]): 
                
                # Eq.2:
                # comad(A_i, A_j) = med[ (A_i - med(A_i)) * (A_j - med(A_j)) ]
                
                centered_product_of_feat_ij = X_centered.iloc[:, i] * X_centered.iloc[:, j] 
                if median_center:
                    coMAD[i,j] = centered_product_of_feat_ij.median()
                else:
                    coMAD[i,j] = centered_product_of_feat_ij.mean()
                
        return pd.DataFrame(coMAD, dtype=np.float64)
    
    
    def coMAD_fast(self, X):        
        
        if type(X) != pd.DataFrame: X = pd.DataFrame(X, dtype=np.float64)
        X = X.apply(pd.to_numeric)
        
        # Eq.3:
        # A_i - med(A_i)
        X_centered = X - self.centering_offset # # 'median' -> comadPCA, 'mean' -> Standard PCA
        
        # Eq.1: 
        # [comad(A_1, A_1)...comad(A_d, A_d)]
        
        coMAD = np.zeros((X_centered.shape[1],X_centered.shape[1]), dtype=np.float64)
        median_center = (self.centering_by=='median')
        
        def calculate_coMAD_ij(combi):
            i,j=combi
            retval=0.
            
            # Eq.2:
            # comad(A_i, A_j) = med[ (A_i - med(A_i)) * (A_j - med(A_j)) ]
            
            centered_product_of_feat_ij = X_centered.iloc[:, i] * X_centered.iloc[:, j] 
            if median_center:
                retval=centered_product_of_feat_ij.median()
            else:
                retval=centered_product_of_feat_ij.mean()
            return (combi, retval)

        def on_result(ret):
            combi, retval = ret
            i,j=combi
            coMAD[i,j] = retval
            coMAD[j,i] = retval
        
        combis = combinations_with_replacement(range(X_centered.shape[1]), 2) # e.g. 2 columns with indexes 0 and 1 -> [(0,0),(0,1),(1,1)]
        t = fast_map_async(calculate_coMAD_ij, list(combis), on_result=on_result, threads_limit=None)
        t.join()
        
        return pd.DataFrame(coMAD, dtype=np.float64)
    
    
    def runPCA(self, A):
        
        A = A.astype(dtype=np.float64)
        
        lamb, v = np.linalg.eig(A)
        pc_idxs = np.argsort(lamb)[::-1]
        evals = lamb[pc_idxs].astype(dtype=np.float64)
        evecs = v.T[pc_idxs].astype(dtype=np.float64)
        if self.sign_flip: evecs = -evecs
        
        evals, evecs = evals[:self.n_components], evecs[:self.n_components, :]
        
        if self.verbose:
            print(f"ComadPCA-runPCA: evals{evals.shape}:\n{evals}")
            print(f"ComadPCA-runPCA: evecs{evecs.shape}:\n{evecs}")     
                
        return evals, evecs
    
    
    def fit(self, X_train):
        
        if type(X_train) != pd.DataFrame: X_train = pd.DataFrame(X_train, dtype=np.float64)   
        X_train = X_train.apply(pd.to_numeric)
        
        self.n_samples_ = X_train.shape[0]
        self.n_features_in_ = X_train.shape[1]
        self.n_features_ = X_train.shape[1]
                
        if self.centering_by=='median':
            self.median_ = np.array(X_train.median(axis=0)) #Eq.3: med(A_i)
            self.centering_offset = self.median_
        else:
            self.mean_ = np.array(X_train.mean(axis=0))
            self.centering_offset = self.mean_ 
        
        A=None
        if self.fast:       
            A = self.coMAD_fast(X_train)
        else:
            A = self.coMAD(X_train)
            
        self.eigenvalues_lambda, self.eigenvectors_v = self.runPCA(A)
        self.explained_variance_, self.components_ = self.eigenvalues_lambda, self.eigenvectors_v
        self.explained_variance_ratio_ = self.eigenvalues_lambda / np.sum(self.eigenvalues_lambda)
        
        return self
        
    
    def transform(self, X_test):
        
        X_trans = None
        
        if self.eigenvalues_lambda is None or self.eigenvectors_v is None:
            print('comadPCA not yet fitted.')
        else:
            X_test_centered = X_test - self.centering_offset
            
            if type(self.n_components)!=int:
                self.n_components = min(self.n_components, 1.0)
                self.n_components = int(math.ceil(self.n_components * self.eigenvectors_v.T.shape[1]))
                print(f"CHANGE: comadPCA on {self.n_components} of {self.eigenvectors_v.T.shape[1]} components")
            
            X_trans = np.dot(self.eigenvectors_v[:self.n_components, :], X_test_centered.T).T
            
        return X_trans    
    
    
    def fit_transform(self, X):
        self = self.fit(X)
        X_trans = self.transform(X)
        return X_trans
    
    
    def inverse_transform(self, X_trans):
                
        if self.eigenvalues_lambda is None or self.eigenvectors_v is None:
            print('comadPCA not yet fitted.')
        else:
            X_orig = np.dot(X_trans, self.eigenvectors_v[:self.n_components, :]) + self.centering_offset
            
        return X_orig
    
    