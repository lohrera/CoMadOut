import numpy as np
import pandas as pd
import copy
import math

from src.utils.stats_utils import kurtosis, zscore
from src.comad_pca import ComadPCA


class CMOPlusFast():
    
    def __init__(self, n_components=2, title='', center_by='median', variant='k', z_thresh_ens=1., ddof_ens=1, random_state=0, verbose=0, p=1, fast=True):
        super(CMOPlusFast, self)
        
        self._variant=variant # based on r - raw scores, k - kurtosis, e - explained variance ratio(evr), ke - kurtosis+evr, Ens - ensemble values of r,k,e,ke
        self._n_components_param=n_components
        self._n_components=n_components
        self._title=title
        self._p=p
        self._center_by=center_by
        self._verbose=verbose
        self._total_components=None
        self.decision_scores_=None
        self._ddof_ens=ddof_ens
        self._z_thresh_ens=z_thresh_ens
        self._random_state=random_state
        self._fast=fast
        self._cpca=None
        
            
    def fit(self, X):
        
        if type(X) != pd.DataFrame: X = pd.DataFrame(X)            
        df_X = copy.deepcopy(X)    
                
        self._total_components = df_X.shape[1]
        if type(self._n_components_param) == float:
            self._n_components = math.ceil(self._n_components_param*df_X.shape[1])
        
        if self._n_components < 2:
            print(f"CMOPlus: _n_components={self._n_components} below minimum -> updated to _n_components=2")
            self._n_components = 2
                
        cpca = ComadPCA(n_components=self._n_components, center_by=self._center_by, sign_flip=False, random_state=self._random_state, verbose=self._verbose, fast=self._fast).fit(df_X)
        eigenvalues = cpca.eigenvalues_lambda
        eigenvectors = cpca.eigenvectors_v
                
        self._cpca = cpca
        self._X_prj = cpca.transform(df_X)

        self._X_prj, self.decision_scores_ = self._outlier_detection(self._X_prj, self._n_components, eigenvectors, eigenvalues, self._verbose)
        
        return self._X_prj, self.decision_scores_
    
    def decision_function(self, df_X):
        
        if self._cpca is None:
            print(f"model not yet fitted.")
            return None
        
        return self.decision_scores_
    
    def predict(self, df_X):
                
        return self.decision_function(df_X)
            
    
    def _outlier_detection(self, X_prj_z, n_components, eigenvectors, eigenvalues, verbose=0):
        
        eps=1e-6
        pc_outlier_scores = np.nan_to_num(np.array(X_prj_z), nan=eps, posinf=1, neginf=-1)
        kurt_scale = np.nan_to_num(kurtosis(pc_outlier_scores), nan=eps, posinf=1, neginf=-1)
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        
        self.accum_pc_outlier_scores_rawsc = pd.DataFrame(np.linalg.norm(pc_outlier_scores, self._p, axis=1))
        self.accum_pc_outlier_scores_kurts = pd.DataFrame(np.linalg.norm(pc_outlier_scores * np.array([kurt_scale]), self._p, axis=1))
        self.accum_pc_outlier_scores_evrsc = pd.DataFrame(np.linalg.norm(np.nan_to_num(pc_outlier_scores / np.array([explained_variance_ratio]), nan=eps, posinf=1, neginf=-1), self._p, axis=1))
        self.accum_pc_outlier_scores_kurts_evr = pd.DataFrame(np.linalg.norm(np.nan_to_num(pc_outlier_scores * np.array([kurt_scale]) / np.array([explained_variance_ratio]), nan=eps, posinf=1, neginf=-1), self._p, axis=1))

        ens = np.array([
            zscore(self.accum_pc_outlier_scores_rawsc.values,ddof=self._ddof_ens),
            zscore(self.accum_pc_outlier_scores_kurts.values,ddof=self._ddof_ens),
            zscore(self.accum_pc_outlier_scores_evrsc.values,ddof=self._ddof_ens),
            zscore(self.accum_pc_outlier_scores_kurts_evr.values,ddof=self._ddof_ens)
        ])

        scmed = np.amax(ens, axis=0)   
        if verbose: print(f"{self._title}: scmed - min: {scmed.min()} max: {scmed.max()} mean: {scmed.mean()} var: {scmed.var()} std: {scmed.std()}")
        self.accum_pc_outlier_scores_enmed = np.zeros((scmed.shape))
        self.accum_pc_outlier_scores_enmed[scmed <= -self._z_thresh_ens] = 1.
        self.accum_pc_outlier_scores_enmed[scmed >= self._z_thresh_ens] = 1.
        self.accum_pc_outlier_scores_enmed = pd.DataFrame(self.accum_pc_outlier_scores_enmed)

        if self._variant == 'k': scores = self.accum_pc_outlier_scores_kurts
        elif self._variant == 'e': scores = self.accum_pc_outlier_scores_evrsc
        elif self._variant == 'ke': scores = self.accum_pc_outlier_scores_kurts_evr
        elif self._variant == 'Ens': scores = self.accum_pc_outlier_scores_enmed
        else: # self._variant == 'r': 
            scores = self.accum_pc_outlier_scores_rawsc
        
        return X_prj_z, scores


