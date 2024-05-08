import numpy as np
import pandas as pd
import copy
import math
import gc

from fast_map import fast_map_async
#from sklearn.decomposition import PCA
from pyod.models.pca import PCA
from src.utils.stats_utils import kurtosis, zscore
from src.comad_pca import ComadPCA
import tracemalloc
tracemalloc.start()

class CMOPlus():
    
    def __init__(self, n_components=2, title='', center_by='median', centermethod='offset', variant='k', z_thresh_ens=1., ddof_ens=1, random_state=0, verbose=0, p=2, fast=False, opt_evals=False, pca=False, scaler=None, tml=[]):
        super(CMOPlus, self)
        
        self._tm=tml        
        cur,_=tracemalloc.get_traced_memory() #2
        self._tm.append(cur)
        
        self._variant=variant # based on r - raw scores, k - kurtosis, e - explained variance ratio(evr), ke - kurtosis+evr, Ens - ensemble values of r,k,e,ke
        self._n_components_param=n_components
        self._n_components=n_components
        self._title=title
        self._p=p
        self._center_by=center_by
        self._centermethod=centermethod
        self._verbose=verbose
        self._total_components=None
        self.decision_scores_=None
        self._ddof_ens=ddof_ens
        self._z_thresh_ens=z_thresh_ens
        self._random_state=random_state
        self._opt_evals=opt_evals
        self._pca=pca
        self._scaler=scaler
        self._fast=fast
        self._cpca=None
        
        cur,_=tracemalloc.get_traced_memory() #3
        self._tm.append(cur)
            
    def fit(self, X):
        
        # -------------
        cur,_=tracemalloc.get_traced_memory() #4
        self._tm.append(cur)
        
        if type(X) != pd.DataFrame: X = pd.DataFrame(X, dtype=np.float64)   
        X = X.apply(pd.to_numeric)   
        df_X = copy.deepcopy(X)    
        
        # -------------
        cur,_=tracemalloc.get_traced_memory() #5
        self._tm.append(cur)
        
        self._total_components = df_X.shape[1]
        if type(self._n_components_param) == float:
            self._n_components = math.ceil(self._n_components_param*df_X.shape[1])
        
        if self._n_components < 2:
            print(f"CMOPlus: _n_components={self._n_components} below minimum -> updated to _n_components=2")
            self._n_components = 2
        
        
        # -------------
        cur,_=tracemalloc.get_traced_memory() #6
        self._tm.append(cur)
        
        if not self._pca: #comadpca    
            
            from sklearn.preprocessing import RobustScaler, StandardScaler
            if self._scaler == 'Robust':    
                df_X = pd.DataFrame(RobustScaler().fit_transform(df_X))
            elif self._scaler == 'Standard': 
                df_X = pd.DataFrame(StandardScaler().fit_transform(df_X))
                
            
            # -------------
            cur,_=tracemalloc.get_traced_memory() #7
            self._tm.append(cur)

                
            cpca = ComadPCA(n_components=self._n_components, center_by=self._center_by, centermethod=self._centermethod, sign_flip=False, random_state=self._random_state, verbose=self._verbose, fast=self._fast).fit(df_X)
            
            # -------------
            cur,_=tracemalloc.get_traced_memory() #8
            self._tm.append(cur)
            
            self._X_prj = cpca.transform(df_X)
            
            eigenvectors = cpca.eigenvectors_v
            eigenvalues = cpca.eigenvalues_lambda
        else:
            
            # -------------
            cur,_=tracemalloc.get_traced_memory() #7
            self._tm.append(cur)
            
            cpca = PCA(n_components=self._n_components, random_state=self._random_state).fit(df_X)
            
            # -------------
            cur,_=tracemalloc.get_traced_memory() #8
            self._tm.append(cur)
            
            self._X_prj = cpca.detector_.transform(df_X)
            
            eigenvectors = cpca.components_
            eigenvalues = cpca.explained_variance_
        
        # -------------
        cur,_=tracemalloc.get_traced_memory() #9
        self._tm.append(cur)

        if self._opt_evals:
            if np.any(cpca.explained_variance_ < 0.): 
                #print(f"!!!! matrix M not positive-semidefinite with eigenvalues {cpca.explained_variance_}") # <- circumstance well known for comad
                eigenvalues = np.abs(cpca.explained_variance_)
            
        if not self._pca: #comad
            cpca.explained_variance_ratio_ = eigenvalues / np.sum(eigenvalues)
            cpca.explained_variance_ = eigenvalues
            cpca.eigenvalues_lambda = eigenvalues
            
        self._cpca = cpca
        df_X=None
        gc.collect()
        
        # -------------
        cur,_=tracemalloc.get_traced_memory() #10
        self._tm.append(cur)
        
        if self._fast:
            self._X_prj, self.decision_scores_ = self._outlier_detection_fast(self._X_prj, self._n_components, eigenvectors, eigenvalues, self._verbose)
        else:
            self._X_prj, self.decision_scores_ = self._outlier_detection(self._X_prj, self._n_components, eigenvectors, eigenvalues, self._verbose)
        
        # -------------
        cur,_=tracemalloc.get_traced_memory() #11
        self._tm.append(cur)
        
        return self._X_prj, self.decision_scores_
    
    def decision_function(self, df_X):
        
        if self._cpca is None:
            print(f"model not yet fitted.")
            return None
        
        return self.decision_scores_
    
    def predict(self, df_X):
                
        return self.decision_function(df_X)
        
    def _get_orthogonal_projection(self, points, line_vector):
        u = line_vector            
        eps = 1e-6 # for numerical stability
        result = [((x @ u) / max((u @ u),eps)) * u for x in points]
        return pd.DataFrame(result)
    
    def _get_subspace_axis_vector_after_transform(self, eigenvectors, pc_idx):
        return np.identity(eigenvectors.shape[0])[pc_idx,:]
    
    def _get_scores(self, pc_outlier_scores, eigenvalues, variant='k', verbose=0):
        
        eps=1e-6
        pc_outlier_scores = np.nan_to_num(np.array(pc_outlier_scores).T, nan=eps, posinf=1, neginf=-1) # pc_outlier_scores = Lp-distances from origin to prj
        kurt_scale = np.nan_to_num(kurtosis(pc_outlier_scores), nan=eps, posinf=1, neginf=-1)   
        ev_lambda = eigenvalues / np.sum(eigenvalues)
        #explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        
        pc_outlier_scores_rawsc = pc_outlier_scores
        pc_outlier_scores_kurts = pc_outlier_scores * kurt_scale 
        #pc_outlier_scores_evrsc = np.nan_to_num(pc_outlier_scores / explained_variance_ratio, nan=eps, posinf=1, neginf=-1)
        #pc_outlier_scores_kurts_evr = np.nan_to_num(pc_outlier_scores * kurt_scale / explained_variance_ratio, nan=eps, posinf=1, neginf=-1)
        pc_outlier_scores_evrsc = np.nan_to_num(pc_outlier_scores / ev_lambda, nan=eps, posinf=1, neginf=-1)
        pc_outlier_scores_kurts_evr = np.nan_to_num(pc_outlier_scores * kurt_scale / ev_lambda, nan=eps, posinf=1, neginf=-1)
                        
        self.accum_pc_outlier_scores_rawsc = pc_outlier_scores_rawsc.sum(axis=1)
        self.accum_pc_outlier_scores_kurts = pc_outlier_scores_kurts.sum(axis=1)
        self.accum_pc_outlier_scores_evrsc = pc_outlier_scores_evrsc.sum(axis=1)
        self.accum_pc_outlier_scores_kurts_evr = pc_outlier_scores_kurts_evr.sum(axis=1)
        
        ens = np.array([
            zscore(self.accum_pc_outlier_scores_rawsc,ddof=self._ddof_ens),
            zscore(self.accum_pc_outlier_scores_kurts,ddof=self._ddof_ens),
            zscore(self.accum_pc_outlier_scores_evrsc,ddof=self._ddof_ens),
            zscore(self.accum_pc_outlier_scores_kurts_evr,ddof=self._ddof_ens)
        ])
        
        scmed = np.amax(ens, axis=0)   
        if self._verbose: print(f"{self._title}: scmed - min: {scmed.min()} max: {scmed.max()} mean: {scmed.mean()} var: {scmed.var()} std: {scmed.std()}")
        self.accum_pc_outlier_scores_enmed = np.zeros((scmed.shape))
        self.accum_pc_outlier_scores_enmed[scmed <= -self._z_thresh_ens] = 1.
        self.accum_pc_outlier_scores_enmed[scmed >= self._z_thresh_ens] = 1.
                
        if variant == 'k': return self.accum_pc_outlier_scores_kurts
        elif variant == 'e': return self.accum_pc_outlier_scores_evrsc
        elif variant == 'ke': return self.accum_pc_outlier_scores_kurts_evr
        elif variant == 'Ens': return self.accum_pc_outlier_scores_enmed
        else: # variant == 'r': 
            return self.accum_pc_outlier_scores_rawsc
    
    def _outlier_detection_fast(self, X_prj_z, n_components, eigenvectors, eigenvalues, verbose=0):
        all_scores_per_dim=[0.]*n_components
        
        def calculate_pc_scores(pc_idx):
            ev_subspace_axis = self._get_subspace_axis_vector_after_transform(eigenvectors, pc_idx)
            z_orth_prj = self._get_orthogonal_projection(X_prj_z, ev_subspace_axis)
            retval = np.linalg.norm(z_orth_prj, self._p, axis=1)
            return (pc_idx, retval)

        def on_pc_score_result(ret):
            pc_idx, retval = ret
            all_scores_per_dim[pc_idx] = retval
        
        th = fast_map_async(calculate_pc_scores, list(range(0, n_components)), on_result=on_pc_score_result, threads_limit=None)
        th.join()
        th=None
        gc.collect()
        
        scores = self._get_scores(all_scores_per_dim, eigenvalues, variant=self._variant, verbose=verbose)
        
        return X_prj_z, scores
    
    def _outlier_detection(self, X_prj_z, n_components, eigenvectors, eigenvalues, verbose=0):
        all_scores_per_dim,scores=[],[]
        
        for pc_idx in range(0, n_components):  
            ev_subspace_axis = self._get_subspace_axis_vector_after_transform(eigenvectors, pc_idx)
            z_orth_prj = self._get_orthogonal_projection(X_prj_z, ev_subspace_axis) 
            all_scores_per_dim.append( np.linalg.norm(z_orth_prj, self._p, axis=1) )
                        
        scores = self._get_scores(all_scores_per_dim, eigenvalues, variant=self._variant, verbose=verbose)
        
        return X_prj_z, scores


