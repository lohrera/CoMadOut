import numpy as np
import pandas as pd
import copy
from fast_map import fast_map_async
from scipy.special import softmax

from matplotlib import pyplot as plt
#from sklearn.decomposition import PCA
from pyod.models.pca import PCA

from src.comad_pca import ComadPCA


class ComadOut():
    
    def __init__(self, n_components=2, title='', noise_margin=True, softmax_scoring=True, center_by='median', centermethod='offset', pca=False, sample_pc_outl_score_selection='mean', random_state=0, verbose=0, fast=False, opt_evals=False, scaler=None):
        super(ComadOut, self)
        
        self._n_components_param=n_components
        self._n_components=n_components
        self._title=title
        self._noise_margin=noise_margin
        self._softmax_scoring=softmax_scoring
        self._sample_pc_outl_score_selection=sample_pc_outl_score_selection
        self._center_by=center_by
        self._centermethod=centermethod
        self._verbose=verbose
        self._pca=pca
        self._outlier_threshold=None
        self._total_components=None
        self.decision_scores_ = None
        self._random_state=random_state
        self._opt_evals=opt_evals
        self._scaler=scaler
        self._fast=fast
        self._cpca=None
            
    def fit(self, X):
        labels,scores=[],[]
        
        if type(X) != pd.DataFrame: X = pd.DataFrame(X, dtype=np.float64)      
        X = X.apply(pd.to_numeric)   
        df_X = copy.deepcopy(X)    
        
        self._total_components = df_X.shape[1]
        if type(self._n_components_param) == float:
            self._n_components = int(self._n_components_param*df_X.shape[1])
            
        if self._n_components < 2:
            print(f"ComadOut: _n_components={self._n_components} below minimum -> updated to _n_components=2")
            self._n_components = 2
                    
        cpca, eigenvalues, eigenvectors = None, None, None
        if not self._pca: #comadpca      
            
            from sklearn.preprocessing import RobustScaler, StandardScaler
            if self._scaler == 'Robust':    
                df_X = pd.DataFrame(RobustScaler().fit_transform(df_X))
            elif self._scaler == 'Standard': 
                df_X = pd.DataFrame(StandardScaler().fit_transform(df_X))
            
            cpca = ComadPCA(n_components=self._n_components, center_by=self._center_by, centermethod=self._centermethod, sign_flip=False, random_state=self._random_state, fast=self._fast).fit(df_X)
            self._X_prj = cpca.transform(df_X)
            
            eigenvectors = cpca.eigenvectors_v
            eigenvalues = cpca.eigenvalues_lambda
        else:
            cpca = PCA(n_components=self._n_components, random_state=self._random_state).fit(df_X)
            self._X_prj = cpca.detector_.transform(df_X)
            
            eigenvectors = cpca.components_
            eigenvalues = cpca.explained_variance_
           
        if self._opt_evals:
            if np.any(cpca.explained_variance_ < 0.): 
                #print(f"!!!! matrix M not positive-semidefinite with eigenvalues {cpca.explained_variance_}") # <- circumstance well known for comad
                eigenvalues = np.abs(cpca.explained_variance_) 
            
        if not self._pca: #comad
            cpca.explained_variance_ratio_ = eigenvalues / np.sum(eigenvalues)
            cpca.explained_variance_ = eigenvalues
            cpca.eigenvalues_lambda = eigenvalues
        
        self._cpca = cpca

        self._X_prj, all_pred_labels, all_pred_scores = self._outlier_detection(
            self._X_prj, self._n_components, eigenvectors, eigenvalues, self._title, self._noise_margin, self._sample_pc_outl_score_selection, self._softmax_scoring, self._verbose)
         
        self.decision_scores_ = all_pred_scores
        
        return self._X_prj, all_pred_labels, all_pred_scores
    
    def decision_function(self, df_X):
        
        if self._cpca is None:
            print(f"model not yet fitted.")
            return None
        
        return self.decision_scores_
    
    def predict(self, df_X):
                
        return self.decision_function(df_X)
        
        
    def _pc_outliers(self, sample_distances, outlier_threshold, pc_idx, verbose=0):
        result = []
        if verbose!=0: print(f"outlier_threshold: {outlier_threshold}")
        for p_dist in sample_distances:
            outlier_detected = (p_dist >= outlier_threshold)
            if verbose!=0: print(f"pc_idx {pc_idx} - p_dist {p_dist} -> outl: {outlier_detected}")
            result.append(outlier_detected)
        return result

    def _get_orthogonal_projection(self, points, line_vector):
        u = line_vector            
        eps = 1e-6 # for numerical stability
        result = [((x @ u) / max((u @ u),eps)) * u for x in points]
        return pd.DataFrame(result)
    
    def _get_subspace_axis_vector_after_transform(self, eigenvectors, pc_idx):
        return np.identity(eigenvectors.shape[0])[pc_idx,:]
    
    def _or(self, mat, sample_idx):
        retval = None
        num_pcs = len(mat)    
        for pc_idx in range(0,num_pcs):
            if retval is None: 
                retval = mat[pc_idx][sample_idx]
            else:
                retval = (retval or mat[pc_idx][sample_idx])
        return retval
    
    def _min(self, mat, margins, sample_pc_outl_score_selection, sample_idx):

        num_pcs = len(mat)
        lst_sample_pc_outl_scores=[]
        for pc_idx in range(0,num_pcs):
            lst_sample_pc_outl_scores.append(max(0., mat[pc_idx][sample_idx] - margins[pc_idx]['ot']))

        sample_pc_outl_scores = np.array(lst_sample_pc_outl_scores)
        sample_pc_outl_scores.sort()
        sample_min_outl_score=0.
        
        if len(np.where((sample_pc_outl_scores > 0)==True)[0]) != 0:
            
            if sample_pc_outl_score_selection == 'min':
            
                min_outl_score_idx = np.where(sample_pc_outl_scores > 0)[0][0]     
                sample_min_outl_score = sample_pc_outl_scores[min_outl_score_idx]
                
            elif sample_pc_outl_score_selection == 'max':
                
                min_outl_score_idx = np.where(sample_pc_outl_scores > 0)[0][-1]    
                sample_min_outl_score = sample_pc_outl_scores[min_outl_score_idx]
            
            elif sample_pc_outl_score_selection == 'median':            
                
                outl_score_idx = np.where(sample_pc_outl_scores > 0)[0]
                sample_min_outl_score = np.median(sample_pc_outl_scores[outl_score_idx])  
            
            elif sample_pc_outl_score_selection == 'mean':  
            
                outl_score_idx = np.where(sample_pc_outl_scores > 0)[0]
                sample_min_outl_score = np.mean(sample_pc_outl_scores[outl_score_idx])  
                            
        return sample_min_outl_score

    def _get_result(self, all_labels, all_scores, margins, sample_pc_outl_score_selection='mean', softmax_scoring=True):
        
        num_samples = len(all_labels[0])
        labels = [self._or(all_labels, i) for i in range(num_samples)]
        scores = [self._min(all_scores, margins, sample_pc_outl_score_selection, i) for i in range(num_samples)]  
                
        if softmax_scoring: # for comad AD only the explained_variance_ correlation magnitude (defensively scaled by sqrt) is relevant
            scores = softmax(np.array(all_scores).T/np.sqrt(np.abs(self._cpca.explained_variance_)), axis=1).max(axis=1)
            
        return labels, scores
    
    def _outlier_detection(self, X_prj, n_components, eigenvectors, eigenvalues, title, noise_margin=True, sample_pc_outl_score_selection='mean', softmax_scoring=True, verbose=False):
        all_scores_per_dim,scores=[],[]
        all_labels_per_dim,labels=[],[]
        eps=1e-6
        margins={}
        
        for pc_idx in range(0, n_components):

            ev_subspace_axis = self._get_subspace_axis_vector_after_transform(eigenvectors, pc_idx)
            prj = self._get_orthogonal_projection(X_prj, ev_subspace_axis)
            distances = np.sqrt((prj**2).sum(axis=1)).astype(dtype=np.float64)
            if verbose: print(distances)
            
            comad_threshold = eigenvalues[pc_idx] #* evec_len # evec (unit vector) length 1
            comad_threshold = max(comad_threshold, eps) # avoiding zero-thresholds
            
            if not noise_margin:
                noise_margin = 0.
                self._outlier_threshold = comad_threshold
            else:
                distances_median = np.median(distances)
                noise_margin = distances_median
                self._outlier_threshold = comad_threshold + noise_margin
                #print(f"{title} - pc[{pc_idx}] outlier_t={self._outlier_threshold} comad_t={comad_threshold} nm={round(noise_margin,6)} dmd={distances_median}")

            margins[pc_idx] = {'ct': comad_threshold, 'nm': noise_margin, 'ot': self._outlier_threshold}
            all_labels_per_dim.append(self._pc_outliers(distances, self._outlier_threshold, pc_idx, verbose=verbose))
            all_scores_per_dim.append(distances)
                        
        self._margins = margins
        
        # get predictions and scores 
        labels, scores = self._get_result(all_labels_per_dim, all_scores_per_dim, margins, sample_pc_outl_score_selection, softmax_scoring)
        
        return X_prj, labels, scores
    

    def draw_dataset(self, df_data, y_true, title=None):
        data4ok = df_data.iloc[[False if idx in list(np.where(y_true)[0]) else True for idx in df_data.index.tolist()], :]
        data4nok = df_data.iloc[[True if idx in list(np.where(y_true)[0]) else False for idx in df_data.index.tolist()], :]

        f = data4ok.plot.scatter(x=0,y=1,c='g',marker='o').get_figure()
        ax = f.get_axes()[0]
        ax.scatter(x=data4nok.iloc[:,0],y=data4nok.iloc[:,1],c='r',s=80,marker=10)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        if title is not None: ax.set_title(title)
