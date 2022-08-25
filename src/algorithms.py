import scipy
import numpy as np
from scipy.stats import chi2
from scipy.special import softmax
from sklearn.covariance import MinCovDet
from sklearn.covariance import EllipticEnvelope
from sklearn.covariance import EmpiricalCovariance


class EllipticEnv():
    
    def __init__(self, random_state=0, support_fraction=None):
        self.decision_scores_=None
        self.random_state=random_state
        self.support_fraction=support_fraction
        
    def fit(self, x):
                
        clf = EllipticEnvelope(random_state=self.random_state, support_fraction=self.support_fraction).fit(x)
        y_score = clf.decision_function(x)
        y_score = softmax(y_score*-1/100)
        self.decision_scores_=y_score
        
    def decision_function(self, x):
        return self.decision_scores_

class MCD():
    
    def __init__(self, random_state=0, support_fraction=None):
        self.decision_scores_=None
        self.random_state=random_state
        self.support_fraction=support_fraction
        
    def fit(self, x):
                    
        cov = MinCovDet(random_state=self.random_state, support_fraction=self.support_fraction).fit(x)
        mcd = cov.covariance_ 
        robust_mean = cov.location_  
        #inv_covmat = scipy.linalg.inv(mcd)   
        #add small noise to avoid singular matrices
        mcd1 = mcd-0.00000001*np.random.rand(mcd.shape[0], mcd.shape[1]) 
        inv_covmat = scipy.linalg.inv(mcd1)

        x_minus_mu = x - robust_mean
        left_term = np.dot(x_minus_mu, inv_covmat)
        mat = np.dot(left_term, x_minus_mu.T)
        rmd = np.sqrt(mat.diagonal())

        #detect outliers
        alpha = 0.001 
        chi2_threshold = np.sqrt(chi2.ppf((1-alpha), df=x.shape[1])) 
        y_pred = [1. if val > chi2_threshold else 0. for val in rmd]
        y_score = np.nan_to_num(rmd, np.median(rmd))
        self.decision_scores_=y_score
        
    def decision_function(self, x):
        return self.decision_scores_


class MLE():
    
    def __init__(self, random_state=0):
        self.decision_scores_=None
        self.random_state=random_state
    
    def fit(self, x):

        ec = EmpiricalCovariance().fit(x)
        mle_cov = ec.covariance_ 
        mle_mean = ec.location_  
        #inv_covmat = scipy.linalg.inv(mle_cov) 
        #add small noise to avoid singular matrices
        mle_cov1 = mle_cov-0.00000001*np.random.rand(mle_cov.shape[0], mle_cov.shape[1]) 
        inv_covmat = scipy.linalg.inv(mle_cov1)

        x_minus_mu = x - mle_mean
        left_term = np.dot(x_minus_mu, inv_covmat)
        mat = np.dot(left_term, x_minus_mu.T)
        md = np.sqrt(mat.diagonal())

        #detect outliers
        alpha = 0.001
        chi2_threshold = np.sqrt(chi2.ppf((1-alpha), df=x.shape[1])) 
        y_pred = [1. if val > chi2_threshold else 0. for val in md]
        y_score = np.nan_to_num(md, np.median(md))
        self.decision_scores_=y_score
        
    def decision_function(self, x):
        return self.decision_scores_
