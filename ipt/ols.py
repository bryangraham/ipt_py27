# Load libraries dependencies
# Ensure "normal" division
from __future__ import division

# Load library dependencies
import numpy as np
import numpy.linalg

import pandas as pd

# Define ols() function
#-----------------------------------------------------------------------------#
def ols(Y, X, c_id = None, sw = None, silent=False):
    """
    AUTHOR: Bryan S. Graham, UC - Berkeley, bgraham@econ.berkeley.edu  
    DATE: 26 May 2016    
    
    This function returns OLS coefficient estimates associated with the
    linear regression fit of Y on X. It reports either heteroscedastic- or
    cluster-robust standard errors as directed by the user. The program
    also allows for the incorporation of sampling weights. While this
    function provides less features that Statsmodels implementation of
    OLS, it is designed to provide easy access to the handful of features
    needed most frequently for cross-section econometric analysis. The
    dependency on Pandas is introduced to provide a convenient way to
    include variable names, which are incorporated into the estimation
    output.    
    
    
    INPUTS:
    -------
    Y        : N X 1 pandas.Series of dependent variable
    X        : N X K pandas.DataFrame of regressors (should include constant if desired)
    c_id     : N X 1 pandas.Series of unique `cluster' id values (assumed to be integer valued) (optional)
               NOTE: Default is to assume independent observations and report heteroscedastic robust 
                     standard errors
    sw       : N X 1 array like vector of sampling weights variable (optional)
    silent   : if set equal to True, then suppress all outcome (optional)
    
    OUTPUTS:
    --------
    beta_hat : K x 1 vector of linear IV estimates of beta
    vcov_hat : K x K cluster-robust variance-covariance estimate

    FUNCTIONS CALLED : None
    ----------------
    
    """
    n       = len(Y)                     # Number of observations
    K       = X.shape[1]                 # Number of regressors
    
    if sw is None:
        sw = 1 
    else:
        sw = sw.reshape((n,1))           # Turn pandas.Series into N x 1 numpy array (if relevant)
        sw = sw/np.mean(sw)              # Normalize sampling weights to have mean 1
    
    # Extract variable names from pandas data objects
    dep_var = Y.name                     # Get dependent variable names
    ind_var = X.columns                  # Get independent variable names
    
    # Transform pandas objects into appropriately sized numpy arrays
    Y       = Y.reshape((n,1))           # Turn pandas.Series into N x 1 numpy array
    X       = np.asarray(X)              # Turn pandas.DataFrame into N x K numpy array
    
    # Compute beta_hat   
    XX  = (sw * X).T.dot(X)
    XY  = (sw * X).T.dot(Y)
    beta_hat = np.linalg.solve(XX, XY)
    
    # Compute estimate of variance-covariance matrix of the sample moment vector
    psi    = sw * X * (Y - X.dot(beta_hat))  # n x K matrix of moment vectors
    
    if c_id is None: 
        
        # Calculate heteroscedastic robust variance-covariance matrix of psi
        fsc   = n/(n-K)                                     # Finite-sample correction factor
        omega = fsc*np.dot(psi.T, psi)                      # K X K variance-covariance of the summed moments
        
        iXX      = np.linalg.inv(XX)
        vcov_hat = iXX.dot(omega).dot(iXX.T)
        
    else:
        
        # Get number and unique list of clusters
        c_list  = np.unique(c_id)            
        N       = len(c_list)    
        
        # Calculate cluster-robust variance-covariance matrix of psi
        # Sum moment vector within clusters
        sum_psi = np.empty((N,K))
    
        for c in range(0,N):
           
            b_cluster    = np.nonzero((c_id == c_list[c]))[0]                   # Observations in c-th cluster 
            sum_psi[c,:] = np.sum(psi[np.ix_(b_cluster, range(0,K))], axis = 0) # Sum over rows within c-th cluster
            
        # Compute variance-covariance matrix of beta_hat
        fsc   = (n/(n-K))*(N/(N-1))                         # Finite-sample correction factor
        omega = fsc*sum_psi.T.dot(sum_psi)                  # K X K variance-covariance of the summed moments
        
        iXX      = np.linalg.inv(XX)
        vcov_hat = iXX.dot(omega).dot(iXX.T)                
 
    if not silent:
        print ""
        print "-----------------------------------------------------------------------"
        print "-                     OLS ESTIMATION RESULTS                          -"
        print "-----------------------------------------------------------------------"
        print "Dependent variable:        " + dep_var
        print "Number of observations, n: " + "%0.0f" % n
        print ""
        print ""
        print "Independent variable       Coef.    ( Std. Err.) "
        print "-----------------------------------------------------------------------"
        
        c = 0
        for names in ind_var:
            print names.ljust(25) + "%10.6f" % beta_hat[c,0] + \
                             " (" + "%10.6f" % np.sqrt(vcov_hat[c,c]) + ")"
            c += 1
        
        print "-----------------------------------------------------------------------"   
        
        if c_id is None:
            print "NOTE: Heteroscedastic-robust standard errors reported"
        else:
            print "NOTE: Cluster-robust standard errors reported"
            print "      Cluster-variable   = " + c_id.name
            print "      Number of clusters = " + "%0.0f" % N

    return [beta_hat,vcov_hat]