def logit(D, X, s_wgt=1, silent=False):

    """
    This function computes the ML estimate of the logit binary choice model:
    Pr(D=1|X=x)= exp(x'delta)/[1+exp(x'delta)]. It is not meant to be a full-service
    logit estimator. It is called by the AST att() estimator in order to construct
    propensity score estimates. This reduces the number of dependencies in the
    ipt package.

    INPUTS
    ------
    D      : N x 1 vector of binary outcomes
    X      : X is a N x K matrix of covariates (without a constant)
    s_wgt  : N x 1 vector of known sampling weights (assumed to have mean
             one, optional)
    silent : when silent = True optimization output is suppressed and
             optimization is by Fisher-scoring with lower tolerances.
             Otherwise optimization output is displayed with tighter convergence
             criteria imposed.         

    OUTPUTS
    -------
    gamma_ml         : ML estimates of logit coefficients 
    vcov_delta_ml    : large sample covariance of estimates (= inverse Infomation)

    Functions called : logit_logl, logit_score, logit_hess
    """
    
    def logit_logl(delta, D, X, s_wgt):
        
        """
        Constructs logit log-likelihood.
        """
        
        delta      = np.reshape(delta,(-1,1))    # NOTE: scipy.optimize treats the parameter as a 1 dimensional array
        X_delta    = np.dot(X, delta)            #       The code below is based on treating it as 2 dimensional vector
        exp_Xdelta = np.exp(X_delta)             #       hence the reshaping.
        logl       = -np.sum(s_wgt * (D * X_delta - np.log(1+exp_Xdelta)))
        
        return logl
                        
    def logit_score(delta, D, X, s_wgt):
        
        """
        Constructs dim(delta) x 1 score vector associated with logit log-likelihood.
        NOTE: scipy.optimize requires that the score vector be returned as a 1 dimensional numpy array, NOT
              a 2 dimensional vector, hence the reshape and ravel calls at the start and end of the function.
        """
        
        delta      = np.reshape(delta,(-1,1))   # Reshape one-dimensional score array into two dimensional vector
        X_delta    = np.dot(X, delta)           # Form score
        exp_Xdelta = np.exp(X_delta)
        score      = -np.dot(X.T, (s_wgt * (D - (exp_Xdelta / (1+exp_Xdelta)))))
        score      = np.ravel(score)            # Return score as 1 dimensional numpy array, not a 2 dimensional vector
        
        return score    
    
    def logit_hess(delta, D, X, s_wgt):
        
        """
        Constructs dim(delta) x dim(delta) hessian matrix associated with logit log-likelihood.
        """
        
        delta      = np.reshape(delta,(-1,1))
        X_delta    = np.dot(X, delta)
        exp_Xdelta = np.exp(X_delta)
        hess       = np.dot((s_wgt * (exp_Xdelta / (1+exp_Xdelta)**2) * X).T, X) 
        
        return hess 
    
    def logit_callback(delta):
        print "Value of -logL = "    + "%.6f" % logit_logl(delta, D, X, s_wgt) + \
              ",  2-norm of score = "+ "%.6f" % numpy.linalg.norm(logit_score(delta, D, X, s_wgt))
    
    #--------------------------------------------------------------------#
    #- STEP 1 : Organize data for estimation                            -#
    #--------------------------------------------------------------------#
                    
    (N, K) = np.shape(X)                                   # Number of observations and covariates
    X      = np.concatenate((np.ones((N,1)), X), axis=1)   # Add a constant to the regressor matrix
    K      = K + 1
     
    #--------------------------------------------------------------------#
    #- STEP 2 : Compute CMLE                                            -#
    #--------------------------------------------------------------------#                   
    
    if silent:
        # Suppress optimization output, use Fisher-Scoring, coarser tolerance values and fewer iterations
                       
        # For starting values set constant to calibrate marginal probability of outcome and
        # all slope coefficients to zero     
        delta_sv    = np.zeros((K,))
        p_hat       = np.mean(D);
        delta_sv[0] = np.log(p_hat/(1-p_hat))
        
        # Compute MLE via Fisher-Scoring
        delta_res_ml = sp.optimize.minimize(logit_logl, delta_sv, args=(D, X, s_wgt), method='Newton-CG', \
                                            jac=logit_score, hess=logit_hess, \
                                            options={'xtol': 1e-6, 'maxiter': 1000, 'disp': False})
        
        delta_ml = delta_res_ml.x
        vcov_delta_ml = np.linalg.inv(logit_hess(delta_ml, D, X, s_wgt)) # Use the inverse observed information to 
                                                                         # estimate the large sample variance of delta_ml 
        
    else:
        # Show optimization output, use Fisher-Scoring, finer tolerance values and more iterations
        
        # For starting values set constant to calibrate marginal probability of outcome and
        # all slope coefficients to zero     
        delta_sv    = np.zeros((K,))
        p_hat       = np.mean(D);
        delta_sv[0] = np.log(p_hat/(1-p_hat))
        
        # Derivative check at starting values
        grad_norm = sp.optimize.check_grad(logit_logl, logit_score, delta_sv, D, X, s_wgt, epsilon = 1e-10)
        print 'Fisher-Scoring Derivative check (2-norm): ' + "%.8f" % grad_norm  
        
        # Solve for MLE
        delta_res_ml = sp.optimize.minimize(logit_logl, delta_sv, args=(D, X, s_wgt), method='Newton-CG', \
                                            jac=logit_score, hess=logit_hess, callback = logit_callback, \
                                            options={'xtol': 1e-16, 'maxiter': 10000, 'disp': True}) 
        delta_ml = delta_res_ml.x
        vcov_delta_ml = np.linalg.inv(logit_hess(delta_ml, D, X, s_wgt)) # Use the inverse observed information to estimate 
                                                                         # the large sample variance of delta_ml 
        
    return [delta_ml, vcov_delta_ml, delta_res_ml.success]                   
