def att(D, Y, r_W, t_W, study_tilt=True, rlgrz = 1, NG=None, s_wgt=1, silent=False, r_W_names=[], t_W_names=[]):
    
    """
    
    AUTHOR: Bryan S. Graham, UC - Berkeley, bgraham@econ.berkeley.edu    
    
    This function estimates the average treatment effect on the treated (ATT)
    using the "auxiliary-to-study tilting" (AST) method described by 
    Graham, Pinto and Egel (2016, Journal of Business and Economic Statistics). 
    The notation below mirrors that in the paper where possible. The Supplemental 
    Web Appendix of the paper describes the estimation algorithm implemented here 
    in detail. A copy of the paper and all supplemental appendices can be found 
    online at http://bryangraham.github.io/econometrics/

    INPUTS
    ------
    D         : N x 1 vector with ith element equal to 1 if ith unit in the merged
                sample is from the study population and zero if from the auxiliary
                population (i.e., D is the "treatment" indicator)
    Y         : N x 1  vector of observed outcomes                  
    r_W       : r(W), N x 1+L matrix of functions of always observed covariates
                (constant included) -- these are the propensity score functions
    t_W       : t(W), N x 1+M matrix of functions of always observed covariates
                (constant included) -- these are the balancing functions     
    study_tilt: If True compute the study sample tilt. This should be set to False 
                if all the elements in t(W) are also contrained in h(W). In that 
                case the study_tilt coincides with its empirical measure.This 
                measure is returned in the pi_s vector when  study_tilt = False.
    rlgrz     : Regularization parameter. Should positive and less than or equal 
                to one. Smaller values correspond to less regularizations, but 
                may cause underflow problems when overlap is poor. The default 
                value will be adequate for most applications.
    NG        : G x 1 vector with gth row equal to the number of units in the gth 
                cluster (optional)
                NOTE: Data are assumed to be pre-sorted by groups.
    s_wgt     : N x 1 vector of known sampling weights (optional)
    silent    : if silent = True display less optimization information and use
                lower tolerance levels (optional)
    r_W_names : List of strings of the same dimension as r_W with variable names 
                (optional, but need for some output to be printed)
    t_W_names : List of strings of the same dimension as t_W with variable names 
                (optional, but need for some output to be printed)     

    OUTPUTS
    -------
    gamma_ast         : AST estimate of gamma (the ATT)
    vcov_gamma_ast    : estimated large sample variance of gamma
    study_test        : ChiSq test statistic of H0 : lambda_s = 0; list with 
                        [statistic, dof, p-val]
                        NOTE: returns [None, None, None] if study_tilt = False
    auxiliar_test     : ChiSq test statistic of H0 : lambda_a = 0; list with 
                        [statistic, dof, p-val]
    pi_eff            : Semiparametrically efficient estimate of F_s(W) 
    pi_s              : Study sample tilt
    pi_a              : Auxiliary sample tilt 
    exitflag          : 1 = success, 2 = can't compute MLE of p-score, 3 = can't compute study/treated tilt,
                        4 = can't compute auxiliary/control tilt

    Functions called  : logit()                             (...logit_logl(), logit_score(), logit_hess()...)
                        ast_crit(), ast_foc(), ast_soc()    (...ast_phi()...)
    """
    
    def ast_phi(lmbda, t_W, p_W_index, NQ):
        
        """
        This function evaluates the regularized phi(v) function for 
        the logit propensity score case (as well as its first and 
        second derivatives) as described in the Supplemental
        Web Appendix of Graham, Pinto and Egel (2016, JBES).

        INPUTS
        ------
        lmbda         : vector of tilting parameters
        t_W           : vector of balancing moments
        p_W_index     : index of estimated logit propensity score
        NQ            : sample size times the marginal probability of missingness
        
        OUTPUTS
        -------
        phi, phi1, phi2 : N x 1 vectors with elements phi(p_W_index + lmbda't_W)
                          and its first and second derivatives w.r.t to 
                          v = p_W_index + lmbda't_W
        """
        
        # Coefficients on quadratic extrapolation of phi(v) used to regularize 
        # the problem
        c = -(NQ - 1)
        b = NQ + (NQ - 1)*np.log(1/(NQ - 1))
        a = -(NQ - 1)*(1 + np.log(1/(NQ - 1)) + 0.5*(np.log(1/(NQ - 1)))**2) 
        
        v_star = np.log(1/(NQ - 1)) 

        # Evaluation of phi(v) and derivatives
        v          =  p_W_index + np.dot(t_W, lmbda)
        phi        =  (v>v_star) * (v - np.exp(-v))   + (v<=v_star) * (a + b*v + 0.5*c*v**2)
        phi1       =  (v>v_star) * (1 + np.exp(-v))   + (v<=v_star) * (b + c*v)
        phi2       =  (v>v_star) * (  - np.exp(-v))   + (v<=v_star) * c
          
        return [phi, phi1, phi2]

    def ast_crit(lmbda, D, p_W, p_W_index, t_W, NQ, s_wgt=1):
        
        """
        This function constructs the AST criterion function
        as described in Graham, Pinto and Egel (2016, JBES).
        
        INPUTS
        ------
        lmbda         : vector of tilting parameters
        D             : N x 1 treatment indicator vector
        p_W           : N x 1 MLEs of the propensity score
        p_W_index     : index of estimated logit propensity score
        t_W           : vector of balancing moments
        NQ            : sample size times the marginal probability of missingness
        s_wgt         : N x 1 vector of known sampling weights (optional)
        
        OUTPUTS
        -------
        crit          : AST criterion function at passed parameter values
        
        Functions called : ast_phi()
        """
        
        lmbda   = np.reshape(lmbda,(-1,1))                                     # make lmda 2-dimensional object
        [phi, phi1, phi2] = ast_phi(lmbda, t_W, p_W_index, NQ)                 # compute phi and 1st/2nd derivatives
        crit    = -np.sum(s_wgt * (D * phi - np.dot(t_W, lmbda)) * (p_W / NQ)) # AST criterion (scalar)
        
        return crit
    
    def ast_foc(lmbda, D, p_W, p_W_index, t_W, NQ, s_wgt):
        
        """
        Returns first derivative vector of AST criterion function with respect
        to lmbda. See the header for ast_crit() for description of parameters.
        """
        
        lmbda   = np.reshape(lmbda,(-1,1))                              # make lmda 2-dimensional object
        [phi, phi1, phi2] = ast_phi(lmbda, t_W, p_W_index, NQ)          # compute phi and 1st/2nd derivatives
        foc     = -np.dot(t_W.T, (s_wgt * (D * phi1 - 1) * (p_W / NQ))) # AST gradient (1+M x 1 vector)
        foc     = np.ravel(foc)                                         # make foc 1-dimensional numpy array
        
        return foc
    
    def ast_soc(lmbda, D, p_W, p_W_index, t_W, NQ, s_wgt):
        
        """
        Returns hessian matrix of AST criterion function with respect
        to lmbda. See the header for ast_crit() for description of parameters.
        """
        
        lmbda   = np.reshape(lmbda,(-1,1))                                # make lmda 2-dimensional object
        [phi, phi1, phi2] = ast_phi(lmbda, t_W, p_W_index, NQ)            # compute phi and 1st/2nd derivatives
        soc     = -np.dot(((s_wgt * D * phi2 * (p_W / NQ)) * t_W).T, t_W) # AST hessian (note use of numpy broadcasting rules)
                                                                          # (1 + M) x (1 + M) "matrix" (numpy array) 
        return [soc]
    
    def ast_study_callback(lmbda):
        print "Value of ast_crit = "   + "%.6f" % ast_crit(lmbda, D, p_W, p_W_index, t_W, NQ, s_wgt) + \
              ",  2-norm of ast_foc = "+ "%.6f" % numpy.linalg.norm(ast_foc(lmbda, D, p_W, p_W_index, t_W, NQ, s_wgt))
    
    def ast_auxiliary_callback(lmbda):
        print "Value of ast_crit = "   + "%.6f" % ast_crit(lmbda, 1-D, p_W, -p_W_index, t_W, NQ, s_wgt) + \
              ",  2-norm of ast_foc = "+ "%.6f" % numpy.linalg.norm(ast_foc(lmbda, 1-D, p_W, -p_W_index, t_W, NQ, s_wgt))

    
    # ----------------------------------------------------------------------------------- #
    # - STEP 1 : ORGANIZE DATA                                                          - #
    # ----------------------------------------------------------------------------------- #

    N       = len(D)                  # Number of units in sample  
    Ns      = np.sum(D)               # Number of study units in the sample (treated units) 
    Na      = N-Ns                    # Number of auxiliary units in the sample (control units) 
    M       = np.shape(t_W)[1] - 1    # Dimension of t_W (excluding constant)
    L       = np.shape(r_W)[1] - 1    # Dimension of r_W (excluding constant)
    s_wgt   = s_wgt/np.mean(s_wgt)    # normalize sample weights to have mean one
    DY      = D * Y                   # D*Y, N x 1  vector of observed outcomes for treated/study units
    mDX     = (1-D) * Y               # (1-D)*X, N x 1  vector of observed outcomes for non-treated/auxiliary units 

    # ----------------------------------------------------------------------------------- #
    # - STEP 2 : ESTIMATE PROPENSITY SCORE PARAMETER BY LOGIT ML                        - #
    # ----------------------------------------------------------------------------------- #

    try:
        if not silent:
            print ""
            print "-------------------------------------------------"
            print "- Computing propensity score by MLE             -"
            print "-------------------------------------------------"
        
        [delta_ml, vcov_delta_ml, success]= logit(D, r_W[:,1:], s_wgt, silent)  # CMLE of p-score coefficients
        delta_ml                 = np.reshape(delta_ml,(-1,1))         # Put delta_ml into 2-dimensional form
        p_W_index                = np.dot(r_W, delta_ml)               # Fitted p-score index 
        p_W                      = (1 + np.exp(-p_W_index))**-1        # Fitted p-score probabilities
        NQ                       = np.sum(s_wgt * p_W)                 # Sum of fitted p-scores
        pi_eff                   = (s_wgt * p_W) / NQ                  # Efficient estimate of F(W)
    
    except:
        print "FATAL ERROR: exitflag = 2, unable to compute propensity score by maximum likelihood."
        
        # Set all returnables to "None" and then exit function
        gamma_ast      = None
        vcov_gamma_ast = None
        study_test     = [None, None, None]
        auxiliary_test = [None, None, None]
        pi_eff         = None
        pi_s           = None
        pi_a           = None
        exitflag       = 2
        
        return [gamma_ast, vcov_gamma_ast, study_test, auxiliary_test, pi_eff, pi_s, pi_a, exitflag]
    
    # ----------------------------------------------------------------------------------- #
    # - STEP 3 : SOLVE FOR AST TILTING PARAMETERS                                       - #
    # ----------------------------------------------------------------------------------- 

    # Set optimization parameters
    if silent:
        # Use Newton-CG solver with vector of zeros as starting values, 
        # low tolerance levels, and smaller number of allowed iterations.
        # Hide iteration output.
        options_set = {'xtol': 1e-6, 'maxiter': 1000, 'disp': False}
    else:
        # Use Newton-CG solver with vector of zeros as starting values, 
        # high tolerance levels, and larger number of allowed iterations.
        # Show iteration output.
        options_set = {'xtol': 1e-16, 'maxiter': 10000, 'disp': True}
        
    lambda_sv = np.zeros(1+M) # use vector of zeros as starting values
  
    #------------------------------#
    #- STUDY TILT                 -#
    #------------------------------#

    # NOTE: Only compute the study_tilt if directed to do so (this is the default). The study_tilt
    #       doesn't need to be computed if all the elements of t(W) are also included in h(W). It
    #       is the users responsibility to check this condition.
    
    if study_tilt:
        # -------------------------------------------------- #
        # - CASE 1: Non-trivial study sample tilt required - #
        # -------------------------------------------------- #
            
        # Compute lamba_s_hat (study or treated sample tilting parameters)
        try:
            if not silent:
                print ""
                print "-------------------------------------------------"
                print "- Computing study/treated sample tilt           -"
                print "-------------------------------------------------"
            
                # Derivative check at starting values
                grad_norm = sp.optimize.check_grad(ast_crit, ast_foc, lambda_sv, D, p_W, p_W_index, t_W, NQ/rlgrz, s_wgt, \
                                                   epsilon = 1e-10)
                print 'Study sample tilt derivative check (2-norm): ' + "%.8f" % grad_norm
                
                # Solve for tilting parameters
                lambda_s_res = sp.optimize.minimize(ast_crit, lambda_sv, args=(D, p_W, p_W_index, t_W, NQ/rlgrz, s_wgt), \
                                                    method='Newton-CG', jac=ast_foc, hess=ast_soc, \
                                                    callback = ast_study_callback, options=options_set)
            else:
                # Solve for tilting parameters
                lambda_s_res = sp.optimize.minimize(ast_crit, lambda_sv, args=(D, p_W, p_W_index, t_W, NQ/rlgrz, s_wgt), \
                                                    method='Newton-CG', jac=ast_foc, hess=ast_soc, \
                                                    options=options_set)
        except:
            print "FATAL ERROR: exitflag = 3, Unable to compute the study/treated vector of tilting parameters."
        
            # Set all returnables to "None" and then exit function
            gamma_ast      = None
            vcov_gamma_ast = None
            study_test     = [None, None, None]
            auxiliary_test = [None, None, None]
            pi_eff         = None
            pi_s           = None
            pi_a           = None
            exitflag       = 3
        
            return [gamma_ast, vcov_gamma_ast, study_test, auxiliary_test, pi_eff, pi_s, pi_a, exitflag]
        
        # Collect study tilt estimation results needed below
        lambda_s_hat = np.reshape(lambda_s_res.x,(-1,1))                           # study/treated sample tilting 
                                                                                   # parameter estimates
        p_W_s = (1+np.exp(-np.dot(r_W, delta_ml) - np.dot(t_W, lambda_s_hat)))**-1 # study/treated sample tilted p-score
        pi_s  = D * pi_eff / p_W_s                                                 # study/treated sample tilt 
    
    else:
        # ------------------------------------------ #
        # - CASE 2: Study sample tilt NOT required - #
        # ------------------------------------------ #
        
        if not silent:
            print ""
            print "----------------------------------------------------------------------"
            print "- Tilt of study sample not required by user (study_tilt = False).    -"
            print "- Validity of this requires all elements of t(W) to be elements of   -"
            print "- h(W) as well. User is advised to verify this condition.            -"
            print "----------------------------------------------------------------------"
            print ""
        
        # Collect study tilt objects needed below
        lambda_s_hat = np.reshape(lambda_sv ,(-1,1)) # study/treated sample tilting parameters set equal to zero
        p_W_s = p_W                                  # study/treated sample tilted p-score equals actual score
        pi_s  = D * pi_eff / p_W_s                   # set pi_s to empirical measure of study sub-sample 
                                                     # (w/o sampling weights this puts mass 1/Ns on each study unit)
   
    #------------------------------#
    #- AUXILIARY TILT             -#
    #------------------------------#
    
    # Compute lamba_a_hat (auxiliary or control sample tilting parameters)
    try:
        if not silent:
            print ""
            print "-------------------------------------------------"
            print "- Computing auxiliary/control sample tilt       -"
            print "-------------------------------------------------"
            
            # Derivative check at starting values
            grad_norm = sp.optimize.check_grad(ast_crit, ast_foc, lambda_sv, 1-D, p_W, -p_W_index, t_W, NQ/rlgrz, s_wgt, \
                                               epsilon = 1e-10)
            print 'Auxiliary sample tilt derivative check (2-norm): ' + "%.8f" % grad_norm 
            
            # Solve for tilting parameters
            lambda_a_res = sp.optimize.minimize(ast_crit, lambda_sv, args=(1-D, p_W, -p_W_index, t_W, NQ/rlgrz, s_wgt), \
                                                method='Newton-CG', jac=ast_foc, hess=ast_soc, \
                                                callback = ast_auxiliary_callback, options=options_set)    
        else:     
            # Solve for tilting parameters
            lambda_a_res = sp.optimize.minimize(ast_crit, lambda_sv, args=(1-D, p_W, -p_W_index, t_W, NQ/rlgrz, s_wgt), \
                                                method='Newton-CG', jac=ast_foc, hess=ast_soc, \
                                                options=options_set)
    except:
        print "FATAL ERROR: exitflag = 4, Unable to compute the auxiliary/control vector of tilting parameters."
        
        # Set returnables to "None" and then exit function
        gamma_ast      = None
        vcov_gamma_ast = None
        study_test     = [None, None, None]
        auxiliary_test = [None, None, None]
        pi_eff         = None
        pi_s           = None
        pi_a           = None
        exitflag       = 4
        
        return [gamma_ast, vcov_gamma_ast, study_test, auxiliary_test, pi_eff, pi_s, pi_a, exitflag]
    
    # Collect auxiliary tilt estimation results needed below
    lambda_a_hat = -np.reshape(lambda_a_res.x,(-1,1))                          # auxiliary/control sample tilting 
                                                                               # parameter estimates 
    p_W_a = (1+np.exp(-np.dot(r_W, delta_ml) - np.dot(t_W, lambda_a_hat)))**-1 # auxiliary sample tilted p-score
    pi_a  = (1-D) * pi_eff / (1-p_W_a)                                         # auxiliary sample tilt
    
    # ----------------------------------------------------------------------------------- #
    # - STEP 4 : SOLVE FOR AST ESTIMATE OF GAMMA (i.e., ATT)                            - #
    # ----------------------------------------------------------------------------------- #

    # AST estimate of gamma -- the ATT %
    gamma_ast = np.sum(s_wgt * p_W * ((D / p_W_s) * DY - (1-D) / (1-p_W_a) * mDX))/NQ;
    
    # ----------------------------------------------------------------------------------- #
    # - STEP 5 : FORM LARGE SAMPLE VARIANCE-COVARIANCE ESTIMATES                        - #
    # ----------------------------------------------------------------------------------- #

    # Form moment vector corresponding to full three step procedure
    m1 = (s_wgt * (D - p_W) * r_W).T                          # 1+L x N matrix of m_1 moments (logit scores)
    m2 = (s_wgt * ((1 - D) / (1 - p_W_a) - 1) * p_W * t_W).T  # 1+M x N matrix of m_2 moments
    m3 = (s_wgt * (D / p_W_s - 1) * p_W * t_W).T              # 1+M x N matrix of m_3 moments
    m4 = (s_wgt * p_W * ((D / p_W_s) * DY - ((1-D) / (1-p_W_a)) * (mDX+gamma_ast))).T  # 1 x N matrix of m_4 moments
    m  = np.concatenate((m1, m2, m3, m4), axis=0)             # 1 + L + 2(1 + M) + 1 x N matrix of all moments                                                                 

    # Calculate covariance matrix of moment vector. Take into account any 
    # within-group dependence/clustering as needed
    if NG is None:
        # Case 1: No cluster dependence to account for when constructing covariance matrix
        G = N
        V_m = np.dot(m, m.T)/G
    else:
        # Case 2: Need to correct for cluster dependence when constructing covariance matrix
        G = len(NG)
        V_m = np.zeros((1+L+2*(1+M)+1,1+L+2*(1+M)+1))
        
        for g in range(0,G):
            # upper & lower bounds for the g-th group
            n1 = np.sum(NG[0:g]) 
            n2 = np.sum(NG[0:g+1])           
            
            # sum of moments for units in g-th cluster
            m_g = np.sum(m[:,n1:n2],1)
            m_g = np.reshape(m_g,(-1,1))
            
            # update variance-covariance matrix
            V_m = V_m + np.dot(m_g, m_g.T) / G    

    # Form Jacobian matrix for entire parameter: theta = (rho, delta, lambda, gamma)
    e_V  = np.exp(np.dot(r_W, delta_ml))
    e_Va = np.exp(np.dot(r_W, delta_ml) + np.dot(t_W, lambda_a_hat))
    e_Vs = np.exp(np.dot(r_W, delta_ml) + np.dot(t_W, lambda_s_hat))

    M1_delta = np.dot((s_wgt * (- e_V / (1 + e_V)**2) * r_W).T, r_W)/N                              # 1 + L x 1 + L
    M2_delta = np.dot((s_wgt * ((1 - D) / (1 - p_W_a) - 1) * (e_V / (1 + e_V)**2) * t_W).T, r_W)/N  # 1 + M x 1 + L     
    M3_delta = np.dot((s_wgt * (D / p_W_s - 1) * (e_V / (1 + e_V)**2) * t_W).T, r_W)/N              # 1 + M x 1 + L     
    M4_delta = np.dot((s_wgt * (e_V / (1 + e_V)**2) * \
                      ((D / p_W_s) * DY - ((1 - D) / (1 - p_W_a)) * (mDX + gamma_ast))).T, r_W)/N   # 1     x 1 + L    

    M2_lambda_a = np.dot(( s_wgt * ((1 - D) / (1 - p_W_a)**2) * p_W * (e_Va / (1 + e_Va)**2) * t_W).T, t_W)/N             # 1 + M x 1 + M
    M4_lambda_a = np.dot((-s_wgt * ((1 - D) / (1 - p_W_a)**2) * p_W * (mDX+gamma_ast) * (e_Va / (1 + e_Va)**2)).T, t_W)/N # 1     x 1 + M                                    

    M3_lambda_s = np.dot((-s_wgt * (D / p_W_s**2) * p_W * (e_Vs / (1 + e_Vs)**2) * t_W).T, t_W)/N  # 1 + M x 1 + M
    M4_lambda_s = np.dot((-s_wgt * (D / p_W_s**2) * p_W * DY * (e_Vs / (1 + e_Vs)**2)).T, t_W)/N   # 1     x 1 + M

    M4_gamma = -(NQ/N).reshape(1,1)                                                                               # 1     x 1  
    
    M1 = np.hstack((M1_delta, np.zeros((1+L,1+M)), np.zeros((1+L,1+M)), np.zeros((1+L,1)))) 
    M2 = np.hstack((M2_delta, M2_lambda_a,         np.zeros((1+M,1+M)), np.zeros((1+M,1))))  
    M3 = np.hstack((M3_delta, np.zeros((1+M,1+M)), M3_lambda_s,         np.zeros((1+M,1))))    
    M4 = np.hstack((M4_delta, M4_lambda_a,         M4_lambda_s,         M4_gamma))              
    
    # Concatenate Jacobian and compute inverse
    M_hat = (N/G)*np.vstack((M1, M2, M3, M4))              
    iM_hat = np.linalg.inv(M_hat)
   
    # Compute sandwich variance estimates
    vcov_theta_ast  = np.dot(np.dot(iM_hat, V_m), iM_hat.T)/G
    vcov_gamma_ast  = vcov_theta_ast[-1,-1]       
    
    exitflag = 1 # AST estimate of the ATT successfully computed!
    
    # ----------------------------------------------------------------------------------- #
    # - STEP 6 : COMPUTE TEST STATISTICS BASED ON TILTING PARAMETER                     - #
    # ----------------------------------------------------------------------------------- #
    
    # Compute propensity score specification test based on study tilt (if applicable)
    if study_tilt:
        iV_lambda_s = np.linalg.inv(vcov_theta_ast[1+L:1+L+1+M,1+L:1+L+1+M])
        ps_test_st  = np.dot(np.dot(lambda_s_hat.T, iV_lambda_s), lambda_s_hat)
        dof_st      = len(lambda_s_hat)
        pval_st     = 1 - sp.stats.chi2.cdf(ps_test_st[0,0], dof_st)
        study_test  = [ps_test_st[0,0], dof_st, pval_st]
    else:
        study_test  = [None, None, None]
        
    # Compute propensity score specification test based on auxiliary tilt (always done)
    iV_lambda_a    = np.linalg.inv(vcov_theta_ast[1+L+1+M:1+L+1+M+1+M,1+L+1+M:1+L+1+M+1+M])
    ps_test_at     = np.dot(np.dot(lambda_a_hat.T, iV_lambda_a), lambda_a_hat)
    dof_at         = len(lambda_a_hat)
    pval_at        = 1 - sp.stats.chi2.cdf(ps_test_at[0,0], dof_at)   
    auxiliary_test = [ps_test_at[0,0], dof_at, pval_at]
    
    # ----------------------------------------------------------------------------------- #
    # - STEP 7 : DISPLAY RESULTS                                                        - #
    # ----------------------------------------------------------------------------------- #
    
    if not silent:
        print ""
        print "-------------------------------------------------"
        print "- Auxiliary-to-Study (AST) estimates of the ATT -"
        print "-------------------------------------------------"
        print "ATT: "  + "%10.6f" % gamma_ast    
        print "     (" + "%10.6f" % np.sqrt(vcov_gamma_ast) + ")"
    
        if r_W_names:
            print ""
            print "-------------------------------------------------"
            print "- Maximum likelihood estimates of the p-score   -"
            print "-------------------------------------------------"
        
            c = 0
            for names in r_W_names:
                print names.ljust(25) + "%10.6f" % delta_ml[c] + \
                                 " (" + "%10.6f" % np.sqrt(vcov_theta_ast[c,c]) + ")"
                c += 1
                
        if t_W_names:
            print ""
            print "-------------------------------------------------"
            print "- Tilting parameter estimates                   -"
            print "-------------------------------------------------"
        
            if study_tilt:
                print ""
                print "Study/Treated sample tilt"
                print "---------------------------------------------------"
                
                c = 0
                for names in t_W_names:
                    print names.ljust(25) + "%10.6f" % lambda_s_hat[c] + \
                                     " (" + "%10.6f" % np.sqrt(vcov_theta_ast[1+L+c,1+L+c]) + ")"
                    c += 1
                    
                print ""
                print "Specification test for p-score (H_0 : lambda_s = 0)"
                print "---------------------------------------------------"
                print "chi-square("+str(dof_st)+") = " + "%10.6f" % ps_test_st + "   p-value: " + "% .6f" % pval_st 
                
                print ""
                print "Summary statistics study/treated re-weighting"
                print "---------------------------------------------------"
                
                j          = np.where(D)[0]        # find indices of treated units
                N_s_eff    = 1/np.sum(pi_s[j]**2)  # Kish's formula for effective sample size
                print "Kish's effective study/treated sample size = " "%0.0f" % N_s_eff
                print ""
                
                print "Percentiles of N_s * pi_s distribution"
                quantiles  = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                qnt_pi_s   = np.percentile(Ns*pi_s[j],quantiles)
            
                c = 0
                for q in quantiles:
                    print "%2.0f" % quantiles[c] + " percentile = " "%2.4f" % qnt_pi_s[c]
                    c += 1     
            
            else:
                print ""
                print "--------------------------------------------------------"
                print "- NOTE: Study tilt not computed (study_tilt = False).  -"
                print "-       Components of t(W) assumed to be also in h(W). -"
                print "--------------------------------------------------------"
                print ""
                
            print ""
            print "Auxiliary/Control sample tilt"
            print "-----------------------------"
            
            c = 0
            for names in t_W_names:
                print names.ljust(25) + "%10.6f" % lambda_a_hat[c] + \
                                 " (" + "%10.6f" % np.sqrt(vcov_theta_ast[1+L+1+M+c,1+L+1+M+c]) + ")"
                c += 1 
                
            print ""
            print "Specification test for p-score (H_0 : lambda_a = 0)"
            print "---------------------------------------------------"
            print "chi-square("+str(dof_at)+") = " + "%10.6f" % ps_test_at + "   p-value: " + "% .6f" % pval_at
            
            print ""
            print "Summary statistics auxiliary/control re-weighting"
            print "---------------------------------------------------"
                
            j          = np.where(1-D)[0]      # find indices of control units
            N_a_eff    = 1/np.sum(pi_a[j]**2)  # Kish's formula for effective sample size
            print "Kish's effective study/treated sample size = " "%0.0f" % N_a_eff
            print ""
            
            print "Percentiles of N_a * pi_a distribution"
            quantiles  = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            qnt_pi_a   = np.percentile(Na*pi_a[j],quantiles)
               
            c = 0
            for q in quantiles:
                print "%2.0f" % quantiles[c] + " percentile = " "%2.4f" % qnt_pi_a[c]
                c += 1     
            
            
            # ------------------------------------------- #
            # Construct "exact balancing" table         - #
            # ------------------------------------------- #
            
            # Compute means of t_W across various distribution function estimates
            # Mean of t(W) across controls
            mu_t_D0      = np.sum(((1-D)/Na) * t_W, axis = 0)
            mu_t_D0_std  = np.sqrt(np.sum(((1-D)/Na) * (t_W - mu_t_D0)**2, axis = 0))
            
            # Mean of t(W) across treated
            mu_t_D1      = np.sum((D/Ns) * t_W, axis = 0)
            mu_t_D1_std  = np.sqrt(np.sum((D/Ns) * (t_W - mu_t_D1)**2, axis = 0))
            
            # Semiparametrically efficient estimate of mean of t(W) across treated
            mu_t_eff     = np.sum(pi_eff * t_W, axis = 0)
            mu_t_eff_std = np.sqrt(np.sum(pi_eff * (t_W - mu_t_eff)**2, axis = 0))
            
            # Mean of t(W) across controls after re-weighting
            mu_t_a     = np.sum(pi_a * t_W, axis = 0)
            mu_t_a_std = np.sqrt(np.sum(pi_a * (t_W - mu_t_a)**2, axis = 0))
            
            # Mean of t(W) across treated after re-weighting
            mu_t_s     = np.sum(pi_s * t_W, axis = 0)
            mu_t_s_std = np.sqrt(np.sum(pi_s * (t_W - mu_t_s)**2, axis = 0))
            
            # Pre-balance table
            print ""
            print "Means & standard deviations of t_W (pre-balance)                                           "
            print "-------------------------------------------------------------------------------------------"
            print "                           D = 0                   D = 1                                   "
            print "-------------------------------------------------------------------------------------------"
            c = 0
            for names in t_W_names:
                print names.ljust(25) + "%8.4f" % mu_t_D0[c]  + " (" + "%8.4f" % mu_t_D0_std[c] + ")    " \
                                      + "%8.4f" % mu_t_D1[c]  + " (" + "%8.4f" % mu_t_D1_std[c] + ")    " 
                c += 1
            
            # Post-balance table
            print ""
            print "Means and standard deviations of t_W (post-balance)                                        "
            print "-------------------------------------------------------------------------------------------"
            print "                            Efficient (D = 1)      Auxiliary (D = 0)      Study (D = 1     "
            print "-------------------------------------------------------------------------------------------"
            c = 0
            for names in t_W_names:
                print names.ljust(25) + "%8.4f" % mu_t_eff[c]  + " (" + "%8.4f" % mu_t_eff_std[c] + ")    " \
                                      + "%8.4f" % mu_t_a[c]    + " (" + "%8.4f" % mu_t_a_std[c]   + ")    " \
                                      + "%8.4f" % mu_t_s[c]    + " (" + "%8.4f" % mu_t_s_std[c]   + ")    "
                c += 1     
    
    return [gamma_ast, vcov_gamma_ast, study_test, auxiliary_test, pi_eff, pi_s, pi_a, exitflag]