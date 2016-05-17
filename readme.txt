ipt: a Python 2.7 package for causal inference by inverse probability tilting
-----------------------------------------------------------------------------
by Bryan S. Graham, UC - Berkeley, e-mail: bgraham@econ.berkeley.edu


This package includes a Python 2.7 implementation of the Average Treatment Effect of the 
Treated (ATT) estimator introduced in Graham, Pinto and Egel (2016). The function att() 
allows for sampling weights as well as "clustered standard errors", but these features have not
yet been extensively tested.

An implementation of the Average Treatment Effect (ATE) estimator introduced in Graham, 
Pinto and Egel (2012) is planned for a future update.

This package is offered "as is", without warranty, implicit or otherwise. While I would
appreciate bug reports, suggestions for improvements and so on, I am unable to provide any
meaningful user-support. Please e-mail me at bgraham@econ.berkeley.edu

Please cite both the code and the underlying source articles listed below when using this 
code in your research.

A simple example script to get started is::

	>>>> # Append location of ipt module root directory to systems path
	>>>> # NOTE: Only required ipt not "permanently" installed
	>>>> import sys
	>>>> sys.path.append('/Users/bgraham/Dropbox/Sites/software/ipt/')

	>>>> # Load ipt package
	>>>> import ipt as ipt

	>>>> # Read nsw data directly from Rajeev Dehejia's webpage into a
	>>>> # Pandas dataframe
	>>>> import numpy as np
	>>>> import pandas as pd

	>>>> nsw=pd.read_stata("http://www.nber.org/~rdehejia/data/nsw_dw.dta")

	>>>> # Extract treatment indicator
	>>>> D = nsw['treat']
	>>>> N = len(D)
	>>>> D = np.array(D).reshape(N,1)

	>>>> # Balancing moments
	>>>> t_W = nsw[['re74','re75']]/1000
	>>>> t_W['re74_sq']     = t_W['re74']**2
	>>>> t_W['re75_sq']     = t_W['re75']**2
	>>>> t_W['re74_X_re75'] = t_W['re74']*t_W['re75']
	>>>> t_W['age']         = nsw['age']/10
	>>>> t_W['education']   = nsw['education']
	>>>> t_W['black']       = nsw['black']
	>>>> t_W['hispanic']    = nsw['hispanic']
	>>>> t_W                = np.concatenate((np.ones((N,1)), t_W), axis=1)

	>>>> # Propensity score variables
	>>>> # NOTE: Propensity score assumed constant, consistent with RCT
	>>>> r_W = np.ones((N,1))

	>>>> # Outcome
	>>>> Y = nsw['re78']
	>>>> Y = np.array(Y).reshape(N,1)

	>>>> # Variable names
	>>>> ps_names  = ['constant']
	>>>> tlt_names = ['constant','earnings 1974','earnings 1975','earnings sq 1974','earnings sq 1975','earnings 74 x 75',\
             		  'age', 'education', 'black', 'hispanic']

	>>>> # Read help file for ipt.att()
	>>>> help(ipt.att)

	>>>> # Compute AST estimate of ATT
	>>>> [gamma_as, vcov_gamma_ast, study_test, auxiliary_test, pi_eff_nsw, pi_s_nsw, pi_a_nsw, exitflag] = \
         			   ipt.att(D, Y, r_W, t_W, study_tilt=True, rlgrz=0.75, NG=None,s_wgt=1, silent=False, \
                    		   r_W_names=ps_names, t_W_names=tlt_names)
    

CODE CITATION
---------------
Graham, Bryan S. (2016). "ipt: a Python 2.7  package for causal inference by inverse probability tilting," (Version 0.1) 
	[Computer program]. Available at https://github.com/bryangraham/ipt (Accessed 04 May 2016) 
	
PAPER CITATIONS
---------------
Graham, Bryan S., Cristine Pinto and Daniel Egel. (2012). “Inverse probability tilting for moment condition models 
	with missing data,” Review of Economic Studies 79 (3): 1053 - 1079


Graham, Bryan S., Cristine Pinto and Daniel Egel. (2016). “Efficient estimation of data combination models by the 
	method of auxiliary-to-study tilting (AST),” Journal of Business and Economic Statistics 31 (2): 288 - 301 	