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
	
	>>>> # View help file
	>>>> help(ipt.att)

	>>>> # Read nsw data directly from Rajeev Dehejia's webpage into a
	>>>> # Pandas dataframe
	>>>> import numpy as np
	>>>> import pandas as pd

	>>>> nsw=pd.read_stata("http://www.nber.org/~rdehejia/data/nsw_dw.dta")
	
	>>>> # Make some adjustments to variable definitions in experimental dataframe
	>>>> nsw['constant'] = 1                # Add constant to observational dataframe
	>>>> nsw['age']      = nsw['age']/10    # Rescale age to be in decades
	>>>> nsw['re74']     = nsw['re74']/1000 # Recale earnings to be in thousands
	>>>> nsw['re75']     = nsw['re75']/1000 # Recale earnings to be in thousands

	>>>> # Treatment indicator
	>>>> D = nsw['treat']

	>>>> # Balancing moments
	>>>> t_W = nsw[['constant','black','hispanic','education','age','re74','re75']]

	>>>> # Propensity score variables
	>>>> r_W = nsw[['constant']]

	>>>> # Outcome
	>>>> Y = nsw['re78']

	>>>> # Compute AST estimate of ATT
	>>>> [gamma_as, vcov_gamma_ast, study_test, auxiliary_test, pi_eff_nsw, pi_s_nsw, pi_a_nsw, exitflag] = \
	>>>>                                                                 ipt.att(D, Y, r_W, t_W, study_tilt=True)


CODE CITATION
---------------
Graham, Bryan S. (2016). "ipt: a Python 2.7  package for causal inference by inverse probability tilting," (Version 0.2.2) 
	[Computer program]. Available at https://github.com/bryangraham/ipt (Accessed 04 May 2016) 
	
PAPER CITATIONS
---------------
Graham, Bryan S., Cristine Pinto and Daniel Egel. (2012). “Inverse probability tilting for moment condition models 
	with missing data,” Review of Economic Studies 79 (3): 1053 - 1079

Graham, Bryan S., Cristine Pinto and Daniel Egel. (2016). “Efficient estimation of data combination models by the 
	method of auxiliary-to-study tilting (AST),” Journal of Business and Economic Statistics 31 (2): 288 - 301 	