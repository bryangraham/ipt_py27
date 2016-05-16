ipt
--------

This package includes a Python 2.7.11 implementation of the Average Treatment Effect of the 
Treated (ATT) estimator introduced in Graham, Pinto and Egel (2016). The function att() 
allows for sampling weights as well as "clustered standard errors", but these features have not
yet been extensively tested.

An implementation of the Average Treatment Effect (ATE) estimator introduced in Graham, 
Pinto and Egel (2012) if planned for a future update.

This package is offered "as is", without warranty, implicit or otherwise. While I would
appreciate bug reports, suggestions for improvements and so on, I am unable to provide any
meaningful user-support.

Please cite the source articles listed below when using this code.

A simple example script to get started is::

    >>> import ipt
    >>> ipt.att()
    

SOURCE