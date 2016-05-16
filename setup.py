from setuptools import setup

def readme():
    with open('README.txt') as f:
        return f.read()

setup(name='ipt',
      version='0.1',
      description='Probability Tilting Methods (IPT) for Causal Inference',
      long_description='Python implementations of Graham, Pinto and Egel (2012, 2016)',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Science/Research'
      ],
      keywords='Inverse Probability Tilting (IPT), Auxiliary-to-Study Tilting (AST)',
      url='http://github.com/bryangraham/ipt',
      author='Bryan S. Graham',
      author_email='bgraham@econ.berkeley.edu',
      license='MIT',
      packages=['ipt'],
      install_requires=[
          'numpy',
          'scipy',
      ],
      include_package_data=True,
      zip_safe=False)
