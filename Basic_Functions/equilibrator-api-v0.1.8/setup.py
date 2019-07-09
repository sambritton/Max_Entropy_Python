#!/usr/bin/python3

__doc__ = "eQuilibrator API - A command-line API with minimal dependencies for calculation of standard thermodynamic potentials of biochemical reactions using the data found on eQuilibrator. Does not require any network connections."
__version__ = '0.1.8'

import os

try:
    import setuptools
except Exception as ex:
    print(ex)
    os.sys.exit(-1)

mydata_files = ['cc_compounds.json',
                'cc_preprocess.npz',
                'cofactors.csv',
                'iJO1366_reactions.csv',
                'kegg_compound_names.tsv',
                'kegg_compound_renaming.tsv'] 
data_files = [('data', 
               [os.path.join('equilibrator_api/data', f) for f in mydata_files])]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='equilibrator_api',
    version=__version__,
    description=__doc__,
    long_description=long_description,
    url='https://gitlab.com/elad.noor/equilibrator-api',
    author='Elad Noor',
    author_email='noor@imsb.biol.ethz.ch',
    license='MIT',
    packages=['equilibrator_api'],
    install_requires=[
        'numpy>=1.15.2',
        'scipy>=1.1.0',
        'optlang>=1.4.3',
        'pandas>=0.23.4',
        'nltk>=3.2.5',
        'pyparsing>=2.2.0',
        'sbtab>=0.9.49',
        'matplotlib>=3.0.0',
        ],
    include_package_data=True,
    data_files = data_files
    )
