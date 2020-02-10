from distutils.core import setup

setup(name='PCA',
      description='Principal Component Analysis in python',
      author='Elizabeth Armstrong',
      packages=['PCA'])

setup(name='clustering',
      description='Clustering techniques and auxiliary functions',
      author='Kamila Zdybal',
      packages=['clustering'])

setup(name='postProcessing',
    description='Post-processing tools',
    author='Kamila Zdybal',
    packages=['postProcessing'])

setup(name='trainingDataGeneration',
    description='Training data generation for machine learning models',
    author='Kamila Zdybal',
    packages=['trainingDataGeneration'])

setup(name='centerAndScale',
    description='Centering and scaling input/output for machine learning models',
    author='Kamila Zdybal',
    packages=['centerAndScale'])
