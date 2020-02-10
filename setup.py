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

setup(name='non-linear-regression',
    description='Non-linear regression tools',
    author='Kamila Zdybal',
    packages=['non-linear-regression.centerAndScale', 'non-linear-regression.trainingDataGeneration'])
