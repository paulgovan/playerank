from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(
    name='playerank',
    version='1.0',
    packages=find_packages(),
    install_requires=[
          'pandas>=1.5.0',
          'scipy>=1.9.0',
          'numpy>=1.23.0',
          'scikit-learn>=1.2.0',
          'joblib>=1.2.0',
      ],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=long_description,
)
