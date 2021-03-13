from setuptools import setup, find_packages

setup(name='chemberta',
      description='Repository for training and evaluating ChemBERTa models on chemical data.',
      license='MIT',
      packages=find_packages(),
      include_package_data=True)