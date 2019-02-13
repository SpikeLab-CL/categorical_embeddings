from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ["scikit-learn==0.20.1",
                      "pandas==0.23.4",
                      "numpy==1.15.4",
                      "keras==2.1.6",
                      "tensorflow==1.8.0",
                      "tqdm"]

setup(
  name='categorical_embedding',
  version='0.1',
  author = 'Matias Aravena',
  author_email = 'matias@spikelab.xyz',
  install_requires=REQUIRED_PACKAGES,
  packages=find_packages(),
  include_package_data=True,
  description='create embeddings from categorical variables using Keras')