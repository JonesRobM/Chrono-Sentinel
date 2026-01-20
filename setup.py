from setuptools import setup, find_packages

setup(
    name='threatsim',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'pandas',
        'numpy',
        'scikit-learn',
        'transformers',
        'matplotlib',
        'seaborn',
        'jupytext',
    ],
)
