from setuptools import setup, find_packages

setup(
    name='threatsim',
    version='0.1.0',
    description='Time-series anomaly detection with transformers and uncertainty quantification',
    author='Your Name',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'torch',
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
    ],
    extras_require={
        'dev': ['jupyter', 'jupytext'],
    },
)
