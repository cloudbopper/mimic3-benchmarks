"""minimal install configuration for pip install"""

from setuptools import find_packages, setup

setup(
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "Keras==2.3.1",
        "tensorflow==2.1.0"
    ],
    name="mimic3benchmarks",
    packages=find_packages(),
    python_requires=">= 3.6"
)
