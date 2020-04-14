"""minimal install configuration for pip install"""

from setuptools import find_packages, setup

setup(
    install_requires=[
        "Keras",
        "numpy",
        "scikit-learn",
        "pandas"
    ],
    name="mimic3benchmarks",
    packages=find_packages(),
    python_requires=">= 3.6"
)
