"""Setup configuration for tri-objective-robust-xai-medimg package."""

from setuptools import find_packages, setup

setup(
    name="tri-objective-robust-xai-medimg",
    version="0.1.0",
    author="Viraj Pankaj Jain",
    author_email="v.jain.1@research.gla.ac.uk",
    description="Adversarially robust and explainable deep learning for medical imaging",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "mlflow>=2.9.0",
        "dvc>=3.30.0",
    ],
)
