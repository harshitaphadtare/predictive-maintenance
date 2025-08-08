#!/usr/bin/env python3
"""
Setup script for Predictive Maintenance with ML (NASA CMAPSS)
"""

from setuptools import setup, find_packages

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="predictive-maintenance-ml",
    version="1.0.0",
    author="Harshita Phadtare",
    author_email="harshita.codewiz@gmail.com",
    description="A comprehensive machine learning pipeline for predictive maintenance using NASA CMAPSS dataset",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/harshitaphadtare/predictive-maintenance",
    project_urls={
        "Bug Reports": "https://github.com/harshitaphadtare/predictive-maintenance/issues",
        "Source": "https://github.com/harshitaphadtare/predictive-maintenance",
        "Documentation": "https://github.com/harshitaphadtare/predictive-maintenance#readme",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.11",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "predictive-maintenance=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="machine-learning, predictive-maintenance, nasa, cmapss, rul, classification, regression",
)
