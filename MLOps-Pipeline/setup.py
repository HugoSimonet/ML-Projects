"""Setup script for MLOps Pipeline."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="mlops-pipeline",
    version="1.0.0",
    author="MLOps Team",
    author_email="mlops@example.com",
    description="Comprehensive MLOps pipeline for ML model deployment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mlops-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
        ],
        'docs': [
            'sphinx>=4.5.0',
            'sphinx-rtd-theme>=1.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'mlops-pipeline=pipeline.mlops_pipeline:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.yaml', '*.yml', '*.json'],
    },
)
