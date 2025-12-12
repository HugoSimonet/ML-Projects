"""
Setup script for Graph Neural Networks package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="graph-neural-networks",
    version="1.0.0",
    author="ML Projects",
    author_email="",
    description="Advanced Graph Neural Networks for Social Network Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Graph-Neural-Networks",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torch-geometric>=2.0.0",
        "torch-scatter>=2.0.0",
        "torch-sparse>=0.6.0",
        "torch-cluster>=1.5.0",
        "networkx>=2.6.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "python-louvain>=0.15",
        "pyyaml>=5.4.0",
        "tensorboard>=2.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "gnn-train=train:main",
            "gnn-evaluate=evaluate:main",
            "gnn-compare=compare_models:main",
        ],
    },
)
