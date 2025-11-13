"""
Setup script for Multi-Modal VLM package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="multi-modal-vlm",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive multi-modal vision-language model implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/multi-modal-vlm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "transformers": [
            "transformers>=4.30.0",
        ],
        "tracking": [
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
        ],
    },
)
