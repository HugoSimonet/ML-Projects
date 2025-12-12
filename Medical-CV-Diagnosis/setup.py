"""
Setup script for Medical CV Diagnosis System
"""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='medical-cv-diagnosis',
    version='1.0.0',
    author='Medical AI Research Team',
    description='Production-ready medical computer vision diagnosis system',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/medical-cv-diagnosis',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
        ],
        'wandb': ['wandb>=0.15.0'],
        'monai': ['monai>=1.2.0'],
    },
    entry_points={
        'console_scripts': [
            'medical-cv-train=train:main',
            'medical-cv-eval=evaluate:main',
        ],
    },
)
