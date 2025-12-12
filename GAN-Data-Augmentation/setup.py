"""
Setup script for GAN Data Augmentation
"""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='gan-data-augmentation',
    version='1.0.0',
    author='GAN Research Team',
    description='Advanced GAN-based data augmentation system',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/gan-data-augmentation',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
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
    },
    entry_points={
        'console_scripts': [
            'train-gan=train_gan:main',
            'evaluate-gan=evaluate:main',
            'generate-samples=generate_samples:main',
        ],
    },
)
