#!/usr/bin/python3

import setuptools

setuptools.setup(
    name="neuwon",
    version="0.0.0",
    license=open("LICENSE.txt", "rt").read(),
    description="Neuroscience simulator",
    long_description=open("README.md", "rt").read(),
    long_description_content_type="text/markdown",
    author="David McDougall",
    author_email = "dam1784@rit.edu",
    # url="https://github.com/pypa/sampleproject",
    install_requires=[
        'numpy',
        'scipy',
        'graph_algorithms',
    ],
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.6',
)
