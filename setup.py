import os
import runpy
from setuptools import setup, find_packages

# Get version
cwd = os.path.abspath(os.path.dirname(__file__))
versionpath = os.path.join(cwd, 'covasim_controller', 'version.py')
version = runpy.run_path(versionpath)['__version__']

# Get the documentation
with open(os.path.join(cwd, 'README.md'), "r") as fh:
    long_description = fh.read()

CLASSIFIERS = [
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: Other/Proprietary License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Development Status :: 4 - Beta",
    "Programming Language :: Python :: 3.7",
]

setup(
    name="inside_schools",
    version=version,
    author="Daniel Klein, Jamie Cohen, Dina Mistry, Cliff Kerr",
    author_email="covasim@idmod.org",
    description="Repository for 'Inside Schools' report",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://covasim.org',
    keywords=["COVID-19", "schools", "reopening", "testing", "covasim"],
    platforms=["OS Independent"],
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "sciris>=1.0.0",
        "covasim>=2.0.0",
        "synthpops",
        "optuna",
    ],
)
