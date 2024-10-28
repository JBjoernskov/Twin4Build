from setuptools import setup
import setuptools
setup(
    name="twin4build",
    python_requires='>3.8',
    version="0.0.0",
    description="A library and framework for modeling Digital Twins of buildings.",
    url="https://github.com/JBjoernskov/Twin4Build",
    keywords="Digital Twins, Energy Modeling, Building Performance Simulation",
    author="Jakob Bjørnskov, SDU Center for Energy Informatics; Avneet, NEC India; Grzegorz Wiszniewski, KMD A/S",
    license="BSD",
    platforms=["Windows", "Linux"],
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={'': ['*.pickle', '*.xlsx', "*.pt", "*.fmu", "*.ini", "*.xlsm", "*.csv"]},
    install_requires=[
        "matplotlib",
        "networkx",
        "seaborn",
        "pandas",
        "torch",
        "openpyxl",
        "pydot<=2.0.0",
        "tqdm",
        "requests",
        "pwlf",
        "fmpy",
        "ptemcee @ git+https://github.com/willvousden/ptemcee.git@c06ffef47eaf9e371a3d629b4d28fb3cecda56b4",
        "scipy",
        "fastapi",
        "numpy",
        "george",
        "uvicorn",
        "prettytable",
        "corner",
        "jupyter",
        "nbformat",
        "pygad"
    ],
    classifiers=["Programming Language :: Python :: 3"],
)