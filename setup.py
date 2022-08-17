from setuptools import setup
import setuptools
setup(
    name="twin4build",
    python_requires='<3.8',
    version="0.0.0",
    description="A library and framework for modeling and creating Digital Twins of buildings.",
    url="https://github.com/JBjoernskov/Twin4Build",
    keywords="Digital Twins, Energy Modeling, Building Performance Simulation",
    author="Jakob Bjørnskov, Center for Energy Informatics SDU",
    author_email="jakob.bjornskov@me.com, jabj@mmmi.sdu.dk",
    license="BSD",
    platforms=["Windows", "Linux"],
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={'': ['*.pickle']},
    install_requires=[
        "matplotlib",
        "networkx",
        "seaborn",
        "pandas",
        "torch",
        "openpyxl",
        "pydot"
    ],
    classifiers=["Programming Language :: Python :: 3"],
)