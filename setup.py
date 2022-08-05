from setuptools import setup
setup(
    name="twin4build",
    python_requires='<3.8',
    version="0.0.0",
    description="A library and framework for modeling and creating Digital Twins of buildings.",
    url="https://github.com/JBjoernskov/Ifc2Graph",
    keywords="Digital Twins, Energy Modeling, Building Performance Simulation",
    author="Jakob Bjørnskov, Center for Energy Informatics SDU",
    author_email="jakob.bjornskov@me.com, jabj@mmmi.sdu.dk",
    license="BSD",
    platforms=["Windows", "Linux"],
    packages=[
        "saref",
        "saref4bldg",
        "saref4syst"
    ],
    include_package_data=True,
    install_requires=[
        "matplotlib",
        "networkx"
    ],
    classifiers=["Programming Language :: Python :: 3"],
)