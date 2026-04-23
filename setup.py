from setuptools import setup, find_packages

setup(
    name="map-converter",
    version="0.1.0",
    packages=find_packages(),
    py_modules=["convert_map"],
    install_requires=[
        "torchdrivesim>=0.2.2",
        "lanelet2",
        "commonroad-io",
        "lxml",
        "pyproj",
        "mgrs",
    ],
)
