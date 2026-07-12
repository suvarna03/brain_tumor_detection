from setuptools import setup, find_packages

setup(
    name= "brain_tumor_detection",
    version= "0.0.1",
    author= "Suvarna Gawali",
    author_email= "suvarna3396@gmail.com",
    packages=find_packages(where='src'),
    package_dir={"":"src"},

)