import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def readlines(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).readlines()

setup(
    name='datascience_starter',
    packages=find_packages(where='./src'),
    package_dir={"": "src"},
    install_requires=readlines('requirements.txt'),
    description='An data science project start up pack.',
    author='Jordan Parker',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
)