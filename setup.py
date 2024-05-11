from setuptools import setup, find_packages
import os

__version__ = None

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open(os.path.join("src", "videogen_hub", "_version.py")) as f:
    exec(f.read(), __version__)

setup(
    name='videogen_hub',
    version=__version__,
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    description='VideoGenHub is a one-stop library to standardize the inference and evaluation of all the conditional '
                'video generation models.',
    long_description=readme,
    author='Max Ku',
    author_email='m3ku@uwaterloo.ca',
    url='https://github.com/TIGER-AI-Lab/VideoGenHub',
    license=license
)
