from setuptools import setup, find_packages

# Come back and update
setup(
    name='JunctionTreeChem',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # Dependencies listed in requirements.txt can also be listed here.
        'numpy',
        'pandas'
    ],
    author='Alex Jimenez',
    author_email='alexxaeljimenez@gmail.com',
    description='This package is a chemical decoder of SMILES objects into graph matricies that are feature encodings of SMILES for further downstream processing. SMILES have limitations in their information encoding since two similar smiles do not imply to similar molecular structures',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/alexjimenez99/JunctionTreeChem',
)