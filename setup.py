from setuptools import find_packages, setup


# Required dependencies
required = [
    # Please keep alphabetized
    'numpy',
    'dowel==0.0.2',
    'torch==1.1.0',
    'torchvision==0.3.0',
]


# Development dependencies
extras = dict()
extras['dev'] = [
    # Please keep alphabetized
    'ipdb',
    'pylint',
    'pytest>=3.6',
]


setup(
    name='meta_agents',
    packages=find_packages(),
    install_requires=required,
    extras_require=extras,
)
