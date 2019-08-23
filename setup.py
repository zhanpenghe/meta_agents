from setuptools import find_packages, setup


# Required dependencies
required = [
    # Please keep alphabetized
    'garage @ https://github.com/rlworkgroup/garage/tarball/586c4eddb28422ff879ea153ae4c9e1081f00357',
    'pyprind',
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
    author="Zhanpeng He",
    author_email="zhanpeng.he@columbia.edu",
    url='https://github.com/zhanpenghe/meta_agents.git',
    packages=find_packages(),
    install_requires=required,
    extras_require=extras,
)
