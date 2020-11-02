#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md', encoding='utf-8') as history_file:
    history = history_file.read()

install_requires = [
    's3fs>=0.2.2,<0.5',
    'baytune>=0.2.3,<0.3',
    'mlblocks>=0.3.0,<0.4',
    'mlprimitives>=0.2.2,<0.3',
    'mongoengine>=0.16.3,<0.17',
    'numpy>=1.15.4,<1.17',
    'pandas>=0.23.4,<0.25',
    'pymongo>=3.7.2,<4',
    'scikit-learn>=0.20.1,<0.21',
    'tabulate>=0.8.3,<0.9',
    'Keras>=2.1.6,<2.4'
]

setup_requires = [
    'pytest-runner>=2.11.1'
]

tests_require = [
    'pytest>=3.4.2',
    'pytest-cov>=2.6.0',
]

development_requires = [
    # general
    'bumpversion>=0.5.3',
    'pip>=9.0.1',
    'watchdog>=0.8.3',

    # docs
    'm2r>=0.2.0,<0.3',
    'Sphinx>=1.7.1,<3',
    'sphinx_rtd_theme>=0.2.4,<0.5',
    'autodocsumm>=0.1.10',

    # style check
    'flake8>=3.7.7',
    'isort>=4.3.4',

    # fix style issues
    'autoflake>=1.2',
    'autopep8>=1.4.3',

    # distribute on PyPI
    'twine>=1.10.0',
    'wheel>=0.30.0',

    # Advanced testing
    'coverage>=4.5.1',
    'tox>=2.9.1',
]

setup(
    author='MIT Data To AI Lab',
    author_email='dailabmit@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description='Time series forecasting using MLPrimitives',
    entry_points={
        'console_scripts': [
            'pyteller=pyteller.cli:main'
        ],
        'mlblocks': [
            'primitives=pyteller:MLBLOCKS_PRIMITIVES',
            'pipelines=pyteller:MLBLOCKS_PIPELINES'
        ],
    },
    extras_require={
        'test': tests_require,
        'dev': development_requires + tests_require,
    },
    install_package_data=True,
    install_requires=install_requires,
    license='MIT license',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='pyteller pyteller pyteller',
    name='pyteller',
    packages=find_packages(include=['pyteller', 'pyteller.*']),
    python_requires='>=3.5',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/signals-dev/pyteller',
    version='0.1.0',
    zip_safe=False,
)
