import sys
from setuptools import setup

__version__ = '0.0.1'

if len(set(('test', 'easy_install')).intersection(sys.argv)) > 0:
    import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

tests_require = []
extra_setuptools_args = {}
if 'setuptools' in sys.modules:
    tests_require.append('nose')
    extra_setuptools_args = dict(
        test_suite='nose.collector',
        extras_require=dict(
            test='nose>=0.10.1')
    )

setup(
    name="bambi",
    version=__version__,
    description="BAyesian Model Building Interface in Python",
    maintainer='Tal Yarkoni',
    maintainer_email='tyarkoni@gmail.com',
    packages=["bambi"],
    tests_require=tests_require,
    license='MIT',
    **extra_setuptools_args
)
