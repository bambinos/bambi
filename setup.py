import sys
from setuptools import setup, find_packages
from os.path import join, dirname

__version__ = '0.0.2'

if len(set(('test', 'easy_install')).intersection(sys.argv)) > 0:
    import setuptools

requirements, dependency_links = [], []
req_file = join(dirname(__file__), 'requirements.txt')
with open(req_file) as f:
    for dep in f.read().splitlines():
        if '-e' in dep:
            dependency_links.append(dep.split(' ')[1])
        else:
            requirements.append(dep)

tests_require = []
extra_setuptools_args = {}
if 'setuptools' in sys.modules:
    tests_require.append('nose')
    extra_setuptools_args = dict(
        test_suite='nose.collector',
        extras_require=dict(
            test='nose>=0.10.1')
    )

print(requirements)
setup(
    name="bambi",
    version=__version__,
    description="BAyesian Model Building Interface in Python",
    url='http://github.com/bambinos/bambi',
    download_url='https://github.com/bambinos/bambi/archive/%s.tar.gz' % __version__,
    install_requires=requirements,
    dependency_links=dependency_links,
    maintainer='Tal Yarkoni',
    maintainer_email='tyarkoni@gmail.com',
    packages=find_packages(exclude=['tests', 'test_*']),
    package_data={'bambi': ["config/*"]},
    tests_require=tests_require,
    license='MIT',
    **extra_setuptools_args
)
