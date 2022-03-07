import codecs
import os
import sys

from setuptools import find_packages, setup

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
README_FILE = os.path.join(PROJECT_ROOT, "README.md")
VERSION_FILE = os.path.join(PROJECT_ROOT, "bambi", "version.py")
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, "requirements.txt")
REQUIRED_PYTHON_VERSION = (3, 7, 2)


def check_installation(rv):
    current_version = sys.version_info
    if (
        (current_version[0] >= rv[0])
        and (current_version[1] >= rv[1])
        and (current_version[2] >= rv[2])
    ):
        pass
    else:
        sys.stderr.write(
            f"[{sys.argv[0]}] - Error: Your Python interpreter must be "
            + f"{rv[0], rv[1], rv[0]}or greater\n"
        )
        sys.exit(-1)
    return 0


def get_long_description():
    with codecs.open(README_FILE, "rt") as buff:
        return buff.read()


def get_requirements():
    with codecs.open(REQUIREMENTS_FILE) as buff:
        return buff.read().splitlines()


# test Python version to fix NoReturnType error
check_installation(REQUIRED_PYTHON_VERSION)

with open(VERSION_FILE) as buff:
    exec(buff.read())

if len(set(("test", "easy_install")).intersection(sys.argv)) > 0:
    import setuptools

tests_require = []
extra_setuptools_args = {}
if "setuptools" in sys.modules:
    tests_require.append("nose")
    extra_setuptools_args = dict(
        test_suite="nose.collector", extras_require=dict(test="nose>=0.10.1")
    )

setup(
    name="bambi",
    version=__version__,
    description="BAyesian Model Building Interface in Python",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="http://github.com/bambinos/bambi",
    download_url="https://github.com/bambinos/bambi/archive/%s.tar.gz" % __version__,
    install_requires=get_requirements(),
    maintainer="Tal Yarkoni",
    maintainer_email="tyarkoni@gmail.com",
    packages=find_packages(exclude=["tests", "test_*"]),
    package_data={"bambi": ["priors/config/*"]},
    tests_require=tests_require,
    license="MIT",
    **extra_setuptools_args,
)
