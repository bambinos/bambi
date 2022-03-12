import os
import sys

from setuptools import find_packages, setup

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
README_FILE = os.path.join(PROJECT_ROOT, "README.md")
VERSION_FILE = os.path.join(PROJECT_ROOT, "bambi", "version.py")
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, "requirements.txt")
MINIMUM_PYTHON_VERSION = (3, 7, 2)


def check_installation():
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        version = ".".join(str(i) for i in MINIMUM_PYTHON_VERSION)
        sys.stderr.write(
            f"[{sys.argv[0]}] - Error: Your Python interpreter must be {version} or greater.\n"
        )
        sys.exit(-1)


def get_long_description():
    with open(README_FILE, encoding="utf-8") as buff:
        return buff.read()


def get_requirements():
    with open(REQUIREMENTS_FILE, encoding="utf-8") as buff:
        return buff.read().splitlines()


def get_version():
    with open(VERSION_FILE, encoding="utf-8") as buff:
        exec(buff.read())  # pylint: disable=exec-used
    return vars()["__version__"]


check_installation()

__version__ = get_version()


setup(
    name="bambi",
    version=__version__,
    description="BAyesian Model Building Interface in Python",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="http://github.com/bambinos/bambi",
    download_url="https://github.com/bambinos/bambi/archive/%s.tar.gz" % __version__,
    install_requires=get_requirements(),
    maintainer="Tomas Capretto",
    maintainer_email="tomicapretto@gmail.com",
    packages=find_packages(exclude=["tests", "test_*"]),
    package_data={"bambi": ["priors/config/*"]},
    license="MIT",
)
