import os

here = os.path.dirname(os.path.realpath(__file__))


def read_version():
    version_file = os.path.join(here, "version.txt")
    with open(version_file, encoding="utf-8") as buff:
        return buff.read().splitlines()[0]


__version__ = read_version()
