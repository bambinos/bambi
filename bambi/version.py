import os

HERE = os.path.dirname(os.path.realpath(__file__))


def read_version():
    VERSION_FILE = os.path.join(HERE, "VERSION.txt")
    with open(VERSION_FILE, encoding="utf-8") as buff:
        return buff.read()


__version__ = read_version()
