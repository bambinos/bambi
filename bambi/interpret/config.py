import logging

# TODO: passing a name to the logger does not work. I think this is because of
# TODO: the "bambi" logger defined in bambi/__init__.py.


class Config:
    def __init__(self, messages=False):
        self.messages = messages

    def get_logger(self, name=None):
        return logging.getLogger(name)


logger = Config()
