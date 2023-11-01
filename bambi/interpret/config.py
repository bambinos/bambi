import logging


class InterpretLogger:
    def __init__(self, messages=False):
        self.messages = messages

    def get_logger(self, name=None):
        _log = logging.getLogger(name)

        if not logging.root.handlers:
            _log.setLevel(logging.INFO)
            if len(_log.handlers) == 0:
                _log.addHandler(logging.StreamHandler())

        return _log


logger = InterpretLogger()
