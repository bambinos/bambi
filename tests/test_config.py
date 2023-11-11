import pytest

from bambi.config import Config


def test_config():
    config = Config()

    # Evaluate getters
    assert config["INTERPRET_VERBOSE"] is True
    assert config.INTERPRET_VERBOSE is True

    config.INTERPRET_VERBOSE = False
    assert config.INTERPRET_VERBOSE is False

    # Evaluate setters
    with pytest.raises(ValueError, match="anything is not a valid value for 'INTERPRET_VERBOSE'"):
        config.INTERPRET_VERBOSE = "anything"

    with pytest.raises(KeyError, match="'DOESNT_EXIST' is not a valid configuration option"):
        config.DOESNT_EXIST = "anything"
