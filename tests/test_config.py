import pytest

from bambi.config import Config


def test_config():
    config = Config()

    # Evaluate getters
    assert config["INTERPRET_VERBOSE"] is True
    assert config.INTERPRET_VERBOSE is True

    assert config["SPARSE_DOT"] is False
    assert config.SPARSE_DOT is False

    # Evaluate setters
    config.INTERPRET_VERBOSE = False
    assert config.INTERPRET_VERBOSE is False
    config["INTERPRET_VERBOSE"] = True
    assert config.INTERPRET_VERBOSE is True

    config.SPARSE_DOT = True
    assert config.SPARSE_DOT is True
    config["SPARSE_DOT"] = False
    assert config.SPARSE_DOT is False

    with pytest.raises(ValueError, match="anything is not a valid value for 'INTERPRET_VERBOSE'"):
        config.INTERPRET_VERBOSE = "anything"

    with pytest.raises(ValueError, match="cheese is not a valid value for 'SPARSE_DOT'"):
        config.SPARSE_DOT = "cheese"

    with pytest.raises(KeyError, match="'DOESNT_EXIST' is not a valid configuration option"):
        config.DOESNT_EXIST = "anything"


def test_config_unused_priors():
    config = Config()

    assert config["UNUSED_PRIORS"] == "warn"

    for value in ("warn", "ignore", "error"):
        config["UNUSED_PRIORS"] = value
        assert config.UNUSED_PRIORS == value

    with pytest.raises(ValueError, match="silent is not a valid value for 'UNUSED_PRIORS'"):
        config.UNUSED_PRIORS = "silent"
