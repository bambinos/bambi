import bambi as bmb
import numpy as np


def assert_ip_dlogp(model: bmb.Model) -> bool:
    dlogp = model.backend.model.compile_dlogp()
    ip = model.backend.model.initial_point()
    assert not np.isinf(dlogp(ip)).any()
