import bambi as bmb
import numpy as np

from pytensor.graph.traversal import ancestors


def assert_ip_dlogp(model: bmb.Model) -> bool:
    dlogp = model.backend.model.compile_dlogp()
    ip = model.backend.model.initial_point()
    assert not np.isinf(dlogp(ip)).any()


def graph_contains_op(output, op_type):
    return any(
        isinstance(node.owner.op, op_type) for node in ancestors([output]) if node.owner is not None
    )
