CR_DEFAULT_PRIORS = {
    "constant": {"name": "Normal", "mu": 0, "sigma": 2.5},
    "linear": {"name": "Normal", "mu": 0, "sigma": 2.5},
    "curvature": {"name": "Normal", "mu": 0, "sigma": {"name": "HalfNormal", "sigma": 1}},
}

CC_DEFAULT_PRIORS = {
    "constant": {"name": "Normal", "mu": 0, "sigma": 2.5},
    "curvature": {"name": "Normal", "mu": 0, "sigma": {"name": "HalfNormal", "sigma": 1}},
}

TP_DEFAULT_PRIORS = {
    "constant": {"name": "Normal", "mu": 0, "sigma": 2.5},
    "linear": {"name": "Normal", "mu": 0, "sigma": 2.5},
    "curvature": {"name": "Normal", "mu": 0, "sigma": {"name": "HalfNormal", "sigma": 1}},
}
