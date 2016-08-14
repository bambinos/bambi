import pandas as pd
import pymc3 as pm


class ModelResults(object):

    def __init__(self, model, trace):

        self.model = model
        self.terms = list(model.terms.values())
        self.trace = trace
        self.n_terms = len(model.terms)
        self.n_samples = len(trace)
        self._fixed_terms = [t.name for t in self.terms if t.type_=='fixed']
        self._random_terms = [t.name for t in self.terms if t.type_=='random']

    def _select_samples(self, fixed, random, names, burn_in):
        trace = self.trace[burn_in:]
        if names is not None:
            names = []
            if fixed:
                names.extend(self._fixed_terms)
            if random:
                names.extend(self._random_terms)
        return trace, names

    def plot_trace(self, burn_in=0, fixed=True, random=True, names=None,
                   **kwargs):
        trace, names = self._select_samples(fixed, random, names, burn_in)
        return pm.traceplot(trace, varnames=names, **kwargs)

    def summary(self, burn_in=0, fixed=True, random=True, names=None, **kwargs):
        trace, names = self._select_samples(fixed, random, names, burn_in)
        return pm.summary(trace, varnames=names, **kwargs)

