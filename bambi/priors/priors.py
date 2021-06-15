# pylint: disable=no-name-in-module
import json
import re


from copy import deepcopy
from os.path import dirname, join

import numpy as np
from statsmodels.genmod import families as genmod_families


genmod_links = genmod_families.links


class Family:
    """A specification of model family.

    Parameters
    ----------
    name : str
        Family name.
    prior : Prior
        A ``Prior`` instance specifying the model likelihood prior.
    link : str
        The name of the link function transforming the linear model prediction to a parameter of
        the likelihood.
    parent : str
        The name of the prior parameter to set to the link-transformed predicted outcome
        (e.g., ``'mu'``, ``'p'``, etc.).
    """

    def __init__(self, name, prior, link, parent):
        self.smlink = None
        fams = {
            "bernoulli": genmod_families.Binomial,
            "gamma": genmod_families.Gamma,
            "gaussian": genmod_families.Gaussian,
            "wald": genmod_families.InverseGaussian,
            "negativebinomial": genmod_families.NegativeBinomial,
            "poisson": genmod_families.Poisson,
            "t": None,  # not implemented in statsmodels
        }
        self.name = name
        self.prior = prior
        self.link = link
        self.parent = parent
        self.smfamily = fams.get(name, None)

    def _set_link(self, link):
        """Set new link function.
        It updates both ``self.link`` (a string consumed by the backend) and ``self.smlink``
        (the link instance for the statsmodel family)
        """
        links = {
            "identity": genmod_links.identity(),
            "logit": genmod_links.logit(),
            "probit": genmod_links.probit(),
            "cloglog": genmod_links.cloglog(),
            "inverse": genmod_links.inverse_power(),
            "inverse_squared": genmod_links.inverse_squared(),
            "log": genmod_links.log(),
        }
        self.link = link
        if link in links:
            self.smlink = links[link]
        else:
            raise ValueError("Link name is not supported.")

    def __str__(self):
        if self.smfamily is None:
            str_ = "No family set"
        else:
            priors_str = ",\n  ".join(
                [
                    f"{k}: {np.round_(v, 8)}" if not isinstance(v, Prior) else f"{k}: {v}"
                    for k, v in self.prior.args.items()
                    if k not in ["observed", self.parent]
                ]
            )
            str_ = f"Response distribution: {self.smfamily.__name__}\n"
            str_ += f"Link: {self.link}\n"
            str_ += f"Priors:\n  {priors_str}"
        return str_

    def __repr__(self):
        return self.__str__()


class Prior:
    """Abstract specification of a term prior.

    Parameters
    ----------
    name : str
        Name of prior distribution (e.g., ``'Normal'``, ``'Bernoulli'``, etc.)
    kwargs : dict
        Optional keywords specifying the parameters of the named distribution.
    """

    def __init__(self, name, scale=None, **kwargs):
        self.name = name
        self._auto_scale = True
        self.scale = scale
        self.args = {}
        self.update(**kwargs)

    def update(self, **kwargs):
        """Update the model arguments with additional arguments.

        Parameters
        ----------
            kwargs : dict
                Optional keyword arguments to add to prior args.
        """
        # Backends expect numpy arrays, so make sure all numeric values are represented as such.
        kwargs_ = {}
        for key, val in kwargs.items():
            if isinstance(val, (int, float)):
                val = np.array(val)
            elif isinstance(val, np.ndarray):
                val = val.squeeze()
            kwargs_[key] = val
        self.args.update(kwargs_)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        else:
            return self.__dict__ == other.__dict__

    def __str__(self):
        args = ", ".join(
            [
                f"{k}: {np.round_(v, 8)}" if not isinstance(v, Prior) else f"{k}: {v}"
                for k, v in self.args.items()
            ]
        )
        return f"{self.name}({args})"

    def __repr__(self):
        return self.__str__()


class PriorFactory:
    """An object that supports specification and easy retrieval of default priors.

    Parameters
    ----------
    defaults : str or dict
        Optional base configuration containing default priors for distribution, families, and term
        types. If a string, the name of a JSON file containing the config. If a dict, must contain
        keys for ``'dists'``, ``'terms'``, and ``'families'``; see the built-in JSON configuration
        for an example. If ``None``, a built-in set of priors will be used as defaults.
    dists : dict
        Optional specification of named distributions to use as priors. Each key gives the name of
        a newly defined distribution; values are two-element lists, where the first element is the
        name of the built-in distribution to use (``'Normal'``, ``'Cauchy',`` etc.), and the second
        element is a dictionary of parameters on that distribution
        (e.g., ``{'mu': 0, 'sigma': 10}``). Priors can be nested to arbitrary depths by replacing
        any parameter with another prior specification.
    terms : dict
        Optional specification of default priors for different model term types. Valid keys are
        ``'intercept'``, ``'common'``, or ``'group_specific'``. Values are either strings preprended
        by a #, in which case they are interpreted as pointers to distributions named in
        the dists dictionary, or key -> value specifications in the same format as elements in
        the dists dictionary.
    families : dict
        Optional specification of default priors for named family objects. Keys are family names,
        and values are dicts containing mandatory keys for ``'dist'``, ``'link'``, and ``'parent'``.

    Examples
    --------
        >>> dists = {'my_dist': ['Normal', {'mu': 10, 'sigma': 1000}]}
        >>> pf = PriorFactory(dists=dists)

        >>> families = {'normalish': {'dist': ['normal', {sigma: '#my_dist'}],
        >>>             link:'identity', parent: 'mu'}}
        >>> pf = PriorFactory(dists=dists, families=families)
    """

    def __init__(self, defaults=None, dists=None, terms=None, families=None):
        if defaults is None:
            defaults = join(dirname(__file__), "config", "priors.json")

        if isinstance(defaults, str):
            with open(defaults, "r") as ofile:
                defaults = json.load(ofile)

        # Just in case the user plans to use the same defaults elsewhere
        defaults = deepcopy(defaults)

        if isinstance(dists, dict):
            defaults["dists"].update(dists)

        if isinstance(terms, dict):
            defaults["terms"].update(terms)

        if isinstance(families, dict):
            defaults["families"].update(families)

        self.dists = defaults["dists"]
        self.terms = defaults["terms"]
        self.families = defaults["families"]

    def _get_prior(self, spec, **kwargs):
        if isinstance(spec, str):
            spec = re.sub(r"^\#", "", spec)
            return self._get_prior(self.dists[spec])
        elif isinstance(spec, (list, tuple)):
            name, args = spec
            if name.startswith("#"):
                name = re.sub(r"^\#", "", name)
                prior = self._get_prior(self.dists[name])
            else:
                prior = Prior(name, **kwargs)
            args = {k: self._get_prior(v) for (k, v) in args.items()}
            prior.update(**args)
            return prior
        else:
            return spec

    # pylint: disable=inconsistent-return-statements
    def get(self, dist=None, term=None, family=None):
        """Retrieve default prior for a named distribution, term type, or family.

        Parameters
        ----------
        dist : str
            Name of desired distribution. Note that the name is the key in the defaults dictionary,
            not the name of the ``Distribution`` object used to construct the prior.
        term : str
            The type of term family to retrieve defaults for. Must be one of ``'intercept'``,
            ``'common'``, or ``'group_specific'``.
        family : str
            The name of the ``Family`` to retrieve. Must be a value defined internally.
            In the default config, this is one of  ``'gaussian'``, ``'bernoulli'``, ``'poisson'``,
            ``'gama'``, ``'wald'``, or ``'negativebinomial'``.
        """

        if dist is not None:
            if dist not in self.dists:
                raise ValueError(f"{dist} is not a valid distribution name.")
            return self._get_prior(self.dists[dist])
        elif term is not None:
            if term not in self.terms:
                raise ValueError(f"{term} is not a valid term type.")
            return self._get_prior(self.terms[term])
        elif family is not None:
            if family not in self.families:
                raise ValueError(f"{family} is not a valid family name.")
            _f = self.families[family]
            prior = self._get_prior(_f["dist"])
            return Family(family, prior, _f["link"], _f["parent"])
