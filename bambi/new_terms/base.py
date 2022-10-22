from abc import ABC, abstractmethod
from types import NoneType

from bambi.priors.prior import Prior
from bambi.utils import spacify, multilinify


VALID_PRIORS = (Prior, int, float, NoneType)


class BaseTerm(ABC):
    _alias = None
    _categorical = None
    _levels = None
    _name = None
    _prior = None

    @property
    @abstractmethod
    def name(self):
        ...

    @property
    @abstractmethod
    def term(self):
        ...

    @property
    @abstractmethod
    def data(self):
        ...

    @property
    @abstractmethod
    def shape(self):
        ...


    @property
    def name(self):
        return self._name

    @property
    def term(self):
        return self._term

    @property
    def alias(self):
        return self._alias

    @alias.setter
    def alias(self, value):
        assert isinstance(value, str), "Alias must be a string"
        self._alias = value

    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, value):
        assert isinstance(value, VALID_PRIORS), f"Prior must be one of {VALID_PRIORS}"
        self._prior = value

    @property
    def ndim(self):
        return len(self.shape)

    def __str__(self, extras=None):
        args = [
            f"name: {self.name}",
            f"prior: {self.prior}",
            f"shape: {self.shape}",
            f"categorical: {self.categorical}",
        ]

        if self.alias:
            args[0] = f"{args[0]} (alias: {self.alias})"

        if self.categorical:
            args += [f"levels: {self.levels}"]

        if extras:
            args += extras
        return f"{self.__class__.__name__}({spacify(multilinify(args))}\n)"

    def __repr__(self):
        return self.__str__()
