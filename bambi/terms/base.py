from abc import ABC, abstractmethod

from bambi.priors.prior import Prior
from bambi.utils import indentify, multilinify


VALID_PRIORS = (Prior, int, float, type(None))


class BaseTerm(ABC):
    _alias = None
    _prior = None

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
    def name(self):
        ...

    @property
    @abstractmethod
    def shape(self):
        ...

    @property
    @abstractmethod
    def levels(self):
        ...

    @property
    @abstractmethod
    def categorical(self):
        ...

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

    def make_str(self, extras=None):
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
        return f"{self.__class__.__name__}({indentify(multilinify(args))}\n)"

    def __str__(self):
        return self.make_str()

    def __repr__(self):
        return self.__str__()
