# pylint: disable=no-name-in-module
# pylint: disable=too-many-lines
import re
import logging
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arviz.plots import plot_posterior
from arviz.data import from_dict
from patsy import dmatrices, dmatrix, EvalFactor
import pymc3 as pm

import bambi.version as version
from .backends import PyMC3BackEnd
from .external.patsy import Custom_NA
from .priors import Prior, PriorFactory, PriorScaler
from .utils import listify, get_bernoulli_data, extract_label

_log = logging.getLogger("bambi")


class Model:
    """Specification of model class.

    Parameters
    ----------
    data : DataFrame or str
        The dataset to use. Either a pandas DataFrame, or the name of the file containing the data,
        which will be passed to `pd.read_csv()`.
    default_priors : dict or str
        An optional specification of the default priors to use for all model terms. Either a
        dictionary containing named distributions, families, and terms (see the documentation in
        priors.PriorFactory for details), or the name of a JSON file containing the same
        information.
    auto_scale : bool
        If True (default), priors are automatically rescaled to the data (to be weakly informative)
        any time default priors are used. Note that any priors explicitly set by the user will
        always take precedence over default priors.
    dropna : bool
        When True, rows with any missing values in either the predictors or outcome are
        automatically dropped from the dataset in a listwise manner.
    taylor : int
        Order of Taylor expansion to use in approximate variance when constructing the default
        priors. Should be between 1 and 13. Lower values are less accurate, tending to undershoot
        the correct prior width, but are faster to compute and more stable. Odd-numbered values
        tend to work better. Defaults to 5 for Normal models and 1 for non-Normal models. Values
        higher than the defaults are generally not recommended as they can be unstable.
    noncentered : bool
        If True (default), uses a non-centered parameterization for normal hyperpriors on grouped
        parameters. If False, naive (centered) parameterization is used.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        data=None,
        default_priors=None,
        auto_scale=True,
        dropna=False,
        taylor=None,
        noncentered=True,
    ):

        if isinstance(data, str):
            data = pd.read_csv(data, sep=None, engine="python")

        self.default_priors = PriorFactory(default_priors)

        obj_cols = data.select_dtypes(["object"]).columns
        data[obj_cols] = data[obj_cols].apply(lambda x: x.astype("category"))
        self.data = data
        # Some group_specific effects stuff later requires us to make guesses about
        # column groupings into terms based on patsy's naming scheme.
        if re.search(r"[\[\]]+", "".join(data.columns)):
            _log.warning(
                "At least one of the column names in the specified "
                "dataset contain square brackets ('[' or ']')."
                "This may cause unexpected behavior if you specify "
                "models with group specific effects. You are encouraged to "
                "rename your columns to avoid square brackets."
            )
        self.reset()

        self.auto_scale = auto_scale
        self.dropna = dropna
        self.taylor = taylor
        self.noncentered = noncentered
        self._backend_name = None

        # build() will loop over these, calling _add() and _set_priors()
        self.added_terms = []
        self._added_priors = {}

        # if dropna=True, completes gets updated by add() to track complete cases
        self.completes = []
        self.clean_data = None

        # attributes that are set later
        self.y = None  # _add_y()
        self.family = None  # _add_y()
        self.backend = None  # _set_backend()
        self.dm_statistics = None  # build()
        self._diagnostics = None  # build()
        self.built = False  # build()

    def reset(self):
        """Reset list of terms and y-variable."""
        self.terms = OrderedDict()
        self.y = None
        self.backend = None
        self.added_terms = []
        self._added_priors = {}
        self.completes = []
        self.clean_data = None

    def _set_backend(self, backend):
        backend = backend.lower()

        if backend.startswith("pymc"):
            self.backend = PyMC3BackEnd()
        else:
            raise ValueError("At the moment, only the PyMC3 backend is supported.")

        self._backend_name = backend

    def build(self, backend="pymc"):
        """Set up the model for sampling/fitting.

        Performs any steps that require access to all model terms (e.g., scaling priors
        on each term), then calls the BackEnd's build() method.

        Parameters
        ----------
        backend : str
            The name of the backend to use for model fitting. Currently only 'pymc' is supported.
        """

        # retain only the complete cases
        n_total = len(self.data.index)
        if self.completes:
            completes = [set(x) for x in sum(self.completes, [])]
            completes = set.intersection(*completes)
        else:
            completes = range(len(self.data.index))
        self.clean_data = self.data.iloc[list(completes), :]

        # warn the user about any dropped rows
        # NOTE: When this message is shown the rows have already been removed.
        if len(completes) < n_total:
            _log.info(
                "Automatically removing %d/%d rows from the dataset.",
                n_total - len(completes),
                n_total,
            )

        # loop over the added terms and _add() them
        for term_args in self.added_terms:
            self._add(**term_args)

        # set custom priors
        self._set_priors(**self._added_priors)

        # prepare all priors
        for name, term in self.terms.items():
            type_ = (
                "intercept"
                if name == "Intercept"
                else "group_specific"
                if self.terms[name].group_specific
                else "common"
            )
            term.prior = self._prepare_prior(term.prior, type_)

        # check for backend
        if backend is None:
            if self._backend_name is None:
                raise ValueError(
                    "No backend was passed or set in the Model; did you forget to call fit()?"
                )
            backend = self._backend_name

        # check for outcome
        if self.y is None:
            raise ValueError(
                "No outcome (y) variable is set! Please specify "
                "an outcome variable using the formula interface "
                "before build() or fit()."
            )

        # X = common effects design matrix (excluding intercept/constant term)
        # r2_x = 1 - 1/VIF, i.e., R2 for predicting each x from all other x's.
        # only compute these stats if there are multiple terms in the model
        terms = [t for t in self.common_terms.values() if t.name != "Intercept"]

        if len(self.common_terms) > 1:

            x_matrix = [pd.DataFrame(x.data, columns=x.levels) for x in terms]
            x_matrix = pd.concat(x_matrix, axis=1)

            self.dm_statistics = {
                "r2_x": pd.Series(
                    {
                        x: sm.OLS(
                            endog=x_matrix[x],
                            exog=sm.add_constant(x_matrix.drop(x, axis=1))
                            if "Intercept" in self.term_names
                            else x_matrix.drop(x, axis=1),
                        )
                        .fit()
                        .rsquared
                        for x in list(x_matrix.columns)
                    }
                ),
                "sigma_x": x_matrix.std(),
                "mean_x": x_matrix.mean(axis=0),
            }

            # save potentially useful info for diagnostics
            # mat = correlation matrix of X, w/ diagonal replaced by X means
            mat = x_matrix.corr()
            for x_col in list(mat.columns):
                mat.loc[x_col, x_col] = self.dm_statistics["mean_x"][x_col]
            self._diagnostics = {
                # the Variance Inflation Factors (VIF), which is possibly
                # useful for diagnostics
                "VIF": 1 / (1 - self.dm_statistics["r2_x"]),
                "corr_mean_X": mat,
            }

            # throw informative error if perfect collinearity among common fx
            if any(self.dm_statistics["r2_x"] > 0.999):
                raise ValueError(
                    "There is perfect collinearity among the common effects!\n"
                    "Printing some design matrix statistics:\n"
                    + str(self.dm_statistics)
                    + "\n"
                    + str(self._diagnostics)
                )

        # throw informative error message if any categorical predictors have 1 category
        num_cats = [x.data.size for x in self.common_terms.values()]
        if any(np.array(num_cats) == 0):
            raise ValueError("At least one categorical predictor contains only 1 category!")

        # only set priors if there is at least one term in the model
        if self.terms:
            # Get and scale default priors if none are defined yet
            if self.taylor is not None:
                taylor = self.taylor
            else:
                taylor = 5 if self.family.name == "gaussian" else 1
            scaler = PriorScaler(self, taylor=taylor)
            scaler.scale()

        # Tell user which event is being modeled
        if self.family.name == "bernoulli":
            _log.info(
                "Modeling the probability that %s==%s",
                self.y.name,
                str(self.y.success_event),
            )

        self._set_backend(backend)
        self.backend.build(self)
        self.built = True

    def fit(
        self,
        common=None,
        group_specific=None,
        fixed=None,
        random=None,
        priors=None,
        family="gaussian",
        link=None,
        run=True,
        categorical=None,
        omit_offsets=True,
        backend="pymc",
        **kwargs,
    ):
        """Fit the model using the specified BackEnd.

        Parameters
        ----------
        common : str
            Optional formula specification of common effects.
        group_specific : list
            Optional list-based specification of group specific effects.
        priors : dict
            Optional specification of priors for one or more terms. A dict where the keys are the
            names of terms in the model, and the values are either instances of class Prior or
            ints, floats, or strings that specify the width of the priors on a standardized scale.
        family : str or Family
            A specification of the model family (analogous to the family object in R). Either a
            string, or an instance of class priors.Family. If a string is passed, a family with
            the corresponding name be defined in the defaults loaded at Model
            initialization.
            Valid pre-defined families are 'gaussian', 'bernoulli', 'poisson', and 't'.
        link : str
            The model link function to use. Can be either a string (must be one of the options
            defined in the current backend; typically this will include at least 'identity',
            'logit', 'inverse', and 'log'), or a callable that takes a 1D ndarray or theano tensor
            as the sole argument and returns one with the same shape.
        run : bool
            Whether or not to immediately begin fitting the model once any set up of passed
            arguments is complete.
        categorical : str or list
            The names of any variables to treat as categorical. Can be either a single variable
            name, or a list of names. If categorical is None, the data type of the columns in the
            DataFrame will be used to infer handling. In cases where numeric columns are to be
            treated as categoricals (e.g., group specific factors coded as numerical IDs),
            explicitly passing variable names via this argument is recommended.
        omit_offsets: bool
            Omits offset terms in the InferenceData object when the model includes
            group specific effects. Defaults to True.
        backend : str
            The name of the BackEnd to use. Currently only 'pymc' backend is supported.
        """
        if fixed is not None:
            _log.warning("The fixed argument has been deprecated, please use common")
            common = fixed
        if random is not None:
            _log.warning("The random argument has been deprecated, please use group_specific")
            group_specific = random

        if common is not None or group_specific is not None:
            self.add(
                common=common,
                group_specific=group_specific,
                priors=priors,
                family=family,
                link=link,
                categorical=categorical,
                append=False,
            )

        # Run the BackEnd to fit the model.
        if backend is None:
            backend = "pymc" if self._backend_name is None else self._backend_name

        if run:
            if not self.built or backend != self._backend_name:
                self.build(backend)
            return self.backend.run(omit_offsets=omit_offsets, **kwargs)

        self._backend_name = backend
        return None

    def add(
        self,
        common=None,
        group_specific=None,
        priors=None,
        family="gaussian",
        link=None,
        categorical=None,
        append=True,
    ):
        """Add one or more terms to the model via an R-like formula syntax.

        Parameters
        ----------
        common : str
            Optional formula specification of common effects.
        group_specific : list
            Optional list-based specification of group specific effects.
        priors : dict
            Optional specification of priors for one or more terms. A dict where the keys are the
            names of terms in the model, and the values are either instances of class Prior or
            ints, floats, or strings that specify the width of the priors on a standardized scale.
        family : str, Family
            A specification of the model family (analogous to the family object in R).
            Either a string, or an instance of class priors.Family. If a string is passed, a family
            with the corresponding name must be defined in the defaults loaded at Model
            initialization. Valid pre-defined families are 'gaussian', 'bernoulli', 'poisson',
            and 't'.
        link : str
            The model link function to use. Can be either a string (must be one of the options
            defined in the current backend; typically this will include at least 'identity',
            'logit', 'inverse', and 'log'), or a callable that takes a 1D ndarray or theano tensor
            as the sole argument and returns one with the same shape.
        categorical : str or list
            The names of any variables to treat as categorical. Can be either a single variable
            name, or a list of names. If categorical is None, the data type of the columns in the
            DataFrame will be used to infer handling. In cases where numeric columns are to be
            treated as categoricals (e.g., group specific factors coded as numerical IDs),
            explicitly passing variable names via this argument is recommended.
        append : bool
            If True, terms are appended to the existing model rather than replacing any
            existing terms. This allows formula-based specification of the model in stages.
        """
        data = self.data

        # Primitive values (floats, strs) can be overwritten with Prior objects
        # so we need to make sure to copy first to avoid bad things happening
        # if user is re-using same prior dict in multiple models.
        if priors is None:
            priors = {}
        else:
            priors = deepcopy(priors)

        if not append:
            self.reset()

        # Explicitly convert columns to category if desired--though this
        # can also be done within the formula using C().
        if categorical is not None:
            data = data.copy()
            cats = listify(categorical)
            data[cats] = data[cats].apply(lambda x: x.astype("category"))

        # Custom patsy.missing.NAAction class. Similar to patsy drop/raise
        # defaults, but changes the raised message and logs any dropped rows
        NA_handler = Custom_NA(dropna=self.dropna)

        # screen common terms
        # it deletes everything between [] and the brackets too.
        if common is not None:
            if "~" in common:
                clean_fix = re.sub(r"\[.+\]", "", common)
                dmatrices(clean_fix, data=data, NA_action=NA_handler)
            else:
                dmatrix(common, data=data, NA_action=NA_handler)

        # screen group specific terms
        if group_specific is not None:
            for term in listify(group_specific):
                for side in term.split("|"):
                    dmatrix(side, data=data, NA_action=NA_handler)

        # update the running list of complete cases
        if NA_handler.completes:
            self.completes.append(NA_handler.completes)

        # save arguments to pass to _add()
        args = dict(
            zip(
                ["common", "group_specific", "priors", "family", "link", "categorical"],
                [common, group_specific, priors, family, link, categorical],
            )
        )
        self.added_terms.append(args)

        self.built = False

    def _add(
        self,
        common=None,
        group_specific=None,
        priors=None,
        family="gaussian",
        link=None,
        categorical=None,
    ):
        """Internal version of add(), with the same arguments.

        Runs during Model.build()
        """
        # use cleaned data with NAs removed (if user requested)
        data = self.clean_data
        # alter this pandas flag to avoid false positive SettingWithCopyWarnings
        data._is_copy = False  # pylint: disable=protected-access

        # Explicitly convert columns to category if desired--though this
        # can also be done within the formula using C().
        if categorical is not None:
            data = data.copy()
            cats = listify(categorical)
            data[cats] = data[cats].apply(lambda x: x.astype("category"))

        if common is not None:
            self._add_common(common, data, family, link, priors)

        if group_specific is not None:
            self._add_group_specific(listify(group_specific), data, priors)

    # pylint: disable=keyword-arg-before-vararg
    def _add_y(self, vector, prior=None, family="gaussian", link=None, event=None):
        """Add a dependent (or outcome) variable to the model.

        Parameters
        ----------
        variable : str
            The name of the dataset column containing the y values.
        prior : Prior, int, float, str
            Optional specification of prior. Can be an instance of class Prior, a numeric value,
            or a string describing the width. In the numeric case, the distribution specified in
            the defaults will be used, and the passed value will be used to scale the appropriate
            variance parameter. For strings (e.g., 'wide', 'narrow', 'medium', or 'superwide'),
            predefined values will be used.
        family : str or Family
            A specification of the model family (analogous to the family object in R). Either a
            string, or an instance of class priors.Family. If a string is passed, a family with the
            corresponding name must be defined in the defaults loaded at Model initialization.
            Valid pre-defined families are 'gaussian', 'bernoulli', 'poisson', and 't'.
        link : str
            The model link function to use. Can be either a string (must be one of the options
            defined in the current backend; typically this will include at least 'identity',
            'logit', 'inverse', and 'log'), or a callable that takes a 1D ndarray or theano tensor
            as the sole argument and returns one with the same shape.
        """
        if isinstance(family, str):
            family = self.default_priors.get(family=family)
        self.family = family

        # Override family's link if another is explicitly passed
        if link is not None:
            self.family.link = link

        if prior is None:
            prior = self.family.prior

        variable = vector.design_info.term_names[0]

        if self.family.name == "gaussian":
            prior.update(sigma=Prior("HalfStudentT", nu=4, sigma=self.clean_data[variable].std()))

        # Success event when family = 'bernoulli'
        success_event = None
        categorical = False

        if event is not None:
            if self.family.name != "bernoulli":
                raise ValueError("Index notation only available for 'bernoulli' family")
            # pass in new Y data that has 1 if y=event and 0 otherwise
            success_event = event.group(1)
            categorical = True
            data = vector[:, vector.design_info.column_names.index(success_event)]
            # recall group(3) contains 'event' from 'y[event]' notation
            data = pd.DataFrame({event.group(3): data})
        else:
            data = self.clean_data[variable]
            if self.family.name == "bernoulli":
                categorical = True
                data, success_event = get_bernoulli_data(data)

        self.y = ResponseTerm(variable, data, categorical, prior, success_event=success_event)
        self.built = False

    def _add_common(self, common, data, family, link, priors):
        # Create design matrices and add response
        if "~" in common:
            # check to see if formula is using the 'y[event] ~ x' syntax.
            # If so, chop it into groups:
            # 1 = 'y[event]', 2 = 'y', 3 = 'event', 4 = 'x'
            # If this syntax is not being used, event = None
            event = re.match(r"^((\S+)\[(\S+)\])\s*~(.*)$", common)
            if event is not None:
                common = "{}~{}".format(event.group(2), event.group(4))
            y_vector, x_matrix = dmatrices(common, data=data, NA_action="raise")
            self._add_y(y_vector, family=family, link=link, event=event)
        else:
            x_matrix = dmatrix(common, data=data, NA_action="raise")

        # Add predictors
        self._add_common_predictors(x_matrix, priors)

    def _add_group_specific(self, group_specific, data, priors):
        for group_specific_effect in group_specific:

            group_specific_effect = group_specific_effect.strip()

            # Split specification into intercept, predictor, and grouper
            patt = r"^([01]+)*[\s\+]*([^\|]+)*\|(.*)"

            intcpt, pred, grpr = re.search(patt, group_specific_effect).groups()
            label = "{}|{}".format(pred, grpr) if pred else grpr
            prior = priors.pop(label, priors.get("group_specific", None))

            # Treat all grouping variables as categoricals, regardless of
            # their dtype and what the user may have specified in the
            # 'categorical' argument.
            var_names = re.findall(r"(\w+)", grpr)
            for var_name in var_names:
                if var_name in data.columns:
                    data.loc[:, var_name] = data.loc[:, var_name].astype("category")
                    self.clean_data.loc[:, var_name] = data.loc[:, var_name]

            # Default to including group specific intercepts
            intcpt = 1 if intcpt is None else int(intcpt)

            grpr_df = dmatrix(f"0+{grpr}", data, return_type="dataframe", NA_action="raise")

            # If there's no predictor, we must be adding group specific intercepts
            if not pred and grpr not in self.terms:
                name = "1|" + grpr
                pred = np.ones((len(grpr_df), 1))
                term = GroupSpecificTerm(
                    name, grpr_df, pred, grpr_df.values, categorical=True, prior=prior
                )
                self.terms[name] = term
            else:
                pred_df = dmatrix(
                    f"{intcpt}+{pred}", data, return_type="dataframe", NA_action="raise"
                )
                # determine value of the 'constant' attribute
                const = np.atleast_2d(pred_df.T).T.sum(1).var() == 0
                factor_infos = pred_df.design_info.factor_infos

                for col, i in pred_df.design_info.column_name_indexes.items():
                    pred_data = pred_df.iloc[:, i]
                    lev_data = grpr_df.multiply(pred_data, axis=0)

                    # Also rename intercepts and skip if already added.
                    # This can happen if user specifies something like
                    # group_specific=['1|school', 'student|school'].
                    if col == "Intercept":
                        if grpr in self.terms:
                            continue
                        label = f"1|{grpr}"
                    else:
                        label = col + "|" + grpr

                    # Delete everything between brackets and the brackets
                    col = re.sub(r"\[.*?\]\ *", "", col)
                    if EvalFactor(col) in factor_infos:
                        categorical = factor_infos[EvalFactor(col)].type == "categorical"
                    else:
                        categorical = False

                    prior = priors.pop(label, priors.get("group_specific", None))

                    pred_data = pred_data.to_numpy()
                    pred_data = pred_data[:, None]  # Must be 2D later
                    term = GroupSpecificTerm(
                        label,
                        lev_data,
                        pred_data,
                        grpr_df.values,
                        categorical=categorical,
                        constant=const if const else None,
                        prior=prior,
                    )
                    self.terms[label] = term

    def _add_common_predictors(self, x_matrix, priors):
        design_info = x_matrix.design_info

        for term in design_info.terms:
            _slice = design_info.term_slices[term]
            _name = term.name()
            cols = design_info.column_names[_slice]
            data = pd.DataFrame(np.asfortranarray(x_matrix[:, _slice]), columns=cols)

            # General for main or interaction effects.
            # Any interaction with one categorical predictor, is considered categorical.
            categorical = "categorical" in [
                design_info.factor_infos[fct].type for fct in term.factors
            ]

            prior = priors.pop(_name, priors.get("common", None))

            # If there is more than one factor, we have an interaction
            if len(term.factors) > 1:
                term = InteractionTerm(_name, data, categorical=categorical, prior=prior)
            else:
                term = Term(_name, data, categorical=categorical, prior=prior)

            self.terms[_name] = term

    def _match_derived_terms(self, name):
        """Return all (group_specific) terms whose named are derived from the specified string.

        For example, 'condition|subject' should match the terms with names '1|subject',
        'condition[T.1]|subject', and so on. Only works for strings with grouping operator ('|').
        """
        if "|" not in name:
            return None

        patt = r"^([01]+)*[\s\+]*([^\|]+)*\|(.*)"
        intcpt, pred, grpr = re.search(patt, name).groups()
        intcpt = f"1|{grpr}"
        if not pred:
            return [self.terms[intcpt]] if intcpt in self.terms else None

        source = f"{pred}|{grpr}"
        found = [
            t
            for (n, t) in self.terms.items()
            if n == intcpt or re.sub(r"(\[.*?\])", "", n) == source
        ]
        # If only the intercept matches, return None, because we want to err
        # on the side of caution and not consider '1|subject' to be a match for
        # 'condition|subject' if no slopes are found (e.g., the intercept could
        # have been set by some other specification like 'gender|subject').
        return found if found and (len(found) > 1 or found[0].name != intcpt) else None

    def set_priors(self, priors=None, common=None, group_specific=None, match_derived_names=True):
        """Set priors for one or more existing terms.

        Parameters
        ----------
        priors : dict
            Dict of priors to update. Keys are names of terms to update; values are the new priors
            (either a Prior instance, or an int or float that scales the default priors). Note that
            a tuple can be passed as the key, in which case the same prior will be applied to all
            terms named in the tuple.
        common : Prior, int, float or str
            A prior specification to apply to all common terms included in the model.
        group_specific : Prior, int, float or str
            A prior specification to apply to all group specific terms included in the model.
        match_derived_names : bool
            If True, the specified prior(s) will be applied not only to terms that match the
            keyword exactly, but to the levels of group specific effects that were derived from the
            original specification with the passed name. For example,
            `priors={'condition|subject':0.5}` would apply the prior to the terms with names
            '1|subject', 'condition[T.1]|subject', and so on. If False, an exact match is required
            for the prior to be applied.
        """
        # save arguments to pass to _set_priors() at build time
        kwargs = dict(
            zip(
                ["priors", "common", "group_specific", "match_derived_names"],
                [priors, common, group_specific, match_derived_names],
            )
        )
        self._added_priors.update(kwargs)

        self.built = False

    def _set_priors(self, priors=None, common=None, group_specific=None, match_derived_names=True):
        """Internal version of set_priors(), with same arguments.

        Runs during Model.build().
        """
        targets = {}

        if common is not None:
            targets.update({name: common for name in self.common_terms.keys()})

        if group_specific is not None:
            targets.update({name: group_specific for name in self.group_specific_terms.keys()})

        if priors is not None:
            for k, prior in priors.items():
                for name in listify(k):
                    term_names = list(self.terms.keys())
                    msg = f"No terms in model match {name}."
                    if name not in term_names:
                        terms = self._match_derived_terms(name)
                        if not match_derived_names or terms is None:
                            raise ValueError(msg)
                        for term in terms:
                            targets[term.name] = prior
                    else:
                        targets[name] = prior

        for name, prior in targets.items():
            self.terms[name].prior = prior

    def _prepare_prior(self, prior, _type):
        """Helper function to correctly set default priors, auto_scaling, etc.

        Parameters
        ----------
        prior : Prior object, or float, or None.
        _type : string
            accepted values are: 'intercept, 'common', or 'group_specific'.
        """
        if prior is None and not self.auto_scale:
            prior = self.default_priors.get(term=_type + "_flat")

        if isinstance(prior, Prior):
            prior._auto_scale = False  # pylint: disable=protected-access
        else:
            _scale = prior
            prior = self.default_priors.get(term=_type)
            prior.scale = _scale
            if prior.scale is not None:
                prior._auto_scale = False  # pylint: disable=protected-access
        return prior

    def plot(self, draws=5000, var_names=None):
        _log.warning("plot will be deprecated, please use plot_priors")
        return self.plot_priors(draws, var_names)

    def plot_priors(
        self,
        draws=5000,
        var_names=None,
        random_seed=None,
        figsize=None,
        textsize=None,
        hdi_prob=None,
        round_to=2,
        point_estimate="mean",
        kind="kde",
        bins=None,
        omit_offsets=True,
        omit_group_specific=True,
        ax=None,
    ):
        """
        Samples from the prior distribution and plot its marginals.

        Parameters
        ----------
        draws : int
            Number of draws to sample from the prior predictive distribution. Defaults to 5000.
        var_names : str or list
            A list of names of variables for which to compute the posterior predictive
            distribution. Defaults to both observed and unobserved RVs.
        random_seed : int
            Seed for the random number generator.
        figsize: tuple
            Figure size. If None it will be defined automatically.
        textsize: float
            Text size scaling factor for labels, titles and lines. If None it will be autoscaled
            based on figsize.
        hdi_prob: float, optional
            Plots highest density interval for chosen percentage of density.
            Use 'hide' to hide the highest density interval. Defaults to 0.94.
        round_to: int, optional
            Controls formatting of floats. Defaults to 2 or the integer part, whichever is bigger.
        point_estimate: Optional[str]
            Plot point estimate per variable. Values should be 'mean', 'median', 'mode' or None.
            Defaults to 'auto' i.e. it falls back to default set in rcParams.
        kind: str
            Type of plot to display (kde or hist) For discrete variables this argument is ignored
            and a histogram is always used.
        bins: integer or sequence or 'auto', optional
            Controls the number of bins, accepts the same keywords `matplotlib.hist()` does.
            Only works if `kind == hist`. If None (default) it will use `auto` for continuous
            variables and `range(xmin, xmax + 1)` for discrete variables.
        omit_offsets: bool
            Whether to omit offset terms in the plot. Defaults to True.
        omit_group_specific: bool
            Whether to omit group specific effects in the plot. Defaults to True.
        ax: numpy array-like of matplotlib axes or bokeh figures, optional
            A 2D array of locations into which to plot the densities. If not supplied, ArviZ will
            create its own array of plot areas (and return it).
        **kwargs
            Passed as-is to plt.hist() or plt.plot() function depending on the value of `kind`.

        Returns
        -------
        axes: matplotlib axes or bokeh figures
        """
        if not self.built:
            raise ValueError("Cannot plot priors until model is built!")

        unobserved_rvs_names = []
        flat_rvs = []
        for unobserved in self.backend.model.unobserved_RVs:
            if "Flat" in unobserved.__str__():
                flat_rvs.append(unobserved.name)
            else:
                unobserved_rvs_names.append(unobserved.name)
        if var_names is None:
            var_names = pm.util.get_default_varnames(
                unobserved_rvs_names, include_transformed=False
            )
        else:
            flat_rvs = [fv for fv in flat_rvs if fv in var_names]
            var_names = [vn for vn in var_names if vn not in flat_rvs]

        if flat_rvs:
            _log.info(
                "Variables %s have flat priors, and hence they are not plotted", ", ".join(flat_rvs)
            )

        if omit_offsets:
            omitted = [f"{rt}_offset" for rt in self.group_specific_terms]
            var_names = [vn for vn in var_names if vn not in omitted]

        if omit_group_specific:
            omitted = list(self.group_specific_terms)
            var_names = [vn for vn in var_names if vn not in omitted]

        axes = None
        if var_names:
            pps = self.prior_predictive(draws=draws, var_names=var_names, random_seed=random_seed)

            axes = plot_posterior(
                pps,
                group="prior",
                figsize=figsize,
                textsize=textsize,
                hdi_prob=hdi_prob,
                round_to=round_to,
                point_estimate=point_estimate,
                kind=kind,
                bins=bins,
                ax=ax,
            )
        return axes

    def prior_predictive(self, draws=500, var_names=None, omit_offsets=True, random_seed=None):
        """
        Generate samples from the prior predictive distribution.

        Parameters
        ----------
        draws : int
            Number of draws to sample from the prior predictive distribution. Defaults to 500.
        var_names : str or list
            A list of names of variables for which to compute the posterior predictive
            distribution. Defaults to both observed and unobserved RVs.
        random_seed : int
            Seed for the random number generator.

        Returns
        -------
        InferenceData
            InferenceData object with the groups prior, prior_predictive and ovserved_data.
        """
        if var_names is None:
            variables = self.backend.model.unobserved_RVs + self.backend.model.observed_RVs
            variables_names = [v.name for v in variables]
            var_names = pm.util.get_default_varnames(variables_names, include_transformed=False)

        if omit_offsets:
            offset_vars = [f"{rt}_offset" for rt in self.group_specific_terms]
            var_names = [vn for vn in var_names if vn not in offset_vars]

        pps = pm.sample_prior_predictive(
            samples=draws, var_names=var_names, model=self.backend.model, random_seed=random_seed
        )

        y_name = self.y.name

        if y_name in pps:
            prior_predictive = {y_name: np.moveaxis(pps.pop(y_name), 2, 0)}
            observed_data = {y_name: self.y.data.squeeze()}
        else:
            prior_predictive = {}
            observed_data = {}

        prior = {k: v[np.newaxis] for k, v in pps.items()}

        idata = from_dict(
            prior_predictive=prior_predictive,
            prior=prior,
            observed_data=observed_data,
            coords=self.backend.model.coords,  # new line
            attrs={
                "inference_library": self.backend.name,
                "inference_library_version": self.backend.name,
                "modeling_interface": "bambi",
                "modeling_interface_version": version.__version__,
            },
        )

        return idata

    def posterior_predictive(
        self, idata, draws=500, var_names=None, inplace=True, random_seed=None
    ):
        """
        Generate samples from the posterior predictive distribution.

        Parameters
        ----------
        idata : InfereceData
            InfereceData with samples from the posterior distribution.
        draws : int
            Number of draws to sample from the prior predictive distribution. Defaults to 500.
        var_names : str or list
            A list of names of variables for which to compute the posterior predictive
            distribution. Defaults to both observed and unobserved RVs.
        inplace : bool
            If ``True`` it will add a posterior_predictive group to idata, otherwise it will
            return a copy of idata with the added group. If true and idata already have a
            posterior_predictive group it will be overwritted
        random_seed : int
            Seed for the random number generator.

        Returns
        -------
        None or InferenceData
            When ``inplace=True`` add posterior_predictive group inplace to idata and return
            ``None`. Otherwise a copy of idata with a posterior_predictive group.

        """
        if var_names is None:
            variables = self.backend.model.observed_RVs
            variables_names = [v.name for v in variables]
            var_names = pm.util.get_default_varnames(variables_names, include_transformed=False)

        pps = pm.sample_posterior_predictive(
            trace=idata,
            samples=draws,
            var_names=var_names,
            model=self.backend.model,
            random_seed=random_seed,
        )

        if not inplace:
            idata = deepcopy(idata)
        if "posterior_predictive" in idata:
            del idata.posterior_predictive

        idata.add_groups(
            {"posterior_predictive": {k: v.squeeze()[np.newaxis] for k, v in pps.items()}}
        )

        getattr(idata, "posterior_predictive").attrs["modeling_interface"] = "bambi"
        getattr(idata, "posterior_predictive").attrs[
            "modeling_interface_version"
        ] = version.__version__
        if inplace:
            return None
        else:
            return idata

    def _get_pymc_coords(self):
        common_terms = {
            k + "_dim_0": v.cleaned_levels for k, v in self.common_terms.items() if v.categorical
        }
        # Include all group specific terms
        group_specific_terms = {
            k + "_dim_0": v.cleaned_levels for k, v in self.group_specific_terms.items()
        }
        return {**common_terms, **group_specific_terms}

    @property
    def term_names(self):
        """Return names of all terms in order of addition to model."""
        return list(self.terms.keys())

    @property
    def common_terms(self):
        """Return dict of all and only common effects in model."""
        return {k: v for (k, v) in self.terms.items() if not v.group_specific}

    @property
    def group_specific_terms(self):
        """Return dict of all and only group specific effects in model."""
        return {k: v for (k, v) in self.terms.items() if v.group_specific}


class BaseTerm:
    """Base class for all model terms"""

    group_specific = False

    def __init__(self, name, categorical, prior):
        self.name = name
        self.categorical = categorical
        self.prior = prior


class ResponseTerm(BaseTerm):
    """Representation of a single response model term.

    Parameters
    ----------
    name : str
        Name of the term.
    data : (DataFrame, Series, ndarray)
        The term values.
    categorical : bool
        If True, the source variable is interpreted as nominal/categorical. If False, the source
        variable is treated as continuous.
    prior : Prior
        A specification of the prior(s) to use. An instance of class priors.Prior.
    success_event: str or None
        Indicates the success level when the term is a categorical variable.
    """

    def __init__(self, name, data, categorical=False, prior=None, success_event=None):
        super().__init__(name, categorical, prior)

        if isinstance(data, pd.Series):
            data = data.to_frame()
        if isinstance(data, pd.DataFrame):
            self.levels = list(data.columns)
            data = data.values

        self.data = data
        self.constant = np.atleast_2d(data.T).T.sum(1).var() == 0
        self.success_event = str(success_event)
        self.clean_event()

    def clean_event(self):
        event = re.search(r"\[([\S+]+)\]", self.success_event)
        if event is not None:
            self.success_event = event.group(1)


class Term(BaseTerm):
    """Representation of a single (common) model term.

    Parameters
    ----------
    name : str
        Name of the term.
    data : (DataFrame, Series, ndarray)
        The term values.
    categorical : bool
        If True, the source variable is interpreted as nominal/categorical. If False, the source
        variable is treated as continuous.
    prior : Prior
        A specification of the prior(s) to use. An instance of class priors.Prior.
    constant : bool
        indicates whether the term levels collectively act as a constant, in which case the term is
        treated as an intercept for prior distribution purposes.
    """

    def __init__(self, name, data, categorical=False, prior=None, constant=None):
        super().__init__(name, categorical, prior)

        if isinstance(data, pd.Series):
            data = data.to_frame()

        if isinstance(data, pd.DataFrame):
            self.levels = list(data.columns)
            data = data.values
        # Group specific effects pass through here
        else:
            data = np.atleast_2d(data)
            self.levels = list(range(data.shape[1]))

        self.data = data

        # identify and flag intercept and cell-means terms (i.e., full-rank
        # dummy codes), which receive special priors
        if constant is None:
            self.constant = np.atleast_2d(data.T).T.sum(1).var() == 0
        else:
            self.constant = constant
        self.clean_levels()

    def clean_levels(self):
        self.cleaned_levels = [extract_label(level, "common") for level in self.levels]


class InteractionTerm(Term):
    """Representation of a single (common) interaction model term.

    Parameters
    ----------
    name : str
        Name of the term.
    data : (DataFrame, Series, ndarray)
        The term values.
    categorical : bool
        If True, the source variable is interpreted as nominal/categorical. If False, the source
        variable is treated as continuous.
    prior : Prior
        A specification of the prior(s) to use. An instance of class priors.Prior.
    """

    def __init__(self, name, data, categorical=False, prior=None):
        super().__init__(name, data, categorical, prior)

    def clean_levels(self):
        # Delete "T." within square brackets
        self.cleaned_levels = [re.sub("T.(?=[^[]]*\\])", "", level) for level in self.levels]


class GroupSpecificTerm(Term):
    """Representation of a single (group specific) model term.

    Parameters
    ----------
    name : str
        Name of the term.
    data : (DataFrame, Series, ndarray)
        The term values.
    predictor: (DataFrame, Series, ndarray)
        Data of the predictor variable in the group specific term.
    grouper: (DataFrame, Series, ndarray)
        Data of the grouping variable in the group specific term.
    categorical : bool
        If True, the source variable is interpreted as nominal/categorical. If False, the source
        variable is treated as continuous.
    prior : Prior
        A specification of the prior(s) to use. An instance of class priors.Prior.
    constant : bool
        indicates whether the term levels collectively act as a constant, in which case the term is
        treated as an intercept for prior distribution purposes.
    """

    group_specific = True

    def __init__(
        self, name, data, predictor, grouper, categorical=False, prior=None, constant=None
    ):
        super().__init__(name, data, categorical, prior, constant)
        self.grouper = grouper
        self.predictor = predictor
        self.group_index = self.invert_dummies(grouper)

    def clean_levels(self):
        self.cleaned_levels = [extract_label(level, "group_specific") for level in self.levels]

    def invert_dummies(self, dummies):
        """
        For the sake of computational efficiency (i.e., to avoid lots of large matrix
        multiplications in the backends), invert the dummy-coding process and represent full-rank
        dummies as a vector of indices into the coefficients.
        """
        vec = np.zeros(len(dummies), dtype=int)
        for i in range(1, dummies.shape[1]):
            vec[dummies[:, i] == 1] = i
        return vec
