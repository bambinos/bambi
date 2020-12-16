# Changelog

### 0.X.X

### New features
* Add posterior predictive sampling (#250)
* Add prior predictive sampling (#244)
* Add gamma, negativebinomial and wald families (#207)

### Maintenance and fixes
* Use pm.sample_prior_predictive function to sample and plot from prior (#238)
* Fix FutureWarning: Support for multi-dimensional indexing (#242)
* Use last version of black (#245)
* fix broken link increase Python version (#227)
* Add black style check on lint (#220)
* Some linting while re-reading library (#219)
* Remove future warning when converting the trace to InferenceData (#213)
* Include missing files for sdist (#204)
* Fixed if-else comparison that prevented HalfTStudent prior from being used (#205)
* Sidestep plotting flat priors in `plot_priors()` (#258)
* GLM.fit_constrained in automatic priors now uses start_params = None (#265)
* Categorical `Term` within `Model` now have `Term.categorical` equal to `True`(#269)
* Use logging instead of warnings (#270)
* Omits ploting group-level effects and offset variables (#276)
* Logistic regression works with no explicit index (#277)
* Add argument to optionally keep offsets in InferenceData (#288)
* Add argument to optionally keep group level effects and offsets variables in `plot_prior` (#288)

### Documentation
* Update example notebooks (#232)
* add missing notebooks (#229)
* Fix notebooks (#222)
* Clean docs (#200)
* Added notebook using Bambi and ArviZ for model comparison (#267)
* Use same color palette in all notebooks (#282)
* Fix divergences in examples (two divergences remaining in Strack RRR example) (#282)

### Deprecation
* Drop support python 3.6 (#218)
* Remove stan backend and replace sd with sigma (#205)
* Deprecate samples argument in favor of draws (#247)

### 0.2.0 The First Python 3 (and ArviZ) Bambino

### New features
* Add laplace approximation (#184) (only for educational use, do not use for real problems)
* Use arviz (#182, #178, #166, #159)

### Maintenance and fixes
* Update requirements (#191)
* Change default sd prior and update docs (#189)
* Add f-strings and support python 3.6+ (#188)
* Fix parallel sampling (#186)
* Lint code (#175, #173, #171, #167)
* Move coverage configuration to setup.cfg (#168)
* Add long description to setup.py; light linting on setup.py (#162)
* Black list external/ and tests/from pylint

### Documentation
* Add missing example (#194)
* Update docs and fix typos (#185, #181)
* Add missing items to readme and code of conduct (#180)
* Simplify readme (#179)
* Unify docstring style and remove not used code (#169)

### Deprecation
* Deprecate Stan backend (#183)

### 0.1.5 (The last legacy Python Bambino)

### New features
* Use a callable as link function (#147)

### Maintenance and fixes
* Update to Python 3, black and some pylint (#158)
* Fix test warnings (#144)
* Reorder requirements; Add matplotlib to requirements.txt (#143)
* Reorder imports; Only import necessary submodules from statsmodels (#142)
* Update travis config (#135)

### Documentation
* Add contributing guide (#146)
* Update notebooks (#140)

### Deprecation
* Last version to support Python 2.7


## 0.1.1 (2017 December 11)
* Minor release for bugfixes and minor improvements. Changes include:
* Bug that was causing an incorrect link function to be used in the PyMC3 backend when fitting logistic models.
* Fixed handling of missing values in categorical variables.
* Fixed bug in set_priors() when passing numerical values for scale.
* Improved internal handling of custom priors.
* Preliminary Sphinx docs (WIP; thanks to @ejolly).

## 0.1.0 (2017 March 31)
This is a major release that introduces several new features, significant API changes, and a large number of bug fixes and minor improvements. Notable changes include:
* Support for Stan as the sampling back-end (in addition to PyMC3), via the PyStan package.
* Dropped support for the `add_term` API; all model specification is now done via formulas.
* Expanded support for arbitrary random effects specifications; any formula now supported by patsy can be passed in as the left-hand side of a random effects specification (e.g., previously, '(a*b)|c' would not have worked).
* Completely refactored `Results` classes that no longer depend on PyMC3, providing a completely generic representation of sampler results, independent of any back-end.
* Refactored plotting and summary methods implemented on the abstract MCMCResults classes rather than at the back-end level.
* *Much* better compilation and sampling performance for models that include random effects with many levels. In many cases, performance should now be comparable to the most efficient native implementations of the models in the respective back-ends.
* All random effects priors now use the "non-centered" parameterization by default, significantly reducing bias for some models.
* Improved naming conventions that are more consistent with other packages (e.g., random effects now include the '|' operator in term names).
* Refactored `Term` class, including a separate subclass for `RandomTerm`s, and a number of other associated changes to the internal object model.
* Updated documentation and notebooks, including two new notebooks featuring well-developed examples (datasets included).
* Improved handling of NA values in continuous columns.
* Support for flat priors everywhere (by setting `auto_scale=False`).
* Numerous bug fixes and minor improvements

## 0.0.5 (2017 January 17)
* Weakly informative default priors now work the same for all response families & link functions
* Minor bug fixes/tweaks

## 0.0.4 (2016 October 11)
* Fixes referencing of Theano ops after PyMC3 namespace clean-up
* Added example Jupyter notebooks
* Improved handling of priors
* Improved prior plots and result summaries
* Improved access to MCMC trace results
* Add handling for datasets with NaN values
* Added travis-ci and coveralls support
* Minor bug fixes/tweaks

## 0.0.3 (2016 September 4)
First official release.
